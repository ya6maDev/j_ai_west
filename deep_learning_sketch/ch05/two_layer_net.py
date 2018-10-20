# coding: utf-8
# ----------------------------------------------
# 誤差逆伝播法によるニューラルネットワークの構築
# ----------------------------------------------

import sys
import os
# 親ディレクトリのファイルをインポートするための設定
sys.path.append(os.pardir)
import numpy as np
from common.layers import *
from common.gradient import numerical_gradient
from collections import OrderedDict


class TwoLayerNet:

    # -------------------------------------------------
    # __init__:初期化を行う
    #     @self
    #     @input_size:入力層のニューロンの数
    #     @hidden_size:隠れ層のニューロンの数
    #     @output_size:出力層のニューロンの数
    #     @weight_init_std:重み初期化時のガウス分布スケール
    # -------------------------------------------------
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 重みの初期化
        self.params = {}
        self.params['W1'] = weight_init_std * \
            np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * \
            np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        # レイヤの生成
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])

        self.lastLayer = SoftmaxWithLoss()

    # -------------------------------------------------
    # predict:認識（推論）を行う
    #     @self
    #     @x:画像データ（入力データ）
    # -------------------------------------------------
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    # -------------------------------------------------
    # loss:損失関数を求める
    #     @self
    #     @x:画像データ（入力データ）
    #     @t:教師データ
    # -------------------------------------------------
    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)

    # -------------------------------------------------
    # accuracy:認識精度を求める
    #     @self
    #     @x:画像データ（入力データ）
    #     @t:教師データ
    # -------------------------------------------------
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    # -------------------------------------------------
    # numerical_gradient:重みパラメータに対する勾配を数値微分によって求める（〜４章までと同様）
    #     @self
    #     @x:画像データ（入力データ）
    #     @t:教師データ
    # -------------------------------------------------
    def numerical_gradient(self, x, t):
        def loss_W(W): return self.loss(x, t)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads

    # -------------------------------------------------
    # gradient:重みパラメータに対する勾配を誤差逆伝播法によって求める
    #     @self
    #     @x:画像データ（入力データ）
    #     @t:教師データ
    # -------------------------------------------------
    def gradient(self, x, t):
        # forward:順伝播
        self.loss(x, t)

        # backward:逆伝播
        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 設定
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db

        return grads
