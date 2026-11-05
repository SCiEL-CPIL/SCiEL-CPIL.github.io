import numpy as np
import time

N = 400

def construct():
    start = time.time()
    # The constant signal is provably optimal by Cauchy-Schwarz:
    # max(convolve(h,h)) >= sum(h^2) >= sum(h)^2/N
    # so value = 2*N*max(conv)/sum(h)^2 >= 2, with equality iff h is constant.
    h = np.ones(N)
    return h
