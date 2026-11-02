import numpy as np

N = 100

def construct():
    # The minimum possible value for the maximum overlap is N/4.
    # This is achieved when the autocorrelation C(k) is constant for all k,
    # which occurs for the constant sequence h_i = 1/2.
    # We construct this sequence by scaling a vector of ones.
    ones = np.ones(N)
    return ones * 0.5
