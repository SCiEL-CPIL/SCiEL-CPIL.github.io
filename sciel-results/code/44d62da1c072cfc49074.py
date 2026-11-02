import numpy as np
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a sequence of step heights h of length N=100, with h_i in [0, 1],
    such that sum(h) = N/2 = 50.
    
    The goal is to minimize the maximum overlap O_k = sum(h_i * (1 - h_{i+k})) for k in 1..N-1.
    
    Analysis:
    Let g_i = 2*h_i - 1. Then g_i in [-1, 1] and sum(g_i) = 0.
    O_k = sum(h_i * (1 - h_{i+k}))
        = sum( (1+g_i)/2 * (1-g_{i+k})/2 )
        = 1/4 * sum( 1 - g_i*g_{i+k} + g_i - g_{i+k} )
        = 1/4 * ( N - sum(g_i*g_{i+k}) + sum(g_i) - sum(g_{i+k}) )
    Since sum(g_i) = 0, O_k = N/4 - 1/4 * C_k, where C_k is the autocorrelation of g at lag k.
    
    We want to minimize max(O_k), which is equivalent to minimizing max( N/4 - C_k/4 )
    = N/4 - min(C_k)/4.
    
    To minimize this, we need to maximize min(C_k).
    
    We know that sum_{k=1}^{N-1} C_k = -sum_{i=0}^{N-1} g_i^2.
    Since g_i^2 >= 0, the sum of autocorrelations is <= 0.
    Therefore, min(C_k) <= average(C_k) = -sum(g_i^2)/(N-1) <= 0.
    
    The maximum possible value for min(C_k) is 0, which occurs if and only if
    sum(g_i^2) = 0, which implies g_i = 0 for all i.
    This corresponds to h_i = 0.5 for all i.
    
    In this case, C_k = 0 for all k, and O_k = N/4 = 25.
    Any deviation from h_i = 0.5 will result in sum(g_i^2) > 0, implying min(C_k) < 0,
    and thus O_max > 25.
    
    Therefore, the constant sequence h_i = 0.5 is the global optimum.
    """
    return np.full(N, 0.5)
# EVOLVE-BLOCK-END
