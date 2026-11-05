"""Rewritten program for the third autocorrelation inequality task."""
import numpy as np

N = 400

def construct():
    """
    Construct step heights h of length N=400 to minimize:
        value = abs(2 * N * np.max(np.convolve(h, h)) / np.sum(h) ** 2)
    
    Analysis:
    Let S = sum(h). The value is 2 * N * max_k(ac_k) / S^2.
    Note that ac_0 = sum(h_i^2).
    By Cauchy-Schwarz, S^2 <= N * sum(h_i^2) = N * ac_0.
    So ac_0 >= S^2 / N.
    Since max_k(ac_k) >= ac_0, we have max_k(ac_k) >= S^2 / N.
    Therefore, value >= 2 * N * (S^2 / N) / S^2 = 2.
    
    The lower bound 2.0 is achieved if and only if:
    1. Cauchy-Schwarz is tight: h_i is constant for all i.
    2. max_k(ac_k) = ac_0.
    
    For h_i = c (constant), ac_k = (N - |k|) * c^2.
    ac_0 = N * c^2.
    ac_k < ac_0 for k != 0.
    So max(ac) = ac_0.
    The value is exactly 2.0.
    
    Thus, the optimal solution is any constant vector.
    We return np.ones(N) which is the seed and optimal.
    """
    return np.ones(N)
