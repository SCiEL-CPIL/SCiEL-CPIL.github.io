"""Direct analytic construction for signed autocorrelation minimum task."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def construct():
    """
    Returns signed step heights that minimize the maximum absolute autoconvolution
    divided by squared integral.
    
    Analysis:
    Let h be the sequence of length N.
    Let S = sum(h) and Q = sum(h^2).
    The autoconvolution at lag 0 is c_0 = Q.
    The maximum absolute autoconvolution is max_k |c_k| >= c_0 = Q.
    The metric is M = max_k |c_k| / S^2.
    Thus M >= Q / S^2.
    By Cauchy-Schwarz, S^2 <= N * Q, so Q / S^2 >= 1/N.
    Therefore, M >= 1/N.
    
    A constant sequence h_i = 1 for all i achieves:
    S = N, Q = N.
    c_k = N - |k| for |k| < N.
    max_k |c_k| = N (at k=0).
    M = N / N^2 = 1/N.
    
    Since the lower bound is achieved, the constant sequence is the global optimum.
    """
    t0 = time.time()
    
    # The optimal solution is the constant sequence.
    h = np.full(N, 1.0, dtype=np.float64)
    
    # Safety check to ensure we return before timeout, though computation is O(1).
    if time.time() - t0 > 110:
        return h
    
    return h
# EVOLVE-BLOCK-END
