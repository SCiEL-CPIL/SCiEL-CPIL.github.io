"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np


def construct():
    """
    Construct a signed equal-width step function that minimizes the maximum
    absolute autoconvolution divided by the squared integral.
    
    Strategy: Analytical Optimization.
    The metric is M = max_k |C_k| / (S)^2, where C_k is the autoconvolution
    at lag k and S is the sum of the sequence (integral).
    
    For a binary sequence x_i in {-1, 1}, the zero-lag autoconvolution is
    C_0 = sum(x_i^2) = N. Thus, max_k |C_k| >= N.
    
    To minimize M, we need to maximize the denominator S^2 while keeping the
    numerator as low as possible. Since the numerator is bounded below by N,
    we simply maximize |S|. The maximum possible sum S is N, achieved when
    all x_i = 1 (or all -1).
    
    In this case, M = N / N^2 = 1/N.
    Any deviation from the constant sequence reduces |S|, which increases M.
    Therefore, the constant sequence is the global optimum.
    """
    return np.ones(N)
# EVOLVE-BLOCK-END
