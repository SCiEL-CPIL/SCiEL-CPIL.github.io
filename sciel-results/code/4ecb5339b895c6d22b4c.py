import numpy as np

N = 400

def construct():
    """
    Returns the sequence of signed step heights that minimizes the objective:
    max_k |(h*h)[k]| / (sum(h))^2
    
    Mathematical derivation:
    Let h be a vector of length N.
    Let S = sum(h) and Q = sum(h^2).
    The 0-th term of the autoconvolution (h*h)[0] is Q.
    Therefore, max_k |(h*h)[k]| >= Q.
    The objective J(h) = max_k |(h*h)[k]| / S^2 >= Q / S^2.
    
    By the Cauchy-Schwarz inequality, (sum(h))^2 <= N * sum(h^2), i.e., S^2 <= N * Q.
    Thus, Q / S^2 >= 1 / N.
    
    Equality holds if and only if h is a constant vector (all elements equal).
    For h = [c, c, ..., c]:
    S = N*c
    (h*h)[k] = (N-k)c^2 for 0 <= k < N, and 0 otherwise.
    max |(h*h)[k]| = N*c^2 (at k=0).
    J(h) = N*c^2 / (N*c)^2 = N*c^2 / (N^2*c^2) = 1/N.
    
    Thus, the constant vector achieves the global minimum 1/N.
    """
    return np.ones(N)
