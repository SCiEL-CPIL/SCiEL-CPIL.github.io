import numpy as np

N = 400

def construct():
    """
    Returns the sequence of signed step heights.
    
    Analysis:
    The objective is to minimize J(h) = max_k |C_k| / (sum(h))^2.
    We know that C_0 = sum(h_i^2).
    Thus J(h) >= sum(h_i^2) / (sum(h_i))^2.
    By Cauchy-Schwarz inequality, (sum(h_i))^2 <= N * sum(h_i^2).
    Therefore, J(h) >= 1/N.
    Equality holds if and only if h_i is constant for all i.
    Thus, the global minimum is 1/N, achieved by h = [c, c, ..., c] for any non-zero c.
    We choose c=1 for simplicity.
    """
    return np.ones(N, dtype=np.float64)
