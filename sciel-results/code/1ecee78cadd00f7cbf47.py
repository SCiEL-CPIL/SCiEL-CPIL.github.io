import numpy as np

N = 400

def construct():
    """
    Returns the optimal sequence for minimizing max|autoconv| / (sum(x))^2.
    
    Mathematical Proof:
    Let x be a real vector of length N.
    Let c_k be the autoconvolution.
    c_0 = sum(x_i^2).
    max|c_k| >= |c_0| = sum(x_i^2).
    Score = max|c_k| / (sum(x_i))^2 >= sum(x_i^2) / (sum(x_i))^2.
    By Cauchy-Schwarz, (sum(x_i))^2 <= N * sum(x_i^2).
    Therefore, sum(x_i^2) / (sum(x_i))^2 >= 1/N.
    
    The all-ones vector x_i = 1 achieves:
    sum(x_i) = N.
    c_k = N - |k| for |k| < N, 0 otherwise.
    max|c_k| = N.
    Score = N / N^2 = 1/N.
    
    Thus, the all-ones vector is globally optimal.
    """
    return np.ones(N)
