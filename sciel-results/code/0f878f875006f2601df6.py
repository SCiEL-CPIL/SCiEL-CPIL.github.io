import numpy as np

N = 100

def construct():
    # Minimize max_k sum_i h_i (1 - h_{i+k}) subject to sum h_i = N/2, 0<=h_i<=1.
    # This equals N/2 - min_k C(k) where C(k) = sum_i h_i h_{i+k} (circular).
    # The average of C(k) over all k is N/4 regardless of h (given sum h = N/2).
    # Hence min_k C(k) <= N/4, so max_k overlap >= N/4.
    # The constant sequence h_i = 1/2 for all i achieves C(k) = N/4 for every k,
    # attaining the lower bound.  Return that optimal solution.
    return np.full(N, 0.5, dtype=np.float64)
