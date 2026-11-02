import numpy as np

N = 100

def construct():
    # The objective is to minimize max_k sum_i h_i (1 - h_{i+k})
    # This is equivalent to N/2 - min_k sum_i h_i h_{i+k}
    # To minimize this, we must maximize min_k C(k) where C(k) is circular autocorrelation.
    # The average value of C(k) over all k is N/4 (for sum h = N/2).
    # The minimum C(k) cannot exceed the average.
    # Equality holds if and only if C(k) is constant for all k.
    # This implies h is a constant sequence.
    # Sum h = N/2 implies h_i = 1/2 for all i.
    # This solution is valid (h in [0,1]) and achieves the theoretical lower bound of N/4.
    return np.full(N, 0.5)
