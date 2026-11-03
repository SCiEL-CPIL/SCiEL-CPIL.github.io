"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Returns the optimal sequence of step heights.
    
    The problem asks to minimize the maximum overlap of h with shifted 1-h.
    Let O_k be the overlap for shift k: sum_i h_i (1 - h_{(i+k)%N}).
    The sum of overlaps over all shifts is:
    sum_k O_k = sum_k sum_i h_i (1 - h_{(i+k)%N})
              = sum_i h_i (N - sum_k h_{(i+k)%N})
              = sum_i h_i (N - sum_j h_j)
              = sum_i h_i (N - N/2)
              = (N/2) * (N/2)
              = N^2 / 4
    
    The average overlap is (N^2 / 4) / N = N / 4.
    Therefore, the maximum overlap must be at least N / 4.
    
    For N = 100, the lower bound is 25.
    The uniform distribution h_i = 0.5 yields an overlap of:
    O_k = sum_i 0.5 * (1 - 0.5) = sum_i 0.25 = 0.25 * N = N / 4 = 25.
    
    Since the uniform distribution achieves the theoretical lower bound,
    it is the global optimum. Any deviation will increase the maximum overlap
    for some shifts, resulting in a score <= 1.0.
    """
    return np.full(N, 0.5)
# EVOLVE-BLOCK-END
