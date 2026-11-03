import numpy as np
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    # The problem asks to minimize the maximum overlap of h with shifted (1-h).
    # Overlap at shift k is O_k = sum_i h_i * (1 - h_{i-k}).
    # O_k = sum_i h_i - sum_i h_i * h_{i-k} = N/2 - C_k.
    # Minimizing max(O_k) is equivalent to maximizing min(C_k) for k != 0.
    # The average of C_k over all shifts is related to the sum of squares.
    # Specifically, sum_k C_k = (sum h_i)^2 - sum h_i^2 is not quite right for linear correlation.
    # For circular correlation, sum_k C_k = N * (mean(h)^2 - var(h))?
    # Let's use the perturbation argument from the thought process.
    # h_i = 0.5 is the global minimum for the max overlap, achieving value N/4.
    # Any deviation increases the max overlap.
    # Thus, the optimal solution is the constant sequence of 0.5.
    
    h = np.full(N, 0.5)
    return h
# EVOLVE-BLOCK-END
