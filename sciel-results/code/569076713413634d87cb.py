"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 100

# EVOLVE-BLOCK-START

def construct():
    """
    Construct the step heights h to minimize the maximum overlap of h with shifted 1-h.
    
    Theoretical analysis:
    - For any h with sum N/2 and 0 <= h_i <= 1:
      C(k) = sum_i h[i] * (1 - h[i+k]) for circular shift k
      Sum over all k: sum_k C(k) = (N/2) * (N - N/2) = N^2/4
      Average C(k) = N/4
      Therefore max_k C(k) >= N/4
    - Uniform h_i = 0.5 gives C(k) = N * 0.25 = N/4 for all k
    - So uniform achieves the lower bound and is optimal.
    """
    h = np.full(N, 0.5)
    return h

# EVOLVE-BLOCK-END
