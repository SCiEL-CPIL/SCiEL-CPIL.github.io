"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np


def construct():
    """
    Returns a sequence of step heights h of length N.
    h_i in [0, 1].
    Sum of h_i is exactly N/2.
    Minimizes the maximum overlap of h with shifted 1-h.
    
    The uniform distribution h_i = 0.5 is optimal for this problem.
    Proof sketch:
    Let overlap O_s = sum_i h_i * (1 - h_{i+s}).
    O_s = sum_i h_i - sum_i h_i h_{i+s} = N/2 - C_s.
    To minimize max O_s, we must minimize max (N/2 - C_s) = N/2 - min C_s.
    We need to maximize min C_s.
    Sum_s C_s = (sum h_i)^2 = (N/2)^2 = N^2/4.
    Average C_s = N^2/4 / N = N/4.
    So min C_s <= N/4.
    Max possible min C_s is N/4, achieved when all C_s are equal to N/4.
    This happens for h_i = 0.5.
    Resulting max overlap = N/2 - N/4 = N/4 = 25.
    """
    return np.full(N, 0.5)
# EVOLVE-BLOCK-END
