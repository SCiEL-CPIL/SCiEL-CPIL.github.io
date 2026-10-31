"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Constructs the optimal signed step function.
    
    The metric is max(|autoconv|) / (sum(signs))^2.
    For binary signs {-1, 1}, the autoconvolution at lag 0 is sum(s_i^2) = N.
    Thus, max(|autoconv|) >= N.
    The denominator (sum(s_i))^2 is maximized when all s_i are identical,
    yielding N^2.
    The minimum possible ratio is therefore N / N^2 = 1/N.
    This lower bound is achieved by constant sequences (all 1s or all -1s).
    """
    return np.ones(N, dtype=np.int32)
# EVOLVE-BLOCK-END
