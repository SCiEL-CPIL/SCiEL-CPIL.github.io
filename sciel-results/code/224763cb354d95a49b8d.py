"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 50

# EVOLVE-BLOCK-START
def construct():
    """
    Maximize ||h||_2^2 / (||h||_1 * ||h*h||_inf) for 50 nonnegative heights.
    
    Strategy: Use a 2-point adjacent cluster with ratio 2:1, L1-normalized.
    This gives a shape-based ratio of 1.25, improving over the uniform seed (1.0).
    """
    h = [0.0] * N
    # 2-point cluster with ratio 2:1, L1-normalized
    h[0] = 2.0 / 3.0
    h[1] = 1.0 / 3.0
    return h
# EVOLVE-BLOCK-END
