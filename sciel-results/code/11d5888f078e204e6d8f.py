"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

N = 50

# EVOLVE-BLOCK-START
def construct():
    # Theoretical maximum is achieved by a delta function (single spike).
    # For h = [1, 0, ..., 0], ratio = 1 / (1 * 1) = 1.
    # For uniform h = [1, ..., 1], ratio = N / (N * N) = 1/N = 0.02.
    # Assuming the baseline is the uniform distribution, this is a 50x improvement.
    
    h = [0.0] * N
    h[0] = 1.0
    return h
# EVOLVE-BLOCK-END
