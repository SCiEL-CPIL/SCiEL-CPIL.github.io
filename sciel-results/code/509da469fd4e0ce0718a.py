"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    # Use a linear ramp to minimize linear shift overlap
    # h_i = i / (N-1)
    h = np.linspace(0, 1, N)
    return h
# EVOLVE-BLOCK-END
