"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    # Try a block sequence: 0s then 1s
    # This minimizes linear asymmetric overlap
    h = np.zeros(N)
    h[N//2:] = 1.0
    
    # Verify sum
    assert np.sum(h) == N/2
    
    return h
# EVOLVE-BLOCK-END
