"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # The target tensor is the all-ones tensor J_{i,j,k} = 1.
    # The rank of the all-ones tensor is 1.
    # U = ones(n*m, 1), V = ones(m*p, 1), W = ones(p*n, 1)
    
    U = np.ones((n * m, 1), dtype=int)
    V = np.ones((m * p, 1), dtype=int)
    W = np.ones((p * n, 1), dtype=int)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
