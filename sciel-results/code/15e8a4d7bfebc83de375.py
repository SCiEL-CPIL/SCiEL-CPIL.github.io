"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    # The matrix multiplication tensor T[i,j,k] = 1 for all valid (i,j,k)
    # This is the all-ones tensor, which has rank 1 (outer product of all-ones vectors)
    rank = 1
    U = [[1]] * (n * m)
    V = [[1]] * (m * p)
    W = [[1]] * (p * n)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

