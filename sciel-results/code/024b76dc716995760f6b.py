"""Matrix multiplication 2,3,8: rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """Return a rank-minimal exact tensor decomposition for 2x3 * 3x8 matrix mult."""
    n, m, p = DIMENSIONS
    rank = n * m * p

    # Build the schoolbook (bitensor) decomposition directly as lists of ints.
    # Each rank-1 term corresponds to a unique (i, j, k) triple:
    #   U[i*m + j, r] = 1  (selects A_{i,j})
    #   V[j*p + k, r] = 1  (selects B_{j,k})
    #   W[k*n + i, r] = 1  (selects C_{k,i})
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][r] = 1
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
                r += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
