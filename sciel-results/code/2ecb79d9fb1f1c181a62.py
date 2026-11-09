import numpy as np
import time

DIMENSIONS = [3, 4, 6]
EVAL_DEADLINE_SECONDS = 118.0


# EVOLVE-BLOCK-START
def construct():
    """
    Return the schoolbook (rank-72) exact decomposition for 3x4x6 matrix multiplication.
    
    The decomposition satisfies:
      T[i, j, k] = sum_r U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
    
    For the schoolbook decomposition, each term corresponds to exactly one (i,j,k) triple.
    Rank = n*m*p = 72.
    """
    t0 = time.time()
    n, m, p = DIMENSIONS

    # Direct (schoolbook) decomposition: rank = n*m*p
    rank = n * m * p

    # Initialize factor matrices as lists of lists (integer entries)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Each rank-1 term has exactly one nonzero in each factor
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    # Safety guard
    if time.time() - t0 > EVAL_DEADLINE_SECONDS:
        raise TimeoutError("construct() exceeded deadline")

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
