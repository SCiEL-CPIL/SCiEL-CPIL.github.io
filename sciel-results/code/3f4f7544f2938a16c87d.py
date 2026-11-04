"""Matrix multiplication 3,3,7 - rank-minimal exact tensor decomposition."""

import time

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a valid rank-63 (schoolbook) decomposition for 3x3 by 3x7 matrix
    multiplication. The factors U, V, W have integer entries (all 0 or 1) and
    exactly decompose the matrix multiplication tensor.
    
    Shapes:
      U: (9, 63)  -- n*m = 9 rows
      V: (21, 63) -- m*p = 21 rows
      W: (21, 63) -- p*n = 21 rows, indexed as W[k*n+i, r]
    """
    n, m, p = 3, 3, 7
    rank = n * m * p  # 63

    U = [[0] * rank for _ in range(n * m)]   # 9 x 63
    V = [[0] * rank for _ in range(m * p)]   # 21 x 63
    W = [[0] * rank for _ in range(p * n)]   # 21 x 63

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
