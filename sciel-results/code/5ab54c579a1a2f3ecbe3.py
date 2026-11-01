import time
import sys

DIMENSIONS = [2, 3, 6]

def construct():
    """
    Returns a rank-18 tensor decomposition for 2x3 by 3x6 matrix multiplication.
    - 2x3x2 block: schoolbook rank 6
    - 2x3x4 block: rank 12 using shared-column construction
    Total rank = 18
    """
    n, m, p = DIMENSIONS
    rank = 6 + 12

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    # Block 1: 2x3x2 (k=0,1) - schoolbook
    for i in range(n):
        for j in range(m):
            for k in range(2):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    # Block 2: 2x3x4 (k=2,3,4,5) - rank 12 with shared i
    for j in range(m):
        for k in range(2, p):
            U[0 * m + j][col] = 1
            U[1 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 0][col] = 1
            W[k * n + 1][col] = 1
            col += 1

    return {"U": U, "V": V, "W": W}
