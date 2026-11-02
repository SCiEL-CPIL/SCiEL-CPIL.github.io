import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # Try rank 28
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Construct rank 28 decomposition
    # j = 2*j1 + j2, j1, j2 in {0, 1}
    # For each j1, we have a 2x2x7 block.
    # Rank of 2x2x7 is 14.
    # We use the identity that 2x2x1 has rank 2.
    # Decomposition of 2x2x1:
    # U1 = [1, 0, 0, 1], V1 = [1, 1], W1 = [1, 1]
    # U2 = [0, 1, -1, 0], V2 = [-1, 1], W2 = [-1, 1]
    # This is not correct.
    
    # I will return schoolbook to be safe.
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
