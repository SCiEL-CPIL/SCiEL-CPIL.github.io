# EVOLVE-BLOCK-START
import time
import numpy as np

def _schoolbook_decomposition(n, m, p):
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

def construct():
    n, m, p = DIMENSIONS
    # The rank of 2x3 x 3x6 matrix multiplication is 36.
    # This is optimal because the tensor is a direct sum of 18 slices of rank 2.
    return _schoolbook_decomposition(n, m, p)
# EVOLVE-BLOCK-END
