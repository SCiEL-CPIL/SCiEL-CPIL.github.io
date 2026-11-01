"""Program for 2x5x5 matrix multiplication tensor decomposition.

Finds an exact tensor decomposition (U, V, W) such that for all A (2x5), B (5x5):
    C = A @ B   where C[i,k] = sum_j A[i,j]*B[j,k]
and the output factor follows the official trace convention W[k*n + i, r].

The schoolbook (trivial) decomposition has rank n*m*p = 50 and is guaranteed
exact. We keep this guaranteed-correct construction as the baseline and guard
against the wall-clock limit.
"""
import time

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p  # schoolbook rank, guaranteed exact
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
# EVOLVE-BLOCK-END
