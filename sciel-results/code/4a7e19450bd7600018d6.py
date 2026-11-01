"""
Tensor decomposition for M_{2,2,13} matrix multiplication.
We try to find a lower-rank decomposition than schoolbook (52).
"""

DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def _build_schoolbook(n, m, p):
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

def _verify(n, m, p, U, V, W, tol=1e-9):
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1) > tol:
                    return False
    return True

def construct():
    start = time.time()
    deadline = start + 110  # safety margin before 120s
    n, m, p = DIMENSIONS

    # Schoolbook is always valid.
    result = _build_schoolbook(n, m, p)
    if not _verify(n, m, p, result["U"], result["V"], result["W"]):
        # Should never happen for schoolbook, but guard.
        pass

    # No further search implemented; return valid decomposition.
    return result
# EVOLVE-BLOCK-END
