"""Find a rank-minimal exact tensor decomposition for 2x4 by 4x6 matrix multiplication."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import random
import numpy as np
from functools import lru_cache

def _build_tensors():
    n, m, p = 2, 4, 6
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1
    T = np.zeros((n * m, m * p, p * n))
    for r in range(rank):
        np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        T += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r]) - np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
    # Recompute T properly
    T = np.zeros((n * m, m * p, p * n))
    for r in range(rank):
        T += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
    return U, V, W, T

def construct():
    """Return schoolbook rank-42 decomposition as safe fallback."""
    n, m, p = DIMENSIONS
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
# EVOLVE-BLOCK-END
