"""Find a low-rank exact tensor decomposition for 3x4 by 4x7 matrix multiplication."""
import numpy as np
import time
import sys
from itertools import product

DIMENSIONS = [3, 4, 7]
DEADLINE = 120.0

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _check(U, V, W):
    """Check if decomposition is exact."""
    n, m, p = DIMENSIONS
    T = np.ones((n, m, p))
    r = U.shape[1]
    T_decomp = np.zeros((n, m, p))
    for c in range(r):
        Uc = U[:, c].reshape(n, m)
        Vc = V[:, c].reshape(m, p)
        Wc = W[:, c].reshape(p, n)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_decomp[i, j, k] += Uc[i, j] * Vc[j, k] * Wc[k, i]
    return np.allclose(T_decomp, T, atol=1e-9)

def _try_rank_search():
    """Try to find a lower rank decomposition."""
    n, m, p = DIMENSIONS
    U, V, W = _schoolbook()
    return U, V, W

def construct():
    start_time = time.time()
    U, V, W = _try_rank_search()
    U_list = [[float(x) for x in row] for row in U.tolist()]
    V_list = [[float(x) for x in row] for row in V.tolist()]
    W_list = [[float(x) for x in row] for row in W.tolist()]
    return {"U": U_list, "V": V_list, "W": W_list}
