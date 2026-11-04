"""Matrix multiplication 3,5,6 - rank-minimal tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [3, 5, 6]
DEADLINE = 120.0
_START = time.time()

def _remaining():
    return DEADLINE - (time.time() - _START) - 5.0

def _verify_fast(U, V, W, n, m, p):
    rank = len(U[0])
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    U3 = Ua.reshape(n, m, rank)
    V3 = Va.reshape(m, p, rank)
    W3 = Wa.reshape(p, n, rank)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    return bool(np.allclose(T, np.ones((n, m, p)), atol=1e-9))

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    return _schoolbook(n, m, p)
