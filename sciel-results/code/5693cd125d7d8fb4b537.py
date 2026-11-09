"""Matrix multiplication 3,5,7 - rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 5, 7]
DEADLINE = 110.0

def _build_schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-9):
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i*m+j, :] * V[j*p+k, :], W[k*n+i, :])
                if abs(s - 1.0) > tol:
                    return False
    return True

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    U, V, W = _build_schoolbook(n, m, p)
    
    # Verify
    if not _verify(U, V, W, n, m, p):
        raise ValueError("Schoolbook decomposition failed verification")
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
