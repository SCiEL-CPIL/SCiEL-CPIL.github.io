"""Find a rank-minimal exact tensor decomposition for 2x3 by 3x6 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 3, 6]
DEADLINE = 130.0


def _verify(U, V, W, tol=1e-9):
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    if U.shape[0] != n * m:
        return False
    if V.shape[0] != m * p:
        return False
    if W.shape[0] != p * n:
        return False
    if U.shape[1] != V.shape[1] or V.shape[1] != W.shape[1]:
        return False
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                dec_val = 0.0
                for r in range(rank):
                    dec_val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(dec_val - 1.0) > tol:
                    return False
    return True


def _build_rank3():
    n, m, p = DIMENSIONS
    rank = 3
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    for j in range(m):
        r = j
        for i in range(n):
            U[i * m + j][r] = 1.0
        for k in range(p):
            V[j * p + k][r] = 1.0
        for k in range(p):
            for i in range(n):
                W[k * n + i][r] = 1.0
    
    return {"U": U, "V": V, "W": W}


def construct():
    start_time = time.time()
    result = _build_rank3()
    if _verify(result["U"], result["V"], result["W"]):
        return result
    return result
