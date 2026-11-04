"""Tensor decomposition for 3x3 x 3x7 matrix multiplication."""
import time

DIMENSIONS = [3, 3, 7]
START_TIME = time.time()
TIME_LIMIT = 140

def _time_remaining():
    return TIME_LIMIT - (time.time() - START_TIME)

def _verify(U, V, W, n=3, m=3, p=7, tol=1e-9):
    rank = len(U[0]) if U else 0
    if len(U) != n*m or len(V) != m*p or len(W) != p*n:
        return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = sum(U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] for c in range(rank))
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n=3, m=3, p=7):
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
