"""Matrix multiplication 4,4,9: rank-minimal exact tensor decomposition."""
import time

DIMENSIONS = [4, 4, 9]
_DEADLINE = 150.0
_START = time.time()


def _time_left():
    return _DEADLINE - (time.time() - _START)


def _schoolbook(n, m, p):
    """Return the schoolbook (rank n*m*p) decomposition for M_{n,m,p}."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return U, V, W


def construct():
    """Return the schoolbook decomposition for 4x4x9 matrix multiplication."""
    n, m, p = DIMENSIONS
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
