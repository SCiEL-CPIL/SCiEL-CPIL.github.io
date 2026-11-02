import numpy as np
import time

DIMENSIONS = [2, 3, 10]
TIME_LIMIT = 110  # seconds
_START_TIME = time.time()

def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)

def construct():
    """
    Returns a rank-55 exact tensor decomposition for 2x3 by 3x10 matrix multiplication.
    """
    if _time_remaining() < 5:
        return _construct_default()
    
    return _construct_default()

def _construct_default():
    """Fallback: construct the rank-55 decomposition without time checks."""
    n, m, p = DIMENSIONS
    rank = 55

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    a00, a01, a10, a11 = 0, 1, 3, 4
    a02, a12 = 2, 5

    for g in range(5):
        k0 = 2 * g
        k1 = 2 * g + 1
        b00, b01 = 0*p+k0, 0*p+k1
        b10, b11 = 1*p+k0, 1*p+k1
        b20, b21 = 2*p+k0, 2*p+k1
        c00, c01 = k0*n+0, k1*n+0
        c10, c11 = k0*n+1, k1*n+1

        U[a00][col] = 1; U[a11][col] = 1
        V[b00][col] = 1; V[b11][col] = 1
        W[c00][col] = 1; W[c11][col] = 1
        col += 1

        U[a10][col] = 1; U[a11][col] = 1
        V[b00][col] = 1
        W[c10][col] = 1; W[c11][col] = -1
        col += 1

        U[a00][col] = 1
        V[b01][col] = 1; V[b11][col] = -1
        W[c01][col] = 1; W[c11][col] = 1
        col += 1

        U[a11][col] = 1
        V[b10][col] = 1; V[b00][col] = -1
        W[c00][col] = 1; W[c10][col] = 1
        col += 1

        U[a00][col] = 1; U[a01][col] = 1
        V[b11][col] = 1
        W[c00][col] = -1; W[c01][col] = 1
        col += 1

        U[a10][col] = 1; U[a00][col] = -1
        V[b00][col] = 1; V[b01][col] = 1
        W[c11][col] = 1
        col += 1

        U[a01][col] = 1; U[a11][col] = -1
        V[b10][col] = 1; V[b11][col] = 1
        W[c00][col] = 1
        col += 1

        U[a02][col] = 1; V[b20][col] = 1; W[c00][col] = 1
        col += 1
        U[a02][col] = 1; V[b21][col] = 1; W[c01][col] = 1
        col += 1
        U[a12][col] = 1; V[b20][col] = 1; W[c10][col] = 1
        col += 1
        U[a12][col] = 1; V[b21][col] = 1; W[c11][col] = 1
        col += 1

    return {"U": U, "V": V, "W": W}
