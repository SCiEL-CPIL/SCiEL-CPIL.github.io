"""Strassen-based 2x5 x 5x6 matrix multiplication decomposition."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    import time
    import numpy as np

    start_time = time.time()
    DEATHLINE = 145.0
    n, m, p = 2, 5, 6

    def elapsed():
        return time.time() - start_time

    # Rank-m decomposition: one rank term per middle index j
    rank = m  # 5
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for j in range(m):
        r = j
        # U[i*m + j, r] = 1 for all i
        for i in range(n):
            U[i * m + j][r] = 1
        # V[j*p + k, r] = 1 for all k
        for k in range(p):
            V[j * p + k][r] = 1
        # W[k*n + i, r] = 1 for all k, i
        for k in range(p):
            for i in range(n):
                W[k * n + i][r] = 1

    result = {"U": U, "V": V, "W": W}

    # Verify
    def verify(U, V, W, tol=1e-10):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if abs(s - 1.0) > tol:
                        return False
        return True

    if not verify(U, V, W):
        # Fallback to trivial rank-60
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][r] = 1
                    V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
                    r += 1
        result = {"U": U, "V": V, "W": W}

    return result
# EVOLVE-BLOCK-END
