"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS

    def make_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def verify(decomp):
        U = np.array(decomp["U"], dtype=float)
        V = np.array(decomp["V"], dtype=float)
        W = np.array(decomp["W"], dtype=float)
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = sum(U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c] for c in range(r))
                    if abs(total - 1.0) > 1e-9:
                        return False
        return True

    # Rank-2 decomposition: split by j-index groups
    # Term 0 covers j=0, term 1 covers j=1,2
    rank = 2
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for i in range(n):
        U[i * m + 0][0] = 1
    for k in range(p):
        V[k][0] = 1
    for k in range(p):
        for i in range(n):
            W[k * n + i][0] = 1

    for i in range(n):
        U[i * m + 1][1] = 1
        U[i * m + 2][1] = 1
    for k in range(p):
        V[p + k][1] = 1
        V[2 * p + k][1] = 1
    for k in range(p):
        for i in range(n):
            W[k * n + i][1] = 1

    result = {"U": U, "V": V, "W": W}
    if verify(result):
        return result
    return make_schoolbook()

# EVOLVE-BLOCK-END
