"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys
DIMENSIONS = [3, 5, 5]
_START_TIME = time.time()
_TIME_LIMIT = 120

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if time.time() - _START_TIME > _TIME_LIMIT - 5:
                    break
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
            if time.time() - _START_TIME > _TIME_LIMIT - 5:
                break
        if time.time() - _START_TIME > _TIME_LIMIT - 5:
            break
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
