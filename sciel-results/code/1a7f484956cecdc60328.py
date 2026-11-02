"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    """Return rank-minimal exact tensor decomposition for 2x3 by 3x10 matrix mult."""
    deadline = time.time() + 140  # safety margin before 150s kill
    n, m, p = DIMENSIONS
    rank = n * m * p  # 60
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
# EVOLVE-BLOCK-END
