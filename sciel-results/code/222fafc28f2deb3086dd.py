"""Program for 3x3 by 3x5 matrix multiplication tensor decomposition."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 5
    rank = n * m * p      # 45

    U = [[0] * rank for _ in range(n * m)]  # 9 x 45
    V = [[0] * rank for _ in range(m * p)]  # 15 x 45
    W = [[0] * rank for _ in range(p * n)]  # 15 x 45

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
