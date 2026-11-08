"""Matrix multiplication 2x3x7 tensor decomposition."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 7
    rank = n * m * p  # 42
    U = [[0] * rank for _ in range(n * m)]  # 6 x 42
    V = [[0] * rank for _ in range(m * p)]  # 21 x 42
    W = [[0] * rank for _ in range(p * n)]  # 14 x 42
    
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
