import time
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 27
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for j in range(m):
        for r in range(9):
            col = j * 9 + r
            for i in range(n):
                U[i * m + j][col] = 1
            for k in range(p):
                V[j * p + k][col] = 1
            for i in range(n):
                for k in range(p):
                    W[k * n + i][col] = 1 / 9
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
