# EVOLVE-BLOCK-START
def construct():
    # Decomposition exploiting row independence: rank n*m
    n, m, p = DIMENSIONS
    rank = n * m
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for i in range(n):
        for j in range(m):
            r = i * m + j
            U[i * m + j][r] = 1
            for k in range(p):
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
