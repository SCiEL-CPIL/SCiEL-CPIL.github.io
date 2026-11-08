"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=6
    rank = m * p  # 24
    
    U = [[0] * rank for _ in range(n * m)]  # 8 x 24
    V = [[0] * rank for _ in range(m * p)]  # 24 x 24
    W = [[0] * rank for _ in range(p * n)]  # 12 x 24
    
    for j in range(m):
        for k in range(p):
            r = j * p + k
            V[j * p + k][r] = 1
            for i in range(n):
                U[i * m + j][r] = 1
                W[k * n + i][r] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
