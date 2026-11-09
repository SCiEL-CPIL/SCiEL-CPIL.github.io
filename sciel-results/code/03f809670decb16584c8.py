"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 10
    rank = n * p  # 20
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for i in range(n):
        for k in range(p):
            col = i * p + k
            # W[k*n + i, col] = 1
            W[k * n + i][col] = 1
            # U[i*m + j, col] = 1 for all j
            for j in range(m):
                U[i * m + j][col] = 1
            # V[j*p + k, col] = 1 for all j
            for j in range(m):
                V[j * p + k][col] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
