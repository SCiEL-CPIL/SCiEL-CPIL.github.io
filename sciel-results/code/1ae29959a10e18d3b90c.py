"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = m * p  # 30
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # V is identity matrix
    for c in range(rank):
        V[c][c] = 1
    
    # U[i*m + j, j*p + k] = 1 for all i, j, k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][j * p + k] = 1
    
    # W[k*n + i, j*p + k] = 1 for all i, j, k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                W[k * n + i][j * p + k] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
