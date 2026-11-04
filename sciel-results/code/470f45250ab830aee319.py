"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m  # 6
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for k in range(m):
            # U[i*m + k, column] = 1
            U[i * m + k][column] = 1
            # V[k*p + j, column] = 1 for all j
            for j in range(p):
                V[k * p + j][column] = 1
            # W[j*n + i, column] = 1 for all j
            for j in range(p):
                W[j * n + i][column] = 1
            column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
