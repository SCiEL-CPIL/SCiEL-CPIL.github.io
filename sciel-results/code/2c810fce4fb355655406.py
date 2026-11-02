"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    # Schoolbook decomposition of rank n*m*p
    n, m, p = DIMENSIONS  # 2, 4, 5
    rank = n * m * p  # 40
    
    U = [[0] * rank for _ in range(n * m)]  # 8 x 40
    V = [[0] * rank for _ in range(m * p)]  # 20 x 40
    W = [[0] * rank for _ in range(p * n)]  # 10 x 40
    
    col = 0
    for i in range(n):      # i = 0, 1 (row of A, row of C)
        for j in range(m):  # j = 0, 1, 2, 3 (col of A, row of B)
            for k in range(p):  # k = 0, 1, 2, 3, 4 (col of B, col of C)
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
