"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = m * p  # 21
    U = [[0] * rank for _ in range(n * m)]  # 6 x 21
    V = [[0] * rank for _ in range(m * p)]  # 21 x 21
    W = [[0] * rank for _ in range(p * n)]  # 14 x 21
    
    for j in range(m):
        for k in range(p):
            c = j * p + k  # column index
            # V[j*p+k, c] = 1
            V[j * p + k][c] = 1
            # U[i*m+j, c] = 1 for all i
            for i in range(n):
                U[i * m + j][c] = 1
            # W[k*n+i, c] = 1 for all i
            for i in range(n):
                W[k * n + i][c] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
