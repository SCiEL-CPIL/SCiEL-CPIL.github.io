"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 2
    # U: shape (n*m, rank) = (4, 2)
    # V: shape (m*p, rank) = (32, 2)
    # W: shape (p*n, rank) = (32, 2)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # For each j in range(m), create a rank-1 term
    # Term j: U[i*m+j, j] = 1 for all i
    #          V[j*p+k, j] = 1 for all k
    #          W[k*n+i, j] = 1 for all k, i
    for j in range(m):
        for i in range(n):
            U[i * m + j][j] = 1
        for k in range(p):
            V[j * p + k][j] = 1
        for k in range(p):
            for i in range(n):
                W[k * n + i][j] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
