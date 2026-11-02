"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is m (the inner dimension)
    # Each term j corresponds to the outer product of column j of A and row j of B
    rank = m
    
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for j in range(m):
        # U[i, j] = 1 for all i
        for i in range(n):
            U[i * m + j][j] = 1
            
        # V[j, k] = 1 for all k
        for k in range(p):
            V[j * p + k][j] = 1
            
        # W[k, i] = 1 for all k, i
        for k in range(p):
            for i in range(n):
                W[k * n + i][j] = 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
