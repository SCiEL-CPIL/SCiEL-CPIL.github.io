"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is n * m = 3 * 3 = 9
    # Because we can decompose each row of A (1x3) multiplying B (3xp) with rank m=3
    rank = n * m
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Fill U, V, W
    # For each row i of A (0 to n-1)
    # We have m terms (j=0 to m-1)
    for i in range(n):
        for j in range(m):
            col = i * m + j
            
            # U[i*m + j, col] = 1
            U[i * m + j][col] = 1
            
            # V[j*p + k, col] = 1 for all k
            for k in range(p):
                V[j * p + k][col] = 1
            
            # W[k*n + i, col] = 1 for all k
            for k in range(p):
                W[k * n + i][col] = 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
