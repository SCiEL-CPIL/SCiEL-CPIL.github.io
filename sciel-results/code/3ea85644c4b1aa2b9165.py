"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is m * p = 5 * 6 = 30
    rank = m * p
    
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Mapping:
    # Column c corresponds to (k, j) where c = k * m + j
    # U[i*m + j, c] = 1 for i in 0..n-1
    # V[j*p + k, c] = 1
    # W[k*n + i, c] = 1 for i in 0..n-1
    
    for k in range(p):
        for j in range(m):
            c = k * m + j
            
            # Set V
            V[j * p + k][c] = 1
            
            # Set U and W for all i
            for i in range(n):
                U[i * m + j][c] = 1
                W[k * n + i][c] = 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
