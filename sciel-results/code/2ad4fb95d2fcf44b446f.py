"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank m * p
    rank = m * p
    
    # Initialize matrices with zeros
    # U: (n*m, rank)
    U = [[0] * rank for _ in range(n * m)]
    # V: (m*p, rank)
    V = [[0] * rank for _ in range(m * p)]
    # W: (p*n, rank)
    W = [[0] * rank for _ in range(p * n)]
    
    # Fill based on the derived decomposition
    # r corresponds to (j, k)
    for j in range(m):
        for k in range(p):
            r = j * p + k
            
            # V[j*p + k, r] = 1
            V[j * p + k][r] = 1
            
            # For all i:
            # U[i*m + j, r] = 1
            # W[k*n + i, r] = 1
            for i in range(n):
                U[i * m + j][r] = 1
                W[k * n + i][r] = 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
