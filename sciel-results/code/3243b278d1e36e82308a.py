"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # The rank is p, as each output column is an independent (n, m, 1) tensor
    # which has rank 1.
    rank = p
    
    # Initialize matrices with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # For each output column k (0-indexed), we assign a rank-1 decomposition
    for k in range(p):
        # U indices: i * m + j. Set to 1 for all i, j
        for i in range(n):
            for j in range(m):
                U[i * m + j][k] = 1
        
        # V indices: j * p + k. Set to 1 for all j (for this specific k)
        for j in range(m):
            V[j * p + k][k] = 1
        
        # W indices: k * n + i. Set to 1 for all i (for this specific k)
        for i in range(n):
            W[k * n + i][k] = 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
