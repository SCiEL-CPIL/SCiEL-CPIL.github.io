"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # The tensor for matrix multiplication C = AB is T[i,j,k] = 1 for all i,j,k.
    # This tensor has rank 1 because T[i,j,k] = 1 * 1 * 1.
    # We can construct a rank-1 decomposition with all entries set to 1.
    
    rank = 1
    
    # U shape: (n*m, rank)
    U = [[1] * rank for _ in range(n * m)]
    
    # V shape: (m*p, rank)
    V = [[1] * rank for _ in range(m * p)]
    
    # W shape: (p*n, rank)
    W = [[1] * rank for _ in range(p * n)]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
