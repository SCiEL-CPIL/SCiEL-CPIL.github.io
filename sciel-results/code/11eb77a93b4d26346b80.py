"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is m * p = 4 * 5 = 20
    rank = m * p
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Iterate over the rank terms
    for r in range(rank):
        # Determine j and k from r
        # r corresponds to (j, k) where j in 0..m-1, k in 0..p-1
        j = r // p
        k = r % p
        
        # V index is j * p + k
        V[j * p + k][r] = 1
        
        # U index is i * m + j
        # For each i in 0..n-1
        for i in range(n):
            U[i * m + j][r] = 1
            
            # W index is k * n + i
            W[k * n + i][r] = 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
