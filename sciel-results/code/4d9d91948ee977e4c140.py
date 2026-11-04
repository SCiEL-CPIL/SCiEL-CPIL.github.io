"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 24 decomposition: 3 multiplications per output column k
    # For each k, use 3 rank-1 terms corresponding to the sum over j=0,1,2
    rank = m * p  # 3 * 8 = 24
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Column index in U/V/W matrices corresponds to r in [0, rank)
    # We map r to (k, j) where k in [0, p), j in [0, m)
    r = 0
    for k in range(p):
        for j in range(m):
            # U[i*m + j, r] = 1 for all i
            for i in range(n):
                U[i * m + j][r] = 1
            
            # V[j*p + k, r] = 1
            V[j * p + k][r] = 1
            
            # W[k*n + i, r] = 1 for all i
            for i in range(n):
                W[k * n + i][r] = 1
                
            r += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
