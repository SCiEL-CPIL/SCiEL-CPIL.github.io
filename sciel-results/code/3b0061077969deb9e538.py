"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is n * m (since for each row i, we need m terms for the 1xm x mxp multiplication)
    # Actually, rank is n * m?
    # For 1 x m x p, rank is m.
    # For n x m x p, it is n independent 1 x m x p multiplications.
    # So rank is n * m.
    rank = n * m
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for i in range(n):
        for j in range(m):
            r = i * m + j
            # U: index i*m + j is 1
            U[i * m + j][r] = 1
            
            # V: index j*p + k is 1 for all k
            for k in range(p):
                V[j * p + k][r] = 1
                
            # W: index k*n + i is 1 for all k
            for k in range(p):
                W[k * n + i][r] = 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
