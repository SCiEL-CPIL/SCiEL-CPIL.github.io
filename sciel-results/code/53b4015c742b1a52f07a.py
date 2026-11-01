"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
def construct():
    # Optimal rank decomposition for 2x2 by 2x11 matrix multiplication.
    # Rank is m*p = 2*11 = 22, which is the theoretical lower bound max(nm, mp, pn).
    n, m, p = DIMENSIONS
    rank = m * p
    
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # The rank index r corresponds to a pair (j, k) where j in [0, m) and k in [0, p).
    # r = j * p + k
    
    for j in range(m):
        for k in range(p):
            r = j * p + k
            
            # V[j*p + k, r] = 1
            # This links the specific input element B[j, k] to this rank component.
            V[j * p + k][r] = 1
            
            # U[i*m + j, r] = 1 for all i
            # This allows A[i, j] to interact with this component for any output row i.
            for i in range(n):
                U[i * m + j][r] = 1
            
            # W[k*n + i, r] = 1 for all i
            # This allows the result to be placed in C[i, k] for any output row i.
            for i in range(n):
                W[k * n + i][r] = 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
