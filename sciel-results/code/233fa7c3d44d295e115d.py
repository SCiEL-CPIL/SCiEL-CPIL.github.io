"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is determined by the middle dimension m*p which is the largest constraint
    # Lower bound is max(n*m, m*p, p*n) = max(6, 24, 16) = 24.
    # We construct a rank-24 decomposition.
    rank = m * p
    
    # Initialize matrices with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Iterate over the indices that define the rank-1 terms
    # Each term corresponds to a unique (j, k) pair from the middle matrix B
    for j in range(m):
        for k in range(p):
            r = j * p + k
            
            # V is the identity on the (j, k) space
            # V index: j * p + k
            V[j * p + k][r] = 1
            
            # U must provide the coefficient for A[i, j]
            # For this term (fixed j, k), it contributes to all i
            # U index: i * m + j
            for i in range(n):
                U[i * m + j][r] = 1
                
            # W must provide the coefficient for C[i, k] (output index k*n+i)
            # For this term (fixed j, k), it contributes to all i
            # W index: k * n + i
            for i in range(n):
                W[k * n + i][r] = 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
