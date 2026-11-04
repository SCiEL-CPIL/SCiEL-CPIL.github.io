"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # The rank of the matrix multiplication tensor T_{n,m,p} is at most min(n*m, m*p, p*n).
    # We use the decomposition with rank R = n*m.
    # This corresponds to fixing the U matrix to be the identity (or a permutation) 
    # and setting V and W to be all-ones in the corresponding blocks.
    
    R = n * m
    
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    # U is the identity matrix (permutation of indices i*m+j)
    for i in range(n):
        for j in range(m):
            idx = i * m + j
            U[idx][idx] = 1
            
    # For each column r (corresponding to i, j), 
    # V[j*p+k, r] = 1 for all k
    # W[k*n+i, r] = 1 for all k
    for i in range(n):
        for j in range(m):
            r = i * m + j
            for k in range(p):
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
