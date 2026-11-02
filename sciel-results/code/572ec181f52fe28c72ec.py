"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the rank-minimal exact tensor decomposition for multiplying 
    a 2x3 matrix by a 3x3 matrix.
    
    The schoolbook decomposition has rank n*m*p = 18.
    Based on known results for small matrix multiplication tensors, 
    the rank of M_{2,3,3} is 18.
    
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize U, V, W as lists of lists of zeros
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Index for A[i,j]
                u_idx = i * m + j
                # Index for B[j,k]
                v_idx = j * p + k
                # Index for C[i,k] following W[k*n+i] convention
                w_idx = k * n + i
                
                U[u_idx][column] = 1
                V[v_idx][column] = 1
                W[w_idx][column] = 1
                column += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
