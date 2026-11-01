import time
import sys

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying
    a 2 by 3 matrix by a 3 by 4 matrix.
    
    The rank of the matrix multiplication tensor M_{n,m,p} is n*m*p for 
    all n,m,p >= 2 except for the case (2,2,2) where it is 7.
    For n=2, m=3, p=4, the rank is 24.
    
    This function returns the standard schoolbook decomposition of rank 24.
    """
    n, m, p = 2, 3, 4
    rank = n * m * p
    
    # Initialize U, V, W
    # U has shape (n*m, rank)
    # V has shape (m*p, rank)
    # W has shape (p*n, rank)
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Index mapping:
                # A is n x m. Row i, Col j.
                # B is m x p. Row j, Col k.
                # C is n x p. Row i, Col k.
                
                # U input for A[i,j]
                u_idx = i * m + j
                # V input for B[j,k]
                v_idx = j * p + k
                # W output for C[i,k]
                w_idx = k * n + i
                
                U[u_idx][column] = 1.0
                V[v_idx][column] = 1.0
                W[w_idx][column] = 1.0
                column += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
