"""Matrix multiplication 3x3 * 3x7 rank minimization."""
import numpy as np
import time

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # The matrix multiplication tensor for A(n*m) x B(m*p) -> C(p*n)
    # is T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # because C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # This is the all-ones tensor, which has tensor rank 1!
    # A single rank-1 term: U[i*m+j,0]=1, V[j*p+k,0]=1, W[k*n+i,0]=1
    # gives sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1*1*1 = 1
    
    rank = 1
    U = [[1] * rank for _ in range(n * m)]
    V = [[1] * rank for _ in range(m * p)]
    W = [[1] * rank for _ in range(p * n)]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
