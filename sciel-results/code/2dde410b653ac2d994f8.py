import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # For matrix multiplication of n x m by m x p:
    # The multiplication tensor has entries T[i, j, k] = 1 if the output index (i,k) 
    # depends on input indices (i,j) and (j,k).
    # Specifically, (A*B)_{i,k} = sum_j A_{i,j} * B_{j,k}
    # So the tensor is T_{(i,j),(j,k),(i,k)} = 1
    # In the factor matrix notation:
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # The decomposition requires: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{i,k} * (something)
    # Actually, for matrix multiplication: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 if (i,j) and (j,k) share j, and output is (i,k)
    # More precisely: the tensor T[i,j,k] (where i in [0,n), j in [0,m), k in [0,p)) 
    # maps to output at position (k*n + i) in W, and inputs at (i*m + j) in U and (j*p + k) in V.
    # The condition is: for each r, the product U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] 
    # should sum over r to give 1 when the indices are consistent, 0 otherwise.
    # Actually, the matrix multiplication tensor satisfies:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 if and only if the "middle" index j is the same
    # Wait, that's always the case since j is shared. Let me think again.
    
    # The matrix multiplication tensor M_{n,m,p} has:
    # M_{(i,j),(j,k),(i,k)} = 1
    # This means: for fixed i, j, k, the tensor entry is 1.
    # A rank-1 decomposition term r contributes:
    # u_r[i,j] * v_r[j,k] * w_r[i,k]
    # We need: sum_r u_r[i,j] * v_r[j,k] * w_r[i,k] = 1 for all valid (i,j,k)
    
    # The schoolbook uses rank = n*m*p with u_r[i,j] = 1 if (i,j) matches, etc.
    
    # For small dimensions like 2x5x5, we can try to find a lower rank decomposition.
    # One approach: use the fact that for fixed j, the subproblem is n x 1 x p matrix multiplication,
    # which has rank min(n*p, something). But we need to combine all j's.
    
    # Let's try a structured approach:
    # For each j in [0, m), we need to decompose the "slice" where the middle index is j.
    # The slice for fixed j is an n x p tensor that is all-ones (since for fixed j, 
    # all (i,k) pairs have T=1). Wait, no. For fixed j, T[i,j,k] = 1 for all i, k.
    # So each slice is an all-ones n x p matrix.
    # But the U, V, W factors are shared across j's.
    
    # Actually, the full tensor is:
    # T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # Wait, is that right? Let me re-read the problem.
    # "Matrix multiplication 2,5,5" means multiplying a 2x5 matrix by a 5x5 matrix.
    # (A*B)_{i,k} = sum_{j=0}^{4} A_{i,j} * B_{j,k}
    # So the tensor has T[i,j,k] = 1 for all valid i, j, k.
    # This means the tensor is indeed all-ones!
    # If T[i,j,k] = 1 for all i, j, k, then the rank is 1:
    # U[i*m+j, 0] = 1 for all i, j
    # V[j*p+k, 0] = 1 for all j, k  
    # W[k*n+i, 0] = 1 for all k, i
    # Then sum_r U * V * W = 1 * 1 * 1 = 1 = T[i,j,k]
    
    # So the rank-minimal decomposition is rank 1!
    
    rank = 1
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for i in range(n):
        for j in range(m):
            U[i * m + j][0] = 1
    
    for j in range(m):
        for k in range(p):
            V[j * p + k][0] = 1
    
    for k in range(p):
        for i in range(n):
            W[k * n + i][0] = 1
    
    return {"U": U, "V": V, "W": W}
