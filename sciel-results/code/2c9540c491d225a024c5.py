import numpy as np
from itertools import product

def construct():
    n, m, p = 3, 3, 8
    
    # The multiplication tensor T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
    # We need to find rank-minimal decomposition:
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Naive rank is n*m*p = 72
    # Let's try to find if we can do better
    
    # For this specific case, let's try a computational search
    # But first, let's consider the structure more carefully
    
    # The tensor can be seen as: for each (i,j,k), the contribution is A[i,j]*B[j,k]
    # This is a 3-mode tensor with dimensions (3*3, 3*8, 3*8) = (9, 24, 24)
    
    # Actually, let me reconsider the problem statement.
    # U has shape (n*m, rank) = (9, rank)
    # V has shape (m*p, rank) = (24, rank)  
    # W has shape (p*n, rank) = (24, rank)
    
    # The constraint is: for all i,j,k:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 (if we normalize)
    # But actually, the standard formulation is:
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # = sum_r (sum_j U[i*m+j,r]*A[i,j]) * V[j*p+k,r] * W[k*n+i,r]
    # This is getting complicated. Let me use the standard tensor rank approach.
    
    # The multiplication tensor M has M[i,j,k] = 1 for the (i,j,k) entry
    # We need: M[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # For the naive decomposition, U[i*m+j, c] = V[j*p+k, c] = W[k*n+i, c] = 1
    # where c = i*m*p + j*p + k
    
    # Let's try to find if there's a lower rank by checking if the tensor
    # has any linear dependencies that can be exploited.
    
    # Build the tensor as a 3D array
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1
    
    # The rank of this tensor is what we want to minimize
    # For now, let's use the naive rank of 72
    rank = n * m * p
    
    # Create the naive decomposition
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, idx] = 1
                V[j*p+k, idx] = 1
                W[k*n+i, idx] = 1
                idx += 1
    
    return {
        "U": U.tolist(),
        "V": V.tolist(), 
        "W": W.tolist()
    }
