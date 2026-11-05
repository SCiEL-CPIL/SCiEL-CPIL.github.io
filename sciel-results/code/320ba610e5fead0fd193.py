import time
import numpy as np

DIMENSIONS = [3, 4, 4]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x4 matrix by a 4x4 matrix.
    
    The matrix multiplication tensor T[i,j,k] for C = AB is defined by C[i,k] = sum_j A[i,j] * B[j,k].
    The tensor T has T[i,j,k] = 1 for all valid indices i, j, k, because the coefficient of A[i,j]B[j,k] 
    in C[i,k] is 1.
    
    The all-ones tensor has rank 1.
    We can construct the decomposition with rank 1:
    U is a (n*m, 1) matrix of ones.
    V is a (m*p, 1) matrix of ones.
    W is a (p*n, 1) matrix of ones.
    
    The product sum is sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 * 1 * 1 = 1.
    """
    n, m, p = DIMENSIONS
    
    # Rank 1 decomposition
    rank = 1
    
    # U shape: (n*m, rank)
    U = np.ones((n * m, rank))
    
    # V shape: (m*p, rank)
    V = np.ones((m * p, rank))
    
    # W shape: (p*n, rank)
    # Index convention: W[k*n + i, r]
    # Since W is all ones, the specific indexing doesn't change the values, 
    # but we ensure the shape is correct.
    W = np.ones((p * n, rank))
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
