import time
import numpy as np

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying 
    a 2 by 5 matrix by a 5 by 5 matrix.
    """
    n, m, p = 2, 5, 5
    
    # The matrix multiplication tensor for C = AB is the all-ones tensor T_{i,j,k} = 1.
    # Since the index i appears in U and W, and the supports for different i are disjoint
    # in the tensor decomposition sum (because W_{k,i,c} is 0 for i != i_c),
    # the rank is the sum of the ranks of the slices for each i.
    # Each slice i is a 1 x m x p all-ones tensor, which has rank 1.
    # Total rank = n * 1 = 2.
    
    rank = n
    
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    # For each i in range(n), create a rank-1 term
    for i in range(n):
        c = i
        
        # U_{i, j, c} = 1 for all j
        for j in range(m):
            U[i * m + j, c] = 1.0
        
        # V_{j, k, c} = 1 for all j, k
        for j in range(m):
            for k in range(p):
                V[j * p + k, c] = 1.0
        
        # W_{k, i, c} = 1 for all k
        for k in range(p):
            W[k * n + i, c] = 1.0
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
