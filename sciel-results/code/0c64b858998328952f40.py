"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    # Tensor T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # Matrix multiplication tensor: (A_{n,m} * B_{m,p})_{i,k} = sum_j A_{i,j} B_{j,k}
    # Tensor T is of shape (n, m, p) where T[i,j,k] = 1
    
    # Schoolbook rank is n*m*p = 2*3*9 = 54
    # We want to find a lower rank decomposition.
    
    # The tensor T_{i,j,k} = 1 for all i,j,k is the all-ones tensor.
    # This tensor has rank 1! Because T_{i,j,k} = 1 * 1 * 1 = u_i * v_j * w_k
    # where u_i = 1, v_j = 1, w_k = 1 for all i,j,k.
    
    # Wait, let me reconsider. The matrix multiplication tensor for multiplying
    # an n x m matrix by an m x p matrix has entries T[i, j, k] = 1 if the
    # (i,k)-th entry of the product uses A[i,j] * B[j,k].
    # So T[i,j,k] = 1 for all valid i, j, k.
    
    # This is indeed the all-ones tensor of shape (n, m, p), which has rank 1.
    # U[i, 0] = 1, V[j, 0] = 1, W[k, 0] = 1
    
    # But wait - the W factor follows the trace convention W[k*n + i, r].
    # Let me re-read the problem statement.
    
    # "The output factor follows the official trace convention W[k*n+i,r]."
    # This means W is indexed as W[k*n + i, r] where k is the column of B (0..p-1)
    # and i is the row of A (0..n-1).
    
    # The tensor decomposition is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # = T[i,j,k] = 1 for all i,j,k
    
    # With rank 1: U[i*m+j, 0] = 1, V[j*p+k, 0] = 1, W[k*n+i, 0] = 1
    # Then sum_0 U[i*m+j,0] * V[j*p+k,0] * W[k*n+i,0] = 1*1*1 = 1 = T[i,j,k] ✓
    
    # So the rank is 1!
    
    rank = 1
    U = [[1] * rank for _ in range(n * m)]
    V = [[1] * rank for _ in range(m * p)]
    W = [[1] * rank for _ in range(p * n)]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
