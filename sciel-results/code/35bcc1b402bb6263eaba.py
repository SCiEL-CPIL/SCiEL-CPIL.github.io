"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x5 matrix.
    The output factor follows the official trace convention W[k*n+i, r].
    """
    n, m, p = 3, 3, 5
    start_time = time.time()
    deadline = 115  # safety margin before 120s limit
    
    # The tensor for matrix multiplication C = A * B where A is n x m and B is m x p
    # T[i, j, k] = A[i, j] * B[j, k]
    # We need to find rank-minimal decomposition: T = sum_r U[:, r] * V[:, r] * W[:, r]
    # where U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    
    # The trivial schoolbook decomposition has rank n*m*p = 45
    # We'll try to find a better decomposition
    
    # Let's start with the schoolbook decomposition and try to improve it
    rank_schoolbook = n * m * p
    
    # Try to find a lower rank decomposition
    # For 3x3x3, the minimal rank is 23 (Strassen-like)
    # For 3x3x5, we need to search
    
    # Let's use a systematic approach:
    # 1. Start with the schoolbook decomposition
    # 2. Try to combine terms to reduce rank
    
    # First, let's create the schoolbook decomposition
    U = np.zeros((n * m, rank_schoolbook))
    V = np.zeros((m * p, rank_schoolbook))
    W = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Now try to find a lower rank decomposition
    # We can use the fact that the multiplication tensor has structure
    
    # For 3x3x5, let's try ranks from 23 upwards
    # Actually, let's think about this more carefully
    
    # The tensor T is in (R^{n*m}) * (R^{m*p}) * (R^{p*n})
    # T_{i,j;k,l} where i in [0,n), j in [0,m), k in [0,m), l in [0,p)
    # T = sum_{j} e_i * e_j * e_j * e_l (for the standard basis)
    
    # Let's try a different approach: use known results
    # For 3x3x3, minimal rank is 23
    # For 3x3x5, we can extend the 3x3x3 result
    
    # Actually, let's just return the schoolbook decomposition for now
    # and try to improve it if we have time
    
    # Convert to lists for the return format
    U_list = [[int(U[i, j]) if abs(U[i, j]) < 1e-10 else float(U[i, j]) for j in range(rank_schoolbook)] for i in range(n * m)]
    V_list = [[int(V[i, j]) if abs(V[i, j]) < 1e-10 else float(V[i, j]) for j in range(rank_schoolbook)] for i in range(m * p)]
    W_list = [[int(W[i, j]) if abs(W[i, j]) < 1e-10 else float(W[i, j]) for j in range(rank_schoolbook)] for i in range(p * n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
