"""Matrix multiplication 2x5x5 — rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Exact tensor decomposition for 2x5x5 matrix multiplication.
    Returns the schoolbook decomposition of rank 50.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Fill with schoolbook terms
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Convert to lists as per interface requirement (integer/half-integer entries)
    # The interface expects {'U', 'V', 'W'} with factor matrices.
    # The seed program returned lists of lists. I will return lists of lists to be safe,
    # or numpy arrays if allowed. The prompt says "factor matrices of shapes... with integer/half-integer entries".
    # The seed used lists. I will stick to lists to match the seed's interface exactly, 
    # but numpy is faster for construction. I'll convert back to lists.
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
