"""Matrix multiplication 2x3x5 - schoolbook decomposition."""
import numpy as np
from fractions import Fraction

def construct():
    """
    Return the matrix multiplication tensor for 2x3 * 3x5 as a rank decomposition.
    U: (n*m, rank) = (6, rank)
    V: (m*p, rank) = (15, rank)
    W: (p*n, rank) = (10, rank)
    
    Indexing conventions:
    U[i*m + j, r] corresponds to A[i,j]
    V[j*p + k, r] corresponds to B[j,k]
    W[k*n + i, r] corresponds to C[i,k]
    
    We use the schoolbook algorithm with rank = n*m*p = 30.
    """
    n, m, p = 2, 3, 5
    
    # Rank for schoolbook
    rank = n * m * p  # 30
    
    # Initialize U, V, W as lists of lists (integer entries)
    U = [[0] * rank for _ in range(n * m)]  # 6 x 30
    V = [[0] * rank for _ in range(m * p)]  # 15 x 30
    W = [[0] * rank for _ in range(p * n)]  # 10 x 30
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U row index: i*m + j
                # V row index: j*p + k
                # W row index: k*n + i
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx][r] = 1
                V[v_idx][r] = 1
                W[w_idx][r] = 1
                r += 1
    
    return {"U": U, "V": V, "W": W}
