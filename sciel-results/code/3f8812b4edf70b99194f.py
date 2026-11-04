"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

import numpy as np
import time
from fractions import Fraction

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 5
    
    # We need to find a rank-minimal exact tensor decomposition for
    # multiplying a 3x3 matrix by a 3x5 matrix.
    
    # The schoolbook algorithm has rank n*m*p = 45.
    # Let's try to find if we can do better using exact rational arithmetic.
    
    # The tensor for matrix multiplication (3x3) @ (3x5) -> (3x5) is:
    # T[i,j,k] = 1 for all i in [0,3), j in [0,3), k in [0,5)
    # where C[i,k] = sum_{j=0}^{2} A[i,j] * B[j,k]
    
    # Decomposition condition:
    # For all i in [0,n), j in [0,m), k in [0,p):
    #   1 = sum_{r=0}^{R-1} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # U has shape (n*m, R) = (9, R)
    # V has shape (m*p, R) = (15, R)  
    # W has shape (p*n, R) = (15, R)
    
    # The schoolbook rank-45 decomposition:
    rank = n * m * p  # 45
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
