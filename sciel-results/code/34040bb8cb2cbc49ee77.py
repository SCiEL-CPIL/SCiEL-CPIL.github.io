"""Matrix multiplication 6x6x6 tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns the schoolbook decomposition of rank 216 for 6x6 matrix multiplication.
    
    For 6x6 matrix multiplication, the schoolbook algorithm has rank n*m*p = 216.
    The tensor is defined by:
      T[i,j,k] = 1 if the indices match the multiplication pattern
    
    The decomposition uses:
      U[i*m+j, c] = 1 for the (i,j,k) triple at position c
      V[j*p+k, c] = 1 for the (i,j,k) triple at position c  
      W[k*n+i, c] = 1 for the (i,j,k) triple at position c
    
    This gives an exact decomposition with integer entries.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p  # 216
    
    # Initialize factor matrices
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    # Fill in the schoolbook decomposition
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    # Convert to lists as required by the interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
