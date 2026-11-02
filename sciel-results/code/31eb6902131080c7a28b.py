"""Matrix multiplication 3x4x5 - rank-minimal decomposition."""
DIMENSIONS = [3, 4, 5]

import numpy as np
import time

# EVOLVE-BLOCK-START

def construct():
    """
    Constructs the rank-minimal tensor decomposition for 3x4x5 matrix multiplication.
    Returns dict with keys 'U', 'V', 'W' as lists of lists with integer/half-integer entries.
    U: shape (n*m, rank), V: shape (m*p, rank), W: shape (p*n, rank)
    Convention: W[k*n + i, r] corresponds to output C[i, k]
    """
    n, m, p = DIMENSIONS
    
    # The tensor T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k
    # Schoolbook decomposition: rank = n*m*p = 60
    # Each rank-1 term corresponds to one (i,j,k) triple
    
    rank = n * m * p
    
    # Build U, V, W as numpy arrays first, then convert to lists
    U_np = np.zeros((n * m, rank), dtype=np.float64)
    V_np = np.zeros((m * p, rank), dtype=np.float64)
    W_np = np.zeros((p * n, rank), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                x = i * m + j      # input A index
                y = j * p + k      # input B index  
                z = k * n + i      # output index (trace convention)
                U_np[x, col] = 1.0
                V_np[y, col] = 1.0
                W_np[z, col] = 1.0
                col += 1
    
    # Convert to lists of lists with integer entries
    U_list = U_np.astype(int).tolist()
    V_list = V_np.astype(int).tolist()
    W_list = W_np.astype(int).tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
