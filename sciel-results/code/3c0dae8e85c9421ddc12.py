import numpy as np

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize NumPy arrays for efficient construction
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col_idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col_idx] = 1
                V[j * p + k, col_idx] = 1
                W[k * n + i, col_idx] = 1
                col_idx += 1
                
    # Convert to lists of lists as per interface requirement
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {
        "U": U_list,
        "V": V_list,
        "W": W_list
    }
