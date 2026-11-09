import numpy as np

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Generate indices
    i_idx, j_idx, k_idx = np.indices((n, m, p))
    
    # Flatten to get term index
    t_idx = (i_idx * m * p + j_idx * p + k_idx).flatten()
    
    # Row indices for U, V, W
    u_rows = (i_idx * m + j_idx).flatten()
    v_rows = (j_idx * p + k_idx).flatten()
    w_rows = (k_idx * n + i_idx).flatten()
    
    # Fill
    U[u_rows, t_idx] = 1
    V[v_rows, t_idx] = 1
    W[w_rows, t_idx] = 1
    
    # Convert to lists as per interface requirement? 
    # The interface says {'U', 'V', 'W'}: factor matrices.
    # The seed returned lists of lists.
    # "integer/half-integer entries"
    # I should return lists to be safe, or numpy arrays if accepted.
    # The seed returned lists. I will return lists to match the type exactly.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
