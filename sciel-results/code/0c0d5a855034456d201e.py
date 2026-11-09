import numpy as np

def construct():
    n, m, p = 2, 4, 6
    rank = n * m * p
    
    # Vectorized construction of the canonical schoolbook decomposition
    i_idx, j_idx, k_idx = np.indices((n, m, p))
    
    # Rank index c = i*m*p + j*p + k
    c_flat = (i_idx * m * p + j_idx * p + k_idx).flatten()
    
    # Row indices for U, V, W
    u_rows = (i_idx * m + j_idx).flatten()
    v_rows = (j_idx * p + k_idx).flatten()
    w_rows = (k_idx * n + i_idx).flatten()
    
    # Initialize factor matrices
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Populate with 1s at the appropriate positions
    U[u_rows, c_flat] = 1
    V[v_rows, c_flat] = 1
    W[w_rows, c_flat] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
