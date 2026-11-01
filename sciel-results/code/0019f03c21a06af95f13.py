import numpy as np

def construct():
    n, m, p = 2, 5, 5
    rank = n * m * p
    
    # Generate all combinations of i, j, k
    i_idx, j_idx, k_idx = np.indices((n, m, p))
    
    # Flatten them to get column indices
    i_flat = i_idx.flatten()
    j_flat = j_idx.flatten()
    k_flat = k_idx.flatten()
    
    # Calculate linear indices for U, V, W
    u_indices = i_flat * m + j_flat
    v_indices = j_flat * p + k_flat
    w_indices = k_flat * n + i_flat
    
    # Initialize matrices
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Set the 1s
    U[u_indices, np.arange(rank)] = 1
    V[v_indices, np.arange(rank)] = 1
    W[w_indices, np.arange(rank)] = 1
    
    return {"U": U, "V": V, "W": W}
