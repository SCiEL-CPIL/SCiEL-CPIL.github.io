import numpy as np

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Generate indices for all combinations
    i_idx, j_idx, k_idx = np.indices((n, m, p))
    
    # Flatten to 1D arrays
    i_flat = i_idx.flatten()
    j_flat = j_idx.flatten()
    k_flat = k_idx.flatten()
    
    # Calculate linear indices
    u_idx = i_flat * m + j_flat
    v_idx = j_flat * p + k_flat
    w_idx = k_flat * n + i_flat
    
    # Create matrices
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Assign 1.0 to the diagonal-like positions
    U[u_idx, np.arange(rank)] = 1.0
    V[v_idx, np.arange(rank)] = 1.0
    W[w_idx, np.arange(rank)] = 1.0
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
