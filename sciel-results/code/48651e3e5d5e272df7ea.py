import numpy as np

def construct():
    n, m, p = 3, 4, 7
    rank = n * m * p
    
    # Create index arrays for the trivial decomposition
    i_idx, j_idx, k_idx = np.indices((n, m, p))
    
    # Flatten to 1D
    i_flat = i_idx.flatten()
    j_flat = j_idx.flatten()
    k_flat = k_idx.flatten()
    
    # Calculate row indices for U, V, W
    # U: (n*m, rank), row index i*m + j
    u_rows = i_flat * m + j_flat
    # V: (m*p, rank), row index j*p + k
    v_rows = j_flat * p + k_flat
    # W: (p*n, rank), row index k*n + i (trace convention)
    w_rows = k_flat * n + i_flat
    
    # Initialize matrices
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Fill with 1s at the calculated positions
    U[u_rows, np.arange(rank)] = 1
    V[v_rows, np.arange(rank)] = 1
    W[w_rows, np.arange(rank)] = 1
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
