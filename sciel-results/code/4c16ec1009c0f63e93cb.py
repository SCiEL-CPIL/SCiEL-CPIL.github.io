import numpy as np

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize U, V, W as zeros
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Vectorized assignment
    # We need to map each (i, j, k) to a column index
    # i in [0, n), j in [0, m), k in [0, p)
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
    # Create indices
    i_idx, j_idx, k_idx = np.indices((n, m, p))
    
    # Flatten to 1D arrays
    i_flat = i_idx.flatten()
    j_flat = j_idx.flatten()
    k_flat = k_idx.flatten()
    
    col_idx = np.arange(rank)
    
    # Compute row indices for U, V, W
    u_rows = i_flat * m + j_flat
    v_rows = j_flat * p + k_flat
    w_rows = k_flat * n + i_flat
    
    # Assign 1s to the corresponding positions
    U[u_rows, col_idx] = 1
    V[v_rows, col_idx] = 1
    W[w_rows, col_idx] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
