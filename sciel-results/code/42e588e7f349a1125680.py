import numpy as np

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Generate indices for all (i, j, k) tuples
    i_vals, j_vals, k_vals = np.indices((n, m, p))
    
    # Flatten the indices
    i_flat = i_vals.flatten()
    j_flat = j_vals.flatten()
    k_flat = k_vals.flatten()
    
    # Calculate row indices for U, V, W
    u_rows = i_flat * m + j_flat
    v_rows = j_flat * p + k_flat
    w_rows = k_flat * n + i_flat
    
    # Column indices are 0 to rank-1
    cols = np.arange(rank)
    
    # Initialize matrices
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Assign 1s at the specified positions
    U[u_rows, cols] = 1
    V[v_rows, cols] = 1
    W[w_rows, cols] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
