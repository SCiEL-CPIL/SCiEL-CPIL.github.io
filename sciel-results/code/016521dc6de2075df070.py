import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Create index grids
    i_idx, j_idx, k_idx = np.indices((n, m, p))
    
    # Flatten the indices to create the column assignments
    # Each (i, j, k) tuple corresponds to a column in U, V, W
    cols = np.arange(rank)
    
    # Calculate row indices for U, V, W
    u_rows = (i_idx * m + j_idx).flatten()
    v_rows = (j_idx * p + k_idx).flatten()
    w_rows = (k_idx * n + i_idx).flatten()
    
    # Initialize sparse matrices or dense lists
    # Since rank is 54 and sizes are small, dense is fine.
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Set the ones
    U[u_rows, cols] = 1
    V[v_rows, cols] = 1
    W[w_rows, cols] = 1
    
    # Convert to lists of lists as required
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
