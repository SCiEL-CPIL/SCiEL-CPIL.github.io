import numpy as np

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Rank is n*m*p for schoolbook
    rank = n * m * p
    
    # Create indices for the schoolbook decomposition
    # We need to map each (i, j, k) to a column index
    # i in [0, n), j in [0, m), k in [0, p)
    
    # Generate all combinations of i, j, k
    # Using np.meshgrid or np.indices for vectorization
    
    # i_idx, j_idx, k_idx will have shape (n, m, p)
    i_idx, j_idx, k_idx = np.indices((n, m, p))
    
    # Flatten to get a list of (i, j, k) tuples
    i_flat = i_idx.flatten()
    j_flat = j_idx.flatten()
    k_flat = k_idx.flatten()
    
    # Number of elements is n*m*p = rank
    num_elements = n * m * p
    
    # Initialize U, V, W as zero matrices
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # For each column c (which corresponds to a specific i, j, k):
    # U[i*m + j, c] = 1
    # V[j*p + k, c] = 1
    # W[k*n + i, c] = 1
    
    # Calculate row indices for U, V, W
    u_rows = i_flat * m + j_flat
    v_rows = j_flat * p + k_flat
    w_rows = k_flat * n + i_flat
    
    # The column index is simply 0 to rank-1
    cols = np.arange(rank)
    
    # Set the values to 1
    U[u_rows, cols] = 1
    V[v_rows, cols] = 1
    W[w_rows, cols] = 1
    
    # Convert to lists of lists as per interface requirement
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
