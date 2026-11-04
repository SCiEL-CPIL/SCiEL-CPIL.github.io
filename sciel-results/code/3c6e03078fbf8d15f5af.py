import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying
    a 3x5 matrix by a 5x5 matrix using a vectorized schoolbook approach.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Generate indices for all (i, j, k) tuples
    i_indices, j_indices, k_indices = np.indices((n, m, p))
    
    # Flatten indices
    i_flat = i_indices.flatten()
    j_flat = j_indices.flatten()
    k_flat = k_indices.flatten()
    
    # Compute linear indices for U, V, W
    # U[i*m + j, c] = 1
    # V[j*p + k, c] = 1
    # W[k*n + i, c] = 1
    u_rows = i_flat * m + j_flat
    v_rows = j_flat * p + k_flat
    w_rows = k_flat * n + i_flat
    
    # Initialize matrices
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Set values
    # Each column c corresponds to the c-th (i,j,k) tuple
    U[u_rows, np.arange(rank)] = 1
    V[v_rows, np.arange(rank)] = 1
    W[w_rows, np.arange(rank)] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
