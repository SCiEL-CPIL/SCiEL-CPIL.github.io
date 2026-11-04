"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    import numpy as np

    n, m, p = DIMENSIONS
    rank = n * m * p

    # Vectorized construction of the schoolbook decomposition
    # This uses Kronecker products to build the factor matrices efficiently.
    
    # U has shape (n*m, rank)
    # V has shape (m*p, rank)
    # W has shape (p*n, rank)
    
    # We iterate over the rank dimension (which corresponds to i, j, k triples)
    # and set the appropriate entries.
    
    # Instead of loops, we can use broadcasting or advanced indexing.
    # However, for clarity and speed in Python, we construct the indices.
    
    # The rank-1 terms are indexed by (i, j, k).
    # U[i*m + j, idx] = 1
    # V[j*p + k, idx] = 1
    # W[k*n + i, idx] = 1
    
    # Generate all combinations of i, j, k
    i_idx, j_idx, k_idx = np.indices((n, m, p))
    
    # Flatten to get the column indices (0 to rank-1)
    # The order doesn't matter as long as it's consistent.
    # np.indices returns arrays of shape (n, m, p).
    # We flatten them to shape (rank,).
    i_flat = i_idx.flatten()
    j_flat = j_idx.flatten()
    k_flat = k_idx.flatten()
    
    # Calculate the row indices for U, V, W
    u_rows = i_flat * m + j_flat
    v_rows = j_flat * p + k_flat
    w_rows = k_flat * n + i_flat
    
    # Initialize matrices with zeros
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Set the entries
    # The column index is simply the index in the flattened array (0 to rank-1)
    col_idx = np.arange(rank)
    
    U[u_rows, col_idx] = 1
    V[v_rows, col_idx] = 1
    W[w_rows, col_idx] = 1
    
    # Convert to lists of lists as required by the interface
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
