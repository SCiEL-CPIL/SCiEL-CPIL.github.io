import numpy as np

def construct():
    n, m, p = 2, 3, 10
    rank = n * m * p
    
    # Use np.indices to generate all (i, j, k) combinations
    # Shape is (n, m, p)
    i_grid, j_grid, k_grid = np.indices((n, m, p))
    
    # Flatten the grids to get 1D arrays of indices
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    k_flat = k_grid.flatten()
    
    # Calculate the row indices in the flattened U, V, W matrices
    # U is (n*m, rank), row index is i*m + j
    u_rows = i_flat * m + j_flat
    # V is (m*p, rank), row index is j*p + k
    v_rows = j_flat * p + k_flat
    # W is (p*n, rank), row index is k*n + i
    w_rows = k_flat * n + i_flat
    
    # Column indices are simply 0 to rank-1
    cols = np.arange(rank)
    
    # Initialize matrices with zeros
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Fill the matrices using advanced indexing
    U[u_rows, cols] = 1
    V[v_rows, cols] = 1
    W[w_rows, cols] = 1
    
    # Convert to lists as required by the interface (if strict) 
    # The seed returns lists of lists. 
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
