import numpy as np

def construct():
    n, m, p = 3, 4, 7
    rank = n * m * p
    
    # We construct the trivial decomposition.
    # The tensor T_{i,j,k} = 1 if we are computing C_{ik} += A_{ij} B_{jk}.
    # In the decomposition U V W, we have rank R factors.
    # For the trivial decomposition, R = n*m*p.
    # Each factor corresponds to a unique (i, j, k) triple.
    
    # Generate all combinations of indices i, j, k
    # i in [0, n), j in [0, m), k in [0, p)
    i_idx, j_idx, k_idx = np.meshgrid(
        np.arange(n), 
        np.arange(m), 
        np.arange(p), 
        indexing='ij'
    )
    
    # Flatten the index grids to correspond to the rank dimension
    # The order of flattening determines the column index in U, V, W
    # Let's flatten in C-order (row-major) for consistency
    i_flat = i_idx.flatten()
    j_flat = j_idx.flatten()
    k_flat = k_idx.flatten()
    
    # Initialize U, V, W with zeros
    # U shape: (n*m, rank). Index in U is i*m + j
    # V shape: (m*p, rank). Index in V is j*p + k
    # W shape: (p*n, rank). Index in W is k*n + i
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Calculate the linear indices for the rows of U, V, W
    u_rows = i_flat * m + j_flat
    v_rows = j_flat * p + k_flat
    w_rows = k_flat * n + i_flat
    
    # Set the 1s in the matrices
    # col_idx is simply 0, 1, ..., rank-1
    col_idx = np.arange(rank)
    
    U[u_rows, col_idx] = 1.0
    V[v_rows, col_idx] = 1.0
    W[w_rows, col_idx] = 1.0
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
