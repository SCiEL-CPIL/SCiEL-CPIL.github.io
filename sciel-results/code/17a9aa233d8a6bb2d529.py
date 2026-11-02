import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the factor matrices for the tensor decomposition of matrix multiplication.
    Dimensions: n x m * m x p = n x p
    n=2, m=4, p=8
    Rank: n * m * p = 64
    Uses NumPy for efficient construction.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Create index arrays
    # U has shape (n*m, rank)
    # V has shape (m*p, rank)
    # W has shape (p*n, rank)
    
    # We iterate over the rank dimension (which corresponds to i, j, k)
    # i in 0..n-1, j in 0..m-1, k in 0..p-1
    
    # Pre-allocate zeros
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    # Generate indices
    # col is the index in the rank dimension
    # We need to map (i, j, k) to col
    # And set U[i*m+j, col] = 1
    #       V[j*p+k, col] = 1
    #       W[k*n+i, col] = 1
    
    # Vectorized approach:
    # Create meshgrid for i, j, k
    i_vals, j_vals, k_vals = np.indices((n, m, p))
    
    # Flatten to get the sequence of (i, j, k)
    i_flat = i_vals.flatten()
    j_flat = j_vals.flatten()
    k_flat = k_vals.flatten()
    
    # The column index for each (i, j, k) tuple is just the linear index in the flattened array
    # Since we iterate i, then j, then k (C-order), the linear index is:
    # idx = i * (m * p) + j * p + k
    # This matches the 'col' increment in the original code.
    
    cols = np.arange(rank)
    
    # Calculate row indices for U, V, W
    u_rows = i_flat * m + j_flat
    v_rows = j_flat * p + k_flat
    w_rows = k_flat * n + i_flat
    
    # Assign 1s
    U[u_rows, cols] = 1
    V[v_rows, cols] = 1
    W[w_rows, cols] = 1
    
    # Convert to lists as required by the interface? 
    # The interface says "factor matrices of shapes...". The seed returned lists of lists.
    # The evaluator might accept NumPy arrays or lists. 
    # "integer/half-integer entries".
    # To be safe and match the seed's return type exactly (lists), we convert.
    # However, NumPy arrays are often accepted. 
    # Let's check the seed: `return {"U": U, "V": V, "W": W}` where U is a list of lists.
    # I will return lists to be strictly compatible with the seed's interface type.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
