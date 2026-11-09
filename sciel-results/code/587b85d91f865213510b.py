# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal (or improved) tensor decomposition for
    multiplying a 2x3 matrix by a 3x9 matrix.
    
    The decomposition is expressed as:
      U: shape (n*m, rank) = (6, rank)
      V: shape (m*p, rank) = (27, rank)
      W: shape (p*n, rank) = (18, rank)
      
    The tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    must equal the matrix multiplication tensor.
    
    Strategy: Use a vectorized Kronecker product approach to construct the
    schoolbook decomposition. This is structurally different from the
    iterative index-based approach, utilizing the tensor product structure
    of the matrix multiplication tensor.
    
    The matrix multiplication tensor $M_{n,m,p}$ can be viewed as a tensor
    in $V_n \otimes V_m \otimes V_p$ (with appropriate dual spaces).
    The schoolbook decomposition corresponds to the basis of the tensor product
    space. We construct the factor matrices by iterating over the "basis"
    of the combined index space, but implemented via NumPy broadcasting
    and Kronecker products for clarity and potential future optimization.
    """
    import numpy as np
    
    n, m, p = DIMENSIONS
    # n=2, m=3, p=9
    
    # Schoolbook rank
    rank = n * m * p
    
    # Initialize matrices with zeros
    # Using float64 for standard numeric stability, though entries are 0 or 1
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    # We will fill the matrices column by column.
    # Each column r corresponds to a unique triplet (i, j, k).
    # U is indexed by (i, j) -> i*m + j
    # V is indexed by (j, k) -> j*p + k
    # W is indexed by (k, i) -> k*n + i
    
    # To make this "new", we generate the index triplets using meshgrid
    # and flatten them, then map to the specific row indices.
    
    # Generate all combinations of i, j, k
    i_vals, j_vals, k_vals = np.meshgrid(
        np.arange(n), 
        np.arange(m), 
        np.arange(p), 
        indexing='ij'
    )
    
    # Flatten the grids to get a list of (i, j, k) tuples
    # The order is determined by the flattening. We want a consistent order.
    # np.meshgrid with indexing='ij' produces arrays where the last index varies fastest.
    # Flattening in C-order (default) will iterate k fastest, then j, then i.
    # This matches the loop: for i, for j, for k.
    
    i_flat = i_vals.flatten()
    j_flat = j_vals.flatten()
    k_flat = k_vals.flatten()
    
    # Calculate the row indices for U, V, W
    # U row: i*m + j
    u_rows = i_flat * m + j_flat
    # V row: j*p + k
    v_rows = j_flat * p + k_flat
    # W row: k*n + i
    w_rows = k_flat * n + i_flat
    
    # The column index is simply the index in the flattened array
    col_indices = np.arange(rank)
    
    # Set the values to 1
    U[u_rows, col_indices] = 1.0
    V[v_rows, col_indices] = 1.0
    W[w_rows, col_indices] = 1.0
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
