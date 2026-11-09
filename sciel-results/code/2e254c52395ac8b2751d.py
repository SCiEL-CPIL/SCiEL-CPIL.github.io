import numpy as np

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the rank decomposition for matrix multiplication M_{3,4,5}.
    Uses a vectorized construction of the standard basis decomposition.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Generate indices for all non-zero terms in the multiplication tensor
    # The tensor T has non-zero entries at (i, j, k) where the multiplication is defined.
    # Specifically, the term (A B)_{i,k} = sum_j A_{i,j} B_{j,k}.
    # The tensor is sum_{i,j,k} e_{i,j} \otimes e_{j,k} \otimes e_{k,i} (in the trace convention W[k*n+i])
    # Actually, the standard decomposition is:
    # T = sum_{i=0}^{n-1} sum_{j=0}^{m-1} sum_{k=0}^{p-1} u_{i,j,k} \otimes v_{i,j,k} \otimes w_{i,j,k}
    # where u is the vector with 1 at (i,j), v is 1 at (j,k), w is 1 at (k,i).
    
    # Create arrays of indices
    # i ranges 0..n-1, j ranges 0..m-1, k ranges 0..p-1
    # We need to create rank columns.
    
    # Flatten the multi-index (i, j, k) to linear indices for U, V, W
    # U index: i * m + j
    # V index: j * p + k
    # W index: k * n + i
    
    # Use meshgrid to get all combinations
    ii, jj, kk = np.meshgrid(np.arange(n), np.arange(m), np.arange(p), indexing='ij')
    
    # Flatten the grids
    ii_flat = ii.flatten()
    jj_flat = jj.flatten()
    kk_flat = kk.flatten()
    
    # Calculate linear indices
    u_indices = ii_flat * m + jj_flat
    v_indices = jj_flat * p + kk_flat
    w_indices = kk_flat * n + ii_flat
    
    # Initialize matrices
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Set the 1s
    # The column index is just 0, 1, ..., rank-1
    cols = np.arange(rank)
    
    U[u_indices, cols] = 1
    V[v_indices, cols] = 1
    W[w_indices, cols] = 1
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
