import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the factor matrices U, V, W for the matrix multiplication tensor
    of dimensions n x m by m x p.
    
    This implementation uses a vectorized approach to generate the schoolbook
    decomposition, which is a different construction strategy than the 
    iterative loop used in the baseline.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Create index grids
    i_idx, j_idx, k_idx = np.indices((n, m, p))
    
    # Flatten indices to create column indices for U, V, W
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
    # We need to map the triple (i, j, k) to a column index c.
    # In the schoolbook algorithm, each (i, j, k) corresponds to a unique column.
    # Let's assign column c = i * (m * p) + j * p + k.
    # Wait, the previous code used a simple counter.
    # The mapping of (i,j,k) to columns doesn't matter as long as it's consistent 
    # and unique for each term.
    
    c_idx = i_idx * (m * p) + j_idx * p + k_idx
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Vectorized assignment
    # U[i*m+j, c] = 1
    # V[j*p+k, c] = 1
    # W[k*n+i, c] = 1
    
    # Flatten the indices
    i_flat = i_idx.flatten()
    j_flat = j_idx.flatten()
    k_flat = k_idx.flatten()
    c_flat = c_idx.flatten()
    
    u_rows = i_flat * m + j_flat
    v_rows = j_flat * p + k_flat
    w_rows = k_flat * n + i_flat
    
    U[u_rows, c_flat] = 1
    V[v_rows, c_flat] = 1
    W[w_rows, c_flat] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
