import numpy as np
import time

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns the schoolbook tensor decomposition for matrix multiplication M_{2,3,7}.
    The rank is n*m*p = 42.
    This implementation uses vectorized numpy operations to generate the factor matrices.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Create index arrays for the decomposition
    # We need to iterate over i in range(n), j in range(m), k in range(p)
    # and assign them to columns in U, V, W.
    
    # Generate the multi-dimensional grid of indices
    # This creates arrays of shape (n, m, p) containing the index values
    i_grid, j_grid, k_grid = np.indices((n, m, p))
    
    # Flatten the grids to match the column order of the decomposition
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    k_flat = k_grid.flatten()
    
    # Initialize U, V, W as zero matrices with integer dtype
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    # The column index for each (i,j,k) tuple is simply the index in the flattened array
    col_indices = np.arange(rank)
    
    # Row indices for U: i * m + j
    u_rows = i_flat * m + j_flat
    # Row indices for V: j * p + k
    v_rows = j_flat * p + k_flat
    # Row indices for W: k * n + i (Following the trace convention W[k*n+i, r])
    w_rows = k_flat * n + i_flat
    
    # Populate the matrices using advanced indexing
    U[u_rows, col_indices] = 1
    V[v_rows, col_indices] = 1
    W[w_rows, col_indices] = 1
    
    # Convert to lists of lists as required by the interface (assuming standard Python types are preferred, 
    # though numpy arrays are often accepted if they are iterable. The prompt says "factor matrices", 
    # but the seed returns lists. To be safe, we return lists.)
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
