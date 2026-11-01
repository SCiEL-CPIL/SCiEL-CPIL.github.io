"""Program for matrix multiplication 2x3 by 3x10 using explicit tensor decomposition."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Construct the exact tensor decomposition for multiplying a 2x3 matrix by a 3x10 matrix
    using a vectorized NumPy approach.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Create index arrays for all (i, j, k) combinations
    i_indices, j_indices, k_indices = np.indices((n, m, p))
    
    # Flatten the indices
    i_flat = i_indices.flatten()
    j_flat = j_indices.flatten()
    k_flat = k_indices.flatten()
    
    # Calculate linear indices for U, V, W
    u_idx = i_flat * m + j_flat
    v_idx = j_flat * p + k_flat
    w_idx = k_flat * n + i_flat
    
    # Initialize sparse-like matrices as dense NumPy arrays
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Set the non-zero elements (all are 1)
    U[u_idx, np.arange(rank)] = 1
    V[v_idx, np.arange(rank)] = 1
    W[w_idx, np.arange(rank)] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
