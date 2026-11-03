"""
Program for matrix multiplication 2x3 by 3x4.
Uses a vectorized index generation approach.
Rank is 24 (trivial decomposition).
"""
import numpy as np

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Generate indices for all (i, j, k) triples
    i_vals, j_vals, k_vals = np.indices((n, m, p))
    
    # Flatten to 1D arrays
    i_flat = i_vals.flatten()
    j_flat = j_vals.flatten()
    k_flat = k_vals.flatten()
    
    # Calculate linear indices
    u_idx = i_flat * m + j_flat
    v_idx = j_flat * p + k_flat
    w_idx = k_flat * n + i_flat
    
    # Assign 1s to the corresponding positions
    cols = np.arange(rank)
    U[u_idx, cols] = 1
    V[v_idx, cols] = 1
    W[w_idx, cols] = 1
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
