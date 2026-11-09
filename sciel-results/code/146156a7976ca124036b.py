import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Create indices for the rank-1 components
    # Each rank-1 component corresponds to a unique (i, j, k) tuple
    i_indices, j_indices, k_indices = np.indices((n, m, p))
    
    # Flatten the indices to get the positions in U, V, W
    # U index: i * m + j
    # V index: j * p + k
    # W index: k * n + i
    
    u_flat = (i_indices.flatten() * m + j_indices.flatten())
    v_flat = (j_indices.flatten() * p + k_indices.flatten())
    w_flat = (k_indices.flatten() * n + i_indices.flatten())
    
    rank_size = rank
    U = np.zeros((n * m, rank_size), dtype=int)
    V = np.zeros((m * p, rank_size), dtype=int)
    W = np.zeros((p * n, rank_size), dtype=int)
    
    # Fill the matrices
    # For each rank component r, U[u_flat[r], r] = 1, etc.
    for r in range(rank_size):
        U[u_flat[r], r] = 1
        V[v_flat[r], r] = 1
        W[w_flat[r], r] = 1
        
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
