"""Matrix multiplication 2,3,9 - rank-minimal exact tensor decomposition."""
import numpy as np

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Generate all combinations of i, j, k
    # t indexes the rank-1 terms
    t = np.arange(rank)
    
    # Map t to i, j, k
    # Order: i is slowest, k is fastest
    # i = t // (m*p)
    # j = (t % (m*p)) // p
    # k = t % p
    
    i_idx = t // (m * p)
    j_idx = (t % (m * p)) // p
    k_idx = t % p
    
    # Compute flat indices for U, V, W
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
    u_idx = i_idx * m + j_idx
    v_idx = j_idx * p + k_idx
    w_idx = k_idx * n + i_idx
    
    # Set 1s in the matrices
    # Since each term has exactly one non-zero entry in each matrix
    # we can assign directly
    U[u_idx, t] = 1
    V[v_idx, t] = 1
    W[w_idx, t] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
