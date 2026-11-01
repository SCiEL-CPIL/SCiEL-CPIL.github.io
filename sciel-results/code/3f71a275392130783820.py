import numpy as np

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Create index arrays
    # U indices: (i, j)
    i_idx = np.repeat(np.arange(n), m * p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    
    # V indices: (j, k)
    j_idx_v = np.repeat(np.arange(m), p)
    k_idx = np.tile(np.arange(p), m)
    # We need to align U, V, W indices for the rank terms.
    # Each term l corresponds to a specific (i, j, k) combination in the naive case.
    # Let's just generate the matrices directly.
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # We can fill them using advanced indexing
    # For each l in range(rank):
    #   i = l // (m*p)
    #   rem = l % (m*p)
    #   j = rem // p
    #   k = rem % p
    
    l_idx = np.arange(rank)
    i_vals = l_idx // (m * p)
    rem = l_idx % (m * p)
    j_vals = rem // p
    k_vals = rem % p
    
    u_rows = i_vals * m + j_vals
    v_rows = j_vals * p + k_vals
    w_rows = k_vals * n + i_vals
    
    U[u_rows, l_idx] = 1
    V[v_rows, l_idx] = 1
    W[w_rows, l_idx] = 1
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
