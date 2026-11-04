import numpy as np

def construct():
    n, m, p = 2, 4, 8
    rank = n * m * p
    
    # Initialize matrices
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Generate all combinations of (i, j, k)
    # i in [0, n), j in [0, m), k in [0, p)
    i_vals, j_vals, k_vals = np.meshgrid(np.arange(n), np.arange(m), np.arange(p), indexing='ij')
    
    # Flatten to 1D arrays of length rank
    i_flat = i_vals.flatten()
    j_flat = j_vals.flatten()
    k_flat = k_vals.flatten()
    
    # The rank index r corresponds to the flattened index
    r_indices = np.arange(rank)
    
    # Set U rows: index is i*m + j
    u_rows = i_flat * m + j_flat
    U[u_rows, r_indices] = 1
    
    # Set V rows: index is j*p + k
    v_rows = j_flat * p + k_flat
    V[v_rows, r_indices] = 1
    
    # Set W rows: index is k*n + i (trace convention)
    w_rows = k_flat * n + i_flat
    W[w_rows, r_indices] = 1
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
