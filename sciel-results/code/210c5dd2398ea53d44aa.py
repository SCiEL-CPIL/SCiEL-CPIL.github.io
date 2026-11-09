import numpy as np

def construct():
    n, m, p = 3, 5, 5
    rank = n * m * p
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    idx = 0
    # Using meshgrid to generate all combinations of i, j, k
    i_vals, j_vals, k_vals = np.meshgrid(range(n), range(m), range(p), indexing='ij')
    i_flat = i_vals.flatten()
    j_flat = j_vals.flatten()
    k_flat = k_vals.flatten()
    
    for i, j, k in zip(i_flat, j_flat, k_flat):
        u_idx = i * m + j
        v_idx = j * p + k
        w_idx = k * n + i
        
        U[u_idx, idx] = 1.0
        V[v_idx, idx] = 1.0
        W[w_idx, idx] = 1.0
        idx += 1
        
    return {'U': U, 'V': V, 'W': W}
