import numpy as np

def construct():
    n, m, p = 2, 4, 4
    rank = 28
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen local factors
    # U_local is 4x7, indices i*2+j
    # V_local is 4x7, indices j*2+k
    # W_local is 4x7, indices k*2+i
    
    U_loc = np.array([
        [1, 1, -1, 0, 1, 0, 1], # i=1, j=1
        [0, 1, 0, 0, 0, 1, 0],  # i=1, j=2
        [0, 0, 1, 0, 0, -1, 1], # i=2, j=1
        [0, 0, 0, 1, 1, 0, 0]   # i=2, j=2
    ], dtype=int)
    
    V_loc = np.array([
        [1, 0, 1, -1, 1, 0, 1],  # j=1, k=1
        [0, 0, 0, 0, 0, 0, -1],  # j=1, k=2
        [-1, 1, 0, 1, 1, 1, 0],  # j=2, k=1
        [0, 0, 0, 0, 0, 1, 0]    # j=2, k=2
    ], dtype=int)
    
    W_loc = np.array([
        [1, 0, 0, 1, -1, 0, 1], # k=1, i=1
        [0, 1, 1, 0, 0, 0, 0],  # k=1, i=2
        [0, 0, 0, 0, 1, 1, 0],  # k=2, i=1
        [1, -1, 1, 1, 0, 0, 0]  # k=2, i=2
    ], dtype=int)
    
    # Define the 4 blocks
    # Block 1: J=0,1 (0-indexed), K=0,1
    # Block 2: J=2,3, K=0,1
    # Block 3: J=0,1, K=2,3
    # Block 4: J=2,3, K=2,3
    
    blocks = [
        (0, 0), # J_start, K_start
        (2, 0),
        (0, 2),
        (2, 2)
    ]
    
    for b_idx, (j_start, k_start) in enumerate(blocks):
        col_start = b_idx * 7
        col_end = col_start + 7
        
        for r in range(7):
            col = col_start + r
            for i in range(n):
                for j_local in range(2):
                    j_global = j_start + j_local
                    u_idx = i * m + j_global
                    U[u_idx, col] = U_loc[i * 2 + j_local, r]
                    
                for k_local in range(2):
                    k_global = k_start + k_local
                    w_idx = k_global * n + i
                    W[w_idx, col] = W_loc[k_local * 2 + i, r]
                    
            for j_local in range(2):
                j_global = j_start + j_local
                for k_local in range(2):
                    k_global = k_start + k_local
                    v_idx = j_global * p + k_global
                    V[v_idx, col] = V_loc[j_local * 2 + k_local, r]
                    
    return {"U": U, "V": V, "W": W}
