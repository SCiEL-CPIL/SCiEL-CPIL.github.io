import numpy as np

def construct():
    n, m, p = 2, 2, 16
    rank = 56
    
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    # Define Strassen terms for 2x2
    # U indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # V indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # W indices: 0:(0,0), 1:(1,0), 2:(0,1), 3:(1,1)  (k*n+i)
    
    terms = [
        # (U_indices, U_vals, V_indices, V_vals, W_indices, W_vals)
        ([0, 3], [1, 1], [0, 3], [1, 1], [0, 3], [1, 1]),       # M1
        ([2, 3], [1, 1], [0],    [1],    [1],    [1]),           # M2
        ([0],    [1],    [1, 3], [1, -1], [2, 3], [1, 1]),       # M3
        ([3],    [1],    [2, 0], [1, -1], [0, 1], [1, 1]),       # M4
        ([0, 1], [1, 1], [3],    [1],    [0, 2], [-1, 1]),       # M5
        ([2, 0], [1, -1], [0, 1], [1, 1], [3],    [1]),          # M6
        ([1, 3], [1, -1], [2, 3], [1, 1], [0],    [1]),          # M7
    ]
    
    num_blocks = p // 2  # 8 blocks
    
    for l in range(num_blocks):
        # Global offsets
        # B columns: 2l, 2l+1
        # C cols: 2l, 2l+1
        
        for t in range(7):
            r = l * 7 + t
            
            u_idx, u_val, v_idx, v_val, w_idx, w_val = terms[t]
            
            # Fill U
            for idx, val in zip(u_idx, u_val):
                U[idx, r] = val
                
            # Fill V
            for idx, val in zip(v_idx, v_val):
                local_j, local_k = divmod(idx, 2) # idx = j*2 + k? No, V is m*p.
                # V index in local 2x2: j*2 + k.
                # Global V index: j * p + (2l + k).
                global_v_idx = local_j * p + (2 * l + local_k)
                V[global_v_idx, r] = val
                
            # Fill W
            for idx, val in zip(w_idx, w_val):
                local_k, local_i = divmod(idx, 2) # idx = k*2 + i.
                # Global W index: (2l + k) * n + i.
                global_w_idx = (2 * l + local_k) * n + local_i
                W[global_w_idx, r] = val
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
