import numpy as np

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # Strassen factors for 2x2x2
    # U, V, W are 4x7
    S_U = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ])
    S_V = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 1],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 0]
    ])
    S_W = np.array([
        [1, 0, 0, 1, -1, 0, -1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 0, 0, 0, 0, 1]
    ])
    
    # Schoolbook factors for 2x2x1
    # U, V, W are 4x4 (for A, B) and 2x4 (for C)
    SB_U = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    SB_V = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 0, 0], # Wait, V index is j*p+k. p=1. j=0,1. k=0.
        [0, 0, 0, 0]  # This is wrong. V size is m*p = 2*1 = 2? No, m=2, p=1. V size 2.
    ])
    # Let's re-verify SB_V size.
    # For 2x2x1, m=2, p=1. V rows = 2*1 = 2.
    # My SB_V above is 4x4. That's for 2x2x2? No.
    # For 2x2x1:
    # U rows: 2*2=4.
    # V rows: 2*1=2.
    # W rows: 1*2=2.
    
    SB_U_221 = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    SB_V_221 = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1]
    ])
    SB_W_221 = np.array([
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ])
    
    # Total rank 36
    rank = 36
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    
    # Helper to add a block
    def add_block(U_f, V_f, W_f, i_off, j_off, k_off, j_stride, k_stride):
        nonlocal col
        num_cols = U_f.shape[1]
        for r in range(num_cols):
            for i in range(2):
                for j in range(2):
                    u_idx = (i + i_off) * m + (j + j_off)
                    U[u_idx, col] += U_f[i*2+j, r]
            for j in range(2):
                for k in range(2):
                    v_idx = (j + j_off) * p + (k + k_off)
                    V[v_idx, col] += V_f[j*2+k, r]
            for k in range(2):
                for i in range(2):
                    w_idx = (k + k_off) * n + (i + i_off)
                    W[w_idx, col] += W_f[k*2+i, r]
            col += 1
            
    def add_block_221(U_f, V_f, W_f, i_off, j_off, k_off):
        nonlocal col
        num_cols = U_f.shape[1]
        for r in range(num_cols):
            for i in range(2):
                for j in range(2):
                    u_idx = (i + i_off) * m + (j + j_off)
                    U[u_idx, col] += U_f[i*2+j, r]
            for j in range(2):
                # k is always 0 in local, mapped to k_off
                v_idx = (j + j_off) * p + k_off
                V[v_idx, col] += V_f[j, r] # V_f is 2x4
            for i in range(2):
                w_idx = k_off * n + (i + i_off)
                W[w_idx, col] += W_f[i, r] # W_f is 2x4
            col += 1

    # Block 1: A(1) B(1)_1. i:0-1, j:0-1, k:0-1
    add_block(S_U, S_V, S_W, 0, 0, 0, 1, 1)
    # Block 2: A(1) B(1)_2. i:0-1, j:0-1, k:2-3
    add_block(S_U, S_V, S_W, 0, 0, 2, 1, 1)
    # Block 3: A(1) B(1)_3. i:0-1, j:0-1, k:4
    add_block_221(SB_U_221, SB_V_221, SB_W_221, 0, 0, 4)
    # Block 4: A(2) B(2)_1. i:0-1, j:2-3, k:0-1
    add_block(S_U, S_V, S_W, 0, 2, 0, 1, 1)
    # Block 5: A(2) B(2)_2. i:0-1, j:2-3, k:2-3
    add_block(S_U, S_V, S_W, 0, 2, 2, 1, 1)
    # Block 6: A(2) B(2)_3. i:0-1, j:2-3, k:4
    add_block_221(SB_U_221, SB_V_221, SB_W_221, 0, 2, 4)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
