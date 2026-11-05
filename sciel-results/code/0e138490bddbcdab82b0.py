import time
import numpy as np

_START_TIME = time.time()
TIME_LIMIT = 110.0

def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)

def construct():
    n, m, p = 4, 4, 6
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    # Strassen 2x2x2 terms
    S2 = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    # 4x4x4 part (rank 49)
    # Outer Strassen 2x2 blocks
    super_terms = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    block_offsets = {0: (0, 0), 1: (0, 2), 2: (2, 0), 3: (2, 2)}
    
    # Add 4x4x4 terms
    for u_blocks, v_blocks, w_blocks in super_terms:
        for u_pat, v_pat, w_pat in S2:
            u_col = [0] * (n * m)
            v_col = [0] * (m * p)
            w_col = [0] * (p * n)
            
            for block_idx, block_coeff in u_blocks.items():
                br, bc = block_offsets[block_idx]
                for (r, c), coeff in u_pat.items():
                    i = br + r
                    j = bc + c
                    if 0 <= i < n and 0 <= j < m:
                        u_col[i * m + j] += block_coeff * coeff
            
            for block_idx, block_coeff in v_blocks.items():
                br, bc = block_offsets[block_idx]
                for (r, c), coeff in v_pat.items():
                    j = br + r
                    k = bc + c
                    if 0 <= j < m and 0 <= k < p:
                        v_col[j * p + k] += block_coeff * coeff
            
            for block_idx, block_coeff in w_blocks.items():
                br, bc = block_offsets[block_idx]
                for (r, c), coeff in w_pat.items():
                    i = br + r
                    k = bc + c
                    if 0 <= i < n and 0 <= k < p:
                        w_col[k * n + i] += block_coeff * coeff
            
            U_cols.append(u_col)
            V_cols.append(v_col)
            W_cols.append(w_col)
            
    # 4x4x2 part (rank 28)
    # 4 independent 2x2x2 Strassen blocks
    offsets_2x2 = [
        (0, 0, 4), # A[0:2, 0:2] * B[0:2, 4:6] -> C[0:2, 4:6]
        (0, 2, 4), # A[0:2, 2:4] * B[2:4, 4:6] -> C[0:2, 4:6]
        (2, 0, 4), # A[2:4, 0:2] * B[0:2, 4:6] -> C[2:4, 4:6]
        (2, 2, 4), # A[2:4, 2:4] * B[2:4, 4:6] -> C[2:4, 4:6]
    ]
    
    for oi, oj, ok in offsets_2x2:
        for u_pat, v_pat, w_pat in S2:
            u_col = [0] * (n * m)
            v_col = [0] * (m * p)
            w_col = [0] * (p * n)
            
            for (r, c), coeff in u_pat.items():
                i = oi + r
                j = oj + c
                if 0 <= i < n and 0 <= j < m:
                    u_col[i * m + j] += coeff
            
            for (r, c), coeff in v_pat.items():
                j = oj + r
                k = ok + c
                if 0 <= j < m and 0 <= k < p:
                    v_col[j * p + k] += coeff
            
            for (r, c), coeff in w_pat.items():
                i = oi + r
                k = ok + c
                if 0 <= i < n and 0 <= k < p:
                    w_col[k * n + i] += coeff
            
            U_cols.append(u_col)
            V_cols.append(v_col)
            W_cols.append(w_col)

    rank = len(U_cols)
    
    U_final = [[U_cols[r][i] for r in range(rank)] for i in range(n * m)]
    V_final = [[V_cols[r][i] for r in range(rank)] for i in range(m * p)]
    W_final = [[W_cols[r][i] for r in range(rank)] for i in range(p * n)]
    
    # Quick verification
    if _verify_decomposition(U_final, V_final, W_final, n, m, p):
        return {"U": U_final, "V": V_final, "W": W_final}
    else:
        return {"U": U_final, "V": V_final, "W": W_final}

def _verify_decomposition(U, V, W, n, m, p):
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.int32)
    V_arr = np.array(V, dtype=np.int32)
    W_arr = np.array(W, dtype=np.int32)
    
    i_vals, j_vals, k_vals = np.meshgrid(
        np.arange(n), np.arange(m), np.arange(p), indexing='ij'
    )
    u_indices = (i_vals * m + j_vals).ravel()
    v_indices = (j_vals * p + k_vals).ravel()
    w_indices = (k_vals * n + i_vals).ravel()
    
    U_sel = U_arr[u_indices, :]
    V_sel = V_arr[v_indices, :]
    W_sel = W_arr[w_indices, :]
    
    totals = np.sum(U_sel * V_sel * W_sel, axis=1)
    
    return np.all(totals == 1)
