import numpy as np
import time

DIMENSIONS = [6, 6, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    n, m, p = DIMENSIONS
    block_size = 2
    n_blocks = n // block_size  # 3
    
    # Total rank is n_blocks^3 * 7
    rank = n_blocks**3 * 7
    
    U = np.zeros((n * m, rank), dtype=np.int8)
    V = np.zeros((m * p, rank), dtype=np.int8)
    W = np.zeros((p * n, rank), dtype=np.int8)
    
    col = 0
    
    def u_idx(ib, kb, r, c):
        return (ib * block_size + r) * m + (kb * block_size + c)
        
    def v_idx(kb, jb, r, c):
        return (kb * block_size + r) * p + (jb * block_size + c)
        
    def w_idx(ib, jb, r, c):
        return (jb * block_size + c) * n + (ib * block_size + r)

    for ib in range(n_blocks):
        for jb in range(n_blocks):
            for kb in range(n_blocks):
                if time.time() > deadline:
                    break
                
                # M1
                U[u_idx(ib, kb, 0, 0), col] += 1
                U[u_idx(ib, kb, 1, 1), col] += 1
                V[v_idx(kb, jb, 0, 0), col] += 1
                V[v_idx(kb, jb, 1, 1), col] += 1
                W[w_idx(ib, jb, 0, 0), col] += 1
                W[w_idx(ib, jb, 1, 1), col] += 1
                col += 1
                
                # M2
                U[u_idx(ib, kb, 1, 0), col] += 1
                U[u_idx(ib, kb, 1, 1), col] += 1
                V[v_idx(kb, jb, 0, 0), col] += 1
                W[w_idx(ib, jb, 1, 0), col] += 1
                W[w_idx(ib, jb, 1, 1), col] -= 1
                col += 1
                
                # M3
                U[u_idx(ib, kb, 0, 0), col] += 1
                V[v_idx(kb, jb, 0, 1), col] += 1
                V[v_idx(kb, jb, 1, 1), col] -= 1
                W[w_idx(ib, jb, 0, 1), col] += 1
                W[w_idx(ib, jb, 1, 1), col] += 1
                col += 1
                
                # M4
                U[u_idx(ib, kb, 1, 1), col] += 1
                V[v_idx(kb, jb, 1, 0), col] += 1
                V[v_idx(kb, jb, 0, 0), col] -= 1
                W[w_idx(ib, jb, 0, 0), col] += 1
                W[w_idx(ib, jb, 1, 0), col] += 1
                col += 1
                
                # M5
                U[u_idx(ib, kb, 0, 0), col] += 1
                U[u_idx(ib, kb, 0, 1), col] += 1
                V[v_idx(kb, jb, 1, 1), col] += 1
                W[w_idx(ib, jb, 0, 0), col] -= 1
                W[w_idx(ib, jb, 0, 1), col] += 1
                col += 1
                
                # M6
                U[u_idx(ib, kb, 1, 0), col] += 1
                U[u_idx(ib, kb, 0, 0), col] -= 1
                V[v_idx(kb, jb, 0, 0), col] += 1
                V[v_idx(kb, jb, 0, 1), col] += 1
                W[w_idx(ib, jb, 1, 1), col] += 1
                col += 1
                
                # M7
                U[u_idx(ib, kb, 0, 1), col] += 1
                U[u_idx(ib, kb, 1, 1), col] -= 1
                V[v_idx(kb, jb, 1, 0), col] += 1
                V[v_idx(kb, jb, 1, 1), col] += 1
                W[w_idx(ib, jb, 0, 0), col] += 1
                col += 1
                
    # Post-processing: Remove redundant columns
    # Check if any column is a scalar multiple of another
    # If U[:, i] = c * U[:, j], V[:, i] = (1/c) * V[:, j], W[:, i] = W[:, j], then i and j are identical terms.
    # This is unlikely in this structured decomposition, but we check for exact duplicates first.
    
    unique_cols = []
    col_map = {}
    
    for i in range(col):
        u_col = tuple(U[:, i].tolist())
        v_col = tuple(V[:, i].tolist())
        w_col = tuple(W[:, i].tolist())
        
        # Check for exact duplicate
        key = (u_col, v_col, w_col)
        if key in col_map:
            continue
        
        # Check for scalar multiple (c = 1, -1 for now, as entries are -1, 0, 1)
        # If U_i = -U_j, V_i = -V_j, W_i = W_j, then U_i V_i W_i = U_j V_j W_j
        # Let's check for c = -1
        key_neg = (tuple(-x for x in u_col), tuple(-x for x in v_col), w_col)
        if key_neg in col_map:
            continue
            
        col_map[key] = len(unique_cols)
        unique_cols.append(i)
        
    if len(unique_cols) < col:
        U = U[:, unique_cols]
        V = V[:, unique_cols]
        W = W[:, unique_cols]
        col = len(unique_cols)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
