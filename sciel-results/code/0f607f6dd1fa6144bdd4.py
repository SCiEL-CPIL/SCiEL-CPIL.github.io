import numpy as np
import time
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    n, m, p = 3, 4, 4
    
    # Build the matrix multiplication tensor
    # T[a,b,c] = 1 if a_col==b_row and c_row==a_row and c_col==b_col
    # a = a_row*m + a_col, b = b_row*p + b_col, c = c_col*n + c_row
    T = np.zeros((n*m, m*p, p*n))
    for a_row in range(n):
        for a_col in range(m):
            for b_col in range(p):
                b_row = a_col
                c_row = a_row
                c_col = b_col
                a = a_row * m + a_col
                b = b_row * p + b_col
                c = c_col * n + c_row
                T[a, b, c] = 1.0
    
    # Current rank-44 solution
    rank = 44
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Local factors for 3x2x2 multiplication
    U_loc = np.zeros((6, 11))
    V_loc = np.zeros((4, 11))
    W_loc = np.zeros((6, 11))
    
    # Strassen part (cols 0-6)
    U_loc[0, 0] = 1; U_loc[3, 0] = 1
    V_loc[0, 0] = 1; V_loc[3, 0] = 1
    W_loc[0, 0] = 1; W_loc[3, 0] = 1
    
    U_loc[2, 1] = 1; U_loc[3, 1] = 1
    V_loc[0, 1] = 1
    W_loc[2, 1] = 1; W_loc[3, 1] = -1
    
    U_loc[0, 2] = 1
    V_loc[1, 2] = 1; V_loc[3, 2] = -1
    W_loc[1, 2] = 1; W_loc[3, 2] = 1
    
    U_loc[3, 3] = 1
    V_loc[2, 3] = 1; V_loc[0, 3] = -1
    W_loc[0, 3] = 1; W_loc[2, 3] = 1
    
    U_loc[0, 4] = 1; U_loc[1, 4] = 1
    V_loc[3, 4] = 1
    W_loc[0, 4] = -1; W_loc[1, 4] = 1
    
    U_loc[2, 5] = 1; U_loc[0, 5] = -1
    V_loc[0, 5] = 1; V_loc[1, 5] = 1
    W_loc[3, 5] = 1
    
    U_loc[1, 6] = 1; U_loc[3, 6] = -1
    V_loc[2, 6] = 1; V_loc[3, 6] = 1
    W_loc[0, 6] = 1
    
    # Extra part (cols 7-10) for Row 2 of A
    U_loc[4, 7] = 1
    V_loc[0, 7] = 1
    W_loc[4, 7] = 1
    
    U_loc[5, 8] = 1
    V_loc[2, 8] = 1
    W_loc[4, 8] = 1
    
    U_loc[4, 9] = 1
    V_loc[1, 9] = 1
    W_loc[5, 9] = 1
    
    U_loc[5, 10] = 1
    V_loc[3, 10] = 1
    W_loc[5, 10] = 1
    
    def add_block(block_idx, a_row_start, a_col_start, b_row_start, b_col_start, c_row_start, c_col_start):
        col_offset = block_idx * 11
        for r in range(3):
            for c in range(2):
                local_a_idx = r * 2 + c
                global_a_idx = (a_row_start + r) * m + (a_col_start + c)
                U[global_a_idx, col_offset:col_offset+11] += U_loc[local_a_idx, :]
        for r in range(2):
            for c in range(2):
                local_b_idx = r * 2 + c
                global_b_idx = (b_row_start + r) * p + (b_col_start + c)
                V[global_b_idx, col_offset:col_offset+11] += V_loc[local_b_idx, :]
        for r in range(3):
            for c in range(2):
                local_c_idx = r * 2 + c
                global_c_idx = (c_col_start + c) * n + (c_row_start + r)
                W[global_c_idx, col_offset:col_offset+11] += W_loc[local_c_idx, :]
    
    add_block(0, 0, 0, 0, 0, 0, 0)
    add_block(1, 0, 2, 2, 0, 0, 0)
    add_block(2, 0, 0, 0, 2, 0, 2)
    add_block(3, 0, 2, 2, 2, 0, 2)
    
    # Verify current solution
    def verify(U_mat, V_mat, W_mat, T_tensor):
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    computed = np.dot(U_mat[a, :], V_mat[b, :] * W_mat[c, :])
                    if abs(computed - T_tensor[a, b, c]) > 1e-10:
                        return False
        return True
    
    if not verify(U, V, W, T):
        # Fallback to trivial
        rank_fallback = n*m*p
        U_f = np.zeros((n*m, rank_fallback))
        V_f = np.zeros((m*p, rank_fallback))
        W_f = np.zeros((p*n, rank_fallback))
        idx = 0
        for a_row in range(n):
            for a_col in range(m):
                for b_col in range(p):
                    b_row = a_col
                    c_row = a_row
                    c_col = b_col
                    a = a_row * m + a_col
                    b = b_row * p + b_col
                    c = c_col * n + c_row
                    U_f[a, idx] = 1
                    V_f[b, idx] = 1
                    W_f[c, idx] = 1
                    idx += 1
        return {"U": U_f.tolist(), "V": V_f.tolist(), "W": W_f.tolist()}
    
    # Try to reduce rank by finding a rank-43 decomposition
    # Strategy: try to merge/eliminate one factor from the 44-factor solution
    # by finding a linear dependency or a clever recombination
    
    # Attempt: search for a rank-43 solution by perturbing and projecting
    # Use the fact that if we have a rank-r decomposition, we can try to
    # find a rank-(r-1) one by checking if any factor is in the span of others
    
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try to eliminate one column by finding if it's redundant
    # A column r is redundant if T = sum_{r' != r} U[:,r']*V[:,r']*W[:,r']
    # i.e., U[:,r]*V[:,r]*W[:,r] = 0 for all (a,b,c)
    # This is hard to check directly. Instead, try random projections.
    
    if time.time() < deadline - 30:
        # Try a more systematic approach: use the 3x2x2 sub-problem
        # and search for a rank-10 decomposition of 3x2x2
        # 3x2x2 tensor: A is 3x2, B is 2x2, C is 3x2
        # Trivial rank: 12, Strassen+trivial: 11
        # Can we get rank 10?
        
        T_sub = np.zeros((6, 4, 6))
        for a_r in range(3):
            for a_c in range(2):
                for b_c in range(2):
                    b_r = a_c
                    c_r = a_r
                    c_c = b_c
                    a = a_r * 2 + a_c
                    b = b_r * 2 + b_c
                    c = c_c * 2 + c_r
                    T_sub[a, b, c] = 1.0
        
        # Try to find rank-10 decomposition of 3x2x2
        # Start from rank-11 and try to eliminate one factor
        U_s = U_loc.copy()
        V_s = V_loc.copy()
        W_s = W_loc.copy()
        
        found_rank10 = False
        for elim_col in range(11):
            if time.time() > deadline - 30:
                break
            # Check if removing column elim_col still gives correct result
            # by trying to compensate with other columns
            U_test = np.delete(U_s, elim_col, axis=1)
            V_test = np.delete(V_s, elim_col, axis=1)
            W_test = np.delete(W_s, elim_col, axis=1)
            
            # Check if the removed factor is zero
            removed = np.einsum('ar,br,cr->abc', U_s[:, elim_col], V_s[:, elim_col], W_s[:, elim_col])
            if np.max(np.abs(removed)) < 1e-10:
                found_rank10 = True
                break
        
        if found_rank10:
            # Rebuild with rank 10 per block, total rank 40
            new_rank = 40
            U_new = np.zeros((n*m, new_rank))
            V_new = np.zeros((m*p, new_rank))
            W_new = np.zeros((p*n, new_rank))
            
            U_s = np.delete(U_loc, elim_col, axis=1)
            V_s = np.delete(V_loc, elim_col, axis=1)
            W_s = np.delete(W_loc, elim_col, axis=1)
            
            def add_block_new(block_idx, a_row_start, a_col_start, b_row_start, b_col_start, c_row_start, c_col_start):
                col_offset = block_idx * 10
                for r in range(3):
                    for c in range(2):
                        local_a_idx = r * 2 + c
                        global_a_idx = (a_row_start + r) * m + (a_col_start + c)
                        U_new[global_a_idx, col_offset:col_offset+10] += U_s[local_a_idx, :]
                for r in range(2):
                    for c in range(2):
                        local_b_idx = r * 2 + c
                        global_b_idx = (b_row_start + r) * p + (b_col_start + c)
                        V_new[global_b_idx, col_offset:col_offset+10] += V_s[local_b_idx, :]
                for r in range(3):
                    for c in range(2):
                        local_c_idx = r * 2 + c
                        global_c_idx = (c_col_start + c) * n + (c_row_start + r)
                        W_new[global_c_idx, col_offset:col_offset+10] += W_s[local_c_idx, :]
            
            add_block_new(0, 0, 0, 0, 0, 0, 0)
            add_block_new(1, 0, 2, 2, 0, 0, 0)
            add_block_new(2, 0, 0, 0, 2, 0, 2)
            add_block_new(3, 0, 2, 2, 2, 0, 2)
            
            if verify(U_new, V_new, W_new, T):
                best_rank = new_rank
                best_U, best_V, best_W = U_new, V_new, W_new
    
    # If we found a better solution, use it
    if best_rank < 44:
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
