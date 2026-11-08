import numpy as np
import time

def construct():
    start_time = time.time()
    timeout = 115.0
    
    n, m, p = 3, 4, 4
    Nm, Mp, Pn = n*m, m*p, p*n  # 12, 16, 12
    
    # Build tensor
    T = np.zeros((Nm, Mp, Pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    def verify_fast(U, V, W):
        R = U.shape[1]
        UV = U[:, None, :] * V[None, :, :]  # (Nm, Mp, R)
        UV_flat = UV.reshape(-1, R)
        T_rec_flat = UV_flat @ W.T
        T_rec = T_rec_flat.reshape(Nm, Mp, Pn)
        return np.allclose(T_rec, T, atol=1e-8)
    
    # Build the rank-44 block decomposition
    def build_block_decomp():
        rank = 44
        U = np.zeros((Nm, rank))
        V = np.zeros((Mp, rank))
        W = np.zeros((Pn, rank))
        
        U_loc = np.zeros((6, 11))
        V_loc = np.zeros((4, 11))
        W_loc = np.zeros((6, 11))
        
        # Strassen for 2x2x2 (rows 0,1 of local A)
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
        
        # Extra for row 2
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
        
        return U, V, W
    
    U_blk, V_blk, W_blk = build_block_decomp()
    
    if not verify_fast(U_blk, V_blk, W_blk):
        # Fallback to direct rank-48
        R = n * m * p
        U = np.zeros((Nm, R))
        V = np.zeros((Mp, R))
        W = np.zeros((Pn, R))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, idx] = 1.0
                    V[j*p+k, idx] = 1.0
                    W[k*n+i, idx] = 1.0
                    idx += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_U, best_V, best_W = U_blk, V_blk, W_blk
    best_rank = 44
    
    # Try to reduce rank by removing redundant columns
    # A column r is redundant if removing it still gives correct decomposition
    # We can try all single-column removals, then re-check
    
    for attempt_round in range(3):
        if time.time() - start_time > timeout - 15:
            break
        
        removed = True
        while removed:
            removed = False
            current_rank = best_U.shape[1]
            for r in range(current_rank):
                if time.time() - start_time > timeout - 10:
                    break
                
                U_test = np.delete(best_U, r, axis=1)
                V_test = np.delete(best_V, r, axis=1)
                W_test = np.delete(best_W, r, axis=1)
                
                if verify_fast(U_test, V_test, W_test):
                    best_U, best_V, best_W = U_test, V_test, W_test
                    best_rank -= 1
                    removed = True
                    break
    
    # Try structured column merging: check if two columns can be combined
    # If col_a and col_b have proportional patterns in U, V, W, they might merge
    # This is harder, so let's try small perturbations on the rank-44 solution
    
    if best_rank == 44 and time.time() - start_time < timeout - 20:
        # Try to see if we can find a rank-43 by perturbing and re-solving
        # For each pair of columns, check if they're "close" to being dependent
        # in the tensor sense
        
        np.random.seed(42)
        for trial in range(50):
            if time.time() - start_time > timeout - 10:
                break
            
            # Try small random perturbations to see if we can find a valid rank-43
            # by checking if any column can be expressed as combination of others
            # in the appropriate tensor factor sense
            
            # Alternative: try different Strassen variants or coefficient choices
            # Let's try if using different signs or coefficients helps
            
            # For now, just try removing pairs of columns
            current_rank = best_U.shape[1]
            for r1 in range(current_rank):
                for r2 in range(r1+1, current_rank):
                    if time.time() - start_time > timeout - 10:
                        break
                    # Can't remove two at once easily, skip for now
                    pass
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
