import numpy as np
import time

def construct():
    start_time = time.time()
    timeout = 145.0  # Safety margin before the 150s kill
    
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
        # T_rec[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        UV = U[:, None, :] * V[None, :, :]  # (Nm, Mp, R)
        UV_flat = UV.reshape(-1, R)  # (Nm*Mp, R)
        T_rec_flat = UV_flat @ W.T  # (Nm*Mp, Pn)
        T_rec = T_rec_flat.reshape(Nm, Mp, Pn)
        return np.allclose(T_rec, T, atol=1e-8)
    
    # Direct decomposition (rank 48)
    def make_direct():
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
        return U, V, W
    
    U_dir, V_dir, W_dir = make_direct()
    assert verify_fast(U_dir, V_dir, W_dir)
    
    # Block decomposition (rank 44) - from inspirations
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
                    global_a_row = a_row_start + r
                    global_a_col = a_col_start + c
                    global_a_idx = global_a_row * m + global_a_col
                    U[global_a_idx, col_offset:col_offset+11] += U_loc[local_a_idx, :]
            for r in range(2):
                for c in range(2):
                    local_b_idx = r * 2 + c
                    global_b_row = b_row_start + r
                    global_b_col = b_col_start + c
                    global_b_idx = global_b_row * p + global_b_col
                    V[global_b_idx, col_offset:col_offset+11] += V_loc[local_b_idx, :]
            for r in range(3):
                for c in range(2):
                    local_c_idx = r * 2 + c
                    global_c_row = c_row_start + r
                    global_c_col = c_col_start + c
                    global_c_idx = global_c_col * n + global_c_row
                    W[global_c_idx, col_offset:col_offset+11] += W_loc[local_c_idx, :]
        
        add_block(0, 0, 0, 0, 0, 0, 0)
        add_block(1, 0, 2, 2, 0, 0, 0)
        add_block(2, 0, 0, 0, 2, 0, 2)
        add_block(3, 0, 2, 2, 2, 0, 2)
        
        return U, V, W
    
    U_blk, V_blk, W_blk = build_block_decomp()
    
    # Verify the block decomposition
    if verify_fast(U_blk, V_blk, W_blk):
        best_U, best_V, best_W = U_blk, V_blk, W_blk
        best_rank = 44
    else:
        best_U, best_V, best_W = U_dir, V_dir, W_dir
        best_rank = 48
    
    # Try to improve: use iterative optimization
    # The equations are: for each (a,b,c), sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
    
    # Try random search with verification for lower ranks
    np.random.seed(123)
    vals = np.array([-2, -1, -0.5, 0, 0, 0, 0.5, 1, 2], dtype=float)
    
    for target_rank in range(best_rank - 1, max(best_rank - 10, 30), -1):
        if time.time() - start_time > timeout - 15:
            break
        
        found = False
        for attempt in range(30):
            if time.time() - start_time > timeout - 10:
                break
            
            U = np.random.choice(vals, size=(Nm, target_rank))
            V = np.random.choice(vals, size=(Mp, target_rank))
            W = np.random.choice(vals, size=(Pn, target_rank))
            
            if verify_fast(U, V, W):
                best_U, best_V, best_W = U, V, W
                best_rank = target_rank
                found = True
                break
        
        if found:
            continue
    
    # Also try: start from block decomp and try to zero out columns
    if best_rank == 44:
        U_try = best_U.copy()
        V_try = best_V.copy()
        W_try = best_W.copy()
        
        # Try to merge columns: if two columns are linearly dependent in some sense
        for r in range(44):
            if time.time() - start_time > timeout - 10:
                break
            # Check if column r is redundant
            U_test = np.delete(U_try, r, axis=1)
            V_test = np.delete(V_try, r, axis=1)
            W_test = np.delete(W_try, r, axis=1)
            if verify_fast(U_test, V_test, W_test):
                best_U, best_V, best_W = U_test, V_test, W_test
                best_rank -= 1
                U_try, V_try, W_try = U_test, V_test, W_test
                # Restart loop since indices shifted
                r = 0
    
    # Try a more systematic search: perturb the block decomposition
    if time.time() - start_time < timeout - 20:
        # Try small perturbations to see if we can reduce rank
        for perturb_rank in range(best_rank - 1, max(best_rank - 5, 30), -1):
            if time.time() - start_time > timeout - 15:
                break
            
            found = False
            for attempt in range(20):
                if time.time() - start_time > timeout - 10:
                    break
                
                # Start from block decomp and try to reduce rank by zeroing some entries
                U_pert = best_U.copy()
                V_pert = best_V.copy()
                W_pert = best_W.copy()
                
                # Try to zero out random entries
                for _ in range(10):
                    if time.time() - start_time > timeout - 10:
                        break
                    row = np.random.randint(0, Nm)
                    col = np.random.randint(0, U_pert.shape[1])
                    U_pert[row, col] = 0
                    if verify_fast(U_pert, V_pert, W_pert):
                        break
                
                if verify_fast(U_pert, V_pert, W_pert) and U_pert.shape[1] < best_rank:
                    # Check if we can reduce the rank
                    # Remove zero columns
                    mask = np.any(U_pert != 0, axis=0) | np.any(V_pert != 0, axis=0) | np.any(W_pert != 0, axis=0)
                    if np.sum(mask) < best_rank:
                        U_pert = U_pert[:, mask]
                        V_pert = V_pert[:, mask]
                        W_pert = W_pert[:, mask]
                        if verify_fast(U_pert, V_pert, W_pert):
                            best_U, best_V, best_W = U_pert, V_pert, W_pert
                            best_rank = U_pert.shape[1]
                            found = True
                            break
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
