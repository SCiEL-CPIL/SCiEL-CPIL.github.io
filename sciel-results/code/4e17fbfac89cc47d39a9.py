import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill
    
    n, m, p = 3, 4, 4
    rank = 44
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Local factors for 3x2x2 multiplication
    # A is 3x2, B is 2x2, C is 3x2
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
    
    # Tile 4 blocks
    def add_block(block_idx, a_col_start, b_row_start, b_col_start, c_col_start):
        col_off = block_idx * 11
        for r in range(3):
            for c in range(2):
                la = r * 2 + c
                ga = r * m + (a_col_start + c)
                U[ga, col_off:col_off+11] += U_loc[la, :]
        for r in range(2):
            for c in range(2):
                lb = r * 2 + c
                gb = (b_row_start + r) * p + (b_col_start + c)
                V[gb, col_off:col_off+11] += V_loc[lb, :]
        for r in range(3):
            for c in range(2):
                lc = r * 2 + c
                gc = (c_col_start + c) * n + r
                W[gc, col_off:col_off+11] += W_loc[lc, :]
    
    add_block(0, 0, 0, 0, 0)
    add_block(1, 2, 2, 0, 0)
    add_block(2, 0, 0, 2, 2)
    add_block(3, 2, 2, 2, 2)
    
    # Quick verification (vectorized)
    if time.time() < deadline:
        Ua = np.asarray(U)
        Va = np.asarray(V)
        Wa = np.asarray(W)
        
        # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        # For each (a,b), compute T[a,b,:] = sum_r (U[a,r]*V[b,r]) * W[:,r]^T
        # = (U[a,:] * V[b,:]) @ W.T
        # Check against expected
        # Expected: T[a,b,c] = 1 iff a_col==b_row, c_row==a_row, c_col==b_col
        # a = a_r*m + a_c, b = b_r*p + b_c, c = c_c*n + c_r
        # Condition: a_c == b_r and c_r == a_r and c_c == b_c
        
        a_rows = np.arange(n*m)
        b_rows = np.arange(m*p)
        a_rs, a_cs = np.divmod(a_rows, m)
        b_rs, b_cs = np.divmod(b_rows, p)
        
        # For each (a,b) pair, the expected c index is:
        # c_c = b_c, c_r = a_r  (if a_c == b_r), else no valid c
        # c_idx = b_c * n + a_r
        
        valid_mask = (a_cs[:, None] == b_rs[None, :])  # shape (n*m, m*p)
        expected_c_idx = b_cs[None, :] * n + a_rs[:, None]  # shape (n*m, m*p)
        
        # Compute T for all (a,b) pairs
        # T[a,b,:] = (Ua[a,:] * Va[b,:]) @ Wa.T
        # This is: (Ua[:, :, None] * Va[None, :, :]) @ Wa[None, :, :].T
        # Shape: (n*m, m*p, rank) -> @ (rank, p*n) -> (n*m, m*p, p*n)
        # But that's 12*16*16*44 = 135168 ops for the multiply, fine
        
        prod = Ua[:, :, None] * Va[None, :, :]  # (n*m, m*p, rank)
        T = prod @ Wa.T  # (n*m, m*p, p*n)
        
        # Check: for each (a,b) where valid_mask is True, T[a,b,expected_c_idx] should be 1
        # and all other c should be 0
        # For invalid (a,b), all T[a,b,:] should be 0
        
        a_idx, b_idx = np.where(valid_mask)
        if len(a_idx) > 0:
            c_targets = expected_c_idx[a_idx, b_idx]
            vals = T[a_idx, b_idx, c_targets]
            if not np.all(np.abs(vals - 1.0) < 1e-10):
                # Fallback: return trivial rank-48
                rank_fallback = n * m * p  # 48
                Uf = np.zeros((n*m, rank_fallback))
                Vf = np.zeros((m*p, rank_fallback))
                Wf = np.zeros((p*n, rank_fallback))
                k = 0
                for a_i in range(n*m):
                    ar, ac = divmod(a_i, m)
                    for b_i in range(m*p):
                        br, bc = divmod(b_i, p)
                        if ac == br:
                            c_i = bc * n + ar
                            Uf[a_i, k] = 1
                            Vf[b_i, k] = 1
                            Wf[c_i, k] = 1
                            k += 1
                return {"U": Uf.tolist(), "V": Vf.tolist(), "W": Wf.tolist()}
        
        # Also check invalid pairs give zero
        invalid_mask = ~valid_mask
        a_idx_inv, b_idx_inv = np.where(invalid_mask)
        if len(a_idx_inv) > 0:
            T_inv = T[a_idx_inv, b_idx_inv, :]
            if not np.all(np.abs(T_inv) < 1e-10):
                rank_fallback = n * m * p
                Uf = np.zeros((n*m, rank_fallback))
                Vf = np.zeros((m*p, rank_fallback))
                Wf = np.zeros((p*n, rank_fallback))
                k = 0
                for a_i in range(n*m):
                    ar, ac = divmod(a_i, m)
                    for b_i in range(m*p):
                        br, bc = divmod(b_i, p)
                        if ac == br:
                            c_i = bc * n + ar
                            Uf[a_i, k] = 1
                            Vf[b_i, k] = 1
                            Wf[c_i, k] = 1
                            k += 1
                return {"U": Uf.tolist(), "V": Vf.tolist(), "W": Wf.tolist()}
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
