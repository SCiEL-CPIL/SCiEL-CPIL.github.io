import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin before 150s kill
    
    n, m, p = 3, 4, 4
    
    def verify(U, V, W):
        """Verify the decomposition is correct."""
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        rank = U.shape[1]
        
        # Build expected tensor
        expected = np.zeros((n*m, m*p, p*n))
        for a in range(n*m):
            a_row, a_col = divmod(a, m)
            for b in range(m*p):
                b_row, b_col = divmod(b, p)
                if a_col == b_row:
                    c_idx = b_col * n + a_row
                    expected[a, b, c_idx] = 1.0
        
        # Compute actual tensor
        # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # For fixed c: T[:,:,c] = (U * W[c,:][None,:]) @ V.T
        for c_idx in range(p*n):
            actual = (U * W[c_idx, :][None, :]) @ V.T
            if not np.allclose(actual, expected[:, :, c_idx], atol=1e-10):
                return False
        return True
    
    def build_rank44():
        """Build rank-44 decomposition using 4 blocks of 3x2x2 at rank 11."""
        rank = 44
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        # Local 3x2x2 factor with rank 11
        U_loc = np.zeros((6, 11))
        V_loc = np.zeros((4, 11))
        W_loc = np.zeros((6, 11))
        
        # Strassen for rows 0,1 (cols 0-6)
        # M1: (a00+a11)(b00+b11) -> c00+=M1, c11+=M1
        U_loc[0, 0] = 1; U_loc[3, 0] = 1
        V_loc[0, 0] = 1; V_loc[3, 0] = 1
        W_loc[0, 0] = 1; W_loc[3, 0] = 1
        
        # M2: (a10+a11)b00 -> c10+=M2, c11-=M2
        U_loc[2, 1] = 1; U_loc[3, 1] = 1
        V_loc[0, 1] = 1
        W_loc[2, 1] = 1; W_loc[3, 1] = -1
        
        # M3: a00(b01-b11) -> c01+=M3, c11+=M3
        # Note: Standard Strassen M3 is A00(B11-B01), but here we use A00(B01-B11)
        # to match the W signs derived earlier.
        U_loc[0, 2] = 1
        V_loc[1, 2] = 1; V_loc[3, 2] = -1
        W_loc[1, 2] = 1; W_loc[3, 2] = 1
        
        # M4: a11(b10-b00) -> c00+=M4, c10+=M4
        U_loc[3, 3] = 1
        V_loc[2, 3] = 1; V_loc[0, 3] = -1
        W_loc[0, 3] = 1; W_loc[2, 3] = 1
        
        # M5: (a00+a01)b11 -> c00-=M5, c01+=M5
        U_loc[0, 4] = 1; U_loc[1, 4] = 1
        V_loc[3, 4] = 1
        W_loc[0, 4] = -1; W_loc[1, 4] = 1
        
        # M6: (a10-a00)(b00+b01) -> c11+=M6
        U_loc[2, 5] = 1; U_loc[0, 5] = -1
        V_loc[0, 5] = 1; V_loc[1, 5] = 1
        W_loc[3, 5] = 1
        
        # M7: (a01-a11)(b10+b11) -> c00+=M7
        U_loc[1, 6] = 1; U_loc[3, 6] = -1
        V_loc[2, 6] = 1; V_loc[3, 6] = 1
        W_loc[0, 6] = 1
        
        # Trivial for row 2 (cols 7-10)
        U_loc[4, 7] = 1; V_loc[0, 7] = 1; W_loc[4, 7] = 1
        U_loc[5, 8] = 1; V_loc[2, 8] = 1; W_loc[5, 8] = 1
        U_loc[4, 9] = 1; V_loc[1, 9] = 1; W_loc[5, 9] = 1
        U_loc[5, 10] = 1; V_loc[3, 10] = 1; W_loc[5, 10] = 1
        
        def add_block(block_idx, a_col_start, b_row_start, b_col_start, c_col_start):
            col_offset = block_idx * 11
            for r in range(3):
                for c in range(2):
                    la = r * 2 + c
                    ga = r * m + (a_col_start + c)
                    U[ga, col_offset:col_offset+11] += U_loc[la, :]
            for r in range(2):
                for c in range(2):
                    lb = r * 2 + c
                    gb = (b_row_start + r) * p + (b_col_start + c)
                    V[gb, col_offset:col_offset+11] += V_loc[lb, :]
            for r in range(3):
                for c in range(2):
                    lc = r * 2 + c
                    gc = (c_col_start + c) * n + r
                    W[gc, col_offset:col_offset+11] += W_loc[lc, :]
        
        add_block(0, 0, 0, 0, 0)
        add_block(1, 2, 2, 0, 0)
        add_block(2, 0, 0, 2, 2)
        add_block(3, 2, 2, 2, 2)
        
        return U, V, W
    
    # Build initial rank-44 decomposition
    U, V, W = build_rank44()
    
    # Verify
    if not verify(U, V, W):
        # Fallback to schoolbook
        rank_school = n * m * p
        U_fb = np.zeros((n*m, rank_school))
        V_fb = np.zeros((m*p, rank_school))
        W_fb = np.zeros((p*n, rank_school))
        
        idx = 0
        for a in range(n*m):
            a_row, a_col = divmod(a, m)
            for b in range(m*p):
                b_row, b_col = divmod(b, p)
                if a_col == b_row:
                    c_idx = b_col * n + a_row
                    U_fb[a, idx] = 1
                    V_fb[b, idx] = 1
                    W_fb[c_idx, idx] = 1
                    idx += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
