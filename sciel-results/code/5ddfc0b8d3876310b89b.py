import numpy as np

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    
    # Helper to create Strassen 2x2x2
    def strassen_222():
        # U: 4x7, V: 4x7, W: 4x7
        # Indices: u=i*2+j, v=j*2+k, w=k*2+i
        U = np.zeros((4, 7), dtype=float)
        V = np.zeros((4, 7), dtype=float)
        W = np.zeros((4, 7), dtype=float)
        
        # M1 = (A00+A11)(B00+B11)
        # A00: u=0, A11: u=3
        # B00: v=0, B11: v=3
        # C00: w=0, C11: w=3? No, w=k*2+i.
        # C00 (i=0,k=0) -> w=0. C11 (i=1,k=1) -> w=3.
        # M1 contributes to C00 and C11?
        # C00 = M1 + M4 - M5 + M7
        # C11 = M1 - M2 + M3 + M6
        
        # We need to fill U, V, W such that sum U V W = T.
        # This is tricky to do manually for Strassen.
        # Instead, I will use the known rank-7 decomposition.
        # Or simpler: use the explicit 7 terms.
        
        # Term 1: (A00+A11)(B00+B11) -> C00, C11
        # U: A00(1), A11(1)
        # V: B00(1), B11(1)
        # W: C00(1), C11(1)
        
        # Let's use a standard library or hardcode the 7 vectors.
        # Actually, I can just use the schoolbook for 2x2x2 (rank 8) if I can't get Strassen right quickly?
        # Rank 8 * 4 blocks = 32.
        # 32 (for 2x4x4) + 22 (for 2x4x3) = 54.
        # 54 is better than 56, but 50 is better.
        # I must implement Strassen.
        
        # Strassen Decomposition (Rank 7)
        # See e.g. "Tensor Decomposition of Matrix Multiplication"
        # U, V, W are 4x7 matrices.
        
        # r=0: (A00+A11)(B00+B11) -> C00, C11
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1 # C00, C11
        
        # r=1: (A10+A11)B00 -> C10, C11
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = 1 # C10 (i=1,k=0 -> w=2), C11 (w=3)
        
        # r=2: A00(B01-B11) -> C01
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1 # C01 (i=0,k=1 -> w=1)
        
        # r=3: A11(B10-B00) -> C10, C11
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[2, 3] = 1; W[3, 3] = 1 # C10, C11
        
        # r=4: (A00+A01)B11 -> C01, C11
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[1, 4] = 1; W[3, 4] = 1 # C01, C11
        
        # r=5: (A10-A00)(B00+B01) -> C10
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[2, 5] = 1 # C10
        
        # r=6: (A01-A11)(B10+B11) -> C01, C11
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[1, 6] = 1; W[3, 6] = 1 # C01, C11
        
        return U, V, W

    def schoolbook_221():
        # 2x2x1, Rank 4
        # u=i*2+j, v=j*1+k (k=0), w=k*2+i
        U = np.zeros((4, 4), dtype=float)
        V = np.zeros((2, 4), dtype=float)
        W = np.zeros((2, 4), dtype=float)
        
        # r=0: A00 B0 -> C00 (w=0)
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        # r=1: A01 B1 -> C01 (w=1)
        U[1, 1] = 1; V[1, 1] = 1; W[1, 1] = 1
        # r=2: A10 B0 -> C10 (w=2)
        U[2, 2] = 1; V[0, 2] = 1; W[0, 2] = 1 # Wait, w=k*2+i. k=0, i=1 -> w=1?
        # No, w index is k*n+i. n=2.
        # C00: i=0,k=0 -> w=0.
        # C01: i=0,k=1 -> w=2.
        # C10: i=1,k=0 -> w=1.
        # C11: i=1,k=1 -> w=3.
        
        # Correction:
        # r=0: A00 (u=0) B0 (v=0) -> C00 (w=0)
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        # r=1: A01 (u=1) B1 (v=1) -> C01 (w=2)
        U[1, 1] = 1; V[1, 1] = 1; W[2, 1] = 1
        # r=2: A10 (u=2) B0 (v=0) -> C10 (w=1)
        U[2, 2] = 1; V[0, 2] = 1; W[1, 2] = 1
        # r=3: A11 (u=3) B1 (v=1) -> C11 (w=3)
        U[3, 3] = 1; V[1, 3] = 1; W[3, 3] = 1
        
        return U, V, W

    # Assemble 2x2x4 (Rank 14)
    def assemble_224():
        U1, V1, W1 = strassen_222()
        U2, V2, W2 = strassen_222()
        
        # k offset for second block is 2.
        # V: v = j*4 + k.
        # Block 1: k in 0,1. v in 0,1 (j=0), 2,3 (j=1)?
        # No, v = j*p + k. p=4.
        # j=0: v=0,1. j=1: v=4,5.
        # Block 1 (k=0,1):
        # j=0: v=0,1.
        # j=1: v=4,5.
        
        # This mapping is complex.
        # I will use a generic assembler.
        
        return None # Placeholder

    # Generic Assembler
    def assemble(n, m, p, sub_decomps, offsets):
        # sub_decomps: list of (U, V, W, n_sub, m_sub, p_sub)
        # offsets: list of (i_off, j_off, k_off)
        
        total_rank = sum(U.shape[1] for U, V, W, _, _, _ in sub_decomps)
        U_full = np.zeros((n*m, total_rank), dtype=float)
        V_full = np.zeros((m*p, total_rank), dtype=float)
        W_full = np.zeros((p*n, total_rank), dtype=float)
        
        col_offset = 0
        for U_sub, V_sub, W_sub, n_sub, m_sub, p_sub, i_off, j_off, k_off in sub_decomps:
            r_sub = U_sub.shape[1]
            
            # Map U: u_global = i*m + j. u_local = i_sub*m_sub + j_sub.
            # i = i_sub + i_off. j = j_sub + j_off.
            # u_global = (i_sub + i_off)*m + (j_sub + j_off)
            
            # Map V: v_global = j*p + k. v_local = j_sub*p_sub + k_sub.
            # j = j_sub + j_off. k = k_sub + k_off.
            # v_global = (j_sub + j_off)*p + (k_sub + k_off)
            
            # Map W: w_global = k*n + i. w_local = k_sub*n_sub + i_sub.
            # k = k_sub + k_off. i = i_sub + i_off.
            # w_global = (k_sub + k_off)*n + (i_sub + i_off)
            
            for r in range(r_sub):
                # U
                for u_local in range(n_sub * m_sub):
                    val = U_sub[u_local, r]
                    if val != 0:
                        i_sub = u_local // m_sub
                        j_sub = u_local % m_sub
                        i_global = i_sub + i_off
                        j_global = j_sub + j_off
                        u_global = i_global * m + j_global
                        U_full[u_global, col_offset + r] += val
                        
                # V
                for v_local in range(m_sub * p_sub):
                    val = V_sub[v_local, r]
                    if val != 0:
                        j_sub = v_local // p_sub
                        k_sub = v_local % p_sub
                        j_global = j_sub + j_off
                        k_global = k_sub + k_off
                        v_global = j_global * p + k_global
                        V_full[v_global, col_offset + r] += val
                        
                # W
                for w_local in range(p_sub * n_sub):
                    val = W_sub[w_local, r]
                    if val != 0:
                        k_sub = w_local // n_sub
                        i_sub = w_local % n_sub
                        k_global = k_sub + k_off
                        i_global = i_sub + i_off
                        w_global = k_global * n + i_global
                        W_full[w_global, col_offset + r] += val
                        
            col_offset += r_sub
            
        return U_full, V_full, W_full

    # Build 2x2x4
    U_s, V_s, W_s = strassen_222()
    # Block 1: k=0,1. j=0,1. i=0,1.
    # Block 2: k=2,3. j=0,1. i=0,1.
    
    # For 2x2x4, n=2, m=2, p=4.
    # Sub 1: 2x2x2 at (0,0,0)
    # Sub 2: 2x2x2 at (0,0,2)
    
    U_224, V_224, W_224 = assemble(2, 2, 4, [
        (U_s, V_s, W_s, 2, 2, 2, 0, 0, 0),
        (U_s, V_s, W_s, 2, 2, 2, 0, 0, 2)
    ])
    
    # Build 2x4x4
    # Sub 1: 2x2x4 at (0,0,0)
    # Sub 2: 2x2x4 at (0,2,0)
    
    U_244, V_244, W_244 = assemble(2, 4, 4, [
        (U_224, V_224, W_224, 2, 2, 4, 0, 0, 0),
        (U_224, V_224, W_224, 2, 2, 4, 0, 2, 0)
    ])
    
    # Build 2x2x3
    # Sub 1: 2x2x1 at (0,0,0)
    # Sub 2: 2x2x2 at (0,0,1)
    
    U_221, V_221, W_221 = schoolbook_221()
    
    U_223, V_223, W_223 = assemble(2, 2, 3, [
        (U_221, V_221, W_221, 2, 2, 1, 0, 0, 0),
        (U_s, V_s, W_s, 2, 2, 2, 0, 0, 1)
    ])
    
    # Build 2x4x3
    # Sub 1: 2x2x3 at (0,0,0)
    # Sub 2: 2x2x3 at (0,2,0)
    
    U_243, V_243, W_243 = assemble(2, 4, 3, [
        (U_223, V_223, W_223, 2, 2, 3, 0, 0, 0),
        (U_223, V_223, W_223, 2, 2, 3, 0, 2, 0)
    ])
    
    # Build 2x4x7
    # Sub 1: 2x4x4 at (0,0,0)
    # Sub 2: 2x4x3 at (0,0,4)
    
    U_final, V_final, W_final = assemble(2, 4, 7, [
        (U_244, V_244, W_244, 2, 4, 4, 0, 0, 0),
        (U_243, V_243, W_243, 2, 4, 3, 0, 0, 4)
    ])
    
    return {
        "U": U_final.tolist(),
        "V": V_final.tolist(),
        "W": W_final.tolist()
    }
