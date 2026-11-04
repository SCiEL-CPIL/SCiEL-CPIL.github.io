import numpy as np

def construct():
    n, m, p = 3, 3, 7
    # We will build U, V, W as lists of columns, then transpose/reshape
    # U is (n*m, rank) -> (9, R)
    # V is (m*p, rank) -> (21, R)
    # W is (p*n, rank) -> (21, R)
    
    # We will accumulate columns
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_block(U_block, V_block, W_block, i_offset=0, j_offset=0, k_offset=0, k_out_offset=0):
        # U_block is (n_sub * m_sub, r_sub)
        # V_block is (m_sub * p_sub, r_sub)
        # W_block is (p_sub * n_sub, r_sub)
        # Indices in U: i, j
        # Indices in V: j, k
        # Indices in W: k, i
        
        r_sub = U_block.shape[1]
        for c in range(r_sub):
            u_col = np.zeros(n * m)
            v_col = np.zeros(m * p)
            w_col = np.zeros(p * n)
            
            # Fill U
            # U_block rows are indexed by i_sub * m_sub + j_sub
            for idx in range(U_block.shape[0]):
                val = U_block[idx, c]
                if val != 0:
                    i_sub = idx // m_sub
                    j_sub = idx % m_sub
                    i_glob = i_offset + i_sub
                    j_glob = j_offset + j_sub
                    u_col[i_glob * m + j_glob] = val
                    
            # Fill V
            # V_block rows are indexed by j_sub * p_sub + k_sub
            for idx in range(V_block.shape[0]):
                val = V_block[idx, c]
                if val != 0:
                    j_sub = idx // p_sub
                    k_sub = idx % p_sub
                    j_glob = j_offset + j_sub
                    k_glob = k_offset + k_sub
                    v_col[j_glob * p + k_glob] = val
                    
            # Fill W
            # W_block rows are indexed by k_sub * n_sub + i_sub
            for idx in range(W_block.shape[0]):
                val = W_block[idx, c]
                if val != 0:
                    k_sub = idx // n_sub
                    i_sub = idx % n_sub
                    k_glob = k_out_offset + k_sub
                    i_glob = i_offset + i_sub
                    w_col[k_glob * n + i_glob] = val
                    
            U_cols.append(u_col)
            V_cols.append(v_col)
            W_cols.append(w_col)
            
    # 1. M_333 (Rank 23)
    # We need a 23-rank decomposition for 3x3.
    # I will use a known one.
    # If I can't, I'll use schoolbook 27.
    # Let's use schoolbook for now to ensure correctness, then try to improve.
    # Actually, I can use the 7-rank Strassen for 2x2 and pad.
    # M_333 = M_222 (top-left 2x2) + M_122 (top-right) + M_212 (bottom-left) + M_112 (bottom-right)?
    # No, that's not how it works.
    
    # Let's just use schoolbook for 3x3 (Rank 27) for safety, total 61.
    # Then I will try to swap 3x3 block with 23-rank.
    
    def schoolbook(n_sub, m_sub, p_sub):
        r = n_sub * m_sub * p_sub
        U = np.zeros((n_sub * m_sub, r))
        V = np.zeros((m_sub * p_sub, r))
        W = np.zeros((p_sub * n_sub, r))
        c = 0
        for i in range(n_sub):
            for j in range(m_sub):
                for k in range(p_sub):
                    U[i * m_sub + j, c] = 1
                    V[j * p_sub + k, c] = 1
                    W[k * n_sub + i, c] = 1
                    c += 1
        return U, V, W
    
    # Block 1: 3x3x3 (cols 0-2)
    U1, V1, W1 = schoolbook(3, 3, 3)
    add_block(U1, V1, W1, 0, 0, 0, 0)
    
    # Block 2: 3x3x2 (cols 3-4)
    # Decomposed into M_232 (rows 0-1) and M_132 (row 2)
    # M_232 is 2x3 by 3x2. Rank 11.
    # M_132 is 1x3 by 3x2. Rank 6.
    
    def m_232_decomp():
        # 2x3 by 3x2
        # A is 2x3, B is 3x2
        # Use Strassen for top-left 2x2 of A and B?
        # A = [[A00, A01, A02], [A10, A11, A12]]
        # B = [[B00, B01], [B10, B11], [B20, B21]]
        # This is not square.
        # I will use schoolbook for 2x3x2 (Rank 12) for simplicity.
        # Then total rank 27 + 12 + 6 + 12 + 6 = 63. No improvement.
        # I need the 11-rank for 3x2x2.
        # M_322: 3x2 by 2x2.
        # A is 3x2, B is 2x2.
        # Rows 0-1 of A use Strassen (7). Row 2 of A uses schoolbook (4).
        # Total 11.
        # But we have M_232 (2x3 by 3x2).
        # M_232 is isomorphic to M_322.
        # So I can use the M_322 decomposition.
        # Let's construct M_322.
        pass
        
    # I will implement M_322 (3x2 by 2x2) rank 11.
    def m_322_decomp():
        # n=3, m=2, p=2
        # A: 3x2, B: 2x2
        # C: 3x2
        # Rows 0,1 of A: Use Strassen M_222.
        # Row 2 of A: Schoolbook.
        U = np.zeros((3*2, 11))
        V = np.zeros((2*2, 11))
        W = np.zeros((2*3, 11))
        
        # Strassen part for A[0:2, :] and B
        # M_222 indices: i in 0,1; j in 0,1; k in 0,1
        # Standard Strassen:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # U indices: i*2+j
        # V indices: j*2+k
        # W indices: k*3+i
        
        # Col 0: M1
        U[0*2+0, 0] = 1; U[1*2+1, 0] = 1
        V[0*2+0, 0] = 1; V[1*2+1, 0] = 1
        W[0*3+0, 0] = 1; W[0*3+1, 0] = 1; W[1*3+0, 0] = 1; W[1*3+1, 0] = 1
        
        # Col 1: M2
        U[1*2+0, 1] = 1; U[1*2+1, 1] = 1
        V[0*2+0, 1] = 1
        W[0*3+1, 1] = 1; W[1*3+1, 1] = 1
        
        # Col 2: M3
        U[0*2+0, 2] = 1
        V[0*2+1, 2] = 1; V[1*2+1, 2] = -1
        W[1*3+0, 2] = 1; W[1*3+1, 2] = 1
        
        # Col 3: M4
        U[1*2+1, 3] = 1
        V[1*2+0, 3] = 1; V[0*2+0, 3] = -1
        W[0*3+1, 3] = 1; W[1*3+1, 3] = 1
        
        # Col 4: M5
        U[0*2+0, 4] = 1; U[0*2+1, 4] = 1
        V[1*2+1, 4] = 1
        W[1*3+0, 4] = 1; W[1*3+1, 4] = 1
        
        # Col 5: M6
        U[1*2+0, 5] = 1; U[0*2+0, 5] = -1
        V[0*2+0, 5] = 1; V[0*2+1, 5] = 1
        W[0*3+0, 5] = 1; W[1*3+0, 5] = 1
        
        # Col 6: M7
        U[0*2+1, 6] = 1; U[1*2+1, 6] = -1
        V[1*2+0, 6] = 1; V[1*2+1, 6] = 1
        W[1*3+1, 6] = 1; W[1*3+1, 6] = 1 # Wait, W index is k*3+i.
        # M7 contributes to C01 and C11?
        # C01 = M1 + M3 - M5 - M7
        # C10 = M2 + M4 + M6
        # C11 = M1 - M2 + M4 + M5
        # My W mapping:
        # W[k*3+i]
        # C00 (i=0,k=0): W[0]
        # C01 (i=0,k=1): W[3]
        # C10 (i=1,k=0): W[1]
        # C11 (i=1,k=1): W[4]
        
        # Let's re-verify Strassen W indices.
        # C = [[C00, C01], [C10, C11]]
        # C00 = M1 - M4 - M6 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 - M5 + M6
        
        # Correct Strassen:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # C00 = M1 - M4 - M6 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 - M5 + M6
        
        # So W entries:
        # W[0] (C00): M1: 1, M4: -1, M6: -1, M7: 1
        # W[3] (C01): M3: 1, M5: 1
        # W[1] (C10): M2: 1, M4: 1
        # W[4] (C11): M1: 1, M2: -1, M5: -1, M6: 1
        
        # I will rewrite the U,V,W for Strassen correctly.
        U = np.zeros((6, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7)) # For 2x2x2, W is 2x2=4. But here n_sub=2 for the block.
        # Wait, in m_322_decomp, the Strassen part is for rows 0,1 of A (2x2) and B (2x2).
        # So the sub-problem is M_222.
        # n_sub=2, m_sub=2, p_sub=2.
        # U is 4x7, V is 4x7, W is 4x7.
        
        # Col 0 (M1)
        U[0,0]=1; U[3,0]=1 # A00, A11
        V[0,0]=1; V[3,0]=1 # B00, B11
        W[0,0]=1; W[1,0]=-1; W[2,0]=-1; W[3,0]=1 # C00, C10, C01, C11?
        # W index: k*n+i.
        # k=0,i=0 -> 0 (C00)
        # k=0,i=1 -> 1 (C10)
        # k=1,i=0 -> 2 (C01)
        # k=1,i=1 -> 3 (C11)
        
        W[0,0]=1 # M1
        W[1,0]=0 # M1
        W[2,0]=0 # M1
        W[3,0]=1 # M1
        
        # Col 1 (M2)
        U[2,1]=1; U[3,1]=1 # A10, A11
        V[0,1]=1 # B00
        W[0,1]=0
        W[1,1]=1 # M2
        W[2,1]=0
        W[3,1]=-1 # M2
        
        # Col 2 (M3)
        U[0,2]=1 # A00
        V[1,2]=1; V[3,2]=-1 # B01, B11
        W[0,2]=0
        W[1,2]=0
        W[2,2]=1 # M3
        W[3,2]=0
        
        # Col 3 (M4)
        U[3,3]=1 # A11
        V[2,3]=1; V[0,3]=-1 # B10, B00
        W[0,3]=-1 # M4
        W[1,3]=1 # M4
        W[2,3]=0
        W[3,3]=0
        
        # Col 4 (M5)
        U[0,4]=1; U[1,4]=1 # A00, A01
        V[3,4]=1 # B11
        W[0,4]=0
        W[1,4]=0
        W[2,4]=1 # M5
        W[3,4]=-1 # M5
        
        # Col 5 (M6)
        U[2,5]=1; U[0,5]=-1 # A10, A00
        V[0,5]=1; V[1,5]=1 # B00, B01
        W[0,5]=-1 # M6
        W[1,5]=0
        W[2,5]=0
        W[3,5]=1 # M6
        
        # Col 6 (M7)
        U[1,6]=1; U[3,6]=-1 # A01, A11
        V[2,6]=1; V[3,6]=1 # B10, B11
        W[0,6]=1 # M7
        W[1,6]=0
        W[2,6]=0
        W[3,6]=0
        
        # Now add schoolbook for row 2 of A (1x2 by 2x2)
        # A20, A21
        # C20 = A20 B00 + A21 B10
        # C21 = A20 B01 + A21 B11
        # 4 multiplications.
        # Total rank 7+4=11.
        
        U_full = np.zeros((6, 11))
        V_full = np.zeros((4, 11))
        W_full = np.zeros((4, 11)) # Wait, W is for the whole 3x2x2?
        # In m_322_decomp, n_sub=3, m_sub=2, p_sub=2.
        # U is 6x11. V is 4x11. W is 6x11 (2*3).
        
        # Copy Strassen part to U_full[0:4, 0:7], V_full[0:4, 0:7], W_full[0:4, 0:7]
        U_full[0:4, 0:7] = U
        V_full[0:4, 0:7] = V
        W_full[0:4, 0:7] = W
        
        # Schoolbook part for row 2
        # A index: i=2, j=0,1 -> 4,5
        # B index: j=0,1; k=0,1 -> 0,1,2,3
        # C index: k=0,1; i=2 -> 4,5 (k*3+i: 0*3+2=2, 1*3+2=5)
        
        # Col 7: A20 B00
        U_full[4, 7] = 1
        V_full[0, 7] = 1
        W_full[2, 7] = 1 # C20
        
        # Col 8: A21 B10
        U_full[5, 8] = 1
        V_full[2, 8] = 1
        W_full[2, 8] = 1 # C20
        
        # Col 9: A20 B01
        U_full[4, 9] = 1
        V_full[1, 9] = 1
        W_full[5, 9] = 1 # C21
        
        # Col 10: A21 B11
        U_full[5, 10] = 1
        V_full[3, 10] = 1
        W_full[5, 10] = 1 # C21
        
        return U_full, V_full, W_full

    # Add Block 2a: M_322 for rows 0-1 of A, cols 3-4 of B
    U2a, V2a, W2a = m_322_decomp()
    # Note: m_322_decomp assumes n_sub=3, m_sub=2, p_sub=2.
    # But we are applying it to rows 0-1 of A (2 rows) and cols 3-4 of B (2 cols).
    # The A part is 2x3? No, A is 3x3. We are taking rows 0,1. So A_sub is 2x3.
    # But m_322_decomp is for 3x2 by 2x2.
    # We need M_232 (2x3 by 3x2).
    # M_232 is isomorphic to M_322.
    # I will just use schoolbook for M_232 (Rank 12) to be safe.
    # Then total rank 27 + 12 + 6 + 12 + 6 = 63.
    
    # I will stick to 63 for now? No, I must improve.
    # I will use the 11-rank for M_322 and map it.
    # M_322: A(3x2), B(2x2).
    # M_232: A(2x3), B(3x2).
    # This is not the same.
    # I will use schoolbook for M_232.
    
    # So Rank = 27 (M_333) + 12 (M_232) + 6 (M_132) + 12 (M_232) + 6 (M_132) = 63.
    # This is not an improvement.
    
    # I need a better decomposition for M_333 or M_232.
    # If I use 23 for M_333, Rank = 23 + 12 + 6 + 12 + 6 = 59.
    # This is an improvement.
    
    # I will implement 23-rank M_333.
    
    return {"U": np.array(U_cols).T, "V": np.array(V_cols).T, "W": np.array(W_cols).T}

