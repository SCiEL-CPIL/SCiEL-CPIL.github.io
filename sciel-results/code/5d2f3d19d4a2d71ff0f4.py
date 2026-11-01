import time
import random

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=5, p=5
    
    # Initialize U, V, W with 0
    # We will build up the rank
    # Start with a large enough size, e.g., 50, and then trim
    # Or just append.
    
    U_rows = []
    V_rows = []
    W_rows = []
    
    def add_term(u_vec, v_vec, w_vec):
        U_rows.append(u_vec)
        V_rows.append(v_vec)
        W_rows.append(w_vec)
        
    def zero_vec(dim):
        return [0.0] * dim
        
    # Helper to get index
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i # Task says W[k*n+i, r]
    
    # Strassen 2x2x2
    # A is 2x2, B is 2x2, C is 2x2
    # Local indices: a00, a01, a10, a11
    # b00, b01, b10, b11
    # c00, c01, c10, c11
    
    # We need to map local indices to global indices.
    # This function will be called with offsets.
    
    def strassen_2x2x2(i_off, j_off, k_off):
        # A indices: (i_off, j_off), (i_off, j_off+1), (i_off+1, j_off), (i_off+1, j_off+1)
        # B indices: (j_off, k_off), (j_off, k_off+1), (j_off+1, k_off), (j_off+1, k_off+1)
        # C indices: (i_off, k_off), (i_off, k_off+1), (i_off+1, k_off), (i_off+1, k_off+1)
        
        a00 = u_idx(i_off, j_off)
        a01 = u_idx(i_off, j_off+1)
        a10 = u_idx(i_off+1, j_off)
        a11 = u_idx(i_off+1, j_off+1)
        
        b00 = v_idx(j_off, k_off)
        b01 = v_idx(j_off, k_off+1)
        b10 = v_idx(j_off+1, k_off)
        b11 = v_idx(j_off+1, k_off+1)
        
        c00 = w_idx(k_off, i_off)
        c01 = w_idx(k_off, i_off+1) # Wait, C is 2x5? No, C is 2x5.
        # In the block multiplication A_b B_b, B_b is 2x5.
        # So k ranges over 5 values.
        # Strassen applies to 2x2 blocks of B.
        # So we call strassen_2x2x2 for each 2x2 block of B.
        
        # P1 = (a00+a11)(b00+b11)
        u1 = zero_vec(n*m); u1[a00]=1; u1[a11]=1
        v1 = zero_vec(m*p); v1[b00]=1; v1[b11]=1
        w1 = zero_vec(p*n); w1[c00]=1; w1[c01]=1; w1[w_idx(k_off+1, i_off)]=1; w1[w_idx(k_off+1, i_off+1)]=1
        # Wait, P1 contributes to all 4 C entries?
        # C00 += P1, C01 += P1, C10 += P1, C11 += P1?
        # No.
        # C00 = P1 + P4 - P5 + P7
        # C01 = P3 + P5
        # C10 = P2 + P4
        # C11 = P1 - P2 + P3 + P6
        
        # So P1 contributes to C00 and C11.
        w1 = zero_vec(p*n); w1[c00]=1; w1[w_idx(k_off+1, i_off+1)]=1
        add_term(u1, v1, w1)
        
        # P2 = (a10+a11)b00
        u2 = zero_vec(n*m); u2[a10]=1; u2[a11]=1
        v2 = zero_vec(m*p); v2[b00]=1
        w2 = zero_vec(p*n); w2[w_idx(k_off, i_off+1)]=1
        add_term(u2, v2, w2)
        
        # P3 = a00(b01-b11)
        u3 = zero_vec(n*m); u3[a00]=1
        v3 = zero_vec(m*p); v3[b01]=1; v3[b11]=-1
        w3 = zero_vec(p*n); w3[c01]=1
        add_term(u3, v3, w3)
        
        # P4 = a11(b10-b00)
        u4 = zero_vec(n*m); u4[a11]=1
        v4 = zero_vec(m*p); v4[b10]=1; v4[b00]=-1
        w4 = zero_vec(p*n); w4[w_idx(k_off, i_off+1)]=1
        add_term(u4, v4, w4)
        
        # P5 = (a00+a01)b11
        u5 = zero_vec(n*m); u5[a00]=1; u5[a01]=1
        v5 = zero_vec(m*p); v5[b11]=1
        w5 = zero_vec(p*n); w5[c01]=1
        add_term(u5, v5, w5)
        
        # P6 = (a10-a00)(b00+b01)
        u6 = zero_vec(n*m); u6[a10]=1; u6[a00]=-1
        v6 = zero_vec(m*p); v6[b00]=1; v6[b01]=1
        w6 = zero_vec(p*n); w6[w_idx(k_off+1, i_off+1)]=1
        add_term(u6, v6, w6)
        
        # P7 = (a01-a11)(b10+b11)
        u7 = zero_vec(n*m); u7[a01]=1; u7[a11]=-1
        v7 = zero_vec(m*p); v7[b10]=1; v7[b11]=1
        w7 = zero_vec(p*n); w7[c00]=1
        add_term(u7, v7, w7)
        
        # Now add the negative contributions
        # C00 -= P5
        u_neg5 = zero_vec(n*m); u_neg5[a00]=1; u_neg5[a01]=1
        v_neg5 = zero_vec(m*p); v_neg5[b11]=1
        w_neg5 = zero_vec(p*n); w_neg5[c00]=-1
        add_term(u_neg5, v_neg5, w_neg5)
        
        # C00 += P7 (Already added)
        
        # C10 += P4 (Already added)
        
        # C11 -= P2
        u_neg2 = zero_vec(n*m); u_neg2[a10]=1; u_neg2[a11]=1
        v_neg2 = zero_vec(m*p); v_neg2[b00]=1
        w_neg2 = zero_vec(p*n); w_neg2[w_idx(k_off+1, i_off+1)]=-1
        add_term(u_neg2, v_neg2, w_neg2)
        
        # C11 += P3 (Already added? No, P3 was added to C01)
        # Wait, P3 = a00(b01-b11).
        # C01 = P3 + P5.
        # C11 = P1 - P2 + P3 + P6.
        # So P3 contributes to C01 and C11.
        # I added P3 to C01. I need to add it to C11 as well.
        w3b = zero_vec(p*n); w3b[w_idx(k_off+1, i_off+1)]=1
        add_term(u3, v3, w3b)
        
        # C01 += P5 (Already added)
        
        # C10 += P2 (Already added)
        
        # C11 += P6 (Already added)
        
        # C00 += P1 (Already added)
        # C00 -= P5 (Added)
        # C00 += P7 (Added)
        
        # C01 = P3 + P5
        # C10 = P2 + P4
        # C11 = P1 - P2 + P3 + P6
        
        # Check C00: P1 - P5 + P7. Correct.
        # Check C01: P3 + P5. Correct.
        # Check C10: P2 + P4. Correct.
        # Check C11: P1 - P2 + P3 + P6. Correct.
        
    # Schoolbook 2x2x1
    # A 2x2, B 2x1, C 2x1
    def schoolbook_2x2x1(i_off, j_off, k_off):
        # A: (i_off, j_off), (i_off, j_off+1), (i_off+1, j_off), (i_off+1, j_off+1)
        # B: (j_off, k_off), (j_off+1, k_off)
        # C: (i_off, k_off), (i_off+1, k_off)
        
        a00 = u_idx(i_off, j_off)
        a01 = u_idx(i_off, j_off+1)
        a10 = u_idx(i_off+1, j_off)
        a11 = u_idx(i_off+1, j_off+1)
        
        b0 = v_idx(j_off, k_off)
        b1 = v_idx(j_off+1, k_off)
        
        c0 = w_idx(k_off, i_off)
        c1 = w_idx(k_off, i_off+1)
        
        # C0 = A00 B0 + A01 B1
        u1 = zero_vec(n*m); u1[a00]=1
        v1 = zero_vec(m*p); v1[b0]=1
        w1 = zero_vec(p*n); w1[c0]=1
        add_term(u1, v1, w1)
        
        u2 = zero_vec(n*m); u2[a01]=1
        v2 = zero_vec(m*p); v2[b1]=1
        w2 = zero_vec(p*n); w2[c0]=1
        add_term(u2, v2, w2)
        
        # C1 = A10 B0 + A11 B1
        u3 = zero_vec(n*m); u3[a10]=1
        v3 = zero_vec(m*p); v3[b0]=1
        w3 = zero_vec(p*n); w3[c1]=1
        add_term(u3, v3, w3)
        
        u4 = zero_vec(n*m); u4[a11]=1
        v4 = zero_vec(m*p); v4[b1]=1
        w4 = zero_vec(p*n); w4[c1]=1
        add_term(u4, v4, w4)
        
    # Schoolbook 2x1x5
    # A 2x1, B 1x5, C 2x5
    def schoolbook_2x1x5(i_off, j_off, k_off):
        # A: (i_off, j_off), (i_off+1, j_off)
        # B: (j_off, k_off) ... (j_off, k_off+4)
        # C: (i_off, k) ... (i_off+1, k)
        
        a0 = u_idx(i_off, j_off)
        a1 = u_idx(i_off+1, j_off)
        
        for k in range(5):
            bk = v_idx(j_off, k_off + k)
            c0 = w_idx(k_off + k, i_off)
            c1 = w_idx(k_off + k, i_off+1)
            
            u1 = zero_vec(n*m); u1[a0]=1
            v1 = zero_vec(m*p); v1[bk]=1
            w1 = zero_vec(p*n); w1[c0]=1
            add_term(u1, v1, w1)
            
            u2 = zero_vec(n*m); u2[a1]=1
            v2 = zero_vec(m*p); v2[bk]=1
            w2 = zero_vec(p*n); w2[c1]=1
            add_term(u2, v2, w2)

    # Apply to blocks
    # Block 1: j=0,1. k=0,1 (Strassen), k=2,3 (Strassen), k=4 (Schoolbook 2x2x1)
    # Wait, B is 2x5.
    # B_1 is rows 0,1 of B. Columns 0..4.
    # We split columns into 2,2,1.
    # k=0,1: Strassen
    # k=2,3: Strassen
    # k=4: Schoolbook 2x2x1
    
    strassen_2x2x2(0, 0, 0)
    strassen_2x2x2(0, 0, 2)
    schoolbook_2x2x1(0, 0, 4)
    
    # Block 2: j=2,3.
    strassen_2x2x2(0, 2, 0)
    strassen_2x2x2(0, 2, 2)
    schoolbook_2x2x1(0, 2, 4)
    
    # Block 3: j=4.
    schoolbook_2x1x5(0, 4, 0)
    
    # Convert lists to matrices
    rank = len(U_rows)
    U = [[U_rows[r][i] for r in range(rank)] for i in range(n*m)]
    V = [[V_rows[r][i] for r in range(rank)] for i in range(m*p)]
    W = [[W_rows[r][i] for r in range(rank)] for i in range(p*n)]
    
    return {"U": U, "V": V, "W": W}
