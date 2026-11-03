import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Target tensor
    target = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target[a, b, c] = 1

    # Rank 72 construction
    rank = 72
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Helper to add a term
    def add_term(u_vec, v_vec, w_vec):
        nonlocal col
        if col >= rank:
            return
        U[:, col] = u_vec
        V[:, col] = v_vec
        W[:, col] = w_vec
        col += 1

    # Strassen 2x2 terms
    # A indices: 00, 01, 10, 11
    # B indices: 00, 01, 10, 11
    # C indices: 00, 01, 10, 11
    
    def strassen_2x2(a_off, b_off, c_off):
        # a_off, b_off, c_off are offsets in the 4x4, 4x5, 4x5 grids
        # We need to map local 2x2 indices to global indices
        # Local (r, c) -> Global (a_off + r, a_off + c) for A? No, A is 4x4.
        # Let's define the local indices explicitly.
        
        # A local indices:
        # 00: (0,0), 01: (0,1), 10: (1,0), 11: (1,1)
        # Global A index for local (lr, lc) is (a_off_r + lr, a_off_c + lc)
        # A is 4x4, so a_off_r, a_off_c are 0 or 2.
        
        # B local indices:
        # 00: (0,0), 01: (0,1), 10: (1,0), 11: (1,1)
        # Global B index for local (lr, lc) is (b_off_r + lr, b_off_c + lc)
        # B is 4x5, so b_off_r is 0 or 2, b_off_c is 0, 2, or 4 (for 2x1 case).
        
        # C local indices:
        # 00: (0,0), 01: (0,1), 10: (1,0), 11: (1,1)
        # Global C index for local (lr, lc) is (c_off_r + lr, c_off_c + lc)
        # C is 4x5, so c_off_r is 0 or 2, c_off_c is 0, 2, or 4.
        
        pass

    # This is getting complicated to hardcode. Let's use a simpler approach.
    # We can just use the schoolbook for the whole thing if we can't easily generate the 72 terms.
    # But 72 is a solid improvement over 80.
    
    # Let's try to generate the 72 terms programmatically.
    
    # Define the 7 Strassen terms for 2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # We need to define the linear forms for A, B, and C.
    
    def get_strassen_terms(a_r, a_c, b_r, b_c, c_r, c_c):
        # a_r, a_c: row, col offset for A (0 or 2)
        # b_r, b_c: row, col offset for B (0 or 2, 0, 2, or 4)
        # c_r, c_c: row, col offset for C (0 or 2, 0, 2, or 4)
        
        # Local indices 0,1
        # A global indices:
        # A00: (a_r, a_c), A01: (a_r, a_c+1), A10: (a_r+1, a_c), A11: (a_r+1, a_c+1)
        # B global indices:
        # B00: (b_r, b_c), B01: (b_r, b_c+1), B10: (b_r+1, b_c), B11: (b_r+1, b_c+1)
        # C global indices:
        # C00: (c_r, c_c), C01: (c_r, c_c+1), C10: (c_r+1, c_c), C11: (c_r+1, c_c+1)
        
        terms = []
        
        # Helper to create a vector of coefficients for A, B, C
        def make_vec(indices, coeff=1):
            # indices is a list of (row, col) tuples
            vec = np.zeros((n, m, p), dtype=float) # This is not the right shape for U, V, W
            # U is (n*m, rank). Index is i*m + j.
            # V is (m*p, rank). Index is j*p + k.
            # W is (p*n, rank). Index is k*n + i.
            
            # We need to return three vectors: u, v, w
            # u has non-zeros at A indices
            # v has non-zeros at B indices
            # w has non-zeros at C indices
            
            u = np.zeros(n*m)
            v = np.zeros(m*p)
            w = np.zeros(p*n)
            
            # This function is too generic. Let's hardcode the 7 terms for each block.
            pass

    # Given the complexity, let's stick to the schoolbook rank 80 but try to optimize the search for a lower rank using the 72-rank structure as a starting point or just return 80 if we can't easily implement 72.
    # However, 72 is a significant improvement.
    
    # Let's implement the 72-rank decomposition by explicitly defining the terms.
    
    # We will define a function that takes the block offsets and adds the 7 (or 4) terms.
    
    def add_strassen_block(a_r, a_c, b_r, b_c, c_r, c_c):
        nonlocal col
        # A indices
        a00 = a_r * m + a_c
        a01 = a_r * m + a_c + 1
        a10 = (a_r + 1) * m + a_c
        a11 = (a_r + 1) * m + a_c + 1
        
        # B indices
        b00 = b_r * p + b_c
        b01 = b_r * p + b_c + 1
        b10 = (b_r + 1) * p + b_c
        b11 = (b_r + 1) * p + b_c + 1
        
        # C indices
        c00 = c_c * n + c_r
        c01 = c_c * n + c_r + 1
        c10 = (c_c + 1) * n + c_r
        c11 = (c_c + 1) * n + c_r + 1
        
        # M1 = (A00+A11)(B00+B11)
        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        u[a00] = 1; u[a11] = 1
        v[b00] = 1; v[b11] = 1
        w[c00] = 1; w[c01] = 1; w[c10] = 1; w[c11] = 1
        # C00 += M1, C01 += 0, C10 += 0, C11 += M1? No.
        # C00 = M1 + M4 - M5 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M7
        
        # We need to add 7 terms.
        
        # Term 1: (A00+A11)(B00+B11)
        u1 = np.zeros(n*m); v1 = np.zeros(m*p); w1 = np.zeros(p*n)
        u1[a00] = 1; u1[a11] = 1
        v1[b00] = 1; v1[b11] = 1
        w1[c00] = 1; w1[c11] = 1
        add_term(u1, v1, w1)
        
        # Term 2: (A10+A11)B00
        u2 = np.zeros(n*m); v2 = np.zeros(m*p); w2 = np.zeros(p*n)
        u2[a10] = 1; u2[a11] = 1
        v2[b00] = 1
        w2[c10] = 1
        add_term(u2, v2, w2)
        
        # Term 3: A00(B10-B11)
        u3 = np.zeros(n*m); v3 = np.zeros(m*p); w3 = np.zeros(p*n)
        u3[a00] = 1
        v3[b10] = 1; v3[b11] = -1
        w3[c01] = 1
        add_term(u3, v3, w3)
        
        # Term 4: A11(B01-B00)
        u4 = np.zeros(n*m); v4 = np.zeros(m*p); w4 = np.zeros(p*n)
        u4[a11] = 1
        v4[b01] = 1; v4[b00] = -1
        w4[c00] = 1; w4[c10] = 1
        add_term(u4, v4, w4)
        
        # Term 5: (A00+A01)B11
        u5 = np.zeros(n*m); v5 = np.zeros(m*p); w5 = np.zeros(p*n)
        u5[a00] = 1; u5[a01] = 1
        v5[b11] = 1
        w5[c01] = 1; w5[c00] = -1
        add_term(u5, v5, w5)
        
        # Term 6: (A10-A00)(B00+B01)
        u6 = np.zeros(n*m); v6 = np.zeros(m*p); w6 = np.zeros(p*n)
        u6[a10] = 1; u6[a00] = -1
        v6[b00] = 1; v6[b01] = 1
        w6[c00] = -1
        add_term(u6, v6, w6)
        
        # Term 7: (A01-A11)(B10+B11)
        u7 = np.zeros(n*m); v7 = np.zeros(m*p); w7 = np.zeros(p*n)
        u7[a01] = 1; u7[a11] = -1
        v7[b10] = 1; v7[b11] = 1
        w7[c11] = 1
        add_term(u7, v7, w7)

    def add_schoolbook_2x1(a_r, a_c, b_r, b_c, c_r, c_c):
        # A is 2x2, B is 2x1, C is 2x1
        # C0 = A00 B0 + A01 B1
        # C1 = A10 B0 + A11 B1
        # 4 terms
        nonlocal col
        a00 = a_r * m + a_c
        a01 = a_r * m + a_c + 1
        a10 = (a_r + 1) * m + a_c
        a11 = (a_r + 1) * m + a_c + 1
        
        b0 = b_r * p + b_c
        b1 = (b_r + 1) * p + b_c
        
        c0 = c_c * n + c_r
        c1 = c_c * n + c_r + 1
        
        # T1: A00 B0 -> C0
        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        u[a00] = 1; v[b0] = 1; w[c0] = 1
        add_term(u, v, w)
        
        # T2: A01 B1 -> C0
        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        u[a01] = 1; v[b1] = 1; w[c0] = 1
        add_term(u, v, w)
        
        # T3: A10 B0 -> C1
        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        u[a10] = 1; v[b0] = 1; w[c1] = 1
        add_term(u, v, w)
        
        # T4: A11 B1 -> C1
        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        u[a11] = 1; v[b1] = 1; w[c1] = 1
        add_term(u, v, w)

    # Block (0,0) of C: from A(0,0)B(0,0) + A(0,1)B(1,0)
    # A(0,0) is 2x2 at (0,0). B(0,0) is 2x2 at (0,0). C(0,0) is 2x2 at (0,0).
    add_strassen_block(0, 0, 0, 0, 0, 0)
    # A(0,1) is 2x2 at (0,2). B(1,0) is 2x2 at (2,0). C(0,0) is 2x2 at (0,0).
    # Wait, C(0,0) is the top-left 2x2 of C.
    # The second term A(0,1)B(1,0) contributes to C(0,0).
    # So we need to add the Strassen terms for A(0,1)B(1,0) to the same C indices.
    add_strassen_block(0, 2, 2, 0, 0, 0)
    
    # Block (0,1) of C: from A(0,0)B(0,1) + A(0,1)B(1,1)
    # A(0,0) at (0,0). B(0,1) at (0,2). C(0,1) at (0,2).
    add_strassen_block(0, 0, 0, 2, 0, 2)
    # A(0,1) at (0,2). B(1,1) at (2,2). C(0,1) at (0,2).
    add_strassen_block(0, 2, 2, 2, 0, 2)
    
    # Block (0,2) of C: from A(0,0)B(0,2) + A(0,1)B(1,2)
    # A(0,0) at (0,0). B(0,2) at (0,4) (2x1). C(0,2) at (0,4) (2x1).
    add_schoolbook_2x1(0, 0, 0, 4, 0, 4)
    # A(0,1) at (0,2). B(1,2) at (2,4) (2x1). C(0,2) at (0,4) (2x1).
    add_schoolbook_2x1(0, 2, 2, 4, 0, 4)
    
    # Block (1,0) of C: from A(1,0)B(0,0) + A(1,1)B(1,0)
    # A(1,0) at (2,0). B(0,0) at (0,0). C(1,0) at (2,0).
    add_strassen_block(2, 0, 0, 0, 2, 0)
    # A(1,1) at (2,2). B(1,0) at (2,0). C(1,0) at (2,0).
    add_strassen_block(2, 2, 2, 0, 2, 0)
    
    # Block (1,1) of C: from A(1,0)B(0,1) + A(1,1)B(1,1)
    # A(1,0) at (2,0). B(0,1) at (0,2). C(1,1) at (2,2).
    add_strassen_block(2, 0, 0, 2, 2, 2)
    # A(1,1) at (2,2). B(1,1) at (2,2). C(1,1) at (2,2).
    add_strassen_block(2, 2, 2, 2, 2, 2)
    
    # Block (1,2) of C: from A(1,0)B(0,2) + A(1,1)B(1,2)
    # A(1,0) at (2,0). B(0,2) at (0,4). C(1,2) at (2,4).
    add_schoolbook_2x1(2, 0, 0, 4, 2, 4)
    # A(1,1) at (2,2). B(1,2) at (2,4). C(1,2) at (2,4).
    add_schoolbook_2x1(2, 2, 2, 4, 2, 4)
    
    # Total terms: 4 * 14 + 2 * 8 = 56 + 16 = 72.
    
    # Verify
    est = np.einsum('ar,br,cr->abc', U, V, W)
    if np.max(np.abs(est - target)) < 1e-9:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook
        best_rank = n * m * p
        U_sb = np.zeros((n * m, best_rank))
        V_sb = np.zeros((m * p, best_rank))
        W_sb = np.zeros((p * n, best_rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j, col] = 1
                    V_sb[j * p + k, col] = 1
                    W_sb[k * n + i, col] = 1
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
