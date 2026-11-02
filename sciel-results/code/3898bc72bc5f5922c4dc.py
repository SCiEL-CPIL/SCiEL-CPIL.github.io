import time

DIMENSIONS = [3, 4, 7]

def construct():
    start_time = time.time()
    deadline = 145.0
    
    n, m, p = DIMENSIONS
    
    U = []
    V = []
    W = []
    
    def add_term(u_coeffs, v_coeffs, w_coeffs):
        # u_coeffs: dict { (i, j): val }
        # v_coeffs: dict { (j, k): val }
        # w_coeffs: dict { (k, i): val }
        # We need to extend U, V, W to include this rank-1 term.
        # But U, V, W are lists of lists (matrices).
        # It is easier to accumulate in a list of tuples and then convert to matrix at the end.
        # Or just append to the existing matrices if we pre-allocate?
        # We don't know the total rank yet.
        # Let's store terms as lists of (index, value) or just full rows?
        # Better: store the coefficients for each term in a flat list, then build matrices.
        # Actually, just appending to U, V, W as we go is efficient if we treat them as lists of columns.
        # U is (n*m) x Rank.
        # Let's store U_cols, V_cols, W_cols.
        pass

    # We will use a list of terms: (U_row_dict, V_row_dict, W_row_dict)
    # Then at the end, determine max rank, and fill matrices.
    # Since indices are small, dicts are fine.
    
    terms = []
    
    def add_strassen_222(i_off, j_off, k_off):
        # Adds Strassen decomposition for A[i_off:i_off+2, j_off:j_off+2] * B[j_off:j_off+2, k_off:k_off+2]
        # Local indices 0,1 map to global i_off, i_off+1 etc.
        
        # M1
        u = {(i_off, j_off): 1, (i_off+1, j_off+1): 1}
        v = {(j_off, k_off): 1, (j_off+1, k_off+1): 1}
        w = {(k_off, i_off): 1, (k_off+1, i_off+1): 1}
        # This is NOT Strassen. This is schoolbook-like.
        # Strassen M1 = (A00+A11)(B00+B11).
        # U part: A00 and A11 have coeff 1.
        # V part: B00 and B11 have coeff 1.
        # W part: We need to add this term to C00, C01, C10, C11.
        # C00 += M1, C01 += M1, C10 += M1, C11 += M1?
        # No, Strassen reconstructs:
        # C00 = M1 - M4 + M5 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # So we need to add 7 terms, each contributing to specific W entries with specific signs.
        
        # Term 1: M1 = (A00+A11)(B00+B11)
        u1 = {(i_off, j_off): 1, (i_off+1, j_off+1): 1}
        v1 = {(j_off, k_off): 1, (j_off+1, k_off+1): 1}
        w1 = {(k_off, i_off): 1, (k_off+1, i_off): 1, (k_off, i_off+1): 1, (k_off+1, i_off+1): 1}
        # Wait, W index is (k, i).
        # C[k, i] gets contribution.
        # M1 contributes to C[0,0], C[0,1], C[1,0], C[1,1] with coeff 1.
        # So W entries: (k_off, i_off): 1, (k_off, i_off+1): 1, (k_off+1, i_off): 1, (k_off+1, i_off+1): 1.
        
        terms.append((u1, v1, w1))
        
        # Term 2: M2 = (A10+A11)B00
        u2 = {(i_off+1, j_off): 1, (i_off+1, j_off+1): 1}
        v2 = {(j_off, k_off): 1}
        w2 = {(k_off, i_off+1): 1, (k_off+1, i_off+1): 1} # M2 contributes to C10, C11?
        # C10 = M2 + M4. C11 = M1 - M2 + ...
        # So M2 has +1 for C10, -1 for C11.
        # W index (k, i).
        # C10 is i=1, k=0. W[(k_off, i_off+1)] = 1.
        # C11 is i=1, k=1. W[(k_off+1, i_off+1)] = -1.
        w2 = {(k_off, i_off+1): 1, (k_off+1, i_off+1): -1}
        terms.append((u2, v2, w2))
        
        # Term 3: M3 = A00(B10-B11)
        u3 = {(i_off, j_off): 1}
        v3 = {(j_off+1, k_off): 1, (j_off+1, k_off+1): -1}
        w3 = {(k_off, i_off): 1, (k_off+1, i_off): 1} # M3 contributes to C00, C01?
        # C00 = M1 - M4 + M5 - M7. No M3.
        # C01 = M3 + M5.
        # C10 = M2 + M4.
        # C11 = M1 - M2 + M3 + M6.
        # So M3 contributes to C01 (+1) and C11 (+1).
        # C01: i=0, k=1. W[(k_off+1, i_off)] = 1.
        # C11: i=1, k=1. W[(k_off+1, i_off+1)] = 1.
        w3 = {(k_off+1, i_off): 1, (k_off+1, i_off+1): 1}
        terms.append((u3, v3, w3))
        
        # Term 4: M4 = A11(B01-B00)
        u4 = {(i_off+1, j_off+1): 1}
        v4 = {(j_off, k_off+1): 1, (j_off, k_off): -1}
        w4 = {(k_off, i_off): -1, (k_off+1, i_off+1): 1} # M4 contributes to C00 (-1) and C10 (+1)?
        # C00 = M1 - M4 + ... -> -1 for C00.
        # C10 = M2 + M4 -> +1 for C10.
        # C00: i=0, k=0. W[(k_off, i_off)] = -1.
        # C10: i=1, k=0. W[(k_off, i_off+1)] = 1.
        w4 = {(k_off, i_off): -1, (k_off, i_off+1): 1}
        terms.append((u4, v4, w4))
        
        # Term 5: M5 = (A00+A01)B11
        u5 = {(i_off, j_off): 1, (i_off, j_off+1): 1}
        v5 = {(j_off+1, k_off+1): 1}
        w5 = {(k_off+1, i_off): 1, (k_off, i_off): 1} # M5 contributes to C00 (+1) and C01 (+1)?
        # C00 = ... + M5 ... -> +1.
        # C01 = M3 + M5 -> +1.
        # C00: i=0, k=0. W[(k_off, i_off)] = 1.
        # C01: i=0, k=1. W[(k_off+1, i_off)] = 1.
        w5 = {(k_off, i_off): 1, (k_off+1, i_off): 1}
        terms.append((u5, v5, w5))
        
        # Term 6: M6 = (A10-A00)(B00+B01)
        u6 = {(i_off+1, j_off): 1, (i_off, j_off): -1}
        v6 = {(j_off, k_off): 1, (j_off, k_off+1): 1}
        w6 = {(k_off+1, i_off+1): 1} # M6 contributes to C11 (+1)?
        # C11 = ... + M6.
        # C11: i=1, k=1. W[(k_off+1, i_off+1)] = 1.
        w6 = {(k_off+1, i_off+1): 1}
        terms.append((u6, v6, w6))
        
        # Term 7: M7 = (A01-A11)(B10+B11)
        u7 = {(i_off, j_off+1): 1, (i_off+1, j_off+1): -1}
        v7 = {(j_off+1, k_off): 1, (j_off+1, k_off+1): 1}
        w7 = {(k_off, i_off): -1} # M7 contributes to C00 (-1)?
        # C00 = ... - M7.
        # C00: i=0, k=0. W[(k_off, i_off)] = -1.
        w7 = {(k_off, i_off): -1}
        terms.append((u7, v7, w7))

    def add_schoolbook_221(i_off, j_off, k_off):
        # 2x2 A, 2x1 B. C is 2x1.
        # C0 = A00 B0 + A01 B1
        # C1 = A10 B0 + A11 B1
        
        # T1: A00 B0 -> C0
        u = {(i_off, j_off): 1}
        v = {(j_off, k_off): 1}
        w = {(k_off, i_off): 1}
        terms.append((u, v, w))
        
        # T2: A01 B1 -> C0
        u = {(i_off, j_off+1): 1}
        v = {(j_off+1, k_off): 1}
        w = {(k_off, i_off): 1}
        terms.append((u, v, w))
        
        # T3: A10 B0 -> C1
        u = {(i_off+1, j_off): 1}
        v = {(j_off, k_off): 1}
        w = {(k_off, i_off+1): 1}
        terms.append((u, v, w))
        
        # T4: A11 B1 -> C1
        u = {(i_off+1, j_off+1): 1}
        v = {(j_off+1, k_off): 1}
        w = {(k_off, i_off+1): 1}
        terms.append((u, v, w))

    def add_schoolbook_147(i_off, j_off, k_off):
        # 1x4 A, 4x7 B. C is 1x7.
        # Ck = sum_j A_j B_jk
        for j in range(4):
            for k in range(7):
                u = {(i_off, j_off + j): 1}
                v = {(j_off + j, k_off + k): 1}
                w = {(k_off + k, i_off): 1}
                terms.append((u, v, w))

    # Block 1: i in {0,1}, j in {0,1}, k in {0,1}
    add_strassen_222(0, 0, 0)
    
    # Block 2: i in {0,1}, j in {0,1}, k in {2,3,4,5,6}
    # Split k into {2,3}, {4,5}, {6}
    # k=2,3 -> 2x2x2
    add_strassen_222(0, 0, 2)
    # k=4,5 -> 2x2x2
    add_strassen_222(0, 0, 4)
    # k=6 -> 2x2x1
    add_schoolbook_221(0, 0, 6)
    
    # Block 3: i in {0,1}, j in {2,3}, k in {0,1,2,3,4,5,6}
    # Split k into {0,1}, {2,3}, {4,5}, {6}
    add_strassen_222(0, 2, 0)
    add_strassen_222(0, 2, 2)
    add_strassen_222(0, 2, 4)
    add_schoolbook_221(0, 2, 6)
    
    # Block 4: i=2, j in {0,1,2,3}, k in {0,1,2,3,4,5,6}
    add_schoolbook_147(2, 0, 0)
    
    # Now build U, V, W matrices
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, (u_dict, v_dict, w_dict) in enumerate(terms):
        for (i, j), val in u_dict.items():
            U[i * m + j][r] = val
        for (j, k), val in v_dict.items():
            V[j * p + k][r] = val
        for (k, i), val in w_dict.items():
            W[k * n + i][r] = val
            
    return {'U': U, 'V': V, 'W': W}
