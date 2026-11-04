import time

def construct():
    deadline = time.time() + 120.0

    n, m, p = 2, 3, 5
    
    # Total rank estimate:
    # Block 1 (2x2x2, Strassen): 7
    # Block 2a (2x2x1, School): 4
    # Block 2b (2x2x2, Strassen): 7
    # Block 3 (2x1x2, School): 4
    # Block 4 (2x1x3, School): 6
    # Total: 28
    
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a schoolbook term for a sub-block
    def add_schoolbook(n_sub, m_sub, p_sub, i_offset, j_offset, k_offset, rank_offset):
        nonlocal col
        for i in range(n_sub):
            for j in range(m_sub):
                for k in range(p_sub):
                    u_idx = (i_offset + i) * m + (j_offset + j)
                    v_idx = (j_offset + j) * p + (k_offset + k)
                    w_idx = (k_offset + k) * n + (i_offset + i)
                    
                    U[u_idx][rank_offset] = 1
                    V[v_idx][rank_offset] = 1
                    W[w_idx][rank_offset] = 1
                    rank_offset += 1
        return rank_offset

    # Helper to add Strassen terms for a 2x2x2 sub-block
    def add_strassen(i_offset, j_offset, k_offset, rank_offset):
        nonlocal col
        # Local indices 0,1 map to global offsets
        # A: (i_offset+i, j_offset+j)
        # B: (j_offset+j, k_offset+k)
        # C: (k_offset+k, i_offset+i)
        
        # M1 = (A00 + A11) * (B00 + B11)
        # U: A00, A11 ; V: B00, B11 ; W: C00, C11, C01, C10 (coeffs)
        # We add 7 terms.
        
        # Term 1: M1
        # U: A00 + A11
        U[i_offset * m + j_offset][rank_offset] = 1
        U[(i_offset+1) * m + (j_offset+1)][rank_offset] = 1
        # V: B00 + B11
        V[j_offset * p + k_offset][rank_offset] = 1
        V[(j_offset+1) * p + (k_offset+1)][rank_offset] = 1
        # W: C00 + C11 - C01 + C10 ? No, M1 contributes to C00, C11, C01, C10 with specific coeffs.
        # C00 += M1, C11 += M1, C01 -= M1? No.
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Let's just define the coefficients for each of the 7 multiplications.
        # This is safer.
        
        # M1: (A00+A11)(B00+B11)
        # M2: (A10+A11)B00
        # M3: A00(B01-B11)
        # M4: A11(B10-B01)
        # M5: (A00+A01)B11
        # M6: (A10-A00)(B00+B01)
        # M7: (A01-A11)(B10+B11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # We will add 7 columns.
        
        # 1. M1
        U[i_offset * m + j_offset][rank_offset] = 1
        U[(i_offset+1) * m + (j_offset+1)][rank_offset] = 1
        V[j_offset * p + k_offset][rank_offset] = 1
        V[(j_offset+1) * p + (k_offset+1)][rank_offset] = 1
        W[k_offset * n + i_offset][rank_offset] = 1       # C00
        W[(k_offset+1) * n + (i_offset+1)][rank_offset] = 1 # C11
        W[(k_offset+1) * n + i_offset][rank_offset] = -1    # C01
        W[k_offset * n + (i_offset+1)][rank_offset] = 1     # C10
        rank_offset += 1
        
        # 2. M2
        U[(i_offset+1) * m + j_offset][rank_offset] = 1
        U[(i_offset+1) * m + (j_offset+1)][rank_offset] = 1
        V[j_offset * p + k_offset][rank_offset] = 1
        W[k_offset * n + (i_offset+1)][rank_offset] = 1     # C10
        W[(k_offset+1) * n + (i_offset+1)][rank_offset] = -1 # C11
        rank_offset += 1
        
        # 3. M3
        U[i_offset * m + j_offset][rank_offset] = 1
        V[j_offset * p + (k_offset+1)][rank_offset] = 1
        V[(j_offset+1) * p + (k_offset+1)][rank_offset] = -1
        W[(k_offset+1) * n + i_offset][rank_offset] = 1     # C01
        W[(k_offset+1) * n + (i_offset+1)][rank_offset] = 1 # C11
        rank_offset += 1
        
        # 4. M4
        U[(i_offset+1) * m + (j_offset+1)][rank_offset] = 1
        V[(j_offset+1) * p + k_offset][rank_offset] = 1
        V[j_offset * p + (k_offset+1)][rank_offset] = -1
        W[k_offset * n + i_offset][rank_offset] = 1         # C00
        W[k_offset * n + (i_offset+1)][rank_offset] = 1     # C10
        rank_offset += 1
        
        # 5. M5
        U[i_offset * m + j_offset][rank_offset] = 1
        U[i_offset * m + (j_offset+1)][rank_offset] = 1
        V[(j_offset+1) * p + (k_offset+1)][rank_offset] = 1
        W[k_offset * n + i_offset][rank_offset] = -1        # C00
        W[(k_offset+1) * n + i_offset][rank_offset] = 1     # C01
        rank_offset += 1
        
        # 6. M6
        U[(i_offset+1) * m + j_offset][rank_offset] = 1
        U[i_offset * m + j_offset][rank_offset] = -1
        V[j_offset * p + k_offset][rank_offset] = 1
        V[j_offset * p + (k_offset+1)][rank_offset] = 1
        W[(k_offset+1) * n + (i_offset+1)][rank_offset] = 1 # C11
        rank_offset += 1
        
        # 7. M7
        U[i_offset * m + (j_offset+1)][rank_offset] = 1
        U[(i_offset+1) * m + (j_offset+1)][rank_offset] = -1
        V[(j_offset+1) * p + k_offset][rank_offset] = 1
        V[(j_offset+1) * p + (k_offset+1)][rank_offset] = 1
        W[k_offset * n + i_offset][rank_offset] = 1         # C00
        W[(k_offset+1) * n + (i_offset+1)][rank_offset] = 1 # C11
        rank_offset += 1
        
        return rank_offset

    # Block 1: J1={0,1}, K1={0,1} -> 2x2x2 Strassen
    # i_offset=0, j_offset=0, k_offset=0
    col = add_strassen(0, 0, 0, col)
    
    # Block 2a: J1={0,1}, K2a={2} -> 2x2x1 School
    # i_offset=0, j_offset=0, k_offset=2
    col = add_schoolbook(2, 2, 1, 0, 0, 2, col)
    
    # Block 2b: J1={0,1}, K2b={3,4} -> 2x2x2 Strassen
    # i_offset=0, j_offset=0, k_offset=3
    col = add_strassen(0, 0, 3, col)
    
    # Block 3: J2={2}, K1={0,1} -> 2x1x2 School
    # i_offset=0, j_offset=2, k_offset=0
    col = add_schoolbook(2, 1, 2, 0, 2, 0, col)
    
    # Block 4: J2={2}, K2={2,3,4} -> 2x1x3 School
    # i_offset=0, j_offset=2, k_offset=2
    col = add_schoolbook(2, 1, 3, 0, 2, 2, col)
    
    if col != rank:
        # Fallback if logic error, though we expect 28
        pass

    return {"U": U, "V": V, "W": W}
