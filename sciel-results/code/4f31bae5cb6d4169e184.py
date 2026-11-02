"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 7
    
    # Hybrid Strassen/schoolbook decomposition, rank 32
    # A is 2x3, B is 3x7, C is 2x7
    # Split: A = [A_L | A_R], A_L is 2x2 (cols 0,1), A_R is 2x1 (col 2)
    #        B = [B_L; B_R], B_L is 2x7 (rows 0,1), B_R is 1x7 (row 2)
    # C = A_L * B_L + A_R * B_R
    
    # A_L * B_L: 2x2 * 2x7
    #   Split B_L into 3 blocks of 2 cols + 1 block of 1 col
    #   3 * Strassen(2x2x2) + 1 * schoolbook(2x2x1) = 3*7 + 4 = 25
    # A_R * B_R: 2x1 * 1x7
    #   Split B_R into 3 blocks of 2 cols + 1 block of 1 col
    #   3 * schoolbook(2x1x2) + 1 * schoolbook(2x1x1) = 3*2 + 1 = 7
    # Total rank = 32
    
    rank = 32
    U = [[0] * rank for _ in range(n * m)]  # 6 x 32
    V = [[0] * rank for _ in range(m * p)]  # 21 x 32
    W = [[0] * rank for _ in range(p * n)]  # 14 x 32
    
    col = 0
    
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        nonlocal col
        U[u_idx][col] = u_val
        V[v_idx][col] = v_val
        W[w_idx][col] = w_val
        col += 1
    
    # Strassen for 2x2 * 2x2 -> 2x2
    # A_L rows: 0,1; cols: 0,1
    # B_L rows: 0,1; cols: b_c0, b_c1
    # C rows: 0,1; cols: c_c0, c_c1
    # U indices: A[i,j] -> i*m + j = i*3 + j
    # V indices: B[j,k] -> j*p + k = j*7 + k
    # W indices: C[i,k] -> k*n + i = k*2 + i
    
    def add_strassen(a_row_base, b_col0, b_col1, c_col0, c_col1):
        nonlocal col
        # A_L is 2x2: rows 0,1; cols 0,1
        # U indices for A_L elements:
        # A[0,0] -> 0*3+0 = 0
        # A[0,1] -> 0*3+1 = 1
        # A[1,0] -> 1*3+0 = 3
        # A[1,1] -> 1*3+1 = 4
        
        # B_L elements for this block:
        # B[0,b_col0] -> 0*7+b_col0
        # B[0,b_col1] -> 0*7+b_col1
        # B[1,b_col0] -> 1*7+b_col0
        # B[1,b_col1] -> 1*7+b_col1
        
        # C elements:
        # C[0,c_col0] -> c_col0*2+0
        # C[0,c_col1] -> c_col1*2+0
        # C[1,c_col0] -> c_col0*2+1
        # C[1,c_col1] -> c_col1*2+1
        
        u00, u01 = 0, 1
        u10, u11 = 3, 4
        
        v00 = 0*7 + b_col0
        v01 = 0*7 + b_col1
        v10 = 1*7 + b_col0
        v11 = 1*7 + b_col1
        
        w00 = c_col0*2 + 0
        w01 = c_col1*2 + 0
        w10 = c_col0*2 + 1
        w11 = c_col1*2 + 1
        
        # Strassen 7 multiplications:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)(B00)
        # M3 = (A00)(B10-B00)
        # M4 = (A11)(B11-B01)
        # M5 = (A00+A01)(B11)
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # C00 = M1 - M2 + M3 + M7
        # C01 = M4 + M5
        # C10 = M1 - M5 + M6 + M7
        # C11 = M2 - M3 + M4 + M6
        
        # M1: u = A00+A11, v = B00+B11, contributes to C00 and C10
        add_term(u00, 1, v00, 1, w00, 1)
        add_term(u11, 1, v00, 1, w00, 1)
        add_term(u00, 1, v11, 1, w00, 1)
        add_term(u11, 1, v11, 1, w00, 1)
        add_term(u00, 1, v00, 1, w10, 1)
        add_term(u11, 1, v00, 1, w10, 1)
        add_term(u00, 1, v11, 1, w10, 1)
        add_term(u11, 1, v11, 1, w10, 1)
        
        # This approach is wrong. Each "multiplication" in Strassen is a single rank-1 term
        # where the u vector has multiple non-zero entries, v vector has multiple non-zero entries,
        # and w vector has multiple non-zero entries.
        # But in our U/V/W format, each column is a single rank-1 tensor with scalar entries.
        # So we need to expand each Strassen multiplication into its constituent rank-1 terms.
        
        # Actually, re-reading the problem: U, V, W are matrices where each column r gives
        # a rank-1 tensor U[:,r] ⊗ V[:,r] ⊗ W[:,r].
        # So for Strassen, each of the 7 "multiplications" corresponds to ONE column where
        # U[:,r] is the linear combination of A elements, V[:,r] is the linear combination
        # of B elements, and W[:,r] is the linear combination of C elements.
        
        # Let me redo this properly.
        # Reset and use the correct approach.
        pass
    
    # Let me restart with a cleaner implementation
    
    rank = 32
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def set_col(u_entries, v_entries, w_entries):
        """u_entries, v_entries, w_entries are dicts: {index: value}"""
        nonlocal col
        for idx, val in u_entries.items():
            U[idx][col] = val
        for idx, val in v_entries.items():
            V[idx][col] = val
        for idx, val in w_entries.items():
            W[idx][col] = val
        col += 1
    
    # Strassen for 2x2 * 2x2
    # A elements: A00=(0,0), A01=(0,1), A10=(1,0), A11=(1,1) in A_L (cols 0,1)
    # B elements: B00, B01, B10, B11 where B0j is row 0 col j, B1j is row 1 col j
    # C elements: C00, C01, C10, C11
    
    def add_strassen_2x2x2(b_c0, b_c1, c_c0, c_c1):
        nonlocal col
        # U indices for A_L (2x2, cols 0,1 of A):
        # A[0,0]=idx 0, A[0,1]=idx 1, A[1,0]=idx 3, A[1,1]=idx 4
        # V indices for B_L block (2x2, rows 0,1, cols b_c0,b_c1):
        # B[0,b_c0]=idx 0*7+b_c0, B[0,b_c1]=idx 0*7+b_c1
        # B[1,b_c0]=idx 1*7+b_c0, B[1,b_c1]=idx 1*7+b_c1
        # W indices for C block (2x2, rows 0,1, cols c_c0,c_c1):
        # C[0,c_c0]=idx c_c0*2+0, C[0,c_c1]=idx c_c1*2+0
        # C[1,c_c0]=idx c_c0*2+1, C[1,c_c1]=idx c_c1*2+1
        
        u00, u01, u10, u11 = 0, 1, 3, 4
        v00 = b_c0
        v01 = b_c1
        v10 = 7 + b_c0
        v11 = 7 + b_c1
        w00 = 2*c_c0 + 0
        w01 = 2*c_c1 + 0
        w10 = 2*c_c0 + 1
        w11 = 2*c_c1 + 1
        
        # M1 = (A00+A11)(B00+B11) -> contributes to C00 and C10
        set_col({u00:1, u11:1}, {v00:1, v11:1}, {w00:1, w10:1})
        
        # M2 = (A10+A11)(B00) -> contributes to C00 (neg) and C11 (pos)
        set_col({u10:1, u11:1}, {v00:1}, {w00:-1, w11:1})
        
        # M3 = A00*(B10-B00) -> contributes to C00 (pos) and C11 (neg)
        set_col({u00:1}, {v10:1, v00:-1}, {w00:1, w11:-1})
        
        # M4 = A11*(B11-B01) -> contributes to C01 (pos) and C11 (pos)
        set_col({u11:1}, {v11:1, v01:-1}, {w01:1, w11:1})
        
        # M5 = (A00+A01)(B11) -> contributes to C01 (pos) and C10 (neg)
        set_col({u00:1, u01:1}, {v11:1}, {w01:1, w10:-1})
        
        # M6 = (A10-A00)(B00+B01) -> contributes to C10 (pos) and C11 (pos)
        set_col({u10:1, u00:-1}, {v00:1, v01:1}, {w10:1, w11:1})
        
        # M7 = (A01-A11)(B10+B11) -> contributes to C00 (pos) and C10 (pos)
        set_col({u01:1, u11:-1}, {v10:1, v11:1}, {w00:1, w10:1})
        
        # Verify: C00 = M1 - M2 + M3 + M7
        # C00 from M1: +1, M2: -1, M3: +1, M7: +1 => net 0? No.
        # Each set_col adds ONE rank-1 term. The coefficient in W determines the sign.
        # So for M1, W[w00]=1 means this term contributes +M1 to C00.
        # For M2, W[w00]=-1 means this term contributes -M2 to C00.
        # That's correct!
        
        # C00 = M1 - M2 + M3 + M7 ✓
        # C01 = M4 + M5 ✓
        # C10 = M1 - M5 + M6 + M7 ✓
        # C11 = M2 - M3 + M4 + M6 ✓
    
    # Schoolbook for 2x2 * 2x1 -> 2x1
    def add_schoolbook_2x2_x_2x1(b_c, c_c):
        nonlocal col
        # A_L is 2x2, B block is 2x1 (row 0 col b_c, row 1 col b_c)
        # C block is 2x1 (row 0 col c_c, row 1 col c_c)
        # C[0,c_c] = A[0,0]*B[0,b_c] + A[0,1]*B[1,b_c]
        # C[1,c_c] = A[1,0]*B[0,b_c] + A[1,1]*B[1,b_c]
        
        u00, u01, u10, u11 = 0, 1, 3, 4
        v0 = b_c
        v1 = 7 + b_c
        w0 = 2*c_c + 0
        w1 = 2*c_c + 1
        
        # Term 1: A[0,0]*B[0,b_c] -> C[0,c_c]
        set_col({u00:1}, {v0:1}, {w0:1})
        # Term 2: A[0,1]*B[1,b_c] -> C[0,c_c]
        set_col({u01:1}, {v1:1}, {w0:1})
        # Term 3: A[1,0]*B[0,b_c] -> C[1,c_c]
        set_col({u10:1}, {v0:1}, {w1:1})
        # Term 4: A[1,1]*B[1,b_c] -> C[1,c_c]
        set_col({u11:1}, {v1:1}, {w1:1})
    
    # Schoolbook for 2x1 * 1x2 -> 2x2
    def add_schoolbook_2x1_x_1x2(b_c0, b_c1, c_c0, c_c1):
        nonlocal col
        # A_R is 2x1 (col 2 of A): A[0,2]=idx 2, A[1,2]=idx 5
        # B_R block is 1x2 (row 2, cols b_c0, b_c1): B[2,b_c0]=idx 14+b_c0, B[2,b_c1]=idx 14+b_c1
        # C block is 2x2: C[0,c_c0], C[0,c_c1], C[1,c_c0], C[1,c_c1]
        # C[0,c_c0] = A[0,2]*B[2,b_c0]
        # C[0,c_c1] = A[0,2]*B[2,b_c1]
        # C[1,c_c0] = A[1,2]*B[2,b_c0]
        # C[1,c_c1] = A[1,2]*B[2,b_c1]
        
        u0 = 2  # A[0,2]
        u1 = 5  # A[1,2]
        v0 = 14 + b_c0  # B[2,b_c0]
        v1 = 14 + b_c1  # B[2,b_c1]
        w00 = 2*c_c0 + 0
        w01 = 2*c_c1 + 0
        w10 = 2*c_c0 + 1
        w11 = 2*c_c1 + 1
        
        # 4 terms
        set_col({u0:1}, {v0:1}, {w00:1})
        set_col({u0:1}, {v1:1}, {w01:1})
        set_col({u1:1}, {v0:1}, {w10:1})
        set_col({u1:1}, {v1:1}, {w11:1})
    
    # Schoolbook for 2x1 * 1x1 -> 2x1
    def add_schoolbook_2x1_x_1x1(b_c, c_c):
        nonlocal col
        # A_R is 2x1: A[0,2]=idx 2, A[1,2]=idx 5
        # B_R is 1x1: B[2,b_c]=idx 14+b_c
        # C is 2x1: C[0,c_c], C[1,c_c]
        # C[0,c_c] = A[0,2]*B[2,b_c]
        # C[1,c_c] = A[1,2]*B[2,b_c]
        
        u0 = 2
        u1 = 5
        v0 = 14 + b_c
        w0 = 2*c_c + 0
        w1 = 2*c_c + 1
        
        set_col({u0:1}, {v0:1}, {w0:1})
        set_col({u1:1}, {v0:1}, {w1:1})
    
    # Now build the decomposition
    # A_L * B_L: 3 Strassen blocks + 1 schoolbook 2x2x1 block
    # B_L cols: 0-1 (block 0), 2-3 (block 1), 4-5 (block 2), 6 (block 3)
    # C cols:   0-1 (block 0), 2-3 (block 1), 4-5 (block 2), 6 (block 3)
    
    add_strassen_2x2x2(0, 1, 0, 1)
    add_strassen_2x2x2(2, 3, 2, 3)
    add_strassen_2x2x2(4, 5, 4, 5)
    add_schoolbook_2x2_x_2x1(6, 6)
    
    # A_R * B_R: 3 schoolbook 2x1x2 blocks + 1 schoolbook 2x1x1 block
    add_schoolbook_2x1_x_1x2(0, 1, 0, 1)
    add_schoolbook_2x1_x_1x2(2, 3, 2, 3)
    add_schoolbook_2x1_x_1x2(4, 5, 4, 5)
    add_schoolbook_2x1_x_1x1(6, 6)
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
