import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 5, 6
    # Rank estimate: 3 * 18 + 3 * 10 = 84
    rank = 84
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a block decomposition
    # A_block: list of (row_offset, col_start, width)
    # B_block: list of (row_start, col_offset, height)
    # C_block: list of (row_offset, col_offset, width, height)
    # This is too generic. Let's hardcode the specific blocks.
    
    # We partition B into 3 blocks of width 2.
    # B_1: cols 0,1. B_2: cols 2,3. B_3: cols 4,5.
    # A is partitioned into A_top (rows 0,1) and A_bot (row 2).
    
    # For each B block (index b_idx 0..2):
    #   B_block is 5x2.
    #   Partition inner dimension 5 into 2, 2, 1.
    #   A_top (2x5) -> A_t1 (2x2), A_t2 (2x2), A_t3 (2x1)
    #   A_bot (1x5) -> A_b1 (1x2), A_b2 (1x2), A_b3 (1x1)
    
    for b_idx in range(3):
        b_col_start = b_idx * 2
        
        # --- A_top part (2x5 x 5x2) ---
        # Split inner 5 into 2, 2, 1
        # Part 1: 2x2 x 2x2 (Strassen)
        # A rows 0,1. Cols 0,1 (relative to block start? No, global cols of A are 0..4)
        # A_t1: A[0:2, 0:2]
        # B_p1: B[0:2, b_col_start:b_col_start+2]
        # C_t1: C[0:2, b_col_start:b_col_start+2]
        
        col += add_strassen_block(U, V, W, col, 
                                  a_rows=(0,1), a_cols=(0,1), 
                                  b_rows=(0,1), b_cols=(b_col_start, b_col_start+1),
                                  c_rows=(0,1), c_cols=(b_col_start, b_col_start+1))
        
        # Part 2: 2x2 x 2x2 (Strassen)
        # A_t2: A[0:2, 2:4]
        # B_p2: B[2:4, b_col_start:b_col_start+2]
        # C_t2: C[0:2, b_col_start:b_col_start+2]
        
        col += add_strassen_block(U, V, W, col,
                                  a_rows=(0,1), a_cols=(2,3),
                                  b_rows=(2,3), b_cols=(b_col_start, b_col_start+1),
                                  c_rows=(0,1), c_cols=(b_col_start, b_col_start+1))
        
        # Part 3: 2x1 x 1x2 (Schoolbook)
        # A_t3: A[0:2, 4:5]
        # B_p3: B[4:5, b_col_start:b_col_start+2]
        # C_t3: C[0:2, b_col_start:b_col_start+2]
        
        col += add_schoolbook_block(U, V, W, col,
                                    a_rows=(0,1), a_cols=(4,4),
                                    b_rows=(4,4), b_cols=(b_col_start, b_col_start+1),
                                    c_rows=(0,1), c_cols=(b_col_start, b_col_start+1))

        # --- A_bot part (1x5 x 5x2) ---
        # Part 1: 1x2 x 2x2 (Schoolbook)
        # A_b1: A[2:3, 0:2]
        # B_p1: B[0:2, b_col_start:b_col_start+2]
        # C_b1: C[2:3, b_col_start:b_col_start+2]
        
        col += add_schoolbook_block(U, V, W, col,
                                    a_rows=(2,2), a_cols=(0,1),
                                    b_rows=(0,1), b_cols=(b_col_start, b_col_start+1),
                                    c_rows=(2,2), c_cols=(b_col_start, b_col_start+1))
        
        # Part 2: 1x2 x 2x2 (Schoolbook)
        # A_b2: A[2:3, 2:4]
        # B_p2: B[2:4, b_col_start:b_col_start+2]
        # C_b2: C[2:3, b_col_start:b_col_start+2]
        
        col += add_schoolbook_block(U, V, W, col,
                                    a_rows=(2,2), a_cols=(2,3),
                                    b_rows=(2,3), b_cols=(b_col_start, b_col_start+1),
                                    c_rows=(2,2), c_cols=(b_col_start, b_col_start+1))
        
        # Part 3: 1x1 x 1x2 (Schoolbook)
        # A_b3: A[2:3, 4:5]
        # B_p3: B[4:5, b_col_start:b_col_start+2]
        # C_b3: C[2:3, b_col_start:b_col_start+2]
        
        col += add_schoolbook_block(U, V, W, col,
                                    a_rows=(2,2), a_cols=(4,4),
                                    b_rows=(4,4), b_cols=(b_col_start, b_col_start+1),
                                    c_rows=(2,2), c_cols=(b_col_start, b_col_start+1))
                                    
    return {"U": U, "V": V, "W": W}

def add_strassen_block(U, V, W, col, a_rows, a_cols, b_rows, b_cols, c_rows, c_cols):
    # a_rows: (r_start, r_end) inclusive
    # a_cols: (c_start, c_end) inclusive
    # b_rows: (r_start, r_end) inclusive
    # b_cols: (c_start, c_end) inclusive
    # c_rows: (r_start, r_end) inclusive
    # c_cols: (c_start, c_end) inclusive
    
    # Local 2x2 indices
    # A00, A01
    # A10, A11
    ar0, ar1 = a_rows
    ac0, ac1 = a_cols
    br0, br1 = b_rows
    bc0, bc1 = b_cols
    cr0, cr1 = c_rows
    cc0, cc1 = c_cols
    
    def get_A(r, c):
        return (ar0 + r, ac0 + c)
    def get_B(r, c):
        return (br0 + r, bc0 + c)
    def get_C(r, c):
        return (cr0 + r, cc0 + c)
        
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11(B01 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    terms = [
        # (coeff_A, coeff_B, coeff_C)
        # M1
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (0,1,1), (1,0,1), (1,1,-1)]),
        # M2
        ([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,-1)]),
        # M3
        ([(0,0,1)], [(1,0,1), (1,1,-1)], [(0,1,1), (1,1,1)]),
        # M4
        ([(1,1,1)], [(0,1,1), (0,0,-1)], [(0,0,1), (1,0,1)]),
        # M5
        ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)]),
        # M6
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)]),
        # M7
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1), (0,1,-1)]),
    ]
    
    # Wait, C output mapping:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # My terms list above has C coeffs mixed up.
    # Let's define the 7 multiplications and then sum them for C.
    
    # Mult 1: (A00+A11)(B00+B11)
    # Mult 2: (A10+A11)B00
    # Mult 3: A00(B10-B11)
    # Mult 4: A11(B01-B00)
    # Mult 5: (A00+A01)B11
    # Mult 6: (A10-A00)(B00+B01)
    # Mult 7: (A01-A11)(B10+B11)
    
    mults = [
        # A coeffs: list of (r, c, val)
        # B coeffs: list of (r, c, val)
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        ([(1,0,1), (1,1,1)], [(0,0,1)]),
        ([(0,0,1)], [(1,0,1), (1,1,-1)]),
        ([(1,1,1)], [(0,1,1), (0,0,-1)]),
        ([(0,0,1), (0,1,1)], [(1,1,1)]),
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)]),
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)]),
    ]
    
    # C coeffs for each Mult
    # C00: M1(1) M4(1) M5(-1) M7(1)
    # C01: M3(1) M5(1)
    # C10: M2(1) M4(1)
    # C11: M1(1) M2(-1) M3(1) M6(1)
    
    c_coeffs = [
        [(0,0,1), (0,1,1), (1,0,1), (1,1,-1)], # M1
        [(1,0,1), (1,1,-1)], # M2
        [(0,1,1), (1,1,1)], # M3
        [(0,0,1), (1,0,1)], # M4
        [(0,0,-1), (0,1,1)], # M5
        [(1,1,1)], # M6
        [(0,0,1), (0,1,-1)] # M7
    ]
    
    # This is wrong. C00 is a single element.
    # The output C is 2x2.
    # The term adds to specific (r,c) in C.
    
    # Let's restructure.
    # For each of the 7 multiplications, we add a rank-1 term.
    # The term contributes to U, V, and W.
    # W contribution: The result of the multiplication is a scalar $m_i$.
    # This scalar $m_i$ is added to specific positions in C.
    
    # So for each mult i:
    # 1. Update U with A coeffs.
    # 2. Update V with B coeffs.
    # 3. Update W with C coeffs (which are +1 or -1 at specific C positions).
    
    for i in range(7):
        a_coeffs = mults[i][0]
        b_coeffs = mults[i][1]
        c_coeffs = c_coeffs[i]
        
        # Update U
        for r, c, val in a_coeffs:
            u_row = get_A(r, c)
            U[u_row[0] * 5 + u_row[1]][col] += val
            
        # Update V
        for r, c, val in b_coeffs:
            v_row = get_B(r, c)
            V[v_row[0] * 6 + v_row[1]][col] += val
            
        # Update W
        # c_coeffs is a list of (r, c, val)
        # Note: W index is k*n + i.
        # r is row of C (i), c is col of C (k).
        for r, c, val in c_coeffs:
            w_row = get_C(r, c)
            W[w_row[1] * 3 + w_row[0]][col] += val
            
        col += 1
        
    return 7

def add_schoolbook_block(U, V, W, col, a_rows, a_cols, b_rows, b_cols, c_rows, c_cols):
    ar0, ar1 = a_rows
    ac0, ac1 = a_cols
    br0, br1 = b_rows
    bc0, bc1 = b_cols
    cr0, cr1 = c_rows
    cc0, cc1 = c_cols
    
    # Schoolbook: sum over k
    # Here the "inner" dimension is the rows of B / cols of A?
    # No, the block is already isolated.
    # A is (ar1-ar0+1) x (ac1-ac0+1)
    # B is (br1-br0+1) x (bc1-bc0+1)
    # We assume the inner dimension matches: ac1-ac0+1 == br1-br0+1.
    
    # For each output element (r, c) in C block:
    # C[r, c] = sum_k A[r, k] B[k, c]
    
    rank_added = 0
    for r in range(cr1 - cr0 + 1):
        for c in range(cc1 - cc0 + 1):
            # Inner dimension k
            # A index: (ar0 + r, ac0 + k)
            # B index: (br0 + k, bc0 + c)
            # C index: (cr0 + r, cc0 + c)
            
            # This is a sum of products.
            # We need to add rank-1 terms for each k.
            for k in range(ac1 - ac0 + 1):
                a_idx = (ar0 + r, ac0 + k)
                b_idx = (br0 + k, bc0 + c)
                c_idx = (cr0 + r, cc0 + c)
                
                U[a_idx[0] * 5 + a_idx[1]][col] = 1
                V[b_idx[0] * 6 + b_idx[1]][col] = 1
                W[c_idx[1] * 3 + c_idx[0]][col] = 1
                
                col += 1
                rank_added += 1
                
    return rank_added
# EVOLVE-BLOCK-END
