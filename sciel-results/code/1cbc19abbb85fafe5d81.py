"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms for 2x2x2
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = A * B
    # Terms: (U_coeffs, V_coeffs, W_coeffs)
    # U: A indices (0:a, 1:b, 2:c, 3:d)
    # V: B indices (0:e, 1:f, 2:g, 3:h)
    # W: C indices (0:C00, 1:C01, 2:C10, 3:C11)
    
    strassen_terms = [
        # M1 = (a+d)(e+h)
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # M2 = (c+d)e
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # M3 = a(f-h)
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        # M4 = d(g-e)
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        # M5 = (a+b)h
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # M6 = (c-a)(e+f)
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # M7 = (b-d)(g+h)
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    # Helper to add a 2x2x2 block multiplication
    # a_row_offset: offset for A row index (0 or 1)
    # a_col_offset: offset for A col index (0, 1, 2, or 3)
    # b_row_offset: offset for B row index (0, 1, 2, or 3)
    # c_row_offset: offset for C row index (0 or 1)
    # c_col_offset: offset for C col index (0, 1, 2, or 3)
    # start_rank: starting column index in U, V, W
    
    def add_block(a_row_off, a_col_off, b_row_off, c_col_off, start_rank):
        # A is 2x4. Local A is 2x2.
        # A global index: i * 4 + j
        # Local A index (li, lj) -> global (li + a_row_off, lj + a_col_off)
        
        # B is 4x4. Local B is 2x2.
        # B global index: j * 4 + r
        # Local B index (lj, lr) -> global (lj + b_row_off, lr + c_col_off)
        # Note: B rows are j, cols are r.
        # In the 2x2x2 block, B is 2x2.
        # Local B row index maps to global B row index.
        # Local B col index maps to global B col index.
        
        # C is 2x4. Local C is 2x2.
        # C global index: r * 2 + i
        # Local C index (li, lr) -> global (li + c_row_off, lr + c_col_off)
        # Note: C rows are i, cols are r.
        # In the 2x2x2 block, C is 2x2.
        # Local C row index maps to global C row index.
        # Local C col index maps to global C col index.
        
        for s in range(7):
            col = start_rank + s
            u_c, v_c, w_c = strassen_terms[s]
            
            for li, coeff in u_c.items():
                # li is 0..3 for 2x2 A
                # li = 0 -> (0,0), 1 -> (0,1), 2 -> (1,0), 3 -> (1,1)
                row = li // 2
                col_a = li % 2
                g_row = row + a_row_off
                g_col = col_a + a_col_off
                idx = g_row * 4 + g_col
                U[idx][col] += coeff
                
            for li, coeff in v_c.items():
                # li is 0..3 for 2x2 B
                # li = 0 -> (0,0), 1 -> (0,1), 2 -> (1,0), 3 -> (1,1)
                row = li // 2
                col_b = li % 2
                g_row = row + b_row_off
                g_col = col_b + c_col_off
                idx = g_row * 4 + g_col
                V[idx][col] += coeff
                
            for li, coeff in w_c.items():
                # li is 0..3 for 2x2 C
                # li = 0 -> (0,0), 1 -> (0,1), 2 -> (1,0), 3 -> (1,1)
                row = li // 2
                col_c = li % 2
                g_row = row + c_row_off
                g_col = col_c + c_col_off
                # C index is r * 2 + i
                # Here g_row is i (0..1), g_col is r (0..3)
                idx = g_col * 2 + g_row
                W[idx][col] += coeff

    # Block 0: A_L (rows 0,1; cols 0,1) * B_LL (rows 0,1; cols 0,1) -> C_L (rows 0,1; cols 0,1)
    # A_L: a_row_off=0, a_col_off=0
    # B_LL: b_row_off=0, c_col_off=0 (B cols 0,1)
    # C_L: c_row_off=0, c_col_off=0 (C cols 0,1)
    add_block(0, 0, 0, 0, 0)
    
    # Block 1: A_L (rows 0,1; cols 0,1) * B_LR (rows 0,1; cols 2,3) -> C_R (rows 0,1; cols 2,3)
    # A_L: a_row_off=0, a_col_off=0
    # B_LR: b_row_off=0, c_col_off=2 (B cols 2,3)
    # C_R: c_row_off=0, c_col_off=2 (C cols 2,3)
    add_block(0, 0, 0, 2, 7)
    
    # Block 2: A_R (rows 0,1; cols 2,3) * B_RL (rows 2,3; cols 0,1) -> C_L (rows 0,1; cols 0,1)
    # A_R: a_row_off=0, a_col_off=2
    # B_RL: b_row_off=2, c_col_off=0
    # C_L: c_row_off=0, c_col_off=0
    add_block(0, 2, 2, 0, 14)
    
    # Block 3: A_R (rows 0,1; cols 2,3) * B_RR (rows 2,3; cols 2,3) -> C_R (rows 0,1; cols 2,3)
    # A_R: a_row_off=0, a_col_off=2
    # B_RR: b_row_off=2, c_col_off=2
    # C_R: c_row_off=0, c_col_off=2
    add_block(0, 2, 2, 2, 21)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
