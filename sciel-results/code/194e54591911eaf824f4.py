"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 44
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a full column
    def add_column(u_entries, v_entries, w_entries):
        nonlocal col
        for idx, val in u_entries.items():
            U[idx][col] += val
        for idx, val in v_entries.items():
            V[idx][col] += val
        for idx, val in w_entries.items():
            W[idx][col] += val
        col += 1

    # Process two 3x4 blocks
    for block in range(2):
        b_col_offset = block * 4
        c_col_offset = block * 4
        
        # Part 1: A_right * B_bottom (8 terms)
        for i in range(2):
            for k in range(4):
                u_idx = i * 3 + 2
                v_idx = 2 * 4 + (b_col_offset + k)
                w_idx = (b_col_offset + k) * 2 + i
                add_column({u_idx: 1}, {v_idx: 1}, {w_idx: 1})
        
        # Part 2: A_left * B_top (14 terms)
        for sub_block in range(2):
            sb_col_offset = sub_block * 2
            global_b_col_offset = b_col_offset + sb_col_offset
            global_c_col_offset = c_col_offset + sb_col_offset
            
            # Indices for A (local to 2x2 part of 2x3)
            # A00: 0, A01: 1, A10: 3, A11: 4
            
            # Indices for B (local to 2x2 part of 3x4)
            b00 = 0 * 4 + global_b_col_offset
            b01 = 0 * 4 + global_b_col_offset + 1
            b10 = 1 * 4 + global_b_col_offset
            b11 = 1 * 4 + global_b_col_offset + 1
            
            # Indices for C (local to 2x2 part of 2x4)
            c00 = global_c_col_offset * 2 + 0
            c01 = (global_c_col_offset + 1) * 2 + 0
            c10 = global_c_col_offset * 2 + 1
            c11 = (global_c_col_offset + 1) * 2 + 1
            
            # Strassen Terms
            
            # 1. (A00+A11)(B00+B11)
            add_column({0: 1, 4: 1}, {b00: 1, b11: 1}, {c00: 1, c11: 1})
            
            # 2. (A10+A11)B00
            add_column({3: 1, 4: 1}, {b00: 1}, {c10: 1, c11: 1})
            
            # 3. A00(B01-B11)
            add_column({0: 1}, {b01: 1, b11: -1}, {c01: 1, c11: -1})
            
            # 4. A11(B10-B00)
            add_column({4: 1}, {b10: 1, b00: -1}, {c00: -1, c10: 1})
            
            # 5. (A00+A10)B11
            add_column({0: 1, 3: 1}, {b11: 1}, {c01: 1, c11: 1})
            
            # 6. (A01-A00)(B00+B10)
            add_column({1: 1, 0: -1}, {b00: 1, b10: 1}, {c00: 1})
            
            # 7. (A11-A10)(B01+B11)
            add_column({4: 1, 3: -1}, {b01: 1, b11: 1}, {c00: 1, c01: -1, c10: 1})
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
