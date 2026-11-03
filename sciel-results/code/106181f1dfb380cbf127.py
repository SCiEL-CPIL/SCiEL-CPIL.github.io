"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 98
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_indices, u_vals, v_indices, v_vals, w_indices, w_vals):
        nonlocal col
        for idx, val in zip(u_indices, u_vals):
            U[idx][col] += val
        for idx, val in zip(v_indices, v_vals):
            V[idx][col] += val
        for idx, val in zip(w_indices, w_vals):
            W[idx][col] += val
        col += 1

    # Indices for A blocks (2x2)
    # A11: 0,1,4,5 ; A12: 2,3,6,7 ; A21: 8,9,12,13 ; A22: 10,11,14,15
    # Flatten 2x2 block to 4 indices
    A_blocks = {
        '11': [0, 1, 4, 5],
        '12': [2, 3, 6, 7],
        '21': [8, 9, 12, 13],
        '22': [10, 11, 14, 15]
    }
    
    # Indices for B_L blocks (2x2)
    # B11: 0,1,8,9 ; B12: 2,3,10,11 ; B21: 16,17,24,25 ; B22: 18,19,26,27
    B_L_blocks = {
        '11': [0, 1, 8, 9],
        '12': [2, 3, 10, 11],
        '21': [16, 17, 24, 25],
        '22': [18, 19, 26, 27]
    }
    
    # Indices for B_R blocks (2x2)
    # B13: 4,5,12,13 ; B14: 6,7,14,15 ; B23: 20,21,28,29 ; B24: 22,23,30,31
    B_R_blocks = {
        '11': [4, 5, 12, 13],  # Mapped to 11 in Strassen logic
        '12': [6, 7, 14, 15],  # Mapped to 12
        '21': [20, 21, 28, 29], # Mapped to 21
        '22': [22, 23, 30, 31]  # Mapped to 22
    }
    
    # Indices for C_L blocks (2x2)
    # C11: 0,1,4,5 ; C12: 2,3,6,7 ; C21: 8,9,12,13 ; C22: 10,11,14,15
    C_L_blocks = {
        '00': [0, 1, 4, 5],
        '01': [2, 3, 6, 7],
        '10': [8, 9, 12, 13],
        '11': [10, 11, 14, 15]
    }
    
    # Indices for C_R blocks (2x2)
    # C13: 16,17,20,21 ; C14: 18,19,22,23 ; C23: 24,25,28,29 ; C24: 26,27,30,31
    C_R_blocks = {
        '00': [16, 17, 20, 21],
        '01': [18, 19, 22, 23],
        '10': [24, 25, 28, 29],
        '11': [26, 27, 30, 31]
    }

    # Strassen 4x4 * 4x4 definitions
    # X_k, Y_k are linear combos of A_blocks, B_blocks
    # Format: {block_id: coeff}
    strassen_4x4 = [
        ({'11': 1, '22': 1}, {'11': 1, '22': 1}), # M1
        ({'21': 1, '22': 1}, {'11': 1}),           # M2
        ({'11': 1}, {'12': 1, '22': -1}),          # M3
        ({'22': 1}, {'21': 1, '11': -1}),          # M4
        ({'11': 1, '12': 1}, {'22': 1}),           # M5
        ({'21': 1, '11': -1}, {'11': 1, '12': 1}), # M6
        ({'12': 1, '22': -1}, {'21': 1, '22': 1})  # M7
    ]
    
    # Strassen 2x2 * 2x2 multipliers
    # m_i = (x_i) * (y_i)
    # x_i, y_i are linear combos of x_00, x_01, x_10, x_11 and y_00...
    # We need to map x_ij to A blocks.
    # X_k is a 2x2 matrix.
    # X_k[0,0] = sum c_a A_a[0,0]
    # X_k[0,1] = sum c_a A_a[0,1]
    # X_k[1,0] = sum c_a A_a[1,0]
    # X_k[1,1] = sum c_a A_a[1,1]
    
    def get_x_y_indices(x_map, y_map, b_blocks):
        # Returns list of (u_indices, u_vals, v_indices, v_vals) for the 7 Strassen 2x2 mults
        terms = []
        
        # Extract x_00, x_01, x_10, x_11
        # x_00 comes from A_a[0,0]
        x00 = []
        x01 = []
        x10 = []
        x11 = []
        for a_id, c in x_map.items():
            idxs = A_blocks[a_id]
            x00.append((idxs[0], c))
            x01.append((idxs[1], c))
            x10.append((idxs[2], c))
            x11.append((idxs[3], c))
            
        y00 = []
        y01 = []
        y10 = []
        y11 = []
        for b_id, c in y_map.items():
            idxs = b_blocks[b_id]
            y00.append((idxs[0], c))
            y01.append((idxs[1], c))
            y10.append((idxs[2], c))
            y11.append((idxs[3], c))
            
        # Define the 7 multipliers
        # m1 = (x00+x11)(y00+y11)
        terms.append((x00 + x11, y00 + y11))
        # m2 = (x10+x11)y00
        terms.append((x10 + x11, y00))
        # m3 = x00(y01-y11)
        terms.append((x00, [(i, -v) for i, v in y11] + y01))
        # m4 = x11(y10-y00)
        terms.append((x11, [(i, -v) for i, v in y00] + y10))
        # m5 = (x00+x01)y11
        terms.append((x00 + x01, y11))
        # m6 = (x10-x00)(y00+y01)
        terms.append(([(i, -v) for i, v in x00] + x10, y00 + y01))
        # m7 = (x01-x11)(y10+y11)
        terms.append(([(i, -v) for i, v in x11] + x01, y10 + y11))
        
        return terms

    # Process Left Half
    for x_map, y_map in strassen_4x4:
        mults = get_x_y_indices(x_map, y_map, B_L_blocks)
        for u_list, v_list in mults:
            # Determine which C blocks this multiplier contributes to
            # We need to know which M_k this is to assign W indices
            # This is tricky because the same x_map, y_map might be used?
            # No, each M_k is unique.
            # We need to pass the multiplier ID to determine W indices.
            pass # Re-implement with ID

    # Re-implement loop with ID
    col = 0
    for k, (x_map, y_map) in enumerate(strassen_4x4):
        mults = get_x_y_indices(x_map, y_map, B_L_blocks)
        for u_list, v_list in mults:
            # Determine W indices
            # M1: C00, C11
            # M2: C10, C11
            # M3: C01, C11
            # M4: C00, C10
            # M5: C01, C11
            # M6: C00
            # M7: C00, C11
            
            # Reconstruction:
            # C00 = m1 - m4 + m6 - m7
            # C01 = m3 + m5
            # C10 = m2 + m4
            # C11 = m1 - m3 + m5 + m7
            
            w_indices = []
            w_vals = []
            
            if k == 0: # m1
                w_indices = C_L_blocks['00'] + C_L_blocks['11']
                w_vals = [1]*4 + [1]*4
            elif k == 1: # m2
                w_indices = C_L_blocks['10'] + C_L_blocks['11']
                w_vals = [1]*4 + [1]*4
            elif k == 2: # m3
                w_indices = C_L_blocks['01'] + C_L_blocks['11']
                w_vals = [1]*4 + [-1]*4
            elif k == 3: # m4
                w_indices = C_L_blocks['00'] + C_L_blocks['10']
                w_vals = [-1]*4 + [1]*4
            elif k == 4: # m5
                w_indices = C_L_blocks['01'] + C_L_blocks['11']
                w_vals = [1]*4 + [1]*4
            elif k == 5: # m6
                w_indices = C_L_blocks['00']
                w_vals = [1]*4
            elif k == 6: # m7
                w_indices = C_L_blocks['00'] + C_L_blocks['11']
                w_vals = [-1]*4 + [1]*4
            
            add_term([i for i, _ in u_list], [v for _, v in u_list],
                     [i for i, _ in v_list], [v for _, v in v_list],
                     w_indices, w_vals)
                     
    # Process Right Half
    for k, (x_map, y_map) in enumerate(strassen_4x4):
        mults = get_x_y_indices(x_map, y_map, B_R_blocks)
        for u_list, v_list in mults:
            w_indices = []
            w_vals = []
            
            if k == 0: # m1
                w_indices = C_R_blocks['00'] + C_R_blocks['11']
                w_vals = [1]*4 + [1]*4
            elif k == 1: # m2
                w_indices = C_R_blocks['10'] + C_R_blocks['11']
                w_vals = [1]*4 + [1]*4
            elif k == 2: # m3
                w_indices = C_R_blocks['01'] + C_R_blocks['11']
                w_vals = [1]*4 + [-1]*4
            elif k == 3: # m4
                w_indices = C_R_blocks['00'] + C_R_blocks['10']
                w_vals = [-1]*4 + [1]*4
            elif k == 4: # m5
                w_indices = C_R_blocks['01'] + C_R_blocks['11']
                w_vals = [1]*4 + [1]*4
            elif k == 5: # m6
                w_indices = C_R_blocks['00']
                w_vals = [1]*4
            elif k == 6: # m7
                w_indices = C_R_blocks['00'] + C_R_blocks['11']
                w_vals = [-1]*4 + [1]*4
            
            add_term([i for i, _ in u_list], [v for _, v in u_list],
                     [i for i, _ in v_list], [v for _, v in v_list],
                     w_indices, w_vals)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
