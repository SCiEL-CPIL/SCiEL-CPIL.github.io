import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 28 decomposition using 4 Strassen 2x2x2 blocks
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen linear forms for 2x2x2
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # u vectors for A (indices 0:a, 1:b, 2:c, 3:d)
    u_forms = [
        [1, 0, 0, 1], # a+d
        [0, 0, 1, 1], # c+d
        [1, 0, 0, 0], # a
        [0, 0, 0, 1], # d
        [1, 1, 0, 0], # a+b
        [0, 0, 1, -1], # c-a
        [0, 1, 0, -1]  # b-d
    ]
    
    # v vectors for B (indices 0:e, 1:f, 2:g, 3:h)
    v_forms = [
        [1, 0, 0, 1], # e+h
        [1, 0, 0, 0], # e
        [0, 1, 0, -1], # f-h
        [0, 0, 1, -1], # g-e
        [0, 0, 0, 1], # h
        [1, 1, 0, 0], # e+f
        [0, 0, 1, 1]  # g+h
    ]
    
    # w matrices for C (2x2), flattened column-major? 
    # Prompt: W[k*n+i, r]. n=2.
    # C = [[c00, c01], [c10, c11]]
    # c00 -> idx 0*2+0 = 0
    # c01 -> idx 1*2+0 = 2
    # c10 -> idx 0*2+1 = 1
    # c11 -> idx 1*2+1 = 3
    # So W_flat = [c00, c10, c01, c11]
    
    # C00 = M1 + M4 - M5 + M7
    # C10 = M2 + M4
    # C01 = M3 + M5
    # C11 = M1 - M2 + M3 + M6
    
    # w_coeffs[t] = [coeff for C00, C10, C01, C11]
    w_coeffs = [
        [1, 0, 0, 1], # M1
        [0, 1, 0, -1], # M2
        [0, 0, 1, 1], # M3
        [1, 1, 0, 0], # M4
        [-1, 0, 1, 0], # M5
        [0, 0, 0, 1], # M6
        [1, 0, 0, 0]  # M7
    ]

    # Define the 4 sub-problems
    # Each defined by:
    # A_rows: [0, 1] (always)
    # A_cols: [start, end)
    # B_rows: [start, end)
    # B_cols: [start, end)
    # C_rows: [0, 1] (always)
    # C_cols: [start, end)
    
    sub_problems = [
        # 1. A[:, 0:2] * B[0:2, 0:2] -> C[:, 0:2]
        {'a_cols': (0, 2), 'b_rows': (0, 2), 'b_cols': (0, 2), 'c_cols': (0, 2)},
        # 2. A[:, 2:4] * B[0:2, 0:2] -> C[:, 0:2]
        {'a_cols': (2, 4), 'b_rows': (0, 2), 'b_cols': (0, 2), 'c_cols': (0, 2)},
        # 3. A[:, 0:2] * B[2:4, 0:2] -> C[:, 0:2] ? No.
        # C[:, 0:2] = A[:, 0:2]B[0:2, 0:2] + A[:, 2:4]B[2:4, 0:2]
        # So Sub-problem 2 should be A[:, 2:4] * B[2:4, 0:2] -> C[:, 0:2]
        # Let's re-verify the split.
        # C = A B
        # A = [A_L, A_R] (2x2, 2x2)
        # B = [B_T; B_B] (2x4, 2x4)
        # C = A_L B_T + A_R B_B
        # B_T = [B_TL, B_TR] (2x2, 2x2)
        # B_B = [B_BL, B_BR] (2x2, 2x2)
        # C_L (2x2) = A_L B_TL + A_R B_BL
        # C_R (2x2) = A_L B_TR + A_R B_BR
        
        # So:
        # 1. A_L * B_TL -> C_L
        # 2. A_R * B_BL -> C_L
        # 3. A_L * B_TR -> C_R
        # 4. A_R * B_BR -> C_R
    ]
    
    # Correct sub-problems:
    sub_problems = [
        # 1. A_L (cols 0-1) * B_TL (rows 0-1, cols 0-1) -> C_L (cols 0-1)
        {'a_start': 0, 'b_row_start': 0, 'b_col_start': 0, 'c_col_start': 0},
        # 2. A_R (cols 2-3) * B_BL (rows 2-3, cols 0-1) -> C_L (cols 0-1)
        {'a_start': 2, 'b_row_start': 2, 'b_col_start': 0, 'c_col_start': 0},
        # 3. A_L (cols 0-1) * B_TR (rows 0-1, cols 2-3) -> C_R (cols 2-3)
        {'a_start': 0, 'b_row_start': 0, 'b_col_start': 2, 'c_col_start': 2},
        # 4. A_R (cols 2-3) * B_BR (rows 2-3, cols 2-3) -> C_R (cols 2-3)
        {'a_start': 2, 'b_row_start': 2, 'b_col_start': 2, 'c_col_start': 2},
    ]

    current_rank = 0
    for sp in sub_problems:
        a_s = sp['a_start']
        b_rs = sp['b_row_start']
        b_cs = sp['b_col_start']
        c_cs = sp['c_col_start']
        
        for t in range(7):
            r = current_rank
            u_form = u_forms[t]
            v_form = v_forms[t]
            w_form = w_coeffs[t]
            
            # Fill U
            # Local A indices 0..3 map to:
            # 0: (0, a_s)
            # 1: (0, a_s+1)
            # 2: (1, a_s)
            # 3: (1, a_s+1)
            for loc_idx in range(4):
                val = u_form[loc_idx]
                if val != 0:
                    i = loc_idx // 2
                    k_loc = loc_idx % 2
                    k_glob = a_s + k_loc
                    u_idx = i * m + k_glob
                    U[u_idx][r] += val
            
            # Fill V
            # Local B indices 0..3 map to:
            # 0: (b_rs, b_cs)
            # 1: (b_rs, b_cs+1)
            # 2: (b_rs+1, b_cs)
            # 3: (b_rs+1, b_cs+1)
            for loc_idx in range(4):
                val = v_form[loc_idx]
                if val != 0:
                    k_loc = loc_idx // 2
                    j_loc = loc_idx % 2
                    k_glob = b_rs + k_loc
                    j_glob = b_cs + j_loc
                    v_idx = k_glob * p + j_glob
                    V[v_idx][r] += val
            
            # Fill W
            # Local C indices 0..3 map to:
            # 0: C00 -> (0, c_cs)
            # 1: C10 -> (1, c_cs)
            # 2: C01 -> (0, c_cs+1)
            # 3: C11 -> (1, c_cs+1)
            # W index formula: j * n + i
            for loc_idx in range(4):
                val = w_form[loc_idx]
                if val != 0:
                    i = loc_idx % 2 # 0 for C0x, 1 for C1x? 
                    # Wait, my w_form order was [C00, C10, C01, C11]
                    # loc_idx 0: C00 -> i=0, j=0
                    # loc_idx 1: C10 -> i=1, j=0
                    # loc_idx 2: C01 -> i=0, j=1
                    # loc_idx 3: C11 -> i=1, j=1
                    if loc_idx == 0: i, j_loc = 0, 0
                    elif loc_idx == 1: i, j_loc = 1, 0
                    elif loc_idx == 2: i, j_loc = 0, 1
                    else: i, j_loc = 1, 1
                    
                    j_glob = c_cs + j_loc
                    w_idx = j_glob * n + i
                    W[w_idx][r] += val
            
            current_rank += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
