import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=5
    # U: (n*m, rank) = (16, rank)
    # V: (m*p, rank) = (20, rank)
    # W: (p*n, rank) = (20, rank)
    
    rank = 65
    U = np.zeros((16, rank))
    V = np.zeros((20, rank))
    W = np.zeros((20, rank))
    
    # Part 1: 4x4x4 multiplication (Rank 49)
    # Using 2x2 Strassen blocks
    
    # Standard 2x2x2 Strassen decomposition
    # U2, V2, W2 are 4x7
    # Indices for 2x2 matrix: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    # Strassen formulas:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We need the rank-7 decomposition of 2x2x2.
    # One standard integer decomposition:
    # U2:
    # col 0: [1, 0, 0, 1] (A00+A11)
    # col 1: [0, 0, 1, 1] (A10+A11)
    # col 2: [1, 0, 0, 0] (A00)
    # col 3: [0, 0, 0, 1] (A11)
    # col 4: [1, 1, 0, 0] (A00+A01)
    # col 5: [1, 0, -1, 0] (A10-A00) -> [0,0,1,0] - [1,0,0,0] = [-1,0,1,0]
    # col 6: [0, 1, 0, -1] (A01-A11) -> [0,1,0,0] - [0,0,0,1] = [0,1,0,-1]
    
    # V2:
    # col 0: [1, 0, 0, 1] (B00+B11)
    # col 1: [1, 0, 0, 0] (B00)
    # col 2: [0, 1, 0, -1] (B01-B11)
    # col 3: [0, 0, 1, -1] (B10-B00)
    # col 4: [0, 0, 0, 1] (B11)
    # col 5: [1, 1, 0, 0] (B00+B01)
    # col 6: [0, 0, 1, 1] (B10+B11)
    
    # W2:
    # col 0: [1, 0, 0, 1] (C00+C11)
    # col 1: [0, 0, 1, 0] (C10)
    # col 2: [0, 1, 0, 0] (C01)
    # col 3: [0, 0, 1, 0] (C10) -- Wait, M4 contributes to C10 and C11?
    # Let's check C formulas again.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # M1: C00(+1), C11(+1) -> [1,0,0,1]
    # M2: C10(+1) -> [0,0,1,0]
    # M3: C01(+1), C11(+1) -> [0,1,0,1]
    # M4: C00(+1), C10(+1) -> [1,0,1,0]
    # M5: C00(-1), C01(+1) -> [-1,1,0,0]
    # M6: C11(+1) -> [0,0,0,1]
    # M7: C00(+1) -> [1,0,0,0]
    
    # So W2 columns:
    # 0: [1, 0, 0, 1]
    # 1: [0, 0, 1, 0]
    # 2: [0, 1, 0, 1]
    # 3: [1, 0, 1, 0]
    # 4: [-1, 1, 0, 0]
    # 5: [0, 0, 0, 1]
    # 6: [1, 0, 0, 0]
    
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]).T # 4x7
    
    V2 = np.array([
        [1, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, -1, 1, 0, 1]
    ]).T # 4x7
    
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, 0, 1, 0, 0, 1, 0]
    ]).T # 4x7
    
    # Define block mappings for A, B, C
    # A blocks: A00 (0-3), A01 (4-7), A10 (8-11), A11 (12-15)
    # B blocks: B00 (0-3), B01 (4-7), B10 (8-11), B11 (12-15)
    # C blocks: C00 (0-3), C01 (4-7), C10 (8-11), C11 (12-15)
    
    # For each Strassen step k, define the block combinations
    # A_comb[k] is a dict mapping local index (0-3) to list of (global_block, coeff)
    # Actually, it's easier to define the global U vector directly.
    
    # Step 1: M1 = (A00+A11)(B00+B11)
    # A_local = A00 + A11
    # B_local = B00 + B11
    # C_local = M1 -> C00(+1), C11(+1)
    
    # We iterate through the 7 Strassen steps
    # And for each, we iterate through the 7 rank-1 terms of 2x2x2
    
    # Helper to add a term
    def add_term(global_U, global_V, global_W, U_vec, V_vec, W_vec, offset):
        global_U[:, offset] += U_vec
        global_V[:, offset] += V_vec
        global_W[:, offset] += W_vec
        
    # We need to map local 2x2 indices to global 4x4 indices
    # Local index 0 -> (0,0)
    # Local index 1 -> (0,1)
    # Local index 2 -> (1,0)
    # Local index 3 -> (1,1)
    
    # For a given block combination, e.g., A00 + A11:
    # Local 0 (A00,00) maps to Global A[0,0] and A[2,2] (A11,00)
    # Local 1 (A00,01) maps to Global A[0,1] and A[2,3] (A11,01)
    # Local 2 (A00,10) maps to Global A[1,0] and A[3,2] (A11,10)
    # Local 3 (A00,11) maps to Global A[1,1] and A[3,3] (A11,11)
    
    # Let's define the mappings for each of the 7 Strassen steps
    # A_maps[k][local_idx] = list of (global_idx, coeff)
    # B_maps[k][local_idx] = list of (global_idx, coeff)
    # C_maps[k][local_idx] = list of (global_idx, coeff)
    
    # A00: 0,1,2,3
    # A01: 4,5,6,7
    # A10: 8,9,10,11
    # A11: 12,13,14,15
    
    # B00: 0,1,2,3
    # B01: 4,5,6,7
    # B10: 8,9,10,11
    # B11: 12,13,14,15
    
    # C00: 0,1,2,3
    # C01: 4,5,6,7
    # C10: 8,9,10,11
    # C11: 12,13,14,15
    
    # Step 1: (A00+A11)(B00+B11)
    A_map_1 = [
        [(0, 1), (12, 1)], # Local 0
        [(1, 1), (13, 1)], # Local 1
        [(2, 1), (14, 1)], # Local 2
        [(3, 1), (15, 1)]  # Local 3
    ]
    B_map_1 = [
        [(0, 1), (12, 1)],
        [(1, 1), (13, 1)],
        [(2, 1), (14, 1)],
        [(3, 1), (15, 1)]
    ]
    C_map_1 = [
        [(0, 1)], # Local 0 -> C00,00
        [(4, 1)], # Local 1 -> C01,00
        [(8, 1)], # Local 2 -> C10,00
        [(12, 1)] # Local 3 -> C11,00
    ]
    # Wait, M1 contributes to C00 and C11.
    # M1 is a 2x2 matrix.
    # M1[0,0] -> C00[0,0] (Global 0)
    # M1[0,1] -> C01[0,0] (Global 4)
    # M1[1,0] -> C10[0,0] (Global 8)
    # M1[1,1] -> C11[0,0] (Global 12)
    # This is correct.
    
    # Step 2: (A10+A11)B00
    A_map_2 = [
        [(8, 1), (12, 1)],
        [(9, 1), (13, 1)],
        [(10, 1), (14, 1)],
        [(11, 1), (15, 1)]
    ]
    B_map_2 = [
        [(0, 1)],
        [(1, 1)],
        [(2, 1)],
        [(3, 1)]
    ]
    C_map_2 = [
        [(8, 1)], # M2[0,0] -> C10[0,0]
        [(12, 1)], # M2[0,1] -> C11[0,0]
        [(9, 1)], # M2[1,0] -> C10[1,0] (Global 9)
        [(13, 1)] # M2[1,1] -> C11[1,0] (Global 13)
    ]
    # Wait, C10 is rows 2,3 cols 0,1.
    # C10[0,0] is Global 8.
    # C10[0,1] is Global 9.
    # C10[1,0] is Global 10.
    # C10[1,1] is Global 11.
    # M2 is 2x2.
    # M2[0,0] -> C10[0,0] (8)
    # M2[0,1] -> C10[0,1] (9)
    # M2[1,0] -> C10[1,0] (10)
    # M2[1,1] -> C10[1,1] (11)
    # But M2 also contributes to C11?
    # C10 = M2 + M4.
    # C11 = M1 - M2 + M3 + M6.
    # So M2 contributes to C10 (+1) and C11 (-1).
    # So C_map_2 should reflect this.
    # Local 0 (M2[0,0]) -> C10[0,0] (+1), C11[0,0] (-1)
    # Local 1 (M2[0,1]) -> C10[0,1] (+1), C11[0,1] (-1)
    # Local 2 (M2[1,0]) -> C10[1,0] (+1), C11[1,0] (-1)
    # Local 3 (M2[1,1]) -> C10[1,1] (+1), C11[1,1] (-1)
    
    C_map_2 = [
        [(8, 1), (12, -1)],
        [(9, 1), (13, -1)],
        [(10, 1), (14, -1)],
        [(11, 1), (15, -1)]
    ]
    
    # Step 3: A00(B01-B11)
    A_map_3 = [
        [(0, 1)],
        [(1, 1)],
        [(2, 1)],
        [(3, 1)]
    ]
    B_map_3 = [
        [(4, 1), (12, -1)],
        [(5, 1), (13, -1)],
        [(6, 1), (14, -1)],
        [(7, 1), (15, -1)]
    ]
    C_map_3 = [
        [(4, 1), (12, 1)], # M3[0,0] -> C01[0,0] (+1), C11[0,0] (+1)
        [(5, 1), (13, 1)], # M3[0,1] -> C01[0,1] (+1), C11[0,1] (+1)
        [(6, 1), (14, 1)], # M3[1,0] -> C01[1,0] (+1), C11[1,0] (+1)
        [(7, 1), (15, 1)]  # M3[1,1] -> C01[1,1] (+1), C11[1,1] (+1)
    ]
    
    # Step 4: A11(B10-B00)
    A_map_4 = [
        [(12, 1)],
        [(13, 1)],
        [(14, 1)],
        [(15, 1)]
    ]
    B_map_4 = [
        [(8, 1), (0, -1)],
        [(9, 1), (1, -1)],
        [(10, 1), (2, -1)],
        [(11, 1), (3, -1)]
    ]
    C_map_4 = [
        [(0, 1), (8, 1)], # M4[0,0] -> C00[0,0] (+1), C10[0,0] (+1)
        [(4, 1), (9, 1)], # M4[0,1] -> C00[0,1] (+1), C10[0,1] (+1)
        [(1, 1), (10, 1)], # M4[1,0] -> C00[1,0] (+1), C10[1,0] (+1)
        [(5, 1), (11, 1)]  # M4[1,1] -> C00[1,1] (+1), C10[1,1] (+1)
    ]
    
    # Step 5: (A00+A01)B11
    A_map_5 = [
        [(0, 1), (4, 1)],
        [(1, 1), (5, 1)],
        [(2, 1), (6, 1)],
        [(3, 1), (7, 1)]
    ]
    B_map_5 = [
        [(12, 1)],
        [(13, 1)],
        [(14, 1)],
        [(15, 1)]
    ]
    C_map_5 = [
        [(0, -1), (4, 1)], # M5[0,0] -> C00[0,0] (-1), C01[0,0] (+1)
        [(1, -1), (5, 1)], # M5[0,1] -> C00[0,1] (-1), C01[0,1] (+1)
        [(2, -1), (6, 1)], # M5[1,0] -> C00[1,0] (-1), C01[1,0] (+1)
        [(3, -1), (7, 1)]  # M5[1,1] -> C00[1,1] (-1), C01[1,1] (+1)
    ]
    
    # Step 6: (A10-A00)(B00+B01)
    A_map_6 = [
        [(8, 1), (0, -1)],
        [(9, 1), (1, -1)],
        [(10, 1), (2, -1)],
        [(11, 1), (3, -1)]
    ]
    B_map_6 = [
        [(0, 1), (4, 1)],
        [(1, 1), (5, 1)],
        [(2, 1), (6, 1)],
        [(3, 1), (7, 1)]
    ]
    C_map_6 = [
        [(12, 1)], # M6[0,0] -> C11[0,0] (+1)
        [(13, 1)], # M6[0,1] -> C11[0,1] (+1)
        [(14, 1)], # M6[1,0] -> C11[1,0] (+1)
        [(15, 1)]  # M6[1,1] -> C11[1,1] (+1)
    ]
    
    # Step 7: (A01-A11)(B10+B11)
    A_map_7 = [
        [(4, 1), (12, -1)],
        [(5, 1), (13, -1)],
        [(6, 1), (14, -1)],
        [(7, 1), (15, -1)]
    ]
    B_map_7 = [
        [(8, 1), (12, 1)],
        [(9, 1), (13, 1)],
        [(10, 1), (14, 1)],
        [(11, 1), (15, 1)]
    ]
    C_map_7 = [
        [(0, 1)], # M7[0,0] -> C00[0,0] (+1)
        [(1, 1)], # M7[0,1] -> C00[0,1] (+1)
        [(2, 1)], # M7[1,0] -> C00[1,0] (+1)
        [(3, 1)]  # M7[1,1] -> C00[1,1] (+1)
    ]
    
    maps_A = [A_map_1, A_map_2, A_map_3, A_map_4, A_map_5, A_map_6, A_map_7]
    maps_B = [B_map_1, B_map_2, B_map_3, B_map_4, B_map_5, B_map_6, B_map_7]
    maps_C = [C_map_1, C_map_2, C_map_3, C_map_4, C_map_5, C_map_6, C_map_7]
    
    col_offset = 0
    for k in range(7):
        for r in range(7):
            u_vec = np.zeros(16)
            v_vec = np.zeros(20) # Only first 16 used for 4x4 part
            w_vec = np.zeros(20)
            
            u_local = U2[:, r]
            v_local = V2[:, r]
            w_local = W2[:, r]
            
            for l in range(4):
                for (g_idx, coeff) in maps_A[k][l]:
                    u_vec[g_idx] += u_local[l] * coeff
                for (g_idx, coeff) in maps_B[k][l]:
                    v_vec[g_idx] += v_local[l] * coeff
                for (g_idx, coeff) in maps_C[k][l]:
                    w_vec[g_idx] += w_local[l] * coeff
            
            U[:, col_offset] = u_vec
            V[:, col_offset] = v_vec
            W[:, col_offset] = w_vec
            col_offset += 1
            
    # Part 2: 4x4x1 multiplication (Rank 16)
    # C_i = sum_j A_ij B_j,4
    # A indices: i*4 + j
    # B indices: j*5 + 4
    # C indices: i*5 + 4? No, C is 4x5.
    # C row i, col 4.
    # C index: i*5 + 4
    
    for i in range(4):
        for j in range(4):
            u_vec = np.zeros(16)
            v_vec = np.zeros(20)
            w_vec = np.zeros(20)
            
            a_idx = i * 4 + j
            b_idx = j * 5 + 4
            c_idx = i * 5 + 4
            
            u_vec[a_idx] = 1
            v_vec[b_idx] = 1
            w_vec[c_idx] = 1
            
            U[:, col_offset] = u_vec
            V[:, col_offset] = v_vec
            W[:, col_offset] = w_vec
            col_offset += 1
            
    return {"U": U, "V": V, "W": W}
