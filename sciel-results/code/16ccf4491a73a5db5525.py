import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n, m, p = 6, 6, 6
    
    # 1. Construct Rank-7 factors for 2x2 Strassen
    # Indices: 0,1 for A00,A01,A10,A11
    S2_U = [
        [1, 0, 0, 1], # M1: A00+A11
        [0, 0, 1, 1], # M2: A10+A11
        [1, 0, 0, 0], # M3: A00
        [0, 0, 0, 1], # M4: A11
        [1, 1, 0, 0], # M5: A00+A01
        [-1, 0, 1, 0],# M6: A10-A00
        [0, 1, 0, -1] # M7: A01-A11
    ]
    S2_V = [
        [1, 0, 0, 1], # M1: B00+B11
        [1, 0, 0, 0], # M2: B00
        [0, 0, 1, -1],# M3: B10-B11
        [-1, 1, 0, 0],# M4: B01-B00
        [0, 0, 0, 1], # M5: B11
        [1, 1, 0, 0], # M6: B00+B01
        [0, 0, 1, 1]  # M7: B10+B11
    ]
    S2_W = [
        [1, 0, 0, 1], # C00: M1+M4-M5+M7 -> M1(1), M4(1), M5(-1), M7(1)
        [0, 1, 0, 1], # C01: M3+M5 -> M3(1), M5(1)
        [0, 0, 1, 1], # C10: M2+M4 -> M2(1), M4(1)
        [1, -1, 1, 1] # C11: M1-M2+M3+M6 -> M1(1), M2(-1), M3(1), M6(1)
    ]
    # Note: S2_W is 4x7 (Blocks x Rank).
    # My S2_U is 7x4 (Rank x Blocks).
    
    # 2. Construct Rank-21 factors for 4x4 Strassen
    # 4x4 is 2x2 blocks of 2x2.
    # Rank 21 = 7 (top) * 7 (bottom).
    R4 = 21
    S4_U = [[0]*R4 for _ in range(16)] # 4x4 flattened
    S4_V = [[0]*R4 for _ in range(16)]
    S4_W = [[0]*R4 for _ in range(16)]
    
    for s in range(7): # Top Strassen term
        for t in range(7): # Bottom Strassen term
            r = s * 7 + t
            # U
            for b in range(4): # Top block index
                coeff_u = S2_U[s][b]
                if coeff_u == 0: continue
                # Bottom block b corresponds to 2x2 matrix
                # Map to 4x4 indices
                r_base = (b // 2) * 2
                c_base = (b % 2) * 2
                for i in range(2):
                    for j in range(2):
                        idx4 = (r_base + i) * 4 + (c_base + j)
                        idx2 = i * 2 + j
                        S4_U[idx4][r] += coeff_u * S2_U[t][idx2]
            
            # V
            for b in range(4):
                coeff_v = S2_V[s][b]
                if coeff_v == 0: continue
                r_base = (b // 2) * 2
                c_base = (b % 2) * 2
                for i in range(2):
                    for j in range(2):
                        idx4 = (r_base + i) * 4 + (c_base + j)
                        idx2 = i * 2 + j
                        S4_V[idx4][r] += coeff_v * S2_V[t][idx2]
                        
            # W
            for b in range(4):
                coeff_w = S2_W[b][s] # S2_W is Block x Rank
                if coeff_w == 0: continue
                r_base = (b // 2) * 2
                c_base = (b % 2) * 2
                for i in range(2):
                    for j in range(2):
                        idx4 = (r_base + i) * 4 + (c_base + j)
                        idx2 = i * 2 + j
                        S4_W[idx4][r] += coeff_w * S2_W[b][t] # Wait, S2_W is 4x7. b is row, t is col.

    # 3. Construct Global 6x6 Factors
    # Top level: Strassen on 2x2 blocks of 3x3.
    # Bottom level: 3x3 padded to 4x4, using S4 (Rank 21).
    
    R_TOTAL = 7 * 21
    
    U = [[0]*R_TOTAL for _ in range(36)]
    V = [[0]*R_TOTAL for _ in range(36)]
    W = [[0]*R_TOTAL for _ in range(36)]
    
    for s in range(7):
        for t in range(21):
            r = s * 21 + t
            
            # U Global
            for a in range(36):
                row = a // 6
                col = a % 6
                if row < 3 and col < 3:
                    b_idx = 0
                    l_r, l_c = row, col
                elif row < 3 and col >= 3:
                    b_idx = 1
                    l_r, l_c = row, col-3
                elif row >= 3 and col < 3:
                    b_idx = 2
                    l_r, l_c = row-3, col
                else:
                    b_idx = 3
                    l_r, l_c = row-3, col-3
                    
                coeff_top = S2_U[s][b_idx]
                if coeff_top == 0: continue
                
                p_idx = l_r * 4 + l_c
                U[a][r] += coeff_top * S4_U[p_idx][t]
                
            # V Global
            for b in range(36):
                row = b // 6
                col = b % 6
                if row < 3 and col < 3:
                    b_idx = 0
                    l_r, l_c = row, col
                elif row < 3 and col >= 3:
                    b_idx = 1
                    l_r, l_c = row, col-3
                elif row >= 3 and col < 3:
                    b_idx = 2
                    l_r, l_c = row-3, col
                else:
                    b_idx = 3
                    l_r, l_c = row-3, col-3
                    
                coeff_top = S2_V[s][b_idx]
                if coeff_top == 0: continue
                
                p_idx = l_r * 4 + l_c
                V[b][r] += coeff_top * S4_V[p_idx][t]
                
            # W Global
            # W index w = k*6 + i. k is col of C, i is row of C.
            for w in range(36):
                k = w // 6
                i = w % 6
                if i < 3 and k < 3:
                    b_idx = 0
                    l_r, l_c = i, k
                elif i < 3 and k >= 3:
                    b_idx = 1
                    l_r, l_c = i, k-3
                elif i >= 3 and k < 3:
                    b_idx = 2
                    l_r, l_c = i-3, k
                else:
                    b_idx = 3
                    l_r, l_c = i-3, k-3
                    
                coeff_top = S2_W[b_idx][s]
                if coeff_top == 0: continue
                
                p_idx = l_r * 4 + l_c
                W[w][r] += coeff_top * S4_W[p_idx][t]
                
    return {"U": U, "V": V, "W": W}
