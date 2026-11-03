import numpy as np
import time

DIMENSIONS = [3, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    
    # 1. Define Strassen factors for 2x2x2
    # U: (4, 7), V: (4, 7), W: (4, 7)
    # Indices for 2x2 matrix flattened row-major: 00->0, 01->1, 10->2, 11->3
    
    U_222 = np.zeros((4, 7))
    V_222 = np.zeros((4, 7))
    W_222 = np.zeros((4, 7))
    
    # M1 = (A00+A11)(B00+B11)
    U_222[0, 0] = 1; U_222[3, 0] = 1
    V_222[0, 0] = 1; V_222[3, 0] = 1
    # C00 += M1, C11 += M1
    W_222[0, 0] = 1; W_222[3, 0] = 1
    
    # M2 = (A10+A11)B00
    U_222[2, 1] = 1; U_222[3, 1] = 1
    V_222[0, 1] = 1
    # C10 += M2, C11 -= M2
    W_222[2, 1] = 1; W_222[3, 1] = -1
    
    # M3 = A00(B01-B11)
    U_222[0, 2] = 1
    V_222[1, 2] = 1; V_222[3, 2] = -1
    # C01 += M3, C11 += M3
    W_222[1, 2] = 1; W_222[3, 2] = 1
    
    # M4 = A11(B10-B00)
    U_222[3, 3] = 1
    V_222[2, 3] = 1; V_222[0, 3] = -1
    # C00 += M4, C10 += M4
    W_222[0, 3] = 1; W_222[2, 3] = 1
    
    # M5 = (A00+A01)B11
    U_222[0, 4] = 1; U_222[1, 4] = 1
    V_222[3, 4] = 1
    # C00 -= M5, C01 += M5
    W_222[0, 4] = -1; W_222[1, 4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U_222[2, 5] = 1; U_222[0, 5] = -1
    V_222[0, 5] = 1; V_222[1, 5] = 1
    # C11 += M6
    W_222[3, 5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U_222[1, 6] = 1; U_222[3, 6] = -1
    V_222[2, 6] = 1; V_222[3, 6] = 1
    # C00 += M7
    W_222[0, 6] = 1

    # 2. Construct 2x4x4 decomposition (Rank 28)
    # A is 2x4, B is 4x4, C is 2x4
    # Split A into AL (2x2) and AR (2x2)
    # Split B into BL (4x2? No, B is 4x4. We need 2x2 blocks.)
    # B = [[B00, B01], [B10, B11]] where each is 2x2.
    # C = AL * [B00, B01] + AR * [B10, B11]
    # C00_part = AL*B00 + AR*B10
    # C01_part = AL*B01 + AR*B11
    
    # We use 4 Strassen instances:
    # S1: AL * B00 -> C_part1 (2x2)
    # S2: AL * B01 -> C_part2 (2x2)
    # S3: AR * B10 -> C_part3 (2x2)
    # S4: AR * B11 -> C_part4 (2x2)
    
    # Total Rank 28.
    U_244 = np.zeros((8, 28)) # 2*4 = 8 rows
    V_244 = np.zeros((16, 28)) # 4*4 = 16 rows
    W_244 = np.zeros((8, 28)) # 2*4 = 8 rows
    
    # Mapping indices for A (2x4):
    # AL: rows 0,1 of A. Cols 0,1 of A.
    # AR: rows 0,1 of A. Cols 2,3 of A.
    # A_flat indices:
    # Row 0: 0,1,2,3
    # Row 1: 4,5,6,7
    
    # Mapping indices for B (4x4):
    # B_flat indices:
    # Row 0: 0,1,2,3
    # Row 1: 4,5,6,7
    # Row 2: 8,9,10,11
    # Row 3: 12,13,14,15
    
    # Mapping indices for C (2x4):
    # C_flat indices:
    # Row 0: 0,1,2,3
    # Row 1: 4,5,6,7
    
    # S1: AL * B00
    # AL uses A indices: 0,1,4,5 (00,01,10,11)
    # B00 uses B indices: 0,1,4,5 (00,01,10,11)
    # Result C_part1 is top-left 2x2 of C?
    # C = [AL*B00 + AR*B10, AL*B01 + AR*B11]
    # So C_part1 adds to C[0:2, 0:2]
    # C indices for top-left 2x2: 0,1,4,5
    
    for r in range(7):
        col = r
        # U
        for a_idx in range(4):
            if U_222[a_idx, r] != 0:
                # Map local 2x2 index a_idx to global A index
                # Local 0->0, 1->1, 2->4, 3->5
                global_a = [0, 1, 4, 5][a_idx]
                U_244[global_a, col] += U_222[a_idx, r]
        # V
        for b_idx in range(4):
            if V_222[b_idx, r] != 0:
                # Map local 2x2 index b_idx to global B index (B00)
                # Local 0->0, 1->1, 2->4, 3->5
                global_b = [0, 1, 4, 5][b_idx]
                V_244[global_b, col] += V_222[b_idx, r]
        # W
        for c_idx in range(4):
            if W_222[c_idx, r] != 0:
                # Map local 2x2 index c_idx to global C index (Top-Left)
                # Local 0->0, 1->1, 2->4, 3->5
                global_c = [0, 1, 4, 5][c_idx]
                W_244[global_c, col] += W_222[c_idx, r]

    # S2: AL * B01
    # AL uses A indices: 0,1,4,5
    # B01 uses B indices: 2,3,6,7 (Top-Right 2x2 of B)
    # Result adds to C[0:2, 2:4] (Top-Right 2x2 of C)
    # C indices: 2,3,6,7
    
    for r in range(7):
        col = 7 + r
        for a_idx in range(4):
            if U_222[a_idx, r] != 0:
                global_a = [0, 1, 4, 5][a_idx]
                U_244[global_a, col] += U_222[a_idx, r]
        for b_idx in range(4):
            if V_222[b_idx, r] != 0:
                global_b = [2, 3, 6, 7][b_idx]
                V_244[global_b, col] += V_222[b_idx, r]
        for c_idx in range(4):
            if W_222[c_idx, r] != 0:
                global_c = [2, 3, 6, 7][c_idx]
                W_244[global_c, col] += W_222[c_idx, r]

    # S3: AR * B10
    # AR uses A indices: 2,3,6,7 (Bottom-Right 2x2 of A? No, A is 2x4. AR is cols 2,3)
    # A indices for AR: 2,3,6,7
    # B10 uses B indices: 8,9,12,13 (Bottom-Left 2x2 of B)
    # Result adds to C[0:2, 0:2] (Top-Left 2x2 of C)
    # C indices: 0,1,4,5
    
    for r in range(7):
        col = 14 + r
        for a_idx in range(4):
            if U_222[a_idx, r] != 0:
                global_a = [2, 3, 6, 7][a_idx]
                U_244[global_a, col] += U_222[a_idx, r]
        for b_idx in range(4):
            if V_222[b_idx, r] != 0:
                global_b = [8, 9, 12, 13][b_idx]
                V_244[global_b, col] += V_222[b_idx, r]
        for c_idx in range(4):
            if W_222[c_idx, r] != 0:
                global_c = [0, 1, 4, 5][c_idx]
                W_244[global_c, col] += W_222[c_idx, r]

    # S4: AR * B11
    # AR uses A indices: 2,3,6,7
    # B11 uses B indices: 10,11,14,15 (Bottom-Right 2x2 of B)
    # Result adds to C[0:2, 2:4] (Top-Right 2x2 of C)
    # C indices: 2,3,6,7
    
    for r in range(7):
        col = 21 + r
        for a_idx in range(4):
            if U_222[a_idx, r] != 0:
                global_a = [2, 3, 6, 7][a_idx]
                U_244[global_a, col] += U_222[a_idx, r]
        for b_idx in range(4):
            if V_222[b_idx, r] != 0:
                global_b = [10, 11, 14, 15][b_idx]
                V_244[global_b, col] += V_222[b_idx, r]
        for c_idx in range(4):
            if W_222[c_idx, r] != 0:
                global_c = [2, 3, 6, 7][c_idx]
                W_244[global_c, col] += W_222[c_idx, r]

    # 3. Construct 1x4x4 decomposition (Rank 16)
    # A is 1x4, B is 4x4, C is 1x4
    # Schoolbook
    U_144 = np.zeros((4, 16))
    V_144 = np.zeros((16, 16))
    W_144 = np.zeros((4, 16))
    
    for i in range(4): # A col / B row
        for k in range(4): # B col / C col
            col = i * 4 + k
            U_144[i, col] = 1
            V_144[i * 4 + k, col] = 1
            W_144[k, col] = 1

    # 4. Combine for 3x4x8
    # A is 3x4. Split into A_top (2x4) and A_bot (1x4).
    # B is 4x8. Split into B_left (4x4) and B_right (4x4).
    # C is 3x8.
    
    # We need 4 multiplications:
    # 1. A_top * B_left (2x4x4) -> C_top_left (2x4)
    # 2. A_top * B_right (2x4x4) -> C_top_right (2x4)
    # 3. A_bot * B_left (1x4x4) -> C_bot_left (1x4)
    # 4. A_bot * B_right (1x4x4) -> C_bot_right (1x4)
    
    R_total = 28 + 28 + 16 + 16 # 88
    
    U_total = np.zeros((12, R_total)) # 3*4
    V_total = np.zeros((32, R_total)) # 4*8
    W_total = np.zeros((24, R_total)) # 8*3
    
    # Block 1: A_top * B_left
    # U: A_top rows 0,1 of A (indices 0-7)
    # V: B_left cols 0-3 of B (indices 0-15)
    # W: C_top_left rows 0-3 of C (indices 0-11)
    
    # Copy U_244 to U_total[:, 0:28]
    U_total[0:8, 0:28] = U_244
    # Copy V_244 to V_total[0:16, 0:28]
    V_total[0:16, 0:28] = V_244
    # Copy W_244 to W_total[0:8, 0:28]
    # W indices for C (3x8):
    # C is flattened row-major: 0,1,2,3,4,5,6,7, 8,9,10,11,12,13,14,15, 16,17,18,19,20,21,22,23
    # Top-left 2x4 of C corresponds to rows 0,1 of C, cols 0-3.
    # C indices: 0,1,2,3, 4,5,6,7
    W_total[0:8, 0:28] = W_244
    
    # Block 2: A_top * B_right
    # U: A_top (same)
    # V: B_right cols 4-7 of B (indices 16-31)
    # W: C_top_right rows 0-3 of C, cols 4-7.
    # C indices: 8,9,10,11, 12,13,14,15
    
    U_total[0:8, 28:56] = U_244
    V_total[16:32, 28:56] = V_244
    # W mapping: W_244 has indices 0-7. We need to map to 8-15.
    # W_244 is (8, 28).
    # We need to place it in W_total[8:16, 28:56]?
    # Wait, W_total rows correspond to C elements.
    # C_top_right is rows 0,1 of C, cols 4-7.
    # Row 0 of C: 0,1,2,3,4,5,6,7. Cols 4-7 are indices 4,5,6,7.
    # Row 1 of C: 8,9,10,11,12,13,14,15. Cols 4-7 are indices 12,13,14,15.
    # So W_total rows should be 4,5,6,7, 12,13,14,15.
    
    # Let's check W_244 structure.
    # W_244 is 2x4 output. Flattened: 0,1,2,3,4,5,6,7.
    # So row 0 of W_244 maps to C[0,0..3] -> C indices 0,1,2,3.
    # Row 1 of W_244 maps to C[1,0..3] -> C indices 4,5,6,7.
    
    # For Block 1 (Top-Left), C indices were 0,1,2,3 and 4,5,6,7.
    # So W_total[0:4, 0:28] = W_244[0:4, :]
    # W_total[4:8, 0:28] = W_244[4:8, :]
    # This matches W_total[0:8, 0:28] = W_244. Correct.
    
    # For Block 2 (Top-Right), C indices are 4,5,6,7 and 12,13,14,15.
    # W_244[0:4, :] corresponds to C[0,0..3] relative to the block.
    # We need to place this at C[0,4..7] which are global indices 4,5,6,7.
    # And W_244[4:8, :] corresponds to C[1,0..3] relative to the block.
    # We need to place this at C[1,4..7] which are global indices 12,13,14,15.
    
    W_total[4:8, 28:56] = W_244[0:4, :]
    W_total[12:16, 28:56] = W_244[4:8, :]
    
    # Block 3: A_bot * B_left
    # U: A_bot row 2 of A (indices 8-11)
    # V: B_left (same as Block 1)
    # W: C_bot_left rows 2 of C, cols 0-3.
    # C indices: 16,17,18,19
    
    U_total[8:12, 56:72] = U_144
    V_total[0:16, 56:72] = V_144
    # W_144 is (4, 16). Row 0 corresponds to C[0,0..3].
    # We need to place at C[2,0..3] which are global indices 16,17,18,19.
    W_total[16:20, 56:72] = W_144
    
    # Block 4: A_bot * B_right
    # U: A_bot (same)
    # V: B_right (same as Block 2)
    # W: C_bot_right rows 2 of C, cols 4-7.
    # C indices: 20,21,22,23
    
    U_total[8:12, 72:88] = U_144
    V_total[16:32, 72:88] = V_144
    W_total[20:24, 72:88] = W_144

    return {"U": U_total.tolist(), "V": V_total.tolist(), "W": W_total.tolist()}
