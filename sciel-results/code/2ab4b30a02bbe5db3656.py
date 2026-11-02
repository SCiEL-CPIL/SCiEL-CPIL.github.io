import numpy as np
import time

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    # n=2, m=4, p=8
    # U: (n*m, rank) = (8, 56)
    # V: (m*p, rank) = (32, 56)
    # W: (p*n, rank) = (16, 56)
    
    rank = 56
    U = np.zeros((n * m, rank), dtype=np.int8)
    V = np.zeros((m * p, rank), dtype=np.int8)
    W = np.zeros((p * n, rank), dtype=np.int8)
    
    # Local 2x2 Strassen decomposition
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Local indices for A (2x2): 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Local indices for B (2x2): 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Local indices for C (2x2): 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    U_local = np.zeros((4, 7), dtype=np.int8)
    V_local = np.zeros((4, 7), dtype=np.int8)
    W_local = np.zeros((4, 7), dtype=np.int8)
    
    # r=0 (M1)
    U_local[0, 0] = 1; U_local[3, 0] = 1
    V_local[0, 0] = 1; V_local[3, 0] = 1
    W_local[0, 0] = 1; W_local[3, 0] = 1
    
    # r=1 (M2)
    U_local[2, 1] = 1; U_local[3, 1] = 1
    V_local[0, 1] = 1
    W_local[2, 1] = 1; W_local[3, 1] = -1
    
    # r=2 (M3)
    U_local[0, 2] = 1
    V_local[1, 2] = 1; V_local[3, 2] = -1
    W_local[1, 2] = 1; W_local[3, 2] = 1
    
    # r=3 (M4)
    U_local[3, 3] = 1
    V_local[2, 3] = 1; V_local[0, 3] = -1
    W_local[0, 3] = 1; W_local[2, 3] = 1
    
    # r=4 (M5)
    U_local[0, 4] = 1; U_local[1, 4] = 1
    V_local[3, 4] = 1
    W_local[0, 4] = -1; W_local[1, 4] = 1
    
    # r=5 (M6)
    U_local[2, 5] = 1; U_local[0, 5] = -1
    V_local[0, 5] = 1; V_local[1, 5] = 1
    W_local[3, 5] = 1
    
    # r=6 (M7)
    U_local[1, 6] = 1; U_local[3, 6] = -1
    V_local[2, 6] = 1; V_local[3, 6] = 1
    W_local[0, 6] = 1
    
    # Global Mapping
    # We decompose the 2x4x8 multiplication into 8 blocks of 2x2x2?
    # No, we decompose into 8 blocks of 2x2x2 is not quite right.
    # A (2x4) = [A1, A2] where A1, A2 are 2x2.
    # B (4x8) = [[B11, B12], [B21, B22]] where Bji are 2x4.
    # C (2x8) = [C1, C2] where C1, C2 are 2x4.
    # C = A1 * [B11, B12] + A2 * [B21, B22]
    # C1 = A1 B11 + A2 B21
    # C2 = A1 B12 + A2 B22
    
    # Each term like A1 B11 is a 2x2x4 multiplication.
    # 2x2x4 can be split into two 2x2x2 multiplications.
    # B11 (2x4) = [B11_L, B11_R] where B11_L, B11_R are 2x2.
    # A1 B11 = A1 B11_L + A1 B11_R.
    # So C1 = A1 B11_L + A1 B11_R + A2 B21_L + A2 B21_R.
    # There are 4 such 2x2x2 multiplications for C1.
    # Similarly 4 for C2.
    # Total 8 blocks of 2x2x2.
    # Each block has rank 7.
    # Total rank 8 * 7 = 56.
    
    # Block indices:
    # j: 0..3 (column block of C, size 2) -> k_g = 2j, 2j+1
    # s: 0..1 (selector for A row-block / B row-block)
    #   s=0: A1 (rows 0-1 of A? No, A is 2x4. A1 is cols 0-1, A2 is cols 2-3)
    #   Wait, A is 2x4. Splitting columns: A_L (cols 0-1), A_R (cols 2-3).
    #   B is 4x8. Splitting rows: B_T (rows 0-1), B_B (rows 2-3).
    #   C = A_L B_T + A_R B_B.
    #   A_L is 2x2. B_T is 2x8.
    #   A_L B_T is 2x8.
    #   This is one 2x2x8 multiplication.
    #   2x2x8 can be split into four 2x2x2 multiplications.
    #   So total 2 (from A_L B_T and A_R B_B) * 4 = 8 blocks.
    
    # Let's define the 8 blocks.
    # Block b in 0..7
    # b = 0..3 corresponds to A_L B_T
    # b = 4..7 corresponds to A_R B_B
    
    for b in range(8):
        if b < 4:
            a_selector = 0  # A_L (cols 0,1)
            b_selector = 0  # B_T (rows 0,1)
            c_col_offset = (b % 2) * 2 # C cols 0,1 or 2,3
            # Wait, B_T is 2x8. Split into 4 blocks of 2x2.
            # B_T[:, 0:2], B_T[:, 2:4], B_T[:, 4:6], B_T[:, 6:8]
            # b=0: B_T[:, 0:2] -> C[:, 0:2]
            # b=1: B_T[:, 2:4] -> C[:, 2:4]
            # b=2: B_T[:, 4:6] -> C[:, 4:6]
            # b=3: B_T[:, 6:8] -> C[:, 6:8]
        else:
            a_selector = 1  # A_R (cols 2,3)
            b_selector = 1  # B_B (rows 2,3)
            # B_B is 2x8.
            # b=4: B_B[:, 0:2] -> C[:, 0:2]
            # b=5: B_B[:, 2:4] -> C[:, 2:4]
            # b=6: B_B[:, 4:6] -> C[:, 4:6]
            # b=7: B_B[:, 6:8] -> C[:, 6:8]
            
            # Wait, C is sum of A_L B_T and A_R B_B.
            # So C[:, 0:2] gets contribution from b=0 and b=4.
            # This means we sum the W vectors for these blocks.
            
            c_col_offset = (b % 4) * 2
            # Actually, the column offset is determined by the block index within the 4 blocks.
            # b=0,4 -> cols 0,1
            # b=1,5 -> cols 2,3
            # b=2,6 -> cols 4,5
            # b=3,7 -> cols 6,7
            
            col_idx = (b % 4) * 2
            
        # Local A indices (0..3) map to Global A indices (0..7)
        # A is 2x4. Flat index i*4 + j.
        # A_L (a_selector=0): cols 0,1.
        #   local (0,0) -> global (0,0) -> idx 0
        #   local (0,1) -> global (0,1) -> idx 1
        #   local (1,0) -> global (1,0) -> idx 4
        #   local (1,1) -> global (1,1) -> idx 5
        # A_R (a_selector=1): cols 2,3.
        #   local (0,0) -> global (0,2) -> idx 2
        #   local (0,1) -> global (0,3) -> idx 3
        #   local (1,0) -> global (1,2) -> idx 6
        #   local (1,1) -> global (1,3) -> idx 7
        
        if a_selector == 0:
            a_map = [0, 1, 4, 5]
        else:
            a_map = [2, 3, 6, 7]
            
        # Local B indices (0..3) map to Global B indices (0..31)
        # B is 4x8. Flat index i*8 + j.
        # B_T (b_selector=0): rows 0,1.
        #   local (0,0) -> global (0, col_idx)
        #   local (0,1) -> global (0, col_idx+1)
        #   local (1,0) -> global (1, col_idx)
        #   local (1,1) -> global (1, col_idx+1)
        # B_B (b_selector=1): rows 2,3.
        #   local (0,0) -> global (2, col_idx)
        #   local (0,1) -> global (2, col_idx+1)
        #   local (1,0) -> global (3, col_idx)
        #   local (1,1) -> global (3, col_idx+1)
        
        if b_selector == 0:
            row_base = 0
        else:
            row_base = 2
            
        b_map = [
            row_base * 8 + col_idx,
            row_base * 8 + col_idx + 1,
            (row_base + 1) * 8 + col_idx,
            (row_base + 1) * 8 + col_idx + 1
        ]
        
        # Local C indices (0..3) map to Global W indices (0..15)
        # W index k*n + i. k is col (0..7), i is row (0..1).
        # C cols are col_idx, col_idx+1.
        # local (0,0) -> k=col_idx, i=0 -> idx col_idx*2 + 0
        # local (0,1) -> k=col_idx+1, i=0 -> idx (col_idx+1)*2 + 0
        # local (1,0) -> k=col_idx, i=1 -> idx col_idx*2 + 1
        # local (1,1) -> k=col_idx+1, i=1 -> idx (col_idx+1)*2 + 1
        
        c_map = [
            col_idx * 2,
            (col_idx + 1) * 2,
            col_idx * 2 + 1,
            (col_idx + 1) * 2 + 1
        ]
        
        # Base rank for this block
        base_r = b * 7
        
        # Fill U, V, W
        for l in range(7):
            r = base_r + l
            
            for la in range(4):
                val = U_local[la, l]
                if val != 0:
                    U[a_map[la], r] += val
                    
            for lb in range(4):
                val = V_local[lb, l]
                if val != 0:
                    V[b_map[lb], r] += val
                    
            for lc in range(4):
                val = W_local[lc, l]
                if val != 0:
                    W[c_map[lc], r] += val

    if time.time() - start_time > deadline:
        # Should not happen, but safety check
        pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
