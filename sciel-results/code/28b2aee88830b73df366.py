import numpy as np
import time

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    
    # Rank 28 decomposition
    # Structure:
    # A is 2x4. Split into A_L (cols 0-1) and A_R (cols 2-3).
    # B is 4x4. Split into 4 blocks of 2x2:
    # B_LL (rows 0-1, cols 0-1)
    # B_LR (rows 0-1, cols 2-3)
    # B_RL (rows 2-3, cols 0-1)
    # B_RR (rows 2-3, cols 2-3)
    #
    # C is 2x4. Split into C_L (cols 0-1) and C_R (cols 2-3).
    # C_L = A_L B_LL + A_R B_RL
    # C_R = A_L B_LR + A_R B_RR
    #
    # We use 4 Strassen decompositions (rank 7 each).
    # Total rank 28.
    
    rank = 28
    U = np.zeros((n * m, rank), dtype=np.int64) # 8 x 28
    V = np.zeros((m * p, rank), dtype=np.int64) # 16 x 28
    W = np.zeros((p * n, rank), dtype=np.int64) # 8 x 28
    
    # Strassen 2x2x2 factors
    # Indices:
    # A: 0,1,2,3 (row, col) -> 0:00, 1:01, 2:10, 3:11
    # B: 0,1,2,3 (row, col)
    # C: 0,1,2,3 (row, col)
    
    # Strassen steps for 2x2x2
    # M1 = A00 * B00
    # M2 = A10 * B00
    # M3 = A00 * B11
    # M4 = A11 * B11
    # M5 = (A00+A11) * (B00+B11)
    # M6 = (A10+A11) * B10
    # M7 = A00 * (B01-B11)
    
    # Standard Strassen 7 terms for 2x2:
    # Let's use the explicit integer entries for U, V, W for one 2x2 block.
    # U (2x2 inputs, 7 ranks), V (2x2 inputs, 7 ranks), W (2x2 outputs, 7 ranks)
    
    # A indices: 0(0,0), 1(0,1), 2(1,0), 3(1,1)
    # B indices: 0(0,0), 1(0,1), 2(1,0), 3(1,1)
    # C indices: 0(0,0), 1(0,1), 2(1,0), 3(1,1)
    
    # Strassen coefficients (u, v, w) for each of 7 terms
    # Term 1: A00 * B00 -> C00, C10
    # Term 2: A10 * B00 -> C10
    # Term 3: A00 * B11 -> C01, C11
    # Term 4: A11 * B11 -> C11
    # Term 5: (A00+A11) * (B00+B11) -> C00, C01, C10, C11
    # Term 6: (A10+A11) * B10 -> C10, C11
    # Term 7: A00 * (B01-B11) -> C00, C01
    
    # Let's just hardcode the 7 vectors for U, V, W for a 2x2x2 multiplication.
    # U (4x7):
    # Col 0 (A00*B00): [1, 0, 0, 0]
    # Col 1 (A10*B00): [0, 0, 1, 0]
    # Col 2 (A00*B11): [1, 0, 0, 0]
    # Col 3 (A11*B11): [0, 0, 0, 1]
    # Col 4 ((A00+A11)*(B00+B11)): [1, 0, 0, 1]
    # Col 5 ((A10+A11)*B10): [0, 0, 1, 1]
    # Col 6 (A00*(B01-B11)): [1, 0, 0, 0]
    
    # V (4x7):
    # Col 0: [1, 0, 0, 0] (B00)
    # Col 1: [1, 0, 0, 0] (B00)
    # Col 2: [0, 0, 0, 1] (B11)
    # Col 3: [0, 0, 0, 1] (B11)
    # Col 4: [1, 0, 0, 1] (B00+B11)
    # Col 5: [0, 0, 1, 0] (B10)
    # Col 6: [0, 1, 0, -1] (B01-B11)
    
    # W (4x7):
    # C00: M1 - M5 + M6 + M7 ? No, standard reconstruction:
    # C00 = M1 - M5 + M6 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M4 + M5 + M6
    # Wait, standard Strassen:
    # M1 = A00(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = (A00+A01)B11
    # M4 = A11(B10-B00)
    # M5 = (A00+A11)(B00+B11)
    # M6 = (A10+A11)(B01-B11)
    # M7 = (A00-A10)(B00+B10)
    # This is one variant. Let's use the one from the previous code which worked.
    
    # Previous code Strassen:
    # M1: A{0,3} B{0,3} -> C{0,3} (This looks like (A00+A11)(B00+B11))
    # Let's stick to the working `_strassen_28` logic but optimize it.
    
    def get_strassen_block():
        # Returns U, V, W for 2x2x2 with rank 7
        # U (4x7), V (4x7), W (4x7)
        # Using the specific decomposition from the seed that worked.
        
        # The seed used:
        # ({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1})   # M1
        # ({2:1, 3:1}, {0:1}, {2:1, 3:-1})        # M2
        # ({0:1}, {1:1, 3:-1}, {1:1, 3:1})        # M3
        # ({3:1}, {2:1, 0:-1}, {0:1, 2:1})        # M4
        # ({0:1, 1:1}, {3:1}, {0:-1, 1:1})        # M5
        # ({2:1, 0:-1}, {0:1, 1:1}, {3:1})        # M6
        # ({1:1, 3:-1}, {2:1, 3:1}, {0:1})        # M7
        
        # Indices: 0:00, 1:01, 2:10, 3:11
        U_block = np.zeros((4, 7), dtype=np.int64)
        V_block = np.zeros((4, 7), dtype=np.int64)
        W_block = np.zeros((4, 7), dtype=np.int64)
        
        # Term 0
        U_block[0, 0] = 1; U_block[3, 0] = 1
        V_block[0, 0] = 1; V_block[3, 0] = 1
        W_block[0, 0] = 1; W_block[3, 0] = 1
        
        # Term 1
        U_block[2, 1] = 1; U_block[3, 1] = 1
        V_block[0, 1] = 1
        W_block[2, 1] = 1; W_block[3, 1] = -1
        
        # Term 2
        U_block[0, 2] = 1
        V_block[1, 2] = 1; V_block[3, 2] = -1
        W_block[1, 2] = 1; W_block[3, 2] = 1
        
        # Term 3
        U_block[3, 3] = 1
        V_block[2, 3] = 1; V_block[0, 3] = -1
        W_block[0, 3] = 1; W_block[2, 3] = 1
        
        # Term 4
        U_block[0, 4] = 1; U_block[1, 4] = 1
        V_block[3, 4] = 1
        W_block[0, 4] = -1; W_block[1, 4] = 1
        
        # Term 5
        U_block[2, 5] = 1; U_block[0, 5] = -1
        V_block[0, 5] = 1; V_block[1, 5] = 1
        W_block[3, 5] = 1
        
        # Term 6
        U_block[1, 6] = 1; U_block[3, 6] = -1
        V_block[2, 6] = 1; V_block[3, 6] = 1
        W_block[0, 6] = 1
        
        return U_block, V_block, W_block

    U_b, V_b, W_b = get_strassen_block()
    
    # Mapping of global indices to local 2x2 indices
    # A (2x4):
    # A_L (cols 0,1): Global A indices 0,1,4,5?
    # A is 2x4. Row 0: 0,1,2,3. Row 1: 4,5,6,7.
    # A_L uses cols 0,1. Indices: 0,1 (row 0), 4,5 (row 1).
    # Local 2x2 A indices: 0:00, 1:01, 2:10, 3:11.
    # Map Local 0 -> Global 0
    # Map Local 1 -> Global 1
    # Map Local 2 -> Global 4
    # Map Local 3 -> Global 5
    
    # A_R uses cols 2,3. Indices: 2,3 (row 0), 6,7 (row 1).
    # Map Local 0 -> Global 2
    # Map Local 1 -> Global 3
    # Map Local 2 -> Global 6
    # Map Local 3 -> Global 7
    
    # B (4x4):
    # B_LL (rows 0-1, cols 0-1).
    # B indices: Row 0: 0,1,2,3. Row 1: 4,5,6,7. Row 2: 8,9,10,11. Row 3: 12,13,14,15.
    # B_LL:
    # Local 0 (0,0) -> Global 0
    # Local 1 (0,1) -> Global 1
    # Local 2 (1,0) -> Global 4
    # Local 3 (1,1) -> Global 5
    
    # B_RL (rows 2-3, cols 0-1).
    # Local 0 (0,0) -> Global 8 (Row 2, Col 0)
    # Local 1 (0,1) -> Global 9
    # Local 2 (1,0) -> Global 12 (Row 3, Col 0)
    # Local 3 (1,1) -> Global 13
    
    # B_LR (rows 0-1, cols 2-3).
    # Local 0 (0,0) -> Global 2 (Row 0, Col 2)
    # Local 1 (0,1) -> Global 3
    # Local 2 (1,0) -> Global 6
    # Local 3 (1,1) -> Global 7
    
    # B_RR (rows 2-3, cols 2-3).
    # Local 0 (0,0) -> Global 10 (Row 2, Col 2)
    # Local 1 (0,1) -> Global 11
    # Local 2 (1,0) -> Global 14
    # Local 3 (1,1) -> Global 15
    
    # C (2x4):
    # C_L (cols 0-1).
    # Local 0 (0,0) -> Global 0
    # Local 1 (0,1) -> Global 1
    # Local 2 (1,0) -> Global 4
    # Local 3 (1,1) -> Global 5
    
    # C_R (cols 2-3).
    # Local 0 (0,0) -> Global 2
    # Local 1 (0,1) -> Global 3
    # Local 2 (1,0) -> Global 6
    # Local 3 (1,1) -> Global 7
    
    def add_block(u_map, v_map, w_map, offset):
        for r in range(7):
            for li in range(4):
                if U_b[li, r] != 0:
                    U[u_map[li], offset + r] += U_b[li, r]
                if V_b[li, r] != 0:
                    V[v_map[li], offset + r] += V_b[li, r]
                if W_b[li, r] != 0:
                    W[w_map[li], offset + r] += W_b[li, r]
    
    # Block 1: A_L * B_LL -> C_L
    # A_L map: {0:0, 1:1, 2:4, 3:5}
    # B_LL map: {0:0, 1:1, 2:4, 3:5}
    # C_L map: {0:0, 1:1, 2:4, 3:5}
    add_block([0, 1, 4, 5], [0, 1, 4, 5], [0, 1, 4, 5], 0)
    
    # Block 2: A_R * B_RL -> C_L
    # A_R map: {0:2, 1:3, 2:6, 3:7}
    # B_RL map: {0:8, 1:9, 2:12, 3:13}
    # C_L map: {0:0, 1:1, 2:4, 3:5}
    add_block([2, 3, 6, 7], [8, 9, 12, 13], [0, 1, 4, 5], 7)
    
    # Block 3: A_L * B_LR -> C_R
    # A_L map: {0:0, 1:1, 2:4, 3:5}
    # B_LR map: {0:2, 1:3, 2:6, 3:7}
    # C_R map: {0:2, 1:3, 2:6, 3:7}
    add_block([0, 1, 4, 5], [2, 3, 6, 7], [2, 3, 6, 7], 14)
    
    # Block 4: A_R * B_RR -> C_R
    # A_R map: {0:2, 1:3, 2:6, 3:7}
    # B_RR map: {0:10, 1:11, 2:14, 3:15}
    # C_R map: {0:2, 1:3, 2:6, 3:7}
    add_block([2, 3, 6, 7], [10, 11, 14, 15], [2, 3, 6, 7], 21)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
