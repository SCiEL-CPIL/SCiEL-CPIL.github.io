import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Helper to create a zero matrix
    def zeros(rows, cols):
        return [[0.0] * cols for _ in range(rows)]

    # Strassen 2x2x2 decomposition (Rank 7)
    # U, V, W are 4x7
    def strassen_222():
        U = zeros(4, 7)
        V = zeros(4, 7)
        W = zeros(4, 7)
        
        # M1 = (A00+A11)(B00+B11)
        # A00=0, A11=1. B00=0, B11=1.
        # U: A00, A11 -> 1, 1. V: B00, B11 -> 1, 1. W: C00, C11 -> 1, -1? 
        # C00 = M1 + ...
        # C11 = M1 - ...
        # So W for M1 contributes +1 to C00, +1 to C11? 
        # Let's check Strassen formulas:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # M1: A=A00+A11, B=B00+B11.
        # U: idx 0 (A00)=1, idx 1 (A01)=0, idx 2 (A10)=0, idx 3 (A11)=1.
        # V: idx 0 (B00)=1, idx 1 (B01)=0, idx 2 (B10)=0, idx 3 (B11)=1.
        # W: C00 gets +1, C01 gets 0, C10 gets 0, C11 gets +1.
        # W indices: 0(C00), 1(C01), 2(C10), 3(C11).
        
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        # M2 = (A10+A11)B00
        # U: A10=1, A11=1. V: B00=1.
        # W: C10 gets +1.
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1
        
        # M3 = A00(B01-B11)
        # U: A00=1. V: B01=1, B11=-1.
        # W: C01 gets +1, C11 gets +1.
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[1][2] = 1; W[3][2] = 1
        
        # M4 = A11(B10-B00)
        # U: A11=1. V: B10=1, B00=-1.
        # W: C00 gets +1, C10 gets +1.
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[2][3] = 1
        
        # M5 = (A00+A01)B11
        # U: A00=1, A01=1. V: B11=1.
        # W: C00 gets -1, C01 gets +1.
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[1][4] = 1
        
        # M6 = (A10-A00)(B00+B01)
        # U: A10=1, A00=-1. V: B00=1, B01=1.
        # W: C11 gets +1.
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        
        # M7 = (A01-A11)(B10+B11)
        # U: A01=1, A11=-1. V: B10=1, B11=1.
        # W: C00 gets +1.
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1
        
        return U, V, W

    # 2x2x4 decomposition (Rank 14)
    # A is 2x2. B is 2x4. C is 2x4.
    # Split B into B_L (2x2) and B_R (2x2).
    # Split C into C_L (2x2) and C_R (2x2).
    # Two independent 2x2x2 multiplications.
    def mul_224():
        U1, V1, W1 = strassen_222()
        U2, V2, W2 = strassen_222()
        
        U = zeros(4, 14)
        V = zeros(8, 14) # 2x4 = 8 entries
        W = zeros(8, 14) # 2x4 = 8 entries
        
        # Copy first half (B_L, C_L)
        for r in range(7):
            for i in range(4):
                U[i][r] = U1[i][r]
            for i in range(4):
                V[i][r] = V1[i][r] # B_L indices 0-3
            for i in range(4):
                W[i][r] = W1[i][r] # C_L indices 0-3
        
        # Copy second half (B_R, C_R)
        for r in range(7):
            for i in range(4):
                U[i][r+7] = U2[i][r]
            for i in range(4):
                V[i+4][r+7] = V2[i][r] # B_R indices 4-7
            for i in range(4):
                W[i+4][r+7] = W2[i][r] # C_R indices 4-7
                
        return U, V, W

    # 4x4x8 decomposition (Rank 98)
    # A is 4x4. B is 4x8. C is 4x8.
    # Blocks are 2x2 (A), 2x4 (B, C).
    
    # Mapping helpers
    # Global A indices: 0-15.
    # A00: 0,1,4,5. A01: 2,3,6,7. A10: 8,9,12,13. A11: 10,11,14,15.
    # Local A indices: 0,1,2,3.
    
    def expand_U(local_U, mapping):
        # local_U is 4x14. mapping is list of 4 lists of (global_idx, coeff)
        global_U = zeros(16, 14)
        for r in range(14):
            for local_i in range(4):
                val = local_U[local_i][r]
                if val != 0:
                    for g_idx, coeff in mapping[local_i]:
                        global_U[g_idx][r] += val * coeff
        return global_U

    def expand_V(local_V, mapping):
        # local_V is 8x14. mapping is list of 8 lists of (global_idx, coeff)
        global_V = zeros(32, 14)
        for r in range(14):
            for local_i in range(8):
                val = local_V[local_i][r]
                if val != 0:
                    for g_idx, coeff in mapping[local_i]:
                        global_V[g_idx][r] += val * coeff
        return global_V

    def expand_W(local_W, mapping):
        # local_W is 8x14. mapping is list of 8 lists of (global_idx, coeff)
        global_W = zeros(32, 14)
        for r in range(14):
            for local_i in range(8):
                val = local_W[local_i][r]
                if val != 0:
                    for g_idx, coeff in mapping[local_i]:
                        global_W[g_idx][r] += val * coeff
        return global_W

    # Define mappings for each Strassen term
    # Local A indices: 0(00), 1(01), 2(10), 3(11)
    # Global A blocks:
    # A00: [0,1,4,5]
    # A01: [2,3,6,7]
    # A10: [8,9,12,13]
    # A11: [10,11,14,15]
    
    # Local B indices: 0-3 (B_L 2x2), 4-7 (B_R 2x2)
    # Global B blocks:
    # B00: 0-3 (rows 0,1 cols 0,1)
    # B01: 4-7 (rows 0,1 cols 2,3)
    # B10: 8-11 (rows 2,3 cols 0,1)
    # B11: 12-15 (rows 2,3 cols 2,3)
    # Wait, B is 4x8.
    # Rows 0,1 are top half. Rows 2,3 are bottom half.
    # Cols 0,1 are B00? No.
    # B is 4x8.
    # B00 is 2x4? No, in Strassen for 4x4x8, B blocks are 2x4.
    # B00: rows 0,1; cols 0,3? No.
    # B is 4x8.
    # Split rows into 0,1 and 2,3.
    # Split cols into 0,1,2,3 and 4,5,6,7?
    # No, Strassen splits into halves.
    # Rows: 0,1 and 2,3.
    # Cols: 0,1,2,3 and 4,5,6,7.
    # So B00 is 2x4 (rows 0,1; cols 0-3).
    # B01 is 2x4 (rows 0,1; cols 4-7).
    # B10 is 2x4 (rows 2,3; cols 0-3).
    # B11 is 2x4 (rows 2,3; cols 4-7).
    
    # Local B in mul_224 is 2x4.
    # Local B indices 0-3 correspond to cols 0-1 of local B?
    # In mul_224, B is 2x4.
    # Local B_L is 2x2 (cols 0,1). Local B_R is 2x2 (cols 2,3).
    # So local index 0-3 is B_L. 4-7 is B_R.
    
    # Global B indices:
    # B00 (2x4): rows 0,1; cols 0,1,2,3.
    #   Indices: 0,1,2,3 (row 0), 4,5,6,7 (row 1).
    # B01 (2x4): rows 0,1; cols 4,5,6,7.
    #   Indices: 8,9,10,11 (row 0), 12,13,14,15 (row 1).
    # B10 (2x4): rows 2,3; cols 0,1,2,3.
    #   Indices: 16,17,18,19 (row 2), 20,21,22,23 (row 3).
    # B11 (2x4): rows 2,3; cols 4,5,6,7.
    #   Indices: 24,25,26,27 (row 2), 28,29,30,31 (row 3).

    # Local C in mul_224 is 2x4.
    # Same structure as B.

    # Mappings for A
    # Local A 0 (00) -> Global A00
    # Local A 1 (01) -> Global A01
    # Local A 2 (10) -> Global A10
    # Local A 3 (11) -> Global A11
    
    def get_A_mapping(A00_coeff, A01_coeff, A10_coeff, A11_coeff):
        # Returns list of 4 mappings
        map_0 = [(i, A00_coeff) for i in [0,1,4,5]]
        map_1 = [(i, A01_coeff) for i in [2,3,6,7]]
        map_2 = [(i, A10_coeff) for i in [8,9,12,13]]
        map_3 = [(i, A11_coeff) for i in [10,11,14,15]]
        return [map_0, map_1, map_2, map_3]

    def get_B_mapping(B00_coeff, B01_coeff, B10_coeff, B11_coeff):
        # Local B has 8 indices.
        # 0-3: B_L (cols 0,1). 4-7: B_R (cols 2,3).
        # Global B blocks are 2x4.
        # B00 covers cols 0-3.
        # So Local B_L (cols 0,1) maps to first half of B00?
        # No, Local B is 2x4.
        # Local B_L is 2x2 (cols 0,1).
        # Global B00 is 2x4 (cols 0-3).
        # So Local B_L maps to Global B00 cols 0,1.
        # Local B_R maps to Global B00 cols 2,3.
        
        # Global B00 indices: 0,1,2,3 (row 0), 4,5,6,7 (row 1).
        # Col 0: 0, 4. Col 1: 1, 5. Col 2: 2, 6. Col 3: 3, 7.
        
        # Local B_L indices: 0,1 (row 0), 2,3 (row 1).
        # Local B_L[0,0] -> Global B00[0,0] -> 0
        # Local B_L[0,1] -> Global B00[0,1] -> 1
        # Local B_L[1,0] -> Global B00[1,0] -> 4
        # Local B_L[1,1] -> Global B00[1,1] -> 5
        
        # Local B_R indices: 4,5 (row 0), 6,7 (row 1).
        # Local B_R[0,0] -> Global B00[0,2] -> 2
        # Local B_R[0,1] -> Global B00[0,3] -> 3
        # Local B_R[1,0] -> Global B00[1,2] -> 6
        # Local B_R[1,1] -> Global B00[1,3] -> 7
        
        # This logic applies to B00.
        # For B01, indices are shifted by 8.
        # For B10, shifted by 16.
        # For B11, shifted by 24.
        
        def block_map(offset, coeff):
            # Maps local B indices to global B indices for a specific block
            m = []
            # B_L part (local 0-3)
            m.append([(offset + 0, coeff), (offset + 4, coeff)]) # 0->0,4; 1->1,5? No.
            # Local 0 (row 0, col 0) -> Global offset+0 (row 0, col 0)
            # Local 1 (row 0, col 1) -> Global offset+1 (row 0, col 1)
            # Local 2 (row 1, col 0) -> Global offset+4 (row 1, col 0)
            # Local 3 (row 1, col 1) -> Global offset+5 (row 1, col 1)
            
            # Local 4 (row 0, col 0 of R) -> Global offset+2 (row 0, col 2)
            # Local 5 (row 0, col 1 of R) -> Global offset+3 (row 0, col 3)
            # Local 6 (row 1, col 0 of R) -> Global offset+6 (row 1, col 2)
            # Local 7 (row 1, col 1 of R) -> Global offset+7 (row 1, col 3)
            
            return [
                [(offset + 0, coeff)],
                [(offset + 1, coeff)],
                [(offset + 4, coeff)],
                [(offset + 5, coeff)],
                [(offset + 2, coeff)],
                [(offset + 3, coeff)],
                [(offset + 6, coeff)],
                [(offset + 7, coeff)]
            ]
        
        map_0 = block_map(0, B00_coeff)
        map_1 = block_map(8, B01_coeff)
        map_2 = block_map(16, B10_coeff)
        map_3 = block_map(24, B11_coeff)
        
        # Combine
        final_map = []
        for i in range(8):
            combined = []
            if B00_coeff: combined.extend(map_0[i])
            if B01_coeff: combined.extend(map_1[i])
            if B10_coeff: combined.extend(map_2[i])
            if B11_coeff: combined.extend(map_3[i])
            final_map.append(combined)
        return final_map

    def get_W_mapping(C00_coeff, C01_coeff, C10_coeff, C11_coeff):
        return get_B_mapping(C00_coeff, C01_coeff, C10_coeff, C11_coeff)

    # Strassen Terms
    # M1: A00+A11, B00+B11, C00+C11
    # M2: A10+A11, B00, C10
    # M3: A00, B01-B11, C01+C11
    # M4: A11, B10-B00, C00+C10
    # M5: A00+A01, B11, C00-C01? No C00 gets -1, C01 gets 1.
    # M6: A10-A00, B00+B01, C11
    # M7: A01-A11, B10+B11, C00
    
    terms = [
        (get_A_mapping(1, 0, 0, 1), get_B_mapping(1, 0, 0, 1), get_W_mapping(1, 0, 0, 1)), # M1
        (get_A_mapping(0, 0, 1, 1), get_B_mapping(1, 0, 0, 0), get_W_mapping(0, 0, 1, 0)), # M2
        (get_A_mapping(1, 0, 0, 0), get_B_mapping(0, 1, 0, -1), get_W_mapping(0, 1, 0, 1)), # M3
        (get_A_mapping(0, 0, 0, 1), get_B_mapping(-1, 0, 1, 0), get_W_mapping(1, 0, 1, 0)), # M4
        (get_A_mapping(1, 1, 0, 0), get_B_mapping(0, 0, 0, 1), get_W_mapping(-1, 1, 0, 0)), # M5
        (get_A_mapping(-1, 0, 1, 0), get_B_mapping(1, 1, 0, 0), get_W_mapping(0, 0, 0, 1)), # M6
        (get_A_mapping(0, 1, 0, -1), get_B_mapping(0, 0, 1, 1), get_W_mapping(1, 0, 0, 0)), # M7
    ]

    U_total = zeros(16, 98)
    V_total = zeros(32, 98)
    W_total = zeros(32, 98)
    
    U224, V224, W224 = mul_224()
    
    for t_idx, (map_A, map_B, map_C) in enumerate(terms):
        offset = t_idx * 14
        U_exp = expand_U(U224, map_A)
        V_exp = expand_V(V224, map_B)
        W_exp = expand_W(W224, map_C)
        
        for r in range(14):
            for i in range(16):
                U_total[i][offset + r] = U_exp[i][r]
            for i in range(32):
                V_total[i][offset + r] = V_exp[i][r]
                W_total[i][offset + r] = W_exp[i][r]

    return {"U": U_total, "V": V_total, "W": W_total}
# EVOLVE-BLOCK-END
