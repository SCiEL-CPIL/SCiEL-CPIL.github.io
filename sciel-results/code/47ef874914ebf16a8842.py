import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS
    # Rank 114: 49 (Strassen 4x4x4) + 49 (Strassen 4x4x4) + 16 (Schoolbook 4x4x1)
    
    # Helper to build Strassen 2x2x2
    def build_strassen_222():
        # Returns U, V, W for 2x2x2 with rank 7
        # U: 4x7, V: 4x7, W: 4x7
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (A11+A22)(B11+B22)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1 # C11 += M1, C22 += M1
        
        # M2 = (A21+A22)B11
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1 # C21 += M2
        
        # M3 = A11(B12-B22)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1 # C12 += M3
        
        # M4 = A22(B21-B11)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1 # C11 += M4, C21 += M4
        
        # M5 = (A11+A12)B22
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1 # C11 -= M5, C12 += M5
        
        # M6 = (A21-A11)(B11+B12)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1 # C22 += M6
        
        # M7 = (A12-A22)(B21+B22)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1; W[3, 6] = -1 # C11 += M7, C22 -= M7
        
        return U, V, W

    def build_strassen_444():
        # Uses 7 instances of 2x2x2 Strassen
        # U: 16x49, V: 16x49, W: 16x49
        U = np.zeros((16, 49))
        V = np.zeros((16, 49))
        W = np.zeros((16, 49))
        
        U2, V2, W2 = build_strassen_222()
        
        # The 7 multiplications in Strassen 4x4x4
        # Each is a 2x2x2 multiplication
        # We need to map the 2x2 blocks of A, B to the 4x4 indices
        
        # A blocks:
        # A11: 0,1,4,5
        # A12: 2,3,6,7
        # A21: 8,9,12,13
        # A22: 10,11,14,15
        
        # B blocks:
        # B11: 0,1,4,5
        # B12: 2,3,6,7
        # B21: 8,9,12,13
        # B22: 10,11,14,15
        
        # C blocks:
        # C11: 0,1,4,5
        # C12: 2,3,6,7
        # C21: 8,9,12,13
        # C22: 10,11,14,15
        
        # The 7 terms M1..M7
        # M1: A11+A22, B11+B22 -> C11, C22
        # M2: A21+A22, B11 -> C21
        # M3: A11, B12-B22 -> C12
        # M4: A22, B21-B11 -> C11, C21
        # M5: A11+A12, B22 -> C11, C12
        # M6: A21-A11, B11+B12 -> C22
        # M7: A12-A22, B21+B22 -> C11, C22
        
        # We define the linear combinations for A and B for each M
        # Then we use the base 2x2x2 decomposition for each M
        
        # Base 2x2x2 U, V, W map local indices 0,1,2,3 to global block indices
        # Local 0 -> Block 11 (0,1,4,5)
        # Local 1 -> Block 12 (2,3,6,7)
        # Local 2 -> Block 21 (8,9,12,13)
        # Local 3 -> Block 22 (10,11,14,15)
        
        block_maps = {
            0: [0, 1, 4, 5],
            1: [2, 3, 6, 7],
            2: [8, 9, 12, 13],
            3: [10, 11, 14, 15]
        }
        
        # For each of the 7 M's, we have a specific combination of blocks
        # We will iterate through the 7 columns of U2, V2, W2
        # and place them into U, V, W at the appropriate global indices
        
        # The 7 M's correspond to the 7 columns of the base decomposition
        # But the inputs to the base decomposition are the linear combinations
        # So for M1, the "A" input is A11+A22.
        # In the base decomposition, U2[0,0]=1, U2[3,0]=1 means A_local0 + A_local3.
        # So we need to map local0 to A11 and local3 to A22.
        
        # Let's define the mapping of local indices to global block indices for each M
        # M1: A: {0:11, 3:22}, B: {0:11, 3:22}
        # M2: A: {2:21, 3:22}, B: {0:11}
        # M3: A: {0:11}, B: {1:12, 3:22}
        # M4: A: {3:22}, B: {2:21, 0:11}
        # M5: A: {0:11, 1:12}, B: {3:22}
        # M6: A: {2:21, 0:11}, B: {0:11, 1:12}
        # M7: A: {1:12, 3:22}, B: {2:21, 3:22}
        
        # This is getting complex to hardcode.
        # Instead, we can just build the 49 rank decomposition directly by composing the linear forms.
        
        # Let's use a simpler approach: 
        # The Strassen 4x4x4 is a known decomposition. 
        # We can generate it by applying the 2x2x2 Strassen to the 7 pairs of 2x2 matrices.
        
        # Let's create a function to add a 2x2x2 decomposition to the main matrices
        def add_222(U_main, V_main, W_main, offset, a_map, b_map, c_map, coeff=1):
            # a_map: dict local_idx -> global_block_idx
            # b_map: dict local_idx -> global_block_idx
            # c_map: dict local_idx -> global_block_idx
            # offset: column offset in U_main
            
            for col in range(7):
                for local_idx, val in enumerate(U2[:, col]):
                    if val != 0:
                        g_idx = a_map[local_idx]
                        for g in block_maps[g_idx]:
                            U_main[g, offset + col] += val * coeff
                            
                for local_idx, val in enumerate(V2[:, col]):
                    if val != 0:
                        g_idx = b_map[local_idx]
                        for g in block_maps[g_idx]:
                            V_main[g, offset + col] += val * coeff
                            
                for local_idx, val in enumerate(W2[:, col]):
                    if val != 0:
                        g_idx = c_map[local_idx]
                        for g in block_maps[g_idx]:
                            W_main[g, offset + col] += val * coeff

        # Define the mappings for the 7 M's
        # M1: A11+A22, B11+B22 -> C11, C22
        # In base 222, col 0 is M1.
        # U2 col 0: A0=1, A3=1. So local0->A11, local3->A22.
        # V2 col 0: B0=1, B3=1. So local0->B11, local3->B22.
        # W2 col 0: C0=1, C3=1. So local0->C11, local3->C22.
        # Wait, W2 col 0 in my base definition was W[0,0]=1, W[3,0]=1.
        # This corresponds to C11 and C22. Correct.
        
        # We need to define the mapping for each of the 7 columns of the base decomposition.
        # The base decomposition U2, V2, W2 has 7 columns.
        # Col 0: M1
        # Col 1: M2
        # Col 2: M3
        # Col 3: M4
        # Col 4: M5
        # Col 5: M6
        # Col 6: M7
        
        # We will call add_222 for each of the 7 columns with the appropriate mappings.
        
        # M1
        add_222(U, V, W, 0, {0:0, 3:3}, {0:0, 3:3}, {0:0, 3:3})
        # M2
        add_222(U, V, W, 7, {2:2, 3:3}, {0:0}, {2:2})
        # M3
        add_222(U, V, W, 14, {0:0}, {1:1, 3:3}, {1:1})
        # M4
        add_222(U, V, W, 21, {3:3}, {2:2, 0:0}, {0:0, 2:2})
        # M5
        add_222(U, V, W, 28, {0:0, 1:1}, {3:3}, {0:0, 1:1})
        # M6
        add_222(U, V, W, 35, {2:2, 0:0}, {0:0, 1:1}, {3:3})
        # M7
        add_222(U, V, W, 42, {1:1, 3:3}, {2:2, 3:3}, {0:0, 3:3})

        return U, V, W

    def build_schoolbook_441():
        # 4x4x1, rank 16
        U = np.zeros((16, 16))
        V = np.zeros((4, 16)) # m*p = 4*1 = 4
        W = np.zeros((4, 16)) # p*n = 1*4 = 4
        
        for i in range(4):
            for j in range(4):
                col = i * 4 + j
                U[i * 4 + j, col] = 1
                V[j, col] = 1
                W[i, col] = 1
                
        return U, V, W

    # Build components
    U1, V1, W1 = build_strassen_444()
    U2, V2, W2 = build_strassen_444()
    U3, V3, W3 = build_schoolbook_441()
    
    # Combine
    # U: 16 x 114
    U_total = np.hstack([U1, U2, U3])
    
    # V: 36 x 114
    # V1 is for B cols 0-3. V1 shape 16x49.
    # We need to place V1 into V_total[:, 0:49] with correct row indices.
    # V_total rows are j*4+k.
    # V1 rows are j*4+k for k in 0..3.
    # So V1 rows 0..15 map to V_total rows 0..15.
    
    V_total = np.zeros((36, 114))
    V_total[0:16, 0:49] = V1
    
    # V2 is for B cols 4-7.
    # V2 rows 0..15 correspond to j*4+k for k in 0..3.
    # We need to shift k by 4.
    # So row r in V2 (j*4+k) should go to row j*4+(k+4) in V_total.
    # r = j*4+k. Target row = j*4+k+4 = r+4.
    # So V2 rows 0..15 map to V_total rows 4..19.
    V_total[4:20, 49:98] = V2
    
    # V3 is for B col 8.
    # V3 shape 4x16. Rows j.
    # Target row j*4+8.
    # j=0 -> 8, j=1 -> 12, j=2 -> 16, j=3 -> 20.
    V_total[8:24:4, 98:114] = V3
    
    # W: 36 x 114
    # W1 is for C cols 0-3.
    # W1 rows are k*4+i for k in 0..3.
    # W_total rows are k*4+i.
    # So W1 rows 0..15 map to W_total rows 0..15.
    W_total = np.zeros((36, 114))
    W_total[0:16, 0:49] = W1
    
    # W2 is for C cols 4-7.
    # W2 rows 0..15 correspond to k*4+i for k in 0..3.
    # Target row k*4+i + 16 (since k shifts by 4, k*4 shifts by 16).
    # r = k*4+i. Target row = r + 16.
    W_total[16:32, 49:98] = W2
    
    # W3 is for C col 8.
    # W3 shape 4x16. Rows i.
    # Target row 8*4+i = 32+i.
    W_total[32:36, 98:114] = W3
    
    return {"U": U_total.tolist(), "V": V_total.tolist(), "W": W_total.tolist()}

