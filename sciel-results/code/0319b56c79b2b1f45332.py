"""Rewritten program for 4x4x7 matrix multiplication tensor decomposition."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    import numpy as np
    import time
    
    _t0 = time.time()
    _DEADLINE = 145.0
    
    n, m, p = 4, 4, 7
    
    # --- Helper: 2x2x2 Strassen (Rank 7) ---
    def get_2x2x2_strassen():
        # U: (4, 7), V: (4, 7), W: (4, 7)
        # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # Strassen terms
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # Coefficients for A (U)
        # A00: 0, A01: 1, A10: 2, A11: 3
        U[:, 0] = [1, 0, 0, 1] # M1
        U[:, 1] = [0, 0, 1, 1] # M2
        U[:, 2] = [1, 0, 0, 0] # M3
        U[:, 3] = [0, 0, 0, 1] # M4
        U[:, 4] = [1, 1, 0, 0] # M5
        U[:, 5] = [-1, 0, 1, 0] # M6
        U[:, 6] = [0, 1, 0, -1] # M7
        
        # Coefficients for B (V)
        # B00: 0, B01: 1, B10: 2, B11: 3
        V[:, 0] = [1, 0, 0, 1] # M1
        V[:, 1] = [1, 0, 0, 0] # M2
        V[:, 2] = [-1, 0, 1, 0] # M3 (B10-B11 -> -B11+B10? No, B10 is index 2, B11 is index 3. B10-B11 = [0,0,1,-1])
        # Wait, M3 = A00 * (B10 - B11).
        # B10 is row 1, col 0. Index 1*2+0 = 2.
        # B11 is row 1, col 1. Index 1*2+1 = 3.
        # So B10 - B11 is [0, 0, 1, -1].
        V[:, 2] = [0, 0, 1, -1]
        V[:, 3] = [1, -1, 0, 0] # M4: B01-B00. B01 idx 1, B00 idx 0. [ -1, 1, 0, 0 ]? No, B01-B00 is [ -1, 1, 0, 0 ] if order is 0,1,2,3.
        # Let's re-verify indices.
        # B = [[b00, b01], [b10, b11]]
        # Flat: b00(0), b01(1), b10(2), b11(3)
        # M4 = A11 * (B01 - B00). B01 is 1, B00 is 0. So [ -1, 1, 0, 0 ].
        V[:, 3] = [-1, 1, 0, 0]
        V[:, 4] = [0, 0, 0, 1] # M5: B11
        V[:, 5] = [1, 1, 0, 0] # M6: B00+B01
        V[:, 6] = [0, 0, 1, 1] # M7: B10+B11
        
        # Coefficients for C (W)
        # C00: 0, C01: 1, C10: 2, C11: 3
        # C00 = M1+M4-M5+M7
        # C01 = M3+M5
        # C10 = M2+M4
        # C11 = M1-M2+M3+M6
        
        W[0, :] = [1, 0, 0, 1, -1, 0, 1] # C00
        W[1, :] = [0, 0, 1, 0, 1, 0, 0]  # C01
        W[2, :] = [0, 1, 0, 1, 0, 0, 0]  # C10
        W[3, :] = [1, -1, 1, 0, 0, 1, 0] # C11
        
        return U, V, W

    # --- Helper: 2x2x3 Rank 11 ---
    def get_2x2x3_rank11():
        # U: (4, 11), V: (6, 11), W: (6, 11)
        # A: 2x2 (4 entries)
        # B: 2x3 (6 entries). B = [[b00, b01, b02], [b10, b11, b12]]
        # C: 2x3 (6 entries). C = [[c00, c01, c02], [c10, c11, c12]]
        
        # Part 1: 2x2x2 Strassen for columns 0,1
        # B_left = [[b00, b01], [b10, b11]]
        # C_left = [[c00, c01], [c10, c11]]
        U_s, V_s, W_s = get_2x2x2_strassen()
        
        U = np.zeros((4, 11))
        V = np.zeros((6, 11))
        W = np.zeros((6, 11))
        
        # Copy Strassen part
        U[:, 0:7] = U_s
        # Map V_s (4x7) to V (6x7)
        # V_s indices: 0:b00, 1:b01, 2:b10, 3:b11
        # V indices: 0:b00, 1:b01, 2:b02, 3:b10, 4:b11, 5:b12
        # So V_s[0] -> V[0], V_s[1] -> V[1], V_s[2] -> V[3], V_s[3] -> V[4]
        V[0, 0:7] = V_s[0, :]
        V[1, 0:7] = V_s[1, :]
        V[3, 0:7] = V_s[2, :]
        V[4, 0:7] = V_s[3, :]
        
        # Map W_s (4x7) to W (6x7)
        # W_s indices: 0:c00, 1:c01, 2:c10, 3:c11
        # W indices: 0:c00, 1:c01, 2:c02, 3:c10, 4:c11, 5:c12
        W[0, 0:7] = W_s[0, :]
        W[1, 0:7] = W_s[1, :]
        W[3, 0:7] = W_s[2, :]
        W[4, 0:7] = W_s[3, :]
        
        # Part 2: 2x2x1 Schoolbook for column 2
        # B_right = [[b02], [b12]]
        # C_right = [[c02], [c12]]
        # Terms:
        # T8: A00 * B02
        # T9: A01 * B12
        # T10: A10 * B02
        # T11: A11 * B12
        
        # T8
        U[0, 7] = 1 # A00
        U[1, 7] = 0
        U[2, 7] = 0
        U[3, 7] = 0
        V[2, 7] = 1 # B02
        W[2, 7] = 1 # C02
        
        # T9
        U[0, 8] = 0
        U[1, 8] = 1 # A01
        U[2, 8] = 0
        U[3, 8] = 0
        V[5, 8] = 1 # B12
        W[2, 8] = 1 # C02
        
        # T10
        U[0, 9] = 0
        U[1, 9] = 0
        U[2, 9] = 1 # A10
        U[3, 9] = 0
        V[2, 9] = 1 # B02
        W[5, 9] = 1 # C12
        
        # T11
        U[0, 10] = 0
        U[1, 10] = 0
        U[2, 10] = 0
        U[3, 10] = 1 # A11
        V[5, 10] = 1 # B12
        W[5, 10] = 1 # C12
        
        return U, V, W

    # --- Helper: 4x4x4 Rank 49 ---
    def get_4x4x4_rank49():
        # U: (16, 49), V: (16, 49), W: (16, 49)
        # A: 4x4, B: 4x4, C: 4x4
        
        U = np.zeros((16, 49))
        V = np.zeros((16, 49))
        W = np.zeros((16, 49))
        
        # Strassen for 4x4x4
        # 7 main products M1..M7
        # Each Mc is 2x2
        # Each Mc computed by 2x2x2 Strassen (7 sub-terms)
        
        # Define the 7 main Strassen linear forms for A and B
        # A blocks: A00, A01, A10, A11 (each 2x2)
        # B blocks: B00, B01, B10, B11 (each 2x2)
        
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # For each Mc, we need to apply 2x2x2 Strassen.
        # The 2x2x2 Strassen operates on 2x2 matrices.
        # So for each c, we extract the 2x2 matrices A'(c) and B'(c).
        # Then we use the 7 sub-terms of 2x2x2 Strassen.
        
        # We will fill U, V, W for the 49 terms.
        # Term index t = c * 7 + m, where c in 0..6, m in 0..6.
        
        # First, get the 2x2x2 Strassen coefficients
        U_s, V_s, W_s = get_2x2x2_strassen()
        
        # Define A and B block combinations for each c
        # A blocks indices in flat 4x4:
        # A00: 0,1,4,5
        # A01: 2,3,6,7
        # A10: 8,9,12,13
        # A11: 10,11,14,15
        
        # B blocks indices in flat 4x4:
        # B00: 0,1,4,5
        # B01: 2,3,6,7
        # B10: 8,9,12,13
        # B11: 10,11,14,15
        
        # C blocks indices in flat 4x4:
        # C00: 0,1,4,5
        # C01: 2,3,6,7
        # C10: 8,9,12,13
        # C11: 10,11,14,15
        
        # For each c, we define the coefficients for A and B blocks
        # coeff_A[c, block, entry]
        # coeff_B[c, block, entry]
        
        # We'll just iterate through c and m.
        
        for c in range(7):
            # Determine A'(c) and B'(c)
            # A'(c) is a 2x2 matrix.
            # B'(c) is a 2x2 matrix.
            
            # Initialize local 2x2 matrices for A and B
            A_local = np.zeros((2, 2))
            B_local = np.zeros((2, 2))
            
            # Map block indices to local 2x2 indices
            # Block 0 -> A_local[0,0] (2x2)
            # Block 1 -> A_local[0,1]
            # Block 2 -> A_local[1,0]
            # Block 3 -> A_local[1,1]
            
            # We need to know which blocks contribute to A'(c) and B'(c)
            # and with what coefficients.
            
            # Hardcode the block coefficients for Strassen 4x4
            # c=0: M1 = (A00+A11)(B00+B11)
            # A blocks: 0:1, 3:1
            # B blocks: 0:1, 3:1
            # c=1: M2 = (A10+A11)B00
            # A blocks: 2:1, 3:1
            # B blocks: 0:1
            # c=2: M3 = A00(B10-B11)
            # A blocks: 0:1
            # B blocks: 2:1, 3:-1
            # c=3: M4 = A11(B01-B00)
            # A blocks: 3:1
            # B blocks: 1:1, 0:-1
            # c=4: M5 = (A00+A01)B11
            # A blocks: 0:1, 1:1
            # B blocks: 3:1
            # c=5: M6 = (A10-A00)(B00+B01)
            # A blocks: 2:1, 0:-1
            # B blocks: 0:1, 1:1
            # c=6: M7 = (A01-A11)(B10+B11)
            # A blocks: 1:1, 3:-1
            # B blocks: 2:1, 3:1
            
            A_block_coeffs = {
                0: {0: 1, 3: 1},
                1: {2: 1, 3: 1},
                2: {0: 1},
                3: {3: 1},
                4: {0: 1, 1: 1},
                5: {2: 1, 0: -1},
                6: {1: 1, 3: -1}
            }
            B_block_coeffs = {
                0: {0: 1, 3: 1},
                1: {0: 1},
                2: {2: 1, 3: -1},
                3: {1: 1, 0: -1},
                4: {3: 1},
                5: {0: 1, 1: 1},
                6: {2: 1, 3: 1}
            }
            
            # Construct A_local and B_local as linear combinations of A and B entries
            # A_local[i, j] = sum_{block} coeff * A_block[i, j]
            
            # We will add the 7 sub-terms for this c
            for m in range(7):
                t = c * 7 + m
                
                # U, V, W for this term t
                # U is 16x1, V is 16x1, W is 16x1
                
                # The 2x2x2 Strassen term m uses:
                # U_s[:, m], V_s[:, m], W_s[:, m]
                # U_s is 4x1 (coeffs for A_local 2x2)
                # V_s is 4x1 (coeffs for B_local 2x2)
                # W_s is 4x1 (coeffs for C_local 2x2)
                
                # We need to map A_local entries to A entries.
                # A_local[0,0] corresponds to A_block 0 (A00)
                # A_local[0,1] corresponds to A_block 1 (A01)
                # A_local[1,0] corresponds to A_block 2 (A10)
                # A_local[1,1] corresponds to A_block 3 (A11)
                
                # So U_s[0] is coeff for A00, U_s[1] for A01, etc?
                # No, U_s is for 2x2 matrix.
                # U_s[0] -> A_local[0,0]
                # U_s[1] -> A_local[0,1]
                # U_s[2] -> A_local[1,0]
                # U_s[3] -> A_local[1,1]
                
                # A_local[0,0] = sum_{b} A_block_coeffs[c][b] * A00
                # So the coefficient for A00 in U is U_s[0] * A_block_coeffs[c].get(0, 0)
                
                # Let's build U[:, t]
                for i in range(4):
                    for j in range(4):
                        # A[i, j]
                        # Which block?
                        # i<2, j<2 -> Block 0
                        # i<2, j>=2 -> Block 1
                        # i>=2, j<2 -> Block 2
                        # i>=2, j>=2 -> Block 3
                        block = (0 if i < 2 else 2) + (0 if j < 2 else 1)
                        
                        # Local index in 2x2
                        li = i % 2
                        lj = j % 2
                        local_idx = li * 2 + lj
                        
                        # Coefficient from U_s
                        u_coeff = U_s[local_idx, m]
                        
                        # Coefficient from block combination
                        b_coeff = A_block_coeffs[c].get(block, 0)
                        
                        U[i * 4 + j, t] = u_coeff * b_coeff
                        
                # Build V[:, t]
                for i in range(4):
                    for j in range(4):
                        # B[i, j]
                        block = (0 if i < 2 else 2) + (0 if j < 2 else 1)
                        li = i % 2
                        lj = j % 2
                        local_idx = li * 2 + lj
                        
                        v_coeff = V_s[local_idx, m]
                        b_coeff = B_block_coeffs[c].get(block, 0)
                        
                        V[i * 4 + j, t] = v_coeff * b_coeff
                        
                # Build W[:, t]
                # C_local is 2x2.
                # C_local[0,0] -> C00
                # C_local[0,1] -> C01
                # C_local[1,0] -> C10
                # C_local[1,1] -> C11
                # W_s[0] -> C00, W_s[1] -> C01, W_s[2] -> C10, W_s[3] -> C11
                
                # C[i, j]
                for i in range(4):
                    for j in range(4):
                        block = (0 if i < 2 else 2) + (0 if j < 2 else 1)
                        li = i % 2
                        lj = j % 2
                        local_idx = li * 2 + lj
                        
                        w_coeff = W_s[local_idx, m]
                        
                        # No block combination for C, it's direct
                        W[i * 4 + j, t] = w_coeff
                        
        return U, V, W

    # --- Helper: 4x4x3 Rank 44 ---
    def get_4x4x3_rank44():
        # U: (16, 44), V: (12, 44), W: (12, 44)
        # A: 4x4, B: 4x3, C: 4x3
        
        U = np.zeros((16, 44))
        V = np.zeros((12, 44))
        W = np.zeros((12, 44))
        
        # Split A into 2x2 blocks
        # Split B into 2x3 blocks (top 2 rows, bottom 2 rows)
        # C = A_top B_top + A_bot B_bot? No.
        # C = A B.
        # A = [[A00, A01], [A10, A11]]
        # B = [[B_top], [B_bot]] where B_top is 2x3, B_bot is 2x3
        # C = [[A00 B_top + A01 B_bot], [A10 B_top + A11 B_bot]]
        
        # 4 products of 2x2x3
        # 1. A00 B_top
        # 2. A01 B_bot
        # 3. A10 B_top
        # 4. A11 B_bot
        
        # Each 2x2x3 has rank 11.
        # Total 44.
        
        U_223, V_223, W_223 = get_2x2x3_rank11()
        
        # Map to full indices
        # A00: rows 0,1 cols 0,1 -> A[0,0..3]
        # A01: rows 0,1 cols 2,3 -> A[0,4..7]
        # A10: rows 2,3 cols 0,1 -> A[8,0..3]
        # A11: rows 2,3 cols 2,3 -> A[8,4..7]
        
        # B_top: rows 0,1 cols 0,1,2 -> B[0,0..5]
        # B_bot: rows 2,3 cols 0,1,2 -> B[6,0..5]
        
        # C_top: rows 0,1 cols 0,1,2 -> C[0,0..5]
        # C_bot: rows 2,3 cols 0,1,2 -> C[6,0..5]
        
        # Product 1: A00 B_top
        # U: A00 entries
        # V: B_top entries
        # W: C_top entries
        
        # U[:, 0:11] = U_223
        # Map A00 indices:
        # A00[0,0] -> A[0,0] (idx 0)
        # A00[0,1] -> A[0,1] (idx 1)
        # A00[1,0] -> A[0,2] (idx 2)
        # A00[1,1] -> A[0,3] (idx 3)
        
        # Wait, A00 is 2x2.
        # A00[0,0] is A[0,0].
        # A00[0,1] is A[0,1].
        # A00[1,0] is A[1,0] -> idx 4? No, A is 4x4.
        # A[0,0]=0, A[0,1]=1, A[0,2]=2, A[0,3]=3
        # A[1,0]=4, A[1,1]=5, A[1,2]=6, A[1,3]=7
        # A[2,0]=8 ...
        
        # A00 uses A[0,0], A[0,1], A[1,0], A[1,1] -> idx 0,1,4,5
        # A01 uses A[0,2], A[0,3], A[1,2], A[1,3] -> idx 2,3,6,7
        # A10 uses A[2,0], A[2,1], A[3,0], A[3,1] -> idx 8,9,12,13
        # A11 uses A[2,2], A[2,3], A[3,2], A[3,3] -> idx 10,11,14,15
        
        # B_top uses B[0,0], B[0,1], B[0,2], B[1,0], B[1,1], B[1,2] -> idx 0..5
        # B_bot uses B[2,0], B[2,1], B[2,2], B[3,0], B[3,1], B[3,2] -> idx 6..11
        
        # C_top uses C[0,0]..C[1,2] -> idx 0..5
        # C_bot uses C[2,0]..C[3,2] -> idx 6..11
        
        # We will fill U, V, W for the 4 products.
        
        # Product 1: A00 B_top
        # U[0:16, 0:11]
        # Map U_223 (4x11) to U (16x11)
        # U_223[0] -> A00[0,0] -> A[0,0] (idx 0)
        # U_223[1] -> A00[0,1] -> A[0,1] (idx 1)
        # U_223[2] -> A00[1,0] -> A[1,0] (idx 4)
        # U_223[3] -> A00[1,1] -> A[1,1] (idx 5)
        
        U[0, 0:11] = U_223[0, :]
        U[1, 0:11] = U_223[1, :]
        U[4, 0:11] = U_223[2, :]
        U[5, 0:11] = U_223[3, :]
        
        # V_223 (6x11) to V (12x11)
        # V_223[0..5] -> B_top[0..5] -> B[0..5]
        V[0:6, 0:11] = V_223
        
        # W_223 (6x11) to W (12x11)
        # W_223[0..5] -> C_top[0..5] -> C[0..5]
        W[0:6, 0:11] = W_223
        
        # Product 2: A01 B_bot
        # U: A01 entries (idx 2,3,6,7)
        U[2, 11:22] = U_223[0, :]
        U[3, 11:22] = U_223[1, :]
        U[6, 11:22] = U_223[2, :]
        U[7, 11:22] = U_223[3, :]
        
        # V: B_bot entries (idx 6..11)
        V[6:12, 11:22] = V_223
        
        # W: C_top entries (idx 0..5)
        W[0:6, 11:22] = W_223
        
        # Product 3: A10 B_top
        # U: A10 entries (idx 8,9,12,13)
        U[8, 22:33] = U_223[0, :]
        U[9, 22:33] = U_223[1, :]
        U[12, 22:33] = U_223[2, :]
        U[13, 22:33] = U_223[3, :]
        
        # V: B_top entries (idx 0..5)
        V[0:6, 22:33] = V_223
        
        # W: C_bot entries (idx 6..11)
        W[6:12, 22:33] = W_223
        
        # Product 4: A11 B_bot
        # U: A11 entries (idx 10,11,14,15)
        U[10, 33:44] = U_223[0, :]
        U[11, 33:44] = U_223[1, :]
        U[14, 33:44] = U_223[2, :]
        U[15, 33:44] = U_223[3, :]
        
        # V: B_bot entries (idx 6..11)
        V[6:12, 33:44] = V_223
        
        # W: C_bot entries (idx 6..11)
        W[6:12, 33:44] = W_223
        
        return U, V, W

    # --- Combine ---
    U1, V1, W1 = get_4x4x4_rank49()
    U2, V2, W2 = get_4x4x3_rank44()
    
    # U1: (16, 49), V1: (16, 49), W1: (16, 49)
    # U2: (16, 44), V2: (12, 44), W2: (12, 44)
    
    # Full V is (28, R). V1 is for B[0:4, 0:4]. V2 is for B[0:4, 4:7].
    # B is 4x7.
    # V1 indices: j*4+k.
    # V2 indices: j*3+k.
    # Full V indices: j*7+k.
    
    # Map V1 (16x49) to V_full (28x49)
    V_full = np.zeros((28, 93))
    for j in range(4):
        for k in range(4):
            V_full[j * 7 + k, 0:49] = V1[j * 4 + k, :]
            
    # Map V2 (12x44) to V_full (28x44)
    for j in range(4):
        for k in range(3):
            V_full[j * 7 + (4 + k), 49:93] = V2[j * 3 + k, :]
            
    # Map W1 (16x49) to W_full (28x49)
    W_full = np.zeros((28, 93))
    for k in range(4):
        for i in range(4):
            W_full[k * 4 + i, 0:49] = W1[k * 4 + i, :]
            
    # Map W2 (12x44) to W_full (28x44)
    for k in range(3):
        for i in range(4):
            W_full[(4 + k) * 4 + i, 49:93] = W2[k * 4 + i, :]
            
    U_full = np.hstack([U1, U2])
    
    if time.time() - _t0 > _DEADLINE:
        return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
        
    return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
# EVOLVE-BLOCK-END
