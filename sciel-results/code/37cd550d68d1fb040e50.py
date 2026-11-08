import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=4, m=5, p=5
    # We decompose the 4x5x5 tensor into:
    # 1. 4x4x4 core (Strassen, rank 49)
    # 2. 4x4x1 cross (A[:, 0:4] * B[0:4, 4] -> C[4, :]) (Rank 16)
    # 3. 4x1x4 cross (A[:, 4] * B[4, 0:4] -> C[0:4, :]) (Rank 16)
    # 4. 4x1x1 cross (A[:, 4] * B[4, 4] -> C[4, :]) (Rank 4)
    # Total Rank = 49 + 16 + 16 + 4 = 85
    
    rank = 85
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # --- Part 1: 4x4x4 Core (Strassen) ---
    # Strassen factors for 2x2
    # S_U: 4x7, S_V: 4x7, S_W: 4x7
    # We need to lift this to 4x4.
    # The 4x4 multiplication can be seen as 4 blocks of 2x2.
    # A = [[A00, A01], [A10, A11]] where Aij are 2x2.
    # B = [[B00, B01], [B10, B11]]
    # C = A B.
    #
    # The Strassen algorithm computes 7 products M1..M7.
    # Each product is a 2x2 matrix multiplication of the block components.
    # So we need 7 * (rank of 2x2 multiplication) terms?
    # No, the rank of 2x2 multiplication is 7 (Strassen).
    # So R(4,4,4) <= 7 * R(2,2,2) = 7 * 7 = 49.
    #
    # Let's construct the 49 terms.
    # The 7 Strassen terms for 2x2 are:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M6 + M7
    #
    # Each M is a 2x2 matrix product.
    # M = X * Y, where X, Y are 2x2.
    # We can use the standard Strassen decomposition for each M?
    # No, M is just a 2x2 multiplication. We can use the rank-7 Strassen decomposition for each of the 7 products.
    # So we have 7 outer terms, each decomposed into 7 inner terms. Total 49.
    
    # Indices for 4x4:
    # i, j, k in 0..3.
    # i = 2*i1 + i2, j = 2*j1 + j2, k = 2*k1 + k2.
    # i1, j1, k1 in {0,1} (block index)
    # i2, j2, k2 in {0,1} (element index within block)
    
    # Strassen 2x2 factors (for element indices i2, j2, k2)
    # U_2x2: 4x7, V_2x2: 4x7, W_2x2: 4x7
    # Note: The standard Strassen factors are for C = A B.
    # U rows: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # V rows: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # W rows: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    # Let's define the 2x2 Strassen factors explicitly.
    # M1: (A00+A11)(B00+B11)
    # M2: (A10+A11)B00
    # M3: A00(B01-B11)
    # M4: A11(B10-B00)
    # M5: (A00+A01)B11
    # M6: (A10-A00)(B00+B01)
    # M7: (A01-A11)(B10+B11)
    
    # We will iterate over the 7 block-level Strassen terms (t in 0..6)
    # and the 7 element-level Strassen terms (s in 0..6).
    
    # Block-level Strassen coefficients for A, B, C blocks
    # A_blocks: A00, A01, A10, A11
    # B_blocks: B00, B01, B10, B11
    # C_blocks: C00, C01, C10, C11
    
    # Coefficients for M1..M7 in terms of A_blocks
    # M1: A00+A11 -> A00:1, A11:1
    # M2: A10+A11 -> A10:1, A11:1
    # M3: A00 -> A00:1
    # M4: A11 -> A11:1
    # M5: A00+A01 -> A00:1, A01:1
    # M6: A10-A00 -> A10:1, A00:-1
    # M7: A01-A11 -> A01:1, A11:-1
    
    A_block_coeffs = [
        {0: 1, 3: 1},   # M1
        {2: 1, 3: 1},   # M2
        {0: 1},         # M3
        {3: 1},         # M4
        {0: 1, 1: 1},   # M5
        {2: 1, 0: -1},  # M6
        {1: 1, 3: -1}   # M7
    ]
    
    # Coefficients for M1..M7 in terms of B_blocks
    # M1: B00+B11 -> B00:1, B11:1
    # M2: B00 -> B00:1
    # M3: B01-B11 -> B01:1, B11:-1
    # M4: B10-B00 -> B10:1, B00:-1
    # M5: B11 -> B11:1
    # M6: B00+B01 -> B00:1, B01:1
    # M7: B10+B11 -> B10:1, B11:1
    
    B_block_coeffs = [
        {0: 1, 3: 1},   # M1
        {0: 1},         # M2
        {1: 1, 3: -1},  # M3
        {2: 1, 0: -1},  # M4
        {3: 1},         # M5
        {0: 1, 1: 1},   # M6
        {2: 1, 3: 1}    # M7
    ]
    
    # Coefficients for C_blocks in terms of M1..M7
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M6 + M7
    
    C_block_coeffs = [
        {0: 1, 3: -1, 4: 1, 6: -1},  # C00
        {2: 1, 4: 1},                 # C01
        {1: 1, 3: 1},                 # C10
        {0: 1, 1: -1, 5: 1, 6: 1}     # C11
    ]
    
    # Element-level Strassen factors (2x2)
    # We need U_2x2, V_2x2, W_2x2 of size 4x7.
    # The indices 0,1,2,3 correspond to (0,0), (0,1), (1,0), (1,1).
    
    # Let's build the 2x2 Strassen factors.
    # M1: (A00+A11)(B00+B11)
    # A factors for M1: A00(1), A11(1). A01(0), A10(0).
    # U_2x2[:, 0] = [1, 0, 0, 1]
    # B factors for M1: B00(1), B11(1).
    # V_2x2[:, 0] = [1, 0, 0, 1]
    # C factors for M1: C00(1), C11(1).
    # W_2x2[:, 0] = [1, 0, 0, 1]
    
    # M2: (A10+A11)B00
    # A: A10(1), A11(1).
    # U_2x2[:, 1] = [0, 0, 1, 1]
    # B: B00(1).
    # V_2x2[:, 1] = [1, 0, 0, 0]
    # C: C10(1), C11(-1).
    # W_2x2[:, 1] = [0, 0, 1, -1]
    
    # M3: A00(B01-B11)
    # A: A00(1).
    # U_2x2[:, 2] = [1, 0, 0, 0]
    # B: B01(1), B11(-1).
    # V_2x2[:, 2] = [0, 1, 0, -1]
    # C: C01(1).
    # W_2x2[:, 2] = [0, 1, 0, 0]
    
    # M4: A11(B10-B00)
    # A: A11(1).
    # U_2x2[:, 3] = [0, 0, 0, 1]
    # B: B10(1), B00(-1).
    # V_2x2[:, 3] = [-1, 0, 1, 0]
    # C: C00(-1), C10(1).
    # W_2x2[:, 3] = [-1, 0, 1, 0]
    
    # M5: (A00+A01)B11
    # A: A00(1), A01(1).
    # U_2x2[:, 4] = [1, 1, 0, 0]
    # B: B11(1).
    # V_2x2[:, 4] = [0, 0, 0, 1]
    # C: C00(1), C01(1).
    # W_2x2[:, 4] = [1, 1, 0, 0]
    
    # M6: (A10-A00)(B00+B01)
    # A: A10(1), A00(-1).
    # U_2x2[:, 5] = [-1, 0, 1, 0]
    # B: B00(1), B01(1).
    # V_2x2[:, 5] = [1, 1, 0, 0]
    # C: C11(1).
    # W_2x2[:, 5] = [0, 0, 0, 1]
    
    # M7: (A01-A11)(B10+B11)
    # A: A01(1), A11(-1).
    # U_2x2[:, 6] = [0, 1, 0, -1]
    # B: B10(1), B11(1).
    # V_2x2[:, 6] = [0, 0, 1, 1]
    # C: C00(-1), C11(1).
    # W_2x2[:, 6] = [-1, 0, 0, 1]
    
    U_2x2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=int)
    
    V_2x2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=int)
    
    W_2x2 = np.array([
        [1, 0, 0, -1, 1, 0, -1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 0, 0, 0, 1, 1]
    ], dtype=int)
    
    # Now populate U, V, W for the 4x4x4 core.
    # Total 49 terms.
    # Term index t in 0..48.
    # t = 7 * t_block + t_elem.
    # t_block in 0..6 (Strassen block term)
    # t_elem in 0..6 (Strassen element term)
    
    for t_block in range(7):
        for t_elem in range(7):
            col = t_block * 7 + t_elem
            
            # Get coefficients for A_blocks, B_blocks, C_blocks
            a_coeffs = A_block_coeffs[t_block]
            b_coeffs = B_block_coeffs[t_block]
            c_coeffs = C_block_coeffs[t_block]
            
            # Get element-level factors
            u_elem = U_2x2[:, t_elem]
            v_elem = V_2x2[:, t_elem]
            w_elem = W_2x2[:, t_elem]
            
            # Loop over block indices
            # A_block index ab in 0..3 (0:00, 1:01, 2:10, 3:11)
            # ab = 2*ab1 + ab2
            # B_block index bb in 0..3
            # C_block index cb in 0..3
            
            for ab in range(4):
                if ab not in a_coeffs: continue
                ca = a_coeffs[ab]
                ab1, ab2 = divmod(ab, 2)
                
                for bb in range(4):
                    if bb not in b_coeffs: continue
                    cb_val = b_coeffs[bb]
                    bb1, bb2 = divmod(bb, 2)
                    
                    # The product is (ca * A_block[ab]) * (cb_val * B_block[bb])
                    # This contributes to C_block[cb] via the c_coeffs.
                    # Wait, the block multiplication is:
                    # C_block[cb] += sum_{ab, bb} (A_block[ab] * B_block[bb]) * (coeff from Strassen)
                    # Actually, the Strassen formula is:
                    # C_block[cb] = sum_{t_block} (sum_{ab, bb} A_block[ab] B_block[bb] * ... )
                    # No, the Strassen algorithm computes M_t = X_t Y_t.
                    # X_t = sum_{ab} a_coeff_t[ab] A_block[ab]
                    # Y_t = sum_{bb} b_coeff_t[bb] B_block[bb]
                    # Then C_block[cb] = sum_{t_block} c_coeff_t[cb] M_t.
                    
                    # So for a fixed t_block, we form X_t and Y_t.
                    # Then we multiply X_t and Y_t using the 2x2 Strassen decomposition?
                    # No, X_t and Y_t are 2x2 matrices.
                    # The product M_t = X_t Y_t is a 2x2 matrix.
                    # We decompose the multiplication of two 2x2 matrices X_t and Y_t into 7 rank-1 terms.
                    # So for each t_block, we have 7 terms (t_elem).
                    # The factor for A is X_t, which is a linear combination of A_blocks.
                    # The factor for B is Y_t, which is a linear combination of B_blocks.
                    # The factor for C is the contribution to C_block[cb].
                    
                    # So, for a specific t_block and t_elem:
                    # The "A-factor" for a specific element (i2, j2) in A is:
                    # Sum over ab where A_block[ab] contains element (i2, j2)?
                    # No.
                    # A_block[ab] is a 2x2 matrix.
                    # Element (i, j) of A is in A_block[ab] where ab = 2*div(i,2) + div(j,2).
                    # Let's map global indices i, j (0..3) to block indices.
                    # i = 2*i1 + i2
                    # j = 2*j1 + j2
                    # A_block index = 2*i1 + j1.
                    # The element within the block is (i2, j2).
                    
                    # So, for a fixed t_block, t_elem:
                    # We need to fill U[i*m + j, col], V[j*p + k, col], W[k*n + i, col].
                    
                    # Let's iterate over all i, j, k in 0..3.
                    # For U:
                    # i = 2*i1 + i2
                    # j = 2*j1 + j2
                    # A_block index ab = 2*i1 + j1
                    # Element index in 2x2: (i2, j2)
                    # U_2x2 value for element (i2, j2) and term t_elem: U_2x2[i2*2+j2, t_elem]
                    # Coefficient from block: a_coeffs[ab] if ab in a_coeffs else 0.
                    
                    # Similarly for V:
                    # j = 2*j1 + j2
                    # k = 2*k1 + k2
                    # B_block index bb = 2*j1 + k1
                    # Element index: (j2, k2)
                    # V_2x2 value: V_2x2[j2*2+k2, t_elem]
                    # Coefficient: b_coeffs[bb]
                    
                    # Similarly for W:
                    # k = 2*k1 + k2
                    # i = 2*i1 + i2
                    # C_block index cb = 2*k1 + i1
                    # Element index: (k2, i2)
                    # W_2x2 value: W_2x2[k2*2+i2, t_elem]
                    # Coefficient: c_coeffs[cb]
                    
                    # We can precompute the values for U, V, W for this column.
                    
                    for i in range(4):
                        for j in range(4):
                            i1, i2 = divmod(i, 2)
                            j1, j2 = divmod(j, 2)
                            ab = 2 * i1 + j1
                            coeff_a = a_coeffs.get(ab, 0)
                            if coeff_a == 0:
                                continue
                            val_u = coeff_a * U_2x2[i2 * 2 + j2, t_elem]
                            if val_u != 0:
                                U[i * 5 + j, col] = val_u # m=5, but for core j<4. Wait.
                                # U is size 20x85. Rows are i*5+j.
                                # For core, i in 0..3, j in 0..3.
                                # Row index is i*5 + j.
                                # This is correct.
                    
                    for j in range(4):
                        for k in range(4):
                            j1, j2 = divmod(j, 2)
                            k1, k2 = divmod(k, 2)
                            bb = 2 * j1 + k1
                            coeff_b = b_coeffs.get(bb, 0)
                            if coeff_b == 0:
                                continue
                            val_v = coeff_b * V_2x2[j2 * 2 + k2, t_elem]
                            if val_v != 0:
                                V[j * 5 + k, col] = val_v # m=5, p=5. Row j*5+k.
                    
                    for k in range(4):
                        for i in range(4):
                            k1, k2 = divmod(k, 2)
                            i1, i2 = divmod(i, 2)
                            cb = 2 * k1 + i1
                            coeff_c = c_coeffs.get(cb, 0)
                            if coeff_c == 0:
                                continue
                            val_w = coeff_c * W_2x2[k2 * 2 + i2, t_elem]
                            if val_w != 0:
                                W[k * 4 + i, col] = val_w # p=5, n=4. Row k*4+i.
    
    # --- Part 2: 4x4x1 Cross ---
    # A[:, 0:4] * B[0:4, 4] -> C[4, :]
    # C[4, i] = sum_{j=0}^3 A[i, j] B[j, 4]
    # Naive rank 16.
    # Columns 49..64.
    col_offset = 49
    for i in range(4):
        for j in range(4):
            col = col_offset + i * 4 + j
            U[i * 5 + j, col] = 1
            V[j * 5 + 4, col] = 1
            W[4 * 4 + i, col] = 1 # k=4, n=4. Row 4*4+i = 16+i.
    
    # --- Part 3: 4x1x4 Cross ---
    # A[:, 4] * B[4, 0:4] -> C[0:4, :]
    # C[k, i] = A[i, 4] B[4, k]
    # Naive rank 16.
    # Columns 65..80.
    col_offset = 65
    for i in range(4):
        for k in range(4):
            col = col_offset + i * 4 + k
            U[i * 5 + 4, col] = 1
            V[4 * 5 + k, col] = 1
            W[k * 4 + i, col] = 1
    
    # --- Part 4: 4x1x1 Cross ---
    # A[:, 4] * B[4, 4] -> C[4, :]
    # C[4, i] = A[i, 4] B[4, 4]
    # Naive rank 4.
    # Columns 81..84.
    col_offset = 81
    for i in range(4):
        col = col_offset + i
        U[i * 5 + 4, col] = 1
        V[4 * 5 + 4, col] = 1
        W[4 * 4 + i, col] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
