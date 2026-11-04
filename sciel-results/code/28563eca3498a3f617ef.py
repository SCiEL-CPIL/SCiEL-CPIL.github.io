"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    # Strassen-based decomposition for 4x4x8.
    # We view the 4x4 matrices as 2x2 blocks of 2x2 submatrices.
    # The multiplication (4x4) * (4x8) is equivalent to two independent (4x4) * (4x4) multiplications
    # because the second matrix has 8 columns, which can be split into two groups of 4 columns each.
    #
    # The rank of (4x4) * (4x4) using Strassen recursion is 7^2 = 49.
    # Therefore, the total rank for (4x4) * (4x8) is 49 + 49 = 98.
    #
    # The Strassen algorithm for 2x2 * 2x2 (rank 7) has the following structure:
    # Let A = [[A11, A12], [A21, A22]] and B = [[B11, B12], [B21, B22]].
    # The 7 products are:
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    #
    # The resulting 2x2 matrix C = A*B is:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # For the 4x4 case, we apply this 2x2 Strassen structure where each element is a 2x2 matrix.
    # This effectively means we are doing a Kronecker product of the 2x2 Strassen decomposition with itself.
    #
    # Let's define the 2x2 Strassen factors.
    # U_2x2: 4x7 matrix, V_2x2: 4x7 matrix, W_2x2: 4x7 matrix.
    # Indices for 2x2 matrices:
    # A: 00, 01, 10, 11 (flattened as 0, 1, 2, 3)
    # B: 00, 01, 10, 11 (flattened as 0, 1, 2, 3)
    # C: 00, 01, 10, 11 (flattened as 0, 1, 2, 3)
    
    # We will build the 4x4x4 decomposition first, then duplicate it for the 4x4x8 case.
    
    # Strassen 2x2x2 factors
    # U vectors for the 7 products (input A)
    # M1: A11 + A22 -> indices 0 (00) and 3 (11)
    # M2: A21 + A22 -> indices 2 (10) and 3 (11)
    # M3: A11 -> index 0 (00)
    # M4: A22 -> index 3 (11)
    # M5: A11 + A12 -> indices 0 (00) and 1 (01)
    # M6: A21 - A11 -> indices 2 (10) and -1 at 0 (00)
    # M7: A12 - A22 -> indices 1 (01) and -1 at 3 (11)
    
    U_2 = [
        [1, 0, 1, 0, 1, -1, 0], # A11 (00)
        [0, 0, 0, 0, 1, 0, 1],  # A12 (01)
        [0, 1, 0, 0, 0, 1, 0],  # A21 (10)
        [1, 1, 0, 1, 0, 0, -1]  # A22 (11)
    ]
    
    # V vectors for the 7 products (input B)
    # M1: B11 + B22 -> indices 0 (00) and 3 (11)
    # M2: B11 -> index 0 (00)
    # M3: B12 - B22 -> indices 1 (01) and -1 at 3 (11)
    # M4: B21 - B11 -> indices 2 (10) and -1 at 0 (00)
    # M5: B22 -> index 3 (11)
    # M6: B11 + B12 -> indices 0 (00) and 1 (01)
    # M7: B21 + B22 -> indices 2 (10) and 3 (11)
    
    V_2 = [
        [1, 1, -1, 1, 0, 1, 0], # B11 (00)
        [0, 0, 1, 0, 0, 1, 0],  # B12 (01)
        [1, 0, 0, -1, 0, 0, 1], # B21 (10)
        [1, 0, -1, 0, 1, 0, 1]  # B22 (11)
    ]
    
    # W vectors for the 7 products (output C)
    # C11 = M1 + M4 - M5 + M7 -> indices 1, 4, -1 at 5, 1 at 7
    # C12 = M3 + M5 -> indices 3, 1 at 5
    # C21 = M2 + M4 -> indices 2, 1 at 4
    # C22 = M1 - M2 + M3 + M6 -> indices 1, -1 at 2, 1 at 3, 1 at 6
    
    W_2 = [
        [1, 0, 0, 0, -1, 0, 1],  # C11 (00)
        [0, 0, 1, 0, 1, 0, 0],   # C12 (01)
        [0, 1, 0, 1, 0, 0, 0],   # C21 (10)
        [1, -1, 1, 0, 0, 1, 0]   # C22 (11)
    ]
    
    # Now we construct the 4x4x4 decomposition by substituting the 2x2 Strassen factors.
    # This is equivalent to taking the Kronecker product of the factors, but we need to be careful with indices.
    # Alternatively, we can just construct the 4x4 factors directly by observing the block structure.
    # Let's use the Kronecker product approach for clarity and correctness.
    # U_4 = kron(U_2, U_2) ? No, the indices don't align directly like that for a simple block decomposition.
    # Actually, for a block matrix multiplication where each block is multiplied using Strassen,
    # the total rank is 7*7=49.
    # The U matrix for the 4x4 case will have 16 rows and 49 columns.
    # Each column c in 0..48 corresponds to a pair (c1, c2) where c1 = c // 7, c2 = c % 7.
    # The row index in U_4 corresponds to a block (i_block, j_block) and an element (i_elem, j_elem).
    # Row index = (i_block * 2 + i_elem) * 2 + (j_block * 2 + j_elem) = i_block*4 + i_elem*2 + j_block*2 + j_elem.
    
    # Let's build U_4, V_4, W_4 of size 16x49, 16x49, 16x49.
    # Note: For 4x4x4, U is (4*4)x49, V is (4*4)x49, W is (4*4)x49.
    
    U_4 = [[0] * 49 for _ in range(16)]
    V_4 = [[0] * 49 for _ in range(16)]
    W_4 = [[0] * 49 for _ in range(16)]
    
    for c1 in range(7):
        for c2 in range(7):
            c = c1 * 7 + c2
            for i_b in range(2):
                for i_e in range(2):
                    for j_b in range(2):
                        for j_e in range(2):
                            row_u = (i_b * 2 + i_e) * 2 + (j_b * 2 + j_e)
                            U_4[row_u][c] = U_2[i_b * 2 + i_e][c1] * U_2[j_b * 2 + j_e][c2]
                            
                            # V_4 row index: (j_b * 2 + j_e) * 2 + (k_b * 2 + k_e)
                            # But V_4 is indexed by input B, which is 4x4.
                            # The block structure of B is similar.
                            # For V_4, the row index is (j_b * 2 + j_e) * 2 + (k_b * 2 + k_e)?
                            # Wait, V is (m*p)xrank = 16x49.
                            # The rows of V correspond to elements of B.
                            # B is 4x4. Block (j_b, k_b), element (j_e, k_e).
                            # Row index = (j_b * 2 + j_e) * 2 + (k_b * 2 + k_e).
                            for k_b in range(2):
                                for k_e in range(2):
                                    row_v = (j_b * 2 + j_e) * 2 + (k_b * 2 + k_e)
                                    V_4[row_v][c] = V_2[j_b * 2 + j_e][c1] * V_2[k_b * 2 + k_e][c2]
                                    
                                    # W_4 row index: (k_b * 2 + k_e) * 2 + (i_b * 2 + i_e)
                                    # W is (p*n)xrank = 16x49.
                                    # The rows of W correspond to elements of C.
                                    # C is 4x4. Block (k_b, i_b), element (k_e, i_e).
                                    # Row index = (k_b * 2 + k_e) * 2 + (i_b * 2 + i_e).
                                    row_w = (k_b * 2 + k_e) * 2 + (i_b * 2 + i_e)
                                    W_4[row_w][c] = W_2[k_b * 2 + k_e][c1] * W_2[i_b * 2 + i_e][c2]
    
    # Now we have the 4x4x4 decomposition.
    # We need 4x4x8.
    # This is two 4x4x4 multiplications.
    # The input A is 4x4 (16 elements).
    # The input B is 4x8 (32 elements). We split B into two 4x4 matrices B1 and B2.
    # B1 columns 0-3, B2 columns 4-7.
    # The output C is 4x8 (32 elements). C1 is C[:, 0:4], C2 is C[:, 4:8].
    #
    # U_8 will have 16 rows and 98 columns.
    # For the first 49 columns, U_8[row, c] = U_4[row, c].
    # For the next 49 columns, U_8[row, c+49] = U_4[row, c].
    
    V_8 = [[0] * 98 for _ in range(32)]
    # For the first 49 columns, V_8[row, c] = V_4[row, c] for row in 0..15 (B1).
    # V_8[row, c] = 0 for row in 16..31 (B2).
    # For the next 49 columns, V_8[row+16, c+49] = V_4[row, c] for row in 0..15 (B2).
    # V_8[row, c+49] = 0 for row in 0..15 (B1).
    
    W_8 = [[0] * 98 for _ in range(32)]
    # W is (p*n)xrank = 8*4=32 x 98.
    # Wait, the trace convention is W[k*n+i, r].
    # k is column of C (0..7), i is row of C (0..3).
    # So row index = k*4 + i.
    # For the first 49 columns (corresponding to C1 = C[:, 0:4]):
    # C1 is 4x4. k1 in 0..3, i in 0..3.
    # Row index in W_8 = k1*4 + i.
    # This matches the row index in W_4 if W_4 is defined as [k*n+i, c].
    # Let's check W_4 definition.
    # In my construction above, row_w = (k_b * 2 + k_e) * 2 + (i_b * 2 + i_e).
    # k_b*2+k_e is k (0..3), i_b*2+i_e is i (0..3).
    # So row_w = k*2 + i? No, row_w = (k_b*2+k_e)*2 + (i_b*2+i_e) = k*2 + i.
    # But the convention is k*n+i = k*4+i.
    # My W_4 row index is k*2+i, which is wrong for 4x4 output.
    # Let's re-evaluate W_4 row index.
    # C is 4x4. Row i, Column k.
    # In my loop:
    # i_b, i_e define row i = i_b*2 + i_e.
    # k_b, k_e define column k = k_b*2 + k_e.
    # W_4 row index should be k*4 + i = (k_b*2+k_e)*4 + (i_b*2+i_e).
    # My code had: row_w = (k_b * 2 + k_e) * 2 + (i_b * 2 + i_e). This is k*2 + i.
    # This is incorrect. It should be k*4 + i.
    
    # Let's fix W_4 construction.
    W_4 = [[0] * 49 for _ in range(16)]
    # U_4 and V_4 are likely correct, let's re-verify them.
    # U_4 row index: A is 4x4. Row i, Col j. i = i_b*2+i_e, j = j_b*2+j_e.
    # Row index = i*4 + j = (i_b*2+i_e)*4 + (j_b*2+j_e).
    # My code: row_u = (i_b * 2 + i_e) * 2 + (j_b * 2 + j_e). This is i*2 + j.
    # This is also incorrect. It should be i*4 + j.
    
    # V_4 row index: B is 4x4. Row j, Col k. j = j_b*2+j_e, k = k_b*2+k_e.
    # Row index = j*4 + k = (j_b*2+j_e)*4 + (k_b*2+k_e).
    # My code: row_v = (j_b * 2 + j_e) * 2 + (k_b * 2 + k_e). This is j*2 + k.
    # This is also incorrect.
    
    # Okay, I need to rebuild U_4, V_4, W_4 with correct row indices.
    
    U_4 = [[0] * 49 for _ in range(16)]
    V_4 = [[0] * 49 for _ in range(16)]
    W_4 = [[0] * 49 for _ in range(16)]
    
    for c1 in range(7):
        for c2 in range(7):
            c = c1 * 7 + c2
            for i_b in range(2):
                for i_e in range(2):
                    for j_b in range(2):
                        for j_e in range(2):
                            i = i_b * 2 + i_e
                            j = j_b * 2 + j_e
                            row_u = i * 4 + j
                            U_4[row_u][c] = U_2[i_b * 2 + i_e][c1] * U_2[j_b * 2 + j_e][c2]
                            
                            for k_b in range(2):
                                for k_e in range(2):
                                    k = k_b * 2 + k_e
                                    row_v = j * 4 + k
                                    V_4[row_v][c] = V_2[j_b * 2 + j_e][c1] * V_2[k_b * 2 + k_e][c2]
                                    
                                    row_w = k * 4 + i
                                    W_4[row_w][c] = W_2[k_b * 2 + k_e][c1] * W_2[i_b * 2 + i_e][c2]
    
    # Now build U_8, V_8, W_8 for 4x4x8.
    n, m, p = 4, 4, 8
    rank = 98
    U_8 = [[0] * rank for _ in range(n * m)] # 16x98
    V_8 = [[0] * rank for _ in range(m * p)] # 32x98
    W_8 = [[0] * rank for _ in range(p * n)] # 32x98
    
    for r in range(16):
        for c in range(49):
            U_8[r][c] = U_4[r][c]
            U_8[r][c + 49] = U_4[r][c]
            
            # V_8
            # B is 4x8.
            # First 49 cols: B1 (cols 0-3).
            # V_8[r, c] = V_4[r, c] for r in 0..15.
            # V_8[r+16, c] = 0.
            # Next 49 cols: B2 (cols 4-7).
            # V_8[r, c+49] = 0.
            # V_8[r+16, c+49] = V_4[r, c].
            V_8[r][c] = V_4[r][c]
            V_8[r + 16][c + 49] = V_4[r][c]
            
            # W_8
            # C is 4x8.
            # First 49 cols: C1 (cols 0-3).
            # W_8[r, c] = W_4[r, c] for r in 0..15.
            # W_8[r+16, c] = 0.
            # Next 49 cols: C2 (cols 4-7).
            # W_8[r, c+49] = 0.
            # W_8[r+16, c+49] = W_4[r, c].
            W_8[r][c] = W_4[r][c]
            W_8[r + 16][c + 49] = W_4[r][c]
            
    return {"U": U_8, "V": V_8, "W": W_8}
# EVOLVE-BLOCK-END
