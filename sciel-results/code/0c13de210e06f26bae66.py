import numpy as np

def construct():
    n, m, p = 4, 4, 9
    
    # Helper to create Strassen 2x2x2 decomposition
    def get_strassen_2x2x2():
        # U: 4x7, V: 4x7, W: 4x7
        # Indices: A[0,0], A[0,1], A[1,0], A[1,1] -> 0,1,2,3
        # B[0,0], B[0,1], B[1,0], B[1,1] -> 0,1,2,3
        # C[0,0], C[0,1], C[1,0], C[1,1] -> 0,1,2,3
        
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (A00+A11)(B00+B11)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        # C00 += M1, C11 += M1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2 = (A10+A11)B00
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        # C10 += M2
        W[2, 1] = 1
        
        # M3 = A00(B01-B11)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        # C01 += M3
        W[1, 2] = 1
        
        # M4 = A11(B10-B00)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        # C00 -= M4, C10 += M4
        W[0, 3] = -1; W[2, 3] = 1
        
        # M5 = (A00+A01)B11
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        # C01 += M5
        W[1, 4] = 1
        
        # M6 = (A10-A00)(B00+B01)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        # C00 += M6
        W[0, 5] = 1
        
        # M7 = (A01-A11)(B10+B11)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        # C11 += M7
        W[3, 6] = 1
        
        return U, V, W

    # Helper to create 4x4x4 Strassen decomposition
    def get_strassen_4x4x4():
        U2, V2, W2 = get_strassen_2x2x2()
        
        # A blocks: A00 (0-3), A01 (4-7), A10 (8-11), A11 (12-15)
        # B blocks: B00 (0-3), B01 (4-7), B10 (8-11), B11 (12-15)
        
        U = np.zeros((16, 49))
        V = np.zeros((16, 49))
        W = np.zeros((16, 49))
        
        # Mapping of M1..M7 to block combinations
        # M1: A00+A11, B00+B11
        # M2: A10+A11, B00
        # M3: A00, B01-B11
        # M4: A11, B10-B00
        # M5: A00+A01, B11
        # M6: A10-A00, B00+B01
        # M7: A01-A11, B10+B11
        
        # For each M_k, we have 7 sub-multiplications (from 2x2x2)
        # We need to sum the contributions from the 2x2x2 decomposition
        # into the 4x4x4 matrices.
        
        # Let's iterate through the 7 M's
        # M1 uses A00, A11 and B00, B11
        # In the 2x2x2 U, V, W, the indices 0,1,2,3 correspond to the 4 elements of the 2x2 matrix.
        # Here, the "2x2 matrix" for A is formed by linear combinations of the 4 blocks.
        # This is tricky because the 2x2x2 decomposition assumes scalar entries.
        # Here, the "entries" are 2x2 blocks.
        # So U for 4x4x4 is a block matrix of 2x2 blocks?
        # No, U is 16x49.
        
        # Let's build it by expanding the 2x2x2 structure.
        # For each of the 7 M's, we have a 2x2x2 tensor.
        # The U for this tensor is 4x7.
        # The "4" corresponds to the 4 elements of the 2x2 A-block combination.
        # We need to map these 4 elements to the 16 elements of the 4x4 A.
        
        # Example M1:
        # A_comb = [[A00, A01], [A10, A11]]? No.
        # M1 = (A00+A11)(B00+B11).
        # This is a product of two 4x4 matrices? No, A00 is 2x2.
        # So M1 is a 2x2 matrix product? No, M1 is the result of multiplying two 2x2 matrices.
        # Wait, in 4x4x4 Strassen, the "scalar" multiplications are 2x2 matrix multiplications.
        # So M1 is a 2x2 matrix.
        # The 2x2x2 decomposition computes 2x2 matrix products.
        # So for M1, we use the 2x2x2 U, V, W.
        # The U of 2x2x2 has 4 rows. These correspond to the 4 elements of the 2x2 matrix (A00+A11).
        # Let's call the 2x2 matrix X. X = A00+A11.
        # X[0,0] = A00[0,0] + A11[0,0]
        # X[0,1] = A00[0,1] + A11[0,1]
        # X[1,0] = A00[1,0] + A11[1,0]
        # X[1,1] = A00[1,1] + A11[1,1]
        
        # So for each row r in U2 (0..3), we add U2[r, col] * (coefficient for A00 element) to U[global_row, global_col].
        
        # Let's define the mapping for each M_k.
        # M1: A = A00+A11, B = B00+B11
        # M2: A = A10+A11, B = B00
        # M3: A = A00, B = B01-B11
        # M4: A = A11, B = B10-B00
        # M5: A = A00+A01, B = B11
        # M6: A = A10-A00, B = B00+B01
        # M7: A = A01-A11, B = B10+B11
        
        # For each M_k, we have a 2x2 A-matrix and a 2x2 B-matrix.
        # We iterate over the 7 columns of U2, V2, W2.
        
        # Define block offsets
        # A00: 0, A01: 4, A10: 8, A11: 12
        # B00: 0, B01: 4, B10: 8, B11: 12
        
        # For each M_k, define the linear combination of blocks for A and B.
        # A_comb = c00*A00 + c01*A01 + c10*A10 + c11*A11
        # B_comb = d00*B00 + d01*B01 + d10*B10 + d11*B11
        
        # The 2x2 matrix for A is [[A_comb[0,0], A_comb[0,1]], [A_comb[1,0], A_comb[1,1]]]
        # The elements of A_comb are sums of 2x2 blocks.
        
        # Let's just hardcode the contributions.
        # For each M_k (index 0..6), and each 2x2x2 column (index 0..6):
        # Global column = M_k * 7 + col_2x2
        
        for mk in range(7):
            # Determine A and B block coefficients
            # A: [c00, c01, c10, c11] for A00, A01, A10, A11
            # B: [d00, d01, d10, d11] for B00, B01, B10, B11
            
            if mk == 0: # M1
                A_coeffs = [1, 0, 0, 1] # A00 + A11
                B_coeffs = [1, 0, 0, 1] # B00 + B11
            elif mk == 1: # M2
                A_coeffs = [0, 0, 1, 1] # A10 + A11
                B_coeffs = [1, 0, 0, 0] # B00
            elif mk == 2: # M3
                A_coeffs = [1, 0, 0, 0] # A00
                B_coeffs = [0, 1, 0, -1] # B01 - B11
            elif mk == 3: # M4
                A_coeffs = [0, 0, 0, 1] # A11
                B_coeffs = [0, 0, 1, -1] # B10 - B00
            elif mk == 4: # M5
                A_coeffs = [1, 1, 0, 0] # A00 + A01
                B_coeffs = [0, 0, 0, 1] # B11
            elif mk == 5: # M6
                A_coeffs = [0, 0, 1, -1] # A10 - A00? No, A10 is index 2.
                # A10 - A00 -> A_coeffs[2]=1, A_coeffs[0]=-1
                A_coeffs = [-1, 0, 1, 0]
                B_coeffs = [1, 1, 0, 0] # B00 + B01
            elif mk == 6: # M7
                A_coeffs = [0, 1, 0, -1] # A01 - A11
                B_coeffs = [0, 0, 1, 1] # B10 + B11
            
            # Now, for each of the 7 sub-columns in 2x2x2
            for c2 in range(7):
                global_col = mk * 7 + c2
                
                # U contribution
                # U2 has 4 rows. Row r corresponds to element (r//2, r%2) of the 2x2 A matrix.
                # The 2x2 A matrix is formed by A_coeffs.
                # A_matrix[r//2, r%2] = sum(A_coeffs[i] * A_block_i[r//2, r%2])
                # Wait, A_block_i is 2x2.
                # A00 is rows 0-1, cols 0-1 of A.
                # A01 is rows 0-1, cols 2-3 of A.
                # A10 is rows 2-3, cols 0-1 of A.
                # A11 is rows 2-3, cols 2-3 of A.
                
                # The 2x2 A_matrix has elements:
                # [0,0] = A00[0,0]*c00 + A01[0,0]*c01 + A10[0,0]*c10 + A11[0,0]*c11
                # [0,1] = A00[0,1]*c00 + A01[0,1]*c01 + A10[0,1]*c10 + A11[0,1]*c11
                # [1,0] = A00[1,0]*c00 + A01[1,0]*c01 + A10[1,0]*c10 + A11[1,0]*c11
                # [1,1] = A00[1,1]*c00 + A01[1,1]*c01 + A10[1,1]*c10 + A11[1,1]*c11
                
                # U2[r, c2] is the coefficient for A_matrix[r//2, r%2].
                # So we add U2[r, c2] * A_coeffs[i] to U[global_row, global_col]
                # where global_row is the index of the element in A_block_i.
                
                for r in range(4):
                    val = U2[r, c2]
                    if val == 0: continue
                    row_2x2 = r // 2
                    col_2x2 = r % 2
                    
                    # Add to A00, A01, A10, A11
                    for i in range(4):
                        coeff = A_coeffs[i]
                        if coeff == 0: continue
                        # A00: rows 0,1 cols 0,1
                        # A01: rows 0,1 cols 2,3
                        # A10: rows 2,3 cols 0,1
                        # A11: rows 2,3 cols 2,3
                        
                        if i == 0: # A00
                            gr = row_2x2
                            gc = col_2x2
                        elif i == 1: # A01
                            gr = row_2x2
                            gc = 2 + col_2x2
                        elif i == 2: # A10
                            gr = 2 + row_2x2
                            gc = col_2x2
                        elif i == 3: # A11
                            gr = 2 + row_2x2
                            gc = 2 + col_2x2
                        
                        U[gr * 4 + gc, global_col] += val * coeff

                # V contribution
                for r in range(4):
                    val = V2[r, c2]
                    if val == 0: continue
                    row_2x2 = r // 2
                    col_2x2 = r % 2
                    
                    for i in range(4):
                        coeff = B_coeffs[i]
                        if coeff == 0: continue
                        if i == 0: # B00
                            gr = row_2x2
                            gc = col_2x2
                        elif i == 1: # B01
                            gr = row_2x2
                            gc = 2 + col_2x2
                        elif i == 2: # B10
                            gr = 2 + row_2x2
                            gc = col_2x2
                        elif i == 3: # B11
                            gr = 2 + row_2x2
                            gc = 2 + col_2x2
                        
                        V[gr * 4 + gc, global_col] += val * coeff

                # W contribution
                # W2 has 4 rows.
                # C00: 0, C01: 1, C10: 2, C11: 3
                # C is 4x4.
                # C00 is top-left 2x2.
                # C01 is top-right 2x2.
                # C10 is bottom-left 2x2.
                # C11 is bottom-right 2x2.
                
                for r in range(4):
                    val = W2[r, c2]
                    if val == 0: continue
                    row_2x2 = r // 2
                    col_2x2 = r % 2
                    
                    # The 2x2 C_matrix is the result of the 2x2x2 multiplication.
                    # This C_matrix corresponds to one of the 4 blocks of the 4x4 C.
                    # Which block?
                    # M1 contributes to C00 and C11.
                    # M2 contributes to C10.
                    # M3 contributes to C01.
                    # M4 contributes to C00 and C10.
                    # M5 contributes to C01.
                    # M6 contributes to C00.
                    # M7 contributes to C11.
                    
                    # We need to know which block of C the M_k result goes to.
                    # And how it is combined (sign).
                    
                    # Let's define the target block and sign for each M_k.
                    # M1: C00 (+1), C11 (+1)
                    # M2: C10 (+1)
                    # M3: C01 (+1)
                    # M4: C00 (-1), C10 (+1)
                    # M5: C01 (+1)
                    # M6: C00 (+1)
                    # M7: C11 (+1)
                    
                    # Note: M1, M4, M6 contribute to C00.
                    # M3, M5 contribute to C01.
                    # M2, M4 contribute to C10.
                    # M1, M7 contribute to C11.
                    
                    # The W2[r, c2] is the coefficient for C_2x2[r//2, r%2].
                    # We need to add this to the corresponding element in the 4x4 C.
                    
                    # Define mapping from (mk, c_block_in_2x2, r) to (global_row, global_col, sign)
                    # c_block_in_2x2 is determined by which block of C the M_k result is added to.
                    # But M_k might contribute to multiple blocks?
                    # No, each M_k is a 2x2 matrix.
                    # In the 4x4x4 Strassen, the 7 M's are 2x2 matrices.
                    # They are added to the 4x4 C matrix.
                    # C00 = M1 - M4 + M6
                    # C01 = M3 + M5
                    # C10 = M2 + M4
                    # C11 = M1 - M3 + M7
                    
                    # So for a given mk, we know which C_block it goes to and with what sign.
                    # But W2[r, c2] gives the contribution to the 2x2 C_matrix.
                    # So we add val * sign to C_block[row_2x2, col_2x2].
                    
                    # Determine target block and sign for mk
                    targets = []
                    if mk == 0: # M1
                        targets = [(0, 1), (3, 1)] # C00, C11
                    elif mk == 1: # M2
                        targets = [(2, 1)] # C10
                    elif mk == 2: # M3
                        targets = [(1, 1)] # C01
                    elif mk == 3: # M4
                        targets = [(0, -1), (2, 1)] # C00, C10
                    elif mk == 4: # M5
                        targets = [(1, 1)] # C01
                    elif mk == 5: # M6
                        targets = [(0, 1)] # C00
                    elif mk == 6: # M7
                        targets = [(3, 1)] # C11
                    
                    for block_idx, sign in targets:
                        # block_idx: 0=C00, 1=C01, 2=C10, 3=C11
                        # C00: rows 0-1, cols 0-1
                        # C01: rows 0-1, cols 2-3
                        # C10: rows 2-3, cols 0-1
                        # C11: rows 2-3, cols 2-3
                        
                        if block_idx == 0:
                            gr = row_2x2
                            gc = col_2x2
                        elif block_idx == 1:
                            gr = row_2x2
                            gc = 2 + col_2x2
                        elif block_idx == 2:
                            gr = 2 + row_2x2
                            gc = col_2x2
                        elif block_idx == 3:
                            gr = 2 + row_2x2
                            gc = 2 + col_2x2
                        
                        W[gr * 4 + gc, global_col] += val * sign

        return U, V, W

    # Helper to create 4x4x1 trivial decomposition
    def get_trivial_4x4x1():
        U = np.zeros((16, 16))
        V = np.zeros((16, 16)) # Note: B is 4x1, so V should be 4x16? 
        # Wait, V shape is (m*p, rank) = (4*1, 16) = (4, 16).
        # But in the combined 4x4x9, V is (36, rank).
        # Here we just return the local parts.
        V = np.zeros((4, 16))
        W = np.zeros((4, 16)) # C is 4x1, so W is (1*4, 16) = (4, 16).
        
        for i in range(4):
            for j in range(4):
                col = i * 4 + j
                U[i * 4 + j, col] = 1
                V[j, col] = 1
                W[i, col] = 1
        return U, V, W

    # Construct 4x4x9
    rank_total = 49 + 49 + 16
    U = np.zeros((16, rank_total))
    V = np.zeros((36, rank_total))
    W = np.zeros((36, rank_total))
    
    # Part 1: B cols 0-3
    U1, V1, W1 = get_strassen_4x4x4()
    U[:, 0:49] = U1
    # V1 is 16x49. We need to map to 36x49.
    # V1 rows 0-15 correspond to B rows 0-3, cols 0-3.
    # In 4x4x9, B is 4x9.
    # Row j, col k in B corresponds to index j*9+k in V.
    # For B1 (cols 0-3), k is 0-3.
    for j in range(4):
        for k in range(4):
            V[j * 9 + k, 0:49] = V1[j * 4 + k, 0:49]
    # W1 is 16x49. C is 4x4.
    # C rows 0-3, cols 0-3.
    # In 4x4x9, C is 4x9.
    # Row k, col i in C corresponds to index k*4+i in W.
    # For C1 (cols 0-3), k is 0-3.
    for k in range(4):
        for i in range(4):
            W[k * 4 + i, 0:49] = W1[k * 4 + i, 0:49]
            
    # Part 2: B cols 4-7
    U2, V2, W2 = get_strassen_4x4x4()
    U[:, 49:98] = U2
    for j in range(4):
        for k in range(4):
            V[j * 9 + (4 + k), 49:98] = V2[j * 4 + k, 0:49]
    for k in range(4):
        for i in range(4):
            W[k * 4 + i, 49:98] = W2[k * 4 + i, 0:49]
            
    # Part 3: B col 8
    U3, V3, W3 = get_trivial_4x4x1()
    U[:, 98:114] = U3
    for j in range(4):
        V[j * 9 + 8, 98:114] = V3[j, 0:16]
    for i in range(4):
        W[8 * 4 + i, 98:114] = W3[i, 0:16]
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    construct()
