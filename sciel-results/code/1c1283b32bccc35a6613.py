import numpy as np
import time

def construct():
    n, m, p = 4, 4, 5
    start_time = time.time()
    
    # Helper to create a 2x2x2 Strassen decomposition
    def strassen_2x2():
        # Returns U, V, W of shape (4, 7), (4, 7), (4, 7)
        # Indices for 2x2: i in 0,1; j in 0,1; k in 0,1
        # U index: 2*i + j
        # V index: 2*j + k
        # W index: 2*k + i
        
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # 1. M1 = (A00 + A11)(B00 + B11)
        # A terms: A00 (idx 0), A11 (idx 3)
        # B terms: B00 (idx 0), B11 (idx 3)
        U[0, 0] += 1; U[3, 0] += 1
        V[0, 0] += 1; V[3, 0] += 1
        # C terms: C00 (idx 0), C11 (idx 3)
        W[0, 0] += 1; W[3, 0] += 1
        
        # 2. M2 = (A10 + A11) B00
        # A terms: A10 (idx 2), A11 (idx 3)
        # B terms: B00 (idx 0)
        U[2, 1] += 1; U[3, 1] += 1
        V[0, 1] += 1
        # C terms: C10 (idx 1), C11 (idx 3)
        W[1, 1] += 1; W[3, 1] -= 1 # Wait, check formula
        # C10 = M2 + M4. C11 = M1 - M2 + M3 + M6.
        # So M2 contributes +1 to C10 and -1 to C11.
        W[1, 1] = 1
        W[3, 1] = -1
        
        # 3. M3 = A00 (B01 - B11)
        # A terms: A00 (idx 0)
        # B terms: B01 (idx 1), B11 (idx 3)
        U[0, 2] += 1
        V[1, 2] += 1; V[3, 2] -= 1
        # C terms: C01 (idx 2), C11 (idx 3)
        # C01 = M3 + M5. C11 = M1 - M2 + M3 + M6.
        W[2, 2] = 1
        W[3, 2] = 1
        
        # 4. M4 = A11 (B10 - B00)
        # A terms: A11 (idx 3)
        # B terms: B10 (idx 2), B00 (idx 0)
        U[3, 3] += 1
        V[2, 3] += 1; V[0, 3] -= 1
        # C terms: C00 (idx 0), C10 (idx 1)
        # C00 = M1 + M4 - M5 + M7. C10 = M2 + M4.
        W[0, 3] = 1
        W[1, 3] = 1
        
        # 5. M5 = (A00 + A01) B11
        # A terms: A00 (idx 0), A01 (idx 1)
        # B terms: B11 (idx 3)
        U[0, 4] += 1; U[1, 4] += 1
        V[3, 4] += 1
        # C terms: C00 (idx 0), C01 (idx 2)
        # C00 = ... - M5 ... C01 = M3 + M5.
        W[0, 4] = -1
        W[2, 4] = 1
        
        # 6. M6 = (A10 - A00) (B00 + B01)
        # A terms: A10 (idx 2), A00 (idx 0)
        # B terms: B00 (idx 0), B01 (idx 1)
        U[2, 5] += 1; U[0, 5] -= 1
        V[0, 5] += 1; V[1, 5] += 1
        # C terms: C11 (idx 3)
        # C11 = ... + M6.
        W[3, 5] = 1
        
        # 7. M7 = (A01 - A11) (B10 + B11)
        # A terms: A01 (idx 1), A11 (idx 3)
        # B terms: B10 (idx 2), B11 (idx 3)
        U[1, 6] += 1; U[3, 6] -= 1
        V[2, 6] += 1; V[3, 6] += 1
        # C terms: C00 (idx 0)
        # C00 = ... + M7.
        W[0, 6] = 1
        
        return U, V, W

    # Build 4x4x4 decomposition using Strassen on 2x2 blocks
    def build_4x4x4():
        # Rank 49
        U = np.zeros((16, 49))
        V = np.zeros((16, 49)) # m*p = 4*4 = 16
        W = np.zeros((16, 49)) # p*n = 4*4 = 16
        
        U2, V2, W2 = strassen_2x2()
        
        # We have 7 block products M_r (r=0..6)
        # Each M_r is a 2x2 matrix product X_r * Y_r
        # X_r and Y_r are linear combinations of the 2x2 blocks of A and B.
        # We need to express X_r and Y_r in terms of the global A and B entries.
        
        # Block indices I, J in 0,1. Local indices i, j in 0,1.
        # Global A index: 4*I + 2*J + j? No.
        # A is 4x4. Block (I,J) is rows 2I..2I+1, cols 2J..2J+1.
        # A entry (2I+i, 2J+j) is at index 4*(2I+i) + (2J+j) = 8I + 4i + 2J + j.
        
        # Let's define the coefficients for X_r and Y_r.
        # X_r is a 2x2 matrix of coefficients for the A blocks.
        # Y_r is a 2x2 matrix of coefficients for the B blocks.
        
        # Strassen formulas for block level:
        # M0: X = [[1, 0], [0, 1]] (A00+A11? No, A00 and A11 are blocks)
        # Let's stick to the 7 terms defined in strassen_2x2.
        # Term 0: M1 = (A00 + A11)(B00 + B11)
        # Term 1: M2 = (A10 + A11)B00
        # Term 2: M3 = A00(B01 - B11)
        # Term 3: M4 = A11(B10 - B00)
        # Term 4: M5 = (A00 + A01)B11
        # Term 5: M6 = (A10 - A00)(B00 + B01)
        # Term 6: M7 = (A01 - A11)(B10 + B11)
        
        # For each term r, we have a 2x2 matrix product.
        # We use the U2, V2, W2 from strassen_2x2 to decompose this product.
        # The "scalars" in U2, V2, W2 are now 2x2 matrices.
        # We need to expand them.
        
        # Let's map the block-level A indices (0,1,2,3) to global A indices.
        # Block 0 (A00): global indices 0,1,4,5 (rows 0,1; cols 0,1)
        # Block 1 (A01): global indices 2,3,6,7 (rows 0,1; cols 2,3)
        # Block 2 (A10): global indices 8,9,12,13 (rows 2,3; cols 0,1)
        # Block 3 (A11): global indices 10,11,14,15 (rows 2,3; cols 2,3)
        
        # Similarly for B.
        # Block 0 (B00): global indices 0,1,4,5 (rows 0,1; cols 0,1) -- Wait, B is 4x4 here.
        # B index: 4*j + k.
        # Block (J,K) of B: rows 2J..2J+1, cols 2K..2K+1.
        # B00: rows 0,1; cols 0,1. Indices: 0,1,4,5.
        # B01: rows 0,1; cols 2,3. Indices: 2,3,6,7.
        # B10: rows 2,3; cols 0,1. Indices: 8,9,12,13.
        # B11: rows 2,3; cols 2,3. Indices: 10,11,14,15.
        
        # Output C blocks:
        # C00: rows 0,1; cols 0,1. Indices: 0,1,4,5.
        # C01: rows 0,1; cols 2,3. Indices: 2,3,6,7.
        # C10: rows 2,3; cols 0,1. Indices: 8,9,12,13.
        # C11: rows 2,3; cols 2,3. Indices: 10,11,14,15.
        
        # We will iterate through the 7 block-level terms.
        # For each term r (0..6), we have coefficients for A blocks and B blocks.
        # Let's define the block coefficients for A and B for each term.
        
        # A_block_coeffs: 7 x 4 (each 4 is for blocks 0,1,2,3)
        # B_block_coeffs: 7 x 4
        
        A_coeffs = [
            [1, 0, 0, 1], # M1: A00 + A11
            [0, 0, 1, 1], # M2: A10 + A11
            [1, 0, 0, 0], # M3: A00
            [0, 0, 0, 1], # M4: A11
            [1, 1, 0, 0], # M5: A00 + A01
            [-1, 0, 1, 0],# M6: A10 - A00
            [0, 1, 0, -1] # M7: A01 - A11
        ]
        
        B_coeffs = [
            [1, 0, 0, 1], # M1: B00 + B11
            [1, 0, 0, 0], # M2: B00
            [0, 1, 0, -1],# M3: B01 - B11
            [0, 0, 1, -1],# M4: B10 - B00
            [0, 0, 0, 1], # M5: B11
            [1, 1, 0, 0], # M6: B00 + B01
            [0, 0, 1, 1]  # M7: B10 + B11
        ]
        
        # C_block_coeffs: 7 x 4 (for blocks 0,1,2,3 of C)
        # M1: C00 + C11
        # M2: C10 - C11
        # M3: C01 + C11
        # M4: C00 + C10
        # M5: -C00 + C01
        # M6: C11
        # M7: C00
        
        C_coeffs = [
            [1, 0, 0, 1], # M1
            [0, 0, 1, -1],# M2
            [0, 1, 0, 1], # M3
            [1, 0, 1, 0], # M4
            [-1, 1, 0, 0],# M5
            [0, 0, 0, 1], # M6
            [1, 0, 0, 0]  # M7
        ]
        
        # Now, for each block term r, we decompose the 2x2 matrix product
        # X_r * Y_r using the 2x2 Strassen decomposition U2, V2, W2.
        # X_r is formed by A_blocks weighted by A_coeffs[r].
        # Y_r is formed by B_blocks weighted by B_coeffs[r].
        
        # The 2x2 Strassen decomposition U2, V2, W2 operates on 2x2 matrices.
        # U2 has 4 rows (corresponding to the 4 elements of the 2x2 input matrix X_r).
        # V2 has 4 rows (corresponding to the 4 elements of the 2x2 input matrix Y_r).
        # W2 has 4 rows (corresponding to the 4 elements of the 2x2 output matrix Z_r).
        
        # We need to map the "elements" of X_r, Y_r, Z_r to global indices.
        # X_r is a 2x2 matrix. Its element (I,J) is a linear combination of A blocks.
        # Wait, X_r itself is a 2x2 matrix of scalars?
        # No, in the block Strassen, X_r and Y_r are 2x2 matrices of *blocks*.
        # But we are decomposing the product of two 2x2 matrices of blocks.
        # This is equivalent to computing 7 products of 2x2 blocks.
        # Each block product is a 2x2 matrix multiplication.
        # We use strassen_2x2 to decompose each 2x2 block multiplication.
        
        # So, for each block term r (0..6):
        # We have a "virtual" 2x2 matrix multiplication.
        # The inputs are the blocks themselves.
        # We need to expand the block-level Strassen into global scalar multiplications.
        
        # Actually, it's simpler:
        # The block-level Strassen gives 7 terms.
        # Each term is a product of two 2x2 matrices (which are linear combinations of A and B blocks).
        # Let's call these matrices X_r and Y_r.
        # We need to compute Z_r = X_r Y_r.
        # We use the 2x2 Strassen decomposition (U2, V2, W2) to compute Z_r.
        # U2, V2, W2 are for 2x2 matrices.
        # The "scalars" in U2, V2, W2 are actually the blocks.
        # So we need to expand U2, V2, W2 into global indices.
        
        # Let's do this explicitly.
        # For each r in 0..6:
        #   For each c in 0..6 (the 7 sub-terms of the 2x2 Strassen):
        #       We have coefficients from U2[:, c], V2[:, c], W2[:, c].
        #       U2[i, c] is the coefficient of the (i//2, i%2) element of X_r.
        #       The (I,J) element of X_r is sum_{a_block} A_coeffs[r][a_block] * A_block(a_block).
        #       This is getting complicated.
        
        # Alternative: Just construct the 49 terms directly.
        # There are 7 block terms. Each block term is a 2x2 multiplication.
        # A 2x2 multiplication has 7 Strassen terms.
        # So 49 terms total.
        
        # Let's generate the 49 terms.
        # Term index t = r * 7 + c.
        
        # For each r (0..6) and c (0..6):
        #   We need to determine the global A, B, C indices and coefficients.
        
        # Let's precompute the block-to-global mapping.
        # A_block(b) corresponds to 4 global indices.
        # Block 0: [0, 1, 4, 5]
        # Block 1: [2, 3, 6, 7]
        # Block 2: [8, 9, 12, 13]
        # Block 3: [10, 11, 14, 15]
        
        # B_block(b) corresponds to 4 global indices.
        # Block 0: [0, 1, 4, 5]
        # Block 1: [2, 3, 6, 7]
        # Block 2: [8, 9, 12, 13]
        # Block 3: [10, 11, 14, 15]
        
        # C_block(b) corresponds to 4 global indices.
        # Block 0: [0, 1, 4, 5]
        # Block 1: [2, 3, 6, 7]
        # Block 2: [8, 9, 12, 13]
        # Block 3: [10, 11, 14, 15]
        
        # The 2x2 Strassen decomposition U2, V2, W2 has 7 columns.
        # For column c, U2[i, c] is the coefficient for the i-th element of the input matrix X.
        # The i-th element of X is X[i//2, i%2].
        
        # So for a given block term r and sub-term c:
        #   The contribution to A is sum over i of U2[i, c] * X_r[i//2, i%2].
        #   X_r[I, J] = sum_{a_block} A_coeffs[r][a_block] * A_block(a_block)[I, J]?
        #   No, X_r is a 2x2 matrix of blocks.
        #   X_r[I, J] is the block at position (I,J) in X_r.
        #   X_r[I, J] = sum_{a_block} A_coeffs[r][a_block] * delta(a_block == I*2+J) * A_block(a_block)?
        #   No, A_coeffs[r] is a vector of 4 coefficients for the 4 blocks of A.
        #   The block A_block(b) is at position (b//2, b%2) in the 2x2 block matrix.
        #   So X_r[I, J] = A_coeffs[r][I*2+J] * A_block(I*2+J).
        
        # Similarly, Y_r[I, J] = B_coeffs[r][I*2+J] * B_block(I*2+J).
        
        # And Z_r[I, J] = C_coeffs[r][I*2+J] * C_block(I*2+J).
        
        # Now, we use U2, V2, W2 to decompose Z_r = X_r Y_r.
        # For each sub-term c:
        #   U_term = sum_{i=0}^3 U2[i, c] * X_r[i//2, i%2]
        #   V_term = sum_{j=0}^3 V2[j, c] * Y_r[j//2, j%2]
        #   W_term = sum_{k=0}^3 W2[k, c] * Z_r[k//2, k%2]
        
        # We need to add these to the global U, V, W.
        
        for r in range(7):
            for c in range(7):
                col = r * 7 + c
                
                # Compute U contributions
                for i in range(4):
                    coeff_u = U2[i, c]
                    if coeff_u != 0:
                        I = i // 2
                        J = i % 2
                        # X_r[I, J] = A_coeffs[r][I*2+J] * A_block(I*2+J)
                        a_block_idx = I * 2 + J
                        a_coeff = A_coeffs[r][a_block_idx]
                        if a_coeff != 0:
                            # Add to global U
                            # Global indices for A_block(a_block_idx)
                            base_a = a_block_idx * 4
                            for offset in [0, 1, 4, 5]:
                                # Wait, block 0 is 0,1,4,5.
                                # Block 1 is 2,3,6,7.
                                # Block b: rows 2*(b//2), 2*(b//2)+1; cols 2*(b%2), 2*(b%2)+1.
                                # Global index = 4*row + col.
                                row1 = 2 * (a_block_idx // 2)
                                row2 = row1 + 1
                                col1 = 2 * (a_block_idx % 2)
                                col2 = col1 + 1
                                idxs = [4*row1 + col1, 4*row1 + col2, 4*row2 + col1, 4*row2 + col2]
                                for idx in idxs:
                                    U[idx, col] += coeff_u * a_coeff
                                    
                # Compute V contributions
                for j in range(4):
                    coeff_v = V2[j, c]
                    if coeff_v != 0:
                        I = j // 2
                        J = j % 2
                        # Y_r[I, J] = B_coeffs[r][I*2+J] * B_block(I*2+J)
                        b_block_idx = I * 2 + J
                        b_coeff = B_coeffs[r][b_block_idx]
                        if b_coeff != 0:
                            row1 = 2 * (b_block_idx // 2)
                            row2 = row1 + 1
                            col1 = 2 * (b_block_idx % 2)
                            col2 = col1 + 1
                            idxs = [4*row1 + col1, 4*row1 + col2, 4*row2 + col1, 4*row2 + col2]
                            for idx in idxs:
                                V[idx, col] += coeff_v * b_coeff
                                
                # Compute W contributions
                for k in range(4):
                    coeff_w = W2[k, c]
                    if coeff_w != 0:
                        I = k // 2
                        J = k % 2
                        # Z_r[I, J] = C_coeffs[r][I*2+J] * C_block(I*2+J)
                        c_block_idx = I * 2 + J
                        c_coeff = C_coeffs[r][c_block_idx]
                        if c_coeff != 0:
                            row1 = 2 * (c_block_idx // 2)
                            row2 = row1 + 1
                            col1 = 2 * (c_block_idx % 2)
                            col2 = col1 + 1
                            idxs = [4*row1 + col1, 4*row1 + col2, 4*row2 + col1, 4*row2 + col2]
                            for idx in idxs:
                                W[idx, col] += coeff_w * c_coeff
                                
        return U, V, W

    # Build 4x4x1 decomposition
    def build_4x4x1():
        # Rank 16
        U = np.zeros((16, 16))
        V = np.zeros((20, 16)) # m*p = 4*5 = 20
        W = np.zeros((20, 16)) # p*n = 5*4 = 20
        
        col = 0
        for i in range(4):
            for j in range(4):
                U[i * 4 + j, col] = 1
                V[j * 5 + 4, col] = 1 # k=4 is the 5th column (index 4)
                W[4 * 4 + i, col] = 1 # k=4, i is the row in C
                col += 1
                
        return U, V, W

    # Combine
    U_444, V_444, W_444 = build_4x4x4()
    U_441, V_441, W_441 = build_4x4x1()
    
    U_total = np.hstack([U_444, U_441])
    V_total = np.hstack([V_444, V_441])
    W_total = np.hstack([W_444, W_441])
    
    # Convert to lists of lists of integers
    U_list = U_total.astype(int).tolist()
    V_list = V_total.astype(int).tolist()
    W_list = W_total.astype(int).tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
