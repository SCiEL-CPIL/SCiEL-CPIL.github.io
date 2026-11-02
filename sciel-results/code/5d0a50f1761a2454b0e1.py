# EVOLVE-BLOCK-START
def construct():
    import numpy as np
    
    n, m, p = 4, 4, 8
    rank = 98
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen coefficients for 2x2 x 2x2
    # M1: (A11+A22)(B11+B22)
    # M2: (A21+A22)B11
    # M3: A11(B12-B22)
    # M4: A22(B21-B11)
    # M5: (A11+A12)B22
    # M6: (A21-A11)(B11+B12)
    # M7: (A12-A22)(B21+B22)
    
    # A coefficients (2x2) for each M
    # Index 0:11, 1:12, 2:21, 3:22
    A_coeffs = [
        [[1, 0], [0, 1]],  # M1
        [[0, 0], [1, 1]],  # M2
        [[1, 0], [0, 0]],  # M3
        [[0, 0], [0, 1]],  # M4
        [[1, 1], [0, 0]],  # M5
        [[-1, 0], [1, 0]], # M6
        [[0, 1], [0, -1]]  # M7
    ]
    
    # B coefficients (2x2) for each M
    B_coeffs = [
        [[1, 1], [1, 1]],  # M1
        [[1, 0], [0, 0]],  # M2
        [[0, 1], [0, -1]], # M3
        [[-1, 0], [1, 0]], # M4
        [[0, 0], [0, 1]],  # M5
        [[1, 1], [0, 0]],  # M6
        [[0, 1], [1, 1]]   # M7
    ]
    
    # Output reconstruction coefficients for 2x2
    # C11 = M1 - M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 - M3 + M6
    # beta[p, q, v] is the coefficient of M_v in C[p, q]
    # p, q in {0, 1}
    beta = np.zeros((2, 2, 7))
    # M0 (M1)
    beta[0, 0, 0] = 1
    beta[1, 1, 0] = 1
    # M1 (M2)
    beta[1, 0, 1] = 1
    beta[1, 1, 1] = -1
    # M2 (M3)
    beta[0, 1, 2] = 1
    beta[1, 1, 2] = -1
    # M3 (M4)
    beta[0, 0, 3] = -1
    beta[1, 0, 3] = 1
    # M4 (M5)
    beta[0, 0, 4] = -1
    beta[0, 1, 4] = 1
    # M5 (M6)
    beta[1, 1, 5] = 1
    # M6 (M7)
    beta[0, 0, 6] = 1
    
    # Outer Strassen for 4x4 x 4x8
    # A blocks: 0:A11, 1:A12, 2:A21, 3:A22 (each 2x2)
    # B blocks: 0:B11, 1:B12, 2:B21, 3:B22 (each 2x4)
    
    # Outer A coefficients (4x4) for each M_s
    # M1: A11+A22
    outer_A_coeffs = [
        [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]], # M1: A11+A22 (flattened 4x4? No, 4 blocks)
        # Let's represent as 4x4 matrix where block (i,j) is the coeff for A_ij
        # Actually, A_s is a 2x2 matrix of blocks.
        # Let's just use the 4 block indices.
    ]
    
    # Simpler: Define the 2x2 matrix of block coefficients for A_s
    outer_A_blocks = [
        [[1, 0], [0, 1]],  # M1: A11+A22
        [[0, 0], [1, 1]],  # M2: A21+A22
        [[1, 0], [0, 0]],  # M3: A11
        [[0, 0], [0, 1]],  # M4: A22
        [[1, 1], [0, 0]],  # M5: A11+A12
        [[-1, 0], [1, 0]], # M6: A21-A11
        [[0, 1], [0, -1]]  # M7: A12-A22
    ]
    
    # Outer B blocks (2x2 matrix of 2x4 blocks)
    outer_B_blocks = [
        [[1, 1], [1, 1]],  # M1: B11+B22
        [[1, 0], [0, 0]],  # M2: B11
        [[0, 1], [0, -1]], # M3: B12-B22
        [[-1, 0], [1, 0]], # M4: B21-B11
        [[0, 0], [0, 1]],  # M5: B22
        [[1, 1], [0, 0]],  # M6: B11+B12
        [[0, 1], [1, 1]]   # M7: B21+B22
    ]
    
    # Outer output reconstruction
    # C11 (rows 0-1, cols 0-3) = M1 - M4 - M5 + M7
    # C12 (rows 0-1, cols 4-7) = M3 + M5
    # C21 (rows 2-3, cols 0-3) = M2 + M4
    # C22 (rows 2-3, cols 4-7) = M1 - M2 - M3 + M6
    
    # Map s to list of (row_offset, col_offset, sign)
    outer_out_map = {
        0: [(0, 0, 1), (2, 4, 1)],  # M1
        1: [(2, 0, 1), (2, 4, -1)], # M2
        2: [(0, 4, 1), (2, 4, -1)], # M3
        3: [(0, 0, -1), (2, 0, 1)], # M4
        4: [(0, 0, -1), (0, 4, 1)], # M5
        5: [(2, 4, 1)],             # M6
        6: [(0, 0, 1)]              # M7
    }

    for s in range(7):
        # Get A_s block coeffs
        A_s_blocks = outer_A_blocks[s] # 2x2
        # Get B_s block coeffs
        B_s_blocks = outer_B_blocks[s] # 2x2
        
        for u in range(2): # 0: Left, 1: Right
            # B_s_u is a 2x2 matrix of coeffs for the 2x2 sub-blocks of B
            # B_s is 2x4. Left is cols 0-1, Right is cols 2-3.
            # B_s_blocks[i, j] is the coeff for B_block[i, j] which is 2x4.
            # We need the coeffs for the 2x2 sub-blocks.
            # B_block[i, j] has left 2x2 and right 2x2.
            # So B_s_u[i, j] = B_s_blocks[i, j] (since the block is either entirely in u or not? No.)
            # Wait. B_s is a linear combo of B_11, B_12, B_21, B_22.
            # Each B_ij is 2x4.
            # B_s_u is the u-th half of B_s.
            # So B_s_u is a linear combo of the u-th halves of B_ij.
            # The coeffs are the same.
            
            for v in range(7):
                r = s * 14 + u * 7 + v
                
                # 1. Construct U column
                # U is 16x1 (4x4)
                # L_A = sum_{ij} gamma_A[i, j] * (A_s)[i, j]
                # (A_s)[i, j] is a 2x2 block, which is a linear combo of A_11..A_22.
                # (A_s)[i, j] = sum_{k,l} A_s_blocks[i, j] * A_block[k, l] ?
                # No, A_s_blocks[i, j] is the coefficient for the block A_{(i+1)(j+1)}.
                # So (A_s)[i, j] = A_s_blocks[i, j] * A_{block_idx}
                
                # gamma_A is A_coeffs[v] (2x2)
                gamma_A = A_coeffs[v]
                
                U_col = np.zeros(16)
                for i in range(2):
                    for j in range(2):
                        coeff = gamma_A[i, j]
                        if coeff != 0:
                            # (A_s)[i, j] has coefficient A_s_blocks[i, j] for block (i, j)
                            block_coeff = A_s_blocks[i, j]
                            if block_coeff != 0:
                                # Block (i, j) is A_{i*2+j+1}? No, A_11, A_12, A_21, A_22.
                                # Index 0:11, 1:12, 2:21, 3:22.
                                block_idx = i * 2 + j
                                # A_block is 2x2.
                                # A_11 is rows 0-1, cols 0-1.
                                # A_12 is rows 0-1, cols 2-3.
                                # A_21 is rows 2-3, cols 0-1.
                                # A_22 is rows 2-3, cols 2-3.
                                row_off = (block_idx // 2) * 2
                                col_off = (block_idx % 2) * 2
                                for di in range(2):
                                    for dj in range(2):
                                        U_col[(row_off + di) * 4 + (col_off + dj)] += coeff * block_coeff
                U[:, r] = U_col
                
                # 2. Construct V column
                # V is 32x1 (4x8)
                # L_B = sum_{jk} gamma_B[j, k] * (B_s_u)[j, k]
                # (B_s_u)[j, k] is a 2x2 block (half of B_block).
                # B_s_u is derived from B_s_blocks.
                # B_s_blocks[i, j] is coeff for B_{block_idx}.
                # B_{block_idx} is 2x4.
                # Its u-th half is 2x2.
                # So (B_s_u)[j, k] = sum_{i, l} B_s_blocks[i, l] * (B_{block_idx}^{(u)})[j, k]?
                # No, B_s is a matrix of blocks.
                # B_s[j, k] (block) = sum_{i, l} B_s_blocks[i, l] * B_{block_idx(i, l)}
                # We want the (j, k)-th 2x2 sub-block of B_s_u.
                # This corresponds to the (j, k)-th block of B_s, taking its u-th half.
                # So coeff is B_s_blocks[j, k].
                
                gamma_B = B_coeffs[v]
                
                V_col = np.zeros(32)
                for j in range(2):
                    for k in range(2):
                        coeff = gamma_B[j, k]
                        if coeff != 0:
                            block_coeff = B_s_blocks[j, k]
                            if block_coeff != 0:
                                block_idx = j * 2 + k
                                row_off = (block_idx // 2) * 2
                                # Col offset: block is 2x4. u=0 is cols 0-1, u=1 is cols 2-3.
                                col_off = (block_idx % 2) * 4 + u * 2
                                for di in range(2):
                                    for dj in range(2):
                                        V_col[(row_off + di) * 8 + (col_off + dj)] += coeff * block_coeff
                V[:, r] = V_col
                
                # 3. Construct W column
                # W is 32x1 (8x4)
                # M_{s, u, v} contributes to M_s[p, 2u+q] with coeff beta[p, q, v].
                # M_s contributes to C blocks via outer_out_map[s].
                
                W_col = np.zeros(32)
                for p in range(2):
                    for q in range(2):
                        b_coeff = beta[p, q, v]
                        if b_coeff != 0:
                            # M_s entry (p, 2u+q)
                            col_in_Ms = 2 * u + q
                            for (row_off, col_off, sign) in outer_out_map[s]:
                                # C index: row = row_off + p, col = col_off + col_in_Ms
                                c_row = row_off + p
                                c_col = col_off + col_in_Ms
                                # W index: k*n + i = c_col * 4 + c_row
                                w_idx = c_col * 4 + c_row
                                W_col[w_idx] += b_coeff * sign
                W[:, r] = W_col

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
