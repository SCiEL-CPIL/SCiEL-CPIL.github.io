import numpy as np

DIMENSIONS = [4, 4, 7]

def get_strassen_2x2_terms():
    # Returns list of (u_indices, v_indices, w_indices, coeff)
    # u_indices: list of (i, k) for A
    # v_indices: list of (k, j) for B
    # w_indices: list of (i, j) for C
    # For 2x2, i,k,j in {0,1}
    # Strassen:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11) B00
    # M3 = A00 (B01 - B11)
    # M4 = A11 (B10 - B00)
    # M5 = (A00 + A01) B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # We need to express each Mi as sum of rank-1 terms of 2x2 multiplication.
    # A 2x2 multiplication C = AB has rank 7.
    # The standard basis for the tensor is the 8 schoolbook terms.
    # Strassen replaces 8 terms with 7.
    # Actually, the "rank-1 terms" in the tensor decomposition are the scalar multiplications.
    # Each scalar multiplication is $A_{ik} B_{kj}$.
    # In Strassen, we compute linear combinations of A and B, multiply, and combine.
    # The "rank" counts the number of scalar multiplications.
    # So we need to identify the 7 scalar multiplications and their coefficients in U, V, W.
    
    # Let's define the 7 products P1..P7.
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B01-B11)
    # P4 = A11(B10-B00)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    
    # We need to map these to the tensor factors.
    # For each product P_r, we have U_r, V_r, W_r vectors.
    # U_r has non-zeros at A indices involved.
    # V_r has non-zeros at B indices involved.
    # W_r has non-zeros at C indices involved.
    
    # Example P1:
    # U1: A00: 1, A11: 1. Others 0.
    # V1: B00: 1, B11: 1. Others 0.
    # W1: C00: 1, C01: 1, C10: 1, C11: 1.
    # Wait, P1 contributes to all 4 entries of C.
    # C00 += P1, C01 += P1, C10 += P1, C11 += P1.
    # So W1 = [1, 1, 1, 1] (flattened).
    
    # P2:
    # U2: A10: 1, A11: 1.
    # V2: B00: 1.
    # W2: C10: 1, C11: 1. (Since (A10+A11)B00 adds to row 1 of C).
    # C10 += P2, C11 += P2.
    # W2 = [0, 0, 1, 1].
    
    # P3:
    # U3: A00: 1.
    # V3: B01: 1, B11: -1.
    # W3: C01: 1, C11: -1? No.
    # A00 * (B01 - B11) = A00 B01 - A00 B11.
    # This contributes to C01 (from A00 B01) and C00? No.
    # C = [ A00B00 + A01B10, A00B01 + A01B11
    #       A10B00 + A11B10, A10B01 + A11B11 ]
    # P3 = A00 B01 - A00 B11.
    # This term is part of C01 (A00 B01) and ... wait.
    # In Strassen, the reconstruction is:
    # C00 = P1 - P4 + P6 - P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P3 + P5 + P7
    # So we need to check the signs in W.
    
    # Let's define the terms (U, V, W) for the 7 products.
    # Indices:
    # A: 00, 01, 10, 11 -> 0, 1, 2, 3
    # B: 00, 01, 10, 11 -> 0, 1, 2, 3
    # C: 00, 01, 10, 11 -> 0, 1, 2, 3
    
    terms = []
    
    # P1
    u = [1, 0, 0, 1] # A00, A11
    v = [1, 0, 0, 1] # B00, B11
    w = [1, 0, 0, 0] # C00 coeff in P1? No, P1 is added to C00 with coeff 1.
    # We store the contribution of this rank-1 term to the final C.
    # The term is P1.
    # C00 += 1*P1. C01 += 0*P1. C10 += 0*P1. C11 += 0*P1?
    # No, the reconstruction formula says:
    # C00 = P1 - P4 + P6 - P7
    # So for P1, W = [1, 0, 0, 0].
    terms.append((u, v, [1, 0, 0, 0]))
    
    # P2
    u = [0, 0, 1, 1] # A10, A11
    v = [1, 0, 0, 0] # B00
    w = [0, 0, 1, 0] # C10 += P2.
    terms.append((u, v, [0, 0, 1, 0]))
    
    # P3
    u = [1, 0, 0, 0] # A00
    v = [0, 1, 0, -1] # B01, B11
    w = [0, 1, 0, 0] # C01 += P3.
    terms.append((u, v, [0, 1, 0, 0]))
    
    # P4
    u = [0, 0, 0, 1] # A11
    v = [0, 0, 1, -1] # B10, B00
    w = [-1, 0, 1, 0] # C00 -= P4, C10 += P4.
    terms.append((u, v, [-1, 0, 1, 0]))
    
    # P5
    u = [1, 1, 0, 0] # A00, A01
    v = [0, 0, 0, 1] # B11
    w = [0, 1, 0, 1] # C01 += P5, C11 += P5.
    terms.append((u, v, [0, 1, 0, 1]))
    
    # P6
    u = [0, 0, 1, -1] # A10, A00
    v = [1, 1, 0, 0] # B00, B01
    w = [1, 0, 0, 0] # C00 += P6.
    terms.append((u, v, [1, 0, 0, 0]))
    
    # P7
    u = [0, 1, 0, -1] # A01, A11
    v = [0, 0, 1, 1] # B10, B11
    w = [-1, 0, 0, 1] # C00 -= P7, C11 += P7.
    terms.append((u, v, [-1, 0, 0, 1]))
    
    return terms

def construct():
    n, m, p = DIMENSIONS
    # We use Rank 97 decomposition.
    # Part 1: A (4x4) * B1 (4x4). B1 is cols 0-3 of B.
    # Part 2: A (4x4) * B2 (4x3). B2 is cols 4-6 of B.
    
    rank = 97
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: Strassen for 4x4x4
    # We need to map the 2x2 Strassen terms to 4x4.
    # The 4x4 multiplication is composed of 7 products of 2x2 matrices.
    # Each 2x2 product has 7 rank-1 terms.
    # Total 49 terms.
    
    # We need to define the 7 2x2 products M1..M7 for the 4x4 case.
    # A = [[A00, A01], [A10, A11]] where Ai are 2x2.
    # B = [[B00, B01], [B10, B11]] where Bj are 2x2.
    
    # The 7 products are:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # For each Mi, we apply the 2x2 Strassen decomposition.
    # This gives 7 scalar multiplications.
    # We need to map the indices of the 2x2 blocks to the 4x4 indices.
    
    # Block indices for A (4x4):
    # A00: rows 0-1, cols 0-1
    # A01: rows 0-1, cols 2-3
    # A10: rows 2-3, cols 0-1
    # A11: rows 2-3, cols 2-3
    
    # Block indices for B (4x4 part of 4x7):
    # B00: rows 0-1, cols 0-1
    # B01: rows 0-1, cols 2-3
    # B10: rows 2-3, cols 0-1
    # B11: rows 2-3, cols 2-3
    # Note: B is 4x7. B1 is the first 4 columns.
    
    # We define the 7 block products.
    # Each block product is a 2x2 matrix multiplication.
    # We use the get_strassen_2x2_terms() to get the local U, V, W for each.
    
    base_terms = get_strassen_2x2_terms()
    
    # Define the 7 block-level products
    # Each is a tuple of (u_block_coeffs, v_block_coeffs, w_block_coeffs)
    # u_block_coeffs: dict {block_idx: coeff}
    # v_block_coeffs: dict {block_idx: coeff}
    # w_block_coeffs: dict {block_idx: coeff}
    
    block_products = [
        ({'A00': 1, 'A11': 1}, {'B00': 1, 'B11': 1}, {'C00': 1, 'C01': 1, 'C10': 1, 'C11': 1}), # M1? No, W is handled by reconstruction.
        # Wait, the W in the 2x2 terms handles the reconstruction of the 2x2 C.
        # So for M1, the result is a 2x2 matrix M1_mat.
        # M1_mat is added to C blocks.
        # C00 += M1_mat
        # C01 += M1_mat
        # C10 += M1_mat
        # C11 += M1_mat
        # So for each rank-1 term in M1's decomposition, the W vector is scaled by the block's contribution to C.
        
    ]
    
    # Let's re-verify the Strassen reconstruction for 4x4.
    # C00 = M1 - M4 + M6 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M5 + M7
    
    # So for each block product Mi, we know which C blocks it contributes to and with what sign.
    
    mi_info = [
        # Mi, u_blocks, v_blocks, c_block_contributions
        (0, {'A00': 1, 'A11': 1}, {'B00': 1, 'B11': 1}, {'C00': 1, 'C01': 1, 'C10': 1, 'C11': 1}),
        (1, {'A10': 1, 'A11': 1}, {'B00': 1}, {'C10': 1, 'C11': 1}),
        (2, {'A00': 1}, {'B01': 1, 'B11': -1}, {'C01': 1, 'C11': -1}), # Wait, M3 = A00(B01-B11).
        # M3 contributes to C01 and C11?
        # C01 = M3 + M5.
        # C11 = M1 - M3 + M5 + M7.
        # So M3 contributes +1 to C01 and -1 to C11.
        # But M3 is a 2x2 matrix.
        # So for each rank-1 term in M3, W is scaled by +1 for C01 block and -1 for C11 block.
        (3, {'A11': 1}, {'B10': 1, 'B00': -1}, {'C00': -1, 'C10': 1}), # M4 = A11(B10-B00).
        # C00 -= M4, C10 += M4.
        (4, {'A00': 1, 'A01': 1}, {'B11': 1}, {'C01': 1, 'C11': 1}), # M5 = (A00+A01)B11.
        # C01 += M5, C11 += M5.
        (5, {'A10': 1, 'A00': -1}, {'B00': 1, 'B01': 1}, {'C00': 1}), # M6 = (A10-A00)(B00+B01).
        # C00 += M6.
        (6, {'A01': 1, 'A11': -1}, {'B10': 1, 'B11': 1}, {'C00': -1, 'C11': 1}), # M7 = (A01-A11)(B10+B11).
        # C00 -= M7, C11 += M7.
    ]
    
    # Map block names to index ranges in 4x4 matrices
    # A blocks:
    # A00: i in 0-1, k in 0-1
    # A01: i in 0-1, k in 2-3
    # A10: i in 2-3, k in 0-1
    # A11: i in 2-3, k in 2-3
    
    # B blocks (for B1, which is 4x4):
    # B00: k in 0-1, j in 0-1
    # B01: k in 0-1, j in 2-3
    # B10: k in 2-3, j in 0-1
    # B11: k in 2-3, j in 2-3
    
    # C blocks:
    # C00: i in 0-1, j in 0-1
    # C01: i in 0-1, j in 2-3
    # C10: i in 2-3, j in 0-1
    # C11: i in 2-3, j in 2-3
    
    for mi_idx, u_blocks, v_blocks, c_blocks in mi_info:
        # Get the 7 local rank-1 terms for this 2x2 product
        for local_u, local_v, local_w in base_terms:
            # local_u, local_v, local_w are length 4 vectors for 2x2 matrices.
            # We need to expand them to 4x4 (length 16 for U, 16 for V, 16 for W).
            
            u_4x4 = [0.0] * 16
            v_4x4 = [0.0] * 16
            w_4x4 = [0.0] * 16
            
            # Expand U
            # local_u[0] is A00 (block 0,0)
            # local_u[1] is A01 (block 0,1)
            # local_u[2] is A10 (block 1,0)
            # local_u[3] is A11 (block 1,1)
            
            # But u_blocks tells us which 4x4 blocks are involved and with what coeff.
            # Example: u_blocks = {'A00': 1, 'A11': 1}
            # This means the 2x2 matrix A_2x2 is formed by A00 and A11.
            # Wait, the local_u is for the 2x2 matrix that is being multiplied.
            # The 2x2 matrix A_2x2 has entries:
            # A_2x2[0,0] = sum of coeffs for A00? No.
            # The block product is M_i = (Linear Comb of A Blocks) * (Linear Comb of B Blocks).
            # The local_u vector corresponds to the coefficients of the 2x2 matrix A_2x2.
            # But A_2x2 is not a single 2x2 matrix from the original A.
            # A_2x2 is a 2x2 matrix where each entry is a 2x2 block from A.
            # No, Strassen works on 2x2 matrices of scalars.
            # For 4x4, we treat A as a 2x2 matrix of 2x2 blocks.
            # The "2x2 matrix" in the Strassen formula is the matrix of blocks.
            # But the multiplication is block multiplication.
            # (A00+A11) is a 2x2 matrix (sum of two 2x2 blocks).
            # So M1 is the product of two 2x2 matrices.
            # The local_u vector [1, 0, 0, 1] means:
            # The first entry of the 2x2 matrix (top-left) is 1.
            # The third entry (bottom-right) is 1.
            # So the 2x2 matrix is [[1, 0], [0, 1]].
            # This means we are multiplying (A00 + A11) by (B00 + B11).
            # Wait, if the local matrix is I, then the product is (A00+A11) * (B00+B11).
            # This is correct.
            
            # So, for each local rank-1 term, we need to map the local indices to global indices.
            # local_u[r] is the coefficient for the r-th entry of the 2x2 A-matrix.
            # The r-th entry of the 2x2 A-matrix corresponds to a block in A.
            # r=0 -> A00. r=1 -> A01. r=2 -> A10. r=3 -> A11.
            # But u_blocks tells us which blocks are actually used and with what scale.
            # Example: u_blocks = {'A00': 1, 'A11': 1}.
            # This implies that the 2x2 A-matrix for this product is:
            # [[A00, 0], [0, A11]]? No.
            # The formula is M1 = (A00+A11)(B00+B11).
            # This is not a standard matrix multiplication of two 2x2 matrices of blocks.
            # It is a product of two 2x2 matrices (scalars) where the "entries" are blocks?
            # No, Strassen for 4x4 uses 7 products of 2x2 matrices.
            # Each product is a standard 2x2 matrix multiplication.
            # The inputs to these 2x2 multiplications are linear combinations of the 2x2 blocks.
            # So, for M1, the input A-matrix is:
            # A_M1 = [[A00, 0], [0, A11]]? No.
            # A_M1[0,0] = A00. A_M1[0,1] = 0. A_M1[1,0] = 0. A_M1[1,1] = A11.
            # Then A_M1 * B_M1 where B_M1[0,0]=B00, B_M1[1,1]=B11.
            # Result M1 = [[A00B00, 0], [0, A11B11]].
            # But Strassen M1 is (A00+A11)(B00+B11) which is a scalar product?
            # No, for 2x2 matrices, M1 = (A00+A11)(B00+B11) is a product of two scalars.
            # For 4x4 matrices, A00 is a 2x2 matrix.
            # So (A00+A11) is a 2x2 matrix. (B00+B11) is a 2x2 matrix.
            # Their product is a 2x2 matrix.
            # So M1 is a 2x2 matrix.
            # The decomposition of M1 into rank-1 terms involves the entries of (A00+A11) and (B00+B11).
            
            # So, local_u[0] corresponds to the (0,0) entry of the 2x2 A-matrix.
            # The (0,0) entry of the 2x2 A-matrix is A00 (a 2x2 block).
            # So local_u[0] scales the entire block A00.
            
            # Therefore:
            # For r in 0..3:
            #   if local_u[r] != 0:
            #     Find the block name for r.
            #     Get the scale from u_blocks.
            #     Add local_u[r] * scale to the corresponding entries in u_4x4.
            
            # Block mapping for A:
            # 0: A00 (rows 0-1, cols 0-1)
            # 1: A01 (rows 0-1, cols 2-3)
            # 2: A10 (rows 2-3, cols 0-1)
            # 3: A11 (rows 2-3, cols 2-3)
            
            a_block_map = {
                0: (0, 0, 2, 2), # r_start, c_start, r_end, c_end
                1: (0, 2, 2, 4),
                2: (2, 0, 4, 2),
                3: (2, 2, 4, 4)
            }
            
            b_block_map = {
                0: (0, 0, 2, 2), # k_start, j_start, k_end, j_end
                1: (0, 2, 2, 4),
                2: (2, 0, 4, 2),
                3: (2, 2, 4, 4)
            }
            
            c_block_map = {
                0: (0, 0, 2, 2), # i_start, j_start, i_end, j_end
                1: (0, 2, 2, 4),
                2: (2, 0, 4, 2),
                3: (2, 2, 4, 4)
            }
            
            for r in range(4):
                if local_u[r] != 0:
                    # Find which block this local index corresponds to
                    # This is tricky because u_blocks might not have all blocks.
                    # Actually, u_blocks defines the 2x2 A-matrix for this product.
                    # The local_u vector is the coefficient vector for the rank-1 term of the 2x2 multiplication.
                    # The 2x2 A-matrix is fixed by u_blocks.
                    # So local_u[r] multiplies the r-th entry of the 2x2 A-matrix.
                    # The r-th entry of the 2x2 A-matrix is a block (or 0).
                    # We need to know which block is at position r.
                    # The position r in a 2x2 matrix is:
                    # r=0: (0,0)
                    # r=1: (0,1)
                    # r=2: (1,0)
                    # r=3: (1,1)
                    
                    # We need to map (row, col) of the 2x2 block matrix to the block name.
                    # (0,0) -> A00
                    # (0,1) -> A01
                    # (1,0) -> A10
                    # (1,1) -> A11
                    
                    block_name = ['A00', 'A01', 'A10', 'A11'][r]
                    scale = u_blocks.get(block_name, 0)
                    if scale != 0:
                        r_s, c_s, r_e, c_e = a_block_map[r]
                        for i in range(r_s, r_e):
                            for k in range(c_s, c_e):
                                idx = i * 4 + k
                                u_4x4[idx] += local_u[r] * scale
            
            for r in range(4):
                if local_v[r] != 0:
                    block_name = ['B00', 'B01', 'B10', 'B11'][r]
                    scale = v_blocks.get(block_name, 0)
                    if scale != 0:
                        k_s, j_s, k_e, j_e = b_block_map[r]
                        for k in range(k_s, k_e):
                            for j in range(j_s, j_e):
                                idx = k * 7 + j # p=7
                                v_4x4[idx] += local_v[r] * scale
            
            for r in range(4):
                if local_w[r] != 0:
                    block_name = ['C00', 'C01', 'C10', 'C11'][r]
                    scale = c_blocks.get(block_name, 0)
                    if scale != 0:
                        i_s, j_s, i_e, j_e = c_block_map[r]
                        for i in range(i_s, i_e):
                            for j in range(j_s, j_e):
                                idx = j * 4 + i # n=4
                                w_4x4[idx] += local_w[r] * scale
            
            # Add to global U, V, W
            for idx in range(16):
                U[idx][col] = u_4x4[idx]
            for idx in range(16):
                V[idx][col] = v_4x4[idx]
            for idx in range(16):
                W[idx][col] = w_4x4[idx]
            col += 1
            
    # Part 2: Schoolbook for 4x4x3
    # B2 is cols 4-6 of B.
    # j in 4,5,6.
    for i in range(4):
        for k in range(4):
            for j in range(4, 7):
                U[i * 4 + k][col] = 1
                V[k * 7 + j][col] = 1
                W[j * 4 + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
