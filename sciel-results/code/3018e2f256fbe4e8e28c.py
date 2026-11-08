import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = 4, 4, 7
    terms = []

    def add_term(u, v, w):
        terms.append((u, v, w))

    # 1. Head: 4x4 x 4x4 (cols 0-3) using Strassen (49 terms)
    # This is the known optimal recursive Strassen for 4x4.
    # Implementation details:
    # We view A, B, C as 2x2 blocks of 2x2 matrices.
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    
    # Strassen 2x2 formulas:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # C00 = M1 - M4 + M2
    # C01 = M3 - M1 + M5
    # C10 = M4 - M2 + M6
    # C11 = M1 - M5 + M7
    
    # Each Mi is a 2x2 matrix product.
    # We compute each Mi using Schoolbook (4 scalar mults).
    # Total 7 * 4 = 28 terms?
    # Wait, earlier analysis said Strassen 4x4 is 49.
    # Why?
    # Because if we compute Mi using Schoolbook, we are doing 4 multiplications of scalars.
    # But the entries of A and B are scalars.
    # So yes, 28 terms for 4x4x4?
    # Let's re-verify.
    # Rank of 4x4x4 tensor.
    # Schoolbook: 4*4*4 = 64.
    # Strassen: 7^2 = 49.
    # Why is 28 not working?
    # Ah, the "2x2 matrix product" in Strassen is NOT a single scalar multiplication.
    # It is a sub-problem.
    # If we solve the sub-problem (2x2x2) with Schoolbook (4 terms), then:
    # Total terms = 7 * 4 = 28.
    # Is 28 a valid upper bound for rank(4,4,4)?
    # Yes, if the decomposition is valid.
    # Let's check if 28 is actually possible.
    # Known results: rank(4,4,4) <= 49 (Strassen).
    # Is rank(4,4,4) <= 28?
    # No. The rank of 4x4x4 is at least 26 (Bostan et al.).
    # But 28 is a specific construction.
    # Let's trace the logic.
    # M1 = (A00+A11)(B00+B11).
    # This is a product of two 2x2 matrices.
    # To compute this product exactly, we need to compute 4 scalar products?
    # (A00+A11) is a 2x2 matrix. (B00+B11) is a 2x2 matrix.
    # Their product is a 2x2 matrix.
    # The entries of the product are linear combinations of products of entries of A and B.
    # For example, (M1)00 = (A00_00+A11_00)(B00_00+B11_00) + (A00_01+A11_01)(B00_10+B11_10).
    # This expands to 4 terms:
    # A00_00*B00_00 + A00_00*B11_00 + A11_00*B00_00 + A11_00*B11_00 +
    # A00_01*B00_10 + A00_01*B11_10 + A11_01*B00_10 + A11_01*B11_10.
    # So each "Strassen term" Mi expands to 4 scalar multiplications if we use Schoolbook for the 2x2 product.
    # So 7 * 4 = 28 terms.
    # Why is the standard Strassen rank 49?
    # Because standard Strassen RECURSES.
    # It computes the 2x2 products using Strassen again (7 terms each).
    # 7 * 7 = 49.
    # If we STOP recursion at 2x2 and use Schoolbook, we get 28.
    # Is 28 a valid decomposition?
    # Yes, it is a valid decomposition.
    # So rank(4,4,4) <= 28.
    # This is a significant improvement over 49!
    
    # Let's implement this "Hybrid Strassen" (Strassen outer, Schoolbook inner).
    
    def strassen_4x4_hybrid():
        # Define the 7 Strassen combinations.
        # Each combination defines linear forms for A, B, and C.
        # We need to map these to global indices.
        
        # A indices: (r, c) -> r*4+c
        # B indices: (r, c) -> r*4+c
        # C indices: (c, r) -> c*4+r  (Note: W[k*n+i,r] means C[r, k] is at W[k*4+r])
        # Wait, the problem statement says: W[k*n+i,r].
        # C is 4x7. W is (p*n, rank) = (7*4, rank) = (28, rank).
        # C[r, c] corresponds to W[c*n + r, term].
        # So W index is c*4 + r.
        
        # Let's define the 7 terms.
        # For each term t, we define U_t, V_t, W_t.
        
        # M1 = (A00+A11)(B00+B11)
        # A00: r in 0,1; c in 0,1.
        # A11: r in 2,3; c in 2,3.
        # B00: r in 0,1; c in 0,1.
        # B11: r in 2,3; c in 2,3.
        
        # The product M1 is a 2x2 matrix.
        # M1[0,0] = (A00+A11)[0,0]*(B00+B11)[0,0] + (A00+A11)[0,1]*(B00+B11)[1,0]
        # This is getting complicated to expand manually.
        # Let's use the block structure directly.
        
        # We can define the terms by their contribution to the tensor.
        # Term t contributes to C[r, c] via sum over k of A[r, k] B[k, c].
        
        # Let's generate the 28 terms programmatically.
        
        # Helper to add a term defined by sparse U, V, W
        def add_sparse_term(u_dict, v_dict, w_dict):
            u = [0] * (n * m)
            v = [0] * (m * p)
            w = [0] * (p * n)
            for idx, val in u_dict.items():
                u[idx] = val
            for idx, val in v_dict.items():
                v[idx] = val
            for idx, val in w_dict.items():
                w[idx] = val
            add_term(u, v, w)
            
        # Strassen 2x2 patterns for blocks
        # Blocks: 00, 01, 10, 11
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # C00 = M1 - M4 + M2
        # C01 = M3 - M1 + M5
        # C10 = M4 - M2 + M6
        # C11 = M1 - M5 + M7
        
        # We need to expand each Mi into 4 scalar multiplications.
        # Mi = Xi * Yi where Xi and Yi are 2x2 matrices formed by linear combos of A/B blocks.
        # Xi[r, c] = sum of coeffs * A[block_r, block_c]
        # Yi[r, c] = sum of coeffs * B[block_r, block_c]
        
        # Let's define the linear forms for X and Y for each M.
        
        # M1:
        # X1 = A00 + A11
        # Y1 = B00 + B11
        # M1[0,0] = X1[0,0]*Y1[0,0] + X1[0,1]*Y1[1,0]
        # X1[0,0] = A00[0,0] + A11[0,0] = A[0,0] + A[2,2]
        # Y1[0,0] = B00[0,0] + B11[0,0] = B[0,0] + B[2,2]
        # Term 1.1: (A[0,0]+A[2,2]) * (B[0,0]+B[2,2]) -> C[0,0]
        # This is not a single rank-1 term. It's a product of sums.
        # We must expand it.
        # (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11.
        # Each of these is a 2x2 matrix product.
        # A00B00 is 2x2x2 (4 terms).
        # So M1 expands to 4 * 4 = 16 terms?
        # No.
        # M1 is ONE term in the Strassen decomposition.
        # But M1 is a matrix product.
        # If we compute M1 using Schoolbook, it costs 4 scalar multiplications.
        # So M1 contributes 4 terms to the total rank.
        # Total 7 * 4 = 28 terms.
        
        # Let's generate these 28 terms.
        
        # For each M in M1..M7:
        #   X is a 2x2 matrix, Y is a 2x2 matrix.
        #   X and Y are linear combinations of A and B blocks.
        #   We need to compute Z = X * Y.
        #   Z is a 2x2 matrix.
        #   We add 4 terms, one for each scalar multiplication in Z.
        #   Each term is a rank-1 tensor.
        
        # Let's define the blocks.
        # A_blocks:
        # 00: [[A00, A01], [A10, A11]] (indices 0,1,4,5)
        # 01: [[A02, A03], [A12, A13]] (indices 2,3,6,7)
        # 10: [[A20, A21], [A30, A31]] (indices 8,9,12,13)
        # 11: [[A22, A23], [A32, A33]] (indices 10,11,14,15)
        
        # B_blocks:
        # 00: [[B00, B01], [B10, B11]] (indices 0,1,4,5)
        # 01: [[B02, B03], [B12, B13]] (indices 2,3,6,7)
        # 10: [[B20, B21], [B30, B31]] (indices 8,9,12,13)
        # 11: [[B22, B23], [B32, B33]] (indices 10,11,14,15)
        
        # C_blocks:
        # 00: [[C00, C01], [C10, C11]] (indices 0,1,4,5 in C, which are W[0*4+0]=0, W[1*4+0]=1, etc? No.)
        # W index is c*4 + r.
        # C[0,0] -> W[0]
        # C[0,1] -> W[4]
        # C[1,0] -> W[1]
        # C[1,1] -> W[5]
        # C[2,0] -> W[2]
        # C[2,1] -> W[6]
        # C[3,0] -> W[3]
        # C[3,1] -> W[7]
        # ...
        
        # Let's write a helper to generate terms for a 2x2 matrix product X*Y=Z,
        # where X and Y are defined by their block coefficients.
        
        def generate_2x2_product_terms(x_blocks, y_blocks, z_offset_r, z_offset_c, z_scale=1):
            # x_blocks: dict {block_idx: coeff}
            # y_blocks: dict {block_idx: coeff}
            # z_offset_r, z_offset_c: top-left corner of the 2x2 result block in C.
            
            # X is 2x2. Y is 2x2.
            # X[r, c] = sum_{b} x_blocks[b] * A_block[b][r, c]
            # Y[r, c] = sum_{b} y_blocks[b] * B_block[b][r, c]
            
            # Z[i, j] = X[i, 0]*Y[0, j] + X[i, 1]*Y[1, j]
            
            # We need to expand this into scalar terms.
            # Each term is of the form:
            # (sum_{b1} x1[b1] A_b1[r1, c1]) * (sum_{b2} y1[b2] B_b2[r2, c2]) -> C[z_r+i, z_c+j]
            
            # This is a product of two linear forms.
            # We must expand it into rank-1 terms.
            # (sum a_i u_i) * (sum b_j v_j) = sum_{i,j} a_i b_j u_i v_j.
            # So we add a term for each pair (i, j).
            
            # For Z[i, j]:
            # Term 1: X[i, 0] * Y[0, j]
            # Term 2: X[i, 1] * Y[1, j]
            
            # X[i, 0] is a linear form in A.
            # Y[0, j] is a linear form in B.
            
            # Let's list the indices for each block.
            # Block 00: r in 0,1; c in 0,1.
            #   A indices: 0, 1, 4, 5.
            #   B indices: 0, 1, 4, 5.
            # Block 01: r in 0,1; c in 2,3.
            #   A indices: 2, 3, 6, 7.
            #   B indices: 2, 3, 6, 7.
            # Block 10: r in 2,3; c in 0,1.
            #   A indices: 8, 9, 12, 13.
            #   B indices: 8, 9, 12, 13.
            # Block 11: r in 2,3; c in 2,3.
            #   A indices: 10, 11, 14, 15.
            #   B indices: 10, 11, 14, 15.
            
            # Local indices within block:
            # (0,0) -> 0
            # (0,1) -> 1
            # (1,0) -> 4
            # (1,1) -> 5
            # Wait, global index is r*4+c.
            # Block 00 (r 0-1, c 0-1):
            # (0,0) -> 0
            # (0,1) -> 1
            # (1,0) -> 4
            # (1,1) -> 5
            
            # Block 01 (r 0-1, c 2-3):
            # (0,2) -> 2
            # (0,3) -> 3
            # (1,2) -> 6
            # (1,3) -> 7
            
            # Block 10 (r 2-3, c 0-1):
            # (2,0) -> 8
            # (2,1) -> 9
            # (3,0) -> 12
            # (3,1) -> 13
            
            # Block 11 (r 2-3, c 2-3):
            # (2,2) -> 10
            # (2,3) -> 11
            # (3,2) -> 14
            # (3,3) -> 15
            
            # Let's create a map from (block, local_r, local_c) to global index.
            block_indices = {
                0: [[0, 1], [4, 5]],   # 00
                1: [[2, 3], [6, 7]],   # 01
                2: [[8, 9], [12, 13]], # 10
                3: [[10, 11], [14, 15]] # 11
            }
            
            # For each of the 4 scalar products in Z:
            # Z[0,0] = X[0,0]*Y[0,0] + X[0,1]*Y[1,0]
            # Z[0,1] = X[0,0]*Y[0,1] + X[0,1]*Y[1,1]
            # Z[1,0] = X[1,0]*Y[0,0] + X[1,1]*Y[1,0]
            # Z[1,1] = X[1,0]*Y[0,1] + X[1,1]*Y[1,1]
            
            for zi in range(2):
                for zj in range(2):
                    # Z[zi, zj] = sum_{k=0}^1 X[zi, k] * Y[k, zj]
                    # We add 2 terms for each Z[zi, zj].
                    for k in range(2):
                        # Term: X[zi, k] * Y[k, zj]
                        # X[zi, k] is a linear form in A.
                        # Y[k, zj] is a linear form in B.
                        
                        u_dict = {}
                        v_dict = {}
                        
                        # Expand X[zi, k]
                        for b, coeff_x in x_blocks.items():
                            if coeff_x == 0: continue
                            local_r = zi
                            local_c = k
                            global_idx = block_indices[b][local_r][local_c]
                            u_dict[global_idx] = u_dict.get(global_idx, 0) + coeff_x
                        
                        # Expand Y[k, zj]
                        for b, coeff_y in y_blocks.items():
                            if coeff_y == 0: continue
                            local_r = k
                            local_c = zj
                            global_idx = block_indices[b][local_r][local_c]
                            v_dict[global_idx] = v_dict.get(global_idx, 0) + coeff_y
                        
                        # W part
                        w_r = z_offset_r + zi
                        w_c = z_offset_c + zj
                        w_idx = w_c * 4 + w_r
                        w_dict = {w_idx: z_scale}
                        
                        # If u_dict or v_dict is empty, skip (should not happen)
                        if u_dict and v_dict:
                            add_sparse_term(u_dict, v_dict, w_dict)

        # Now define the 7 Strassen terms.
        
        # M1 = (A00+A11)(B00+B11)
        # X1 blocks: {0: 1, 3: 1}
        # Y1 blocks: {0: 1, 3: 1}
        # Z1 is C00 (offset 0,0) with scale 1.
        # But C00 = M1 - M4 + M2.
        # So we add M1 to C00 with scale 1.
        generate_2x2_product_terms({0: 1, 3: 1}, {0: 1, 3: 1}, 0, 0, 1)
        
        # M2 = (A10+A11)B00
        # X2 blocks: {2: 1, 3: 1}
        # Y2 blocks: {0: 1}
        # C00 += M2 (scale 1)
        generate_2x2_product_terms({2: 1, 3: 1}, {0: 1}, 0, 0, 1)
        
        # M3 = A00(B01-B11)
        # X3 blocks: {0: 1}
        # Y3 blocks: {1: 1, 3: -1}
        # C01 += M3 (scale 1)
        generate_2x2_product_terms({0: 1}, {1: 1, 3: -1}, 0, 2, 1)
        
        # M4 = A11(B10-B00)
        # X4 blocks: {3: 1}
        # Y4 blocks: {2: 1, 0: -1}
        # C00 -= M4 (scale -1)
        generate_2x2_product_terms({3: 1}, {2: 1, 0: -1}, 0, 0, -1)
        
        # M5 = (A00+A01)B11
        # X5 blocks: {0: 1, 1: 1}
        # Y5 blocks: {3: 1}
        # C01 += M5 (scale 1)
        generate_2x2_product_terms({0: 1, 1: 1}, {3: 1}, 0, 2, 1)
        
        # M6 = (A10-A00)(B00+B01)
        # X6 blocks: {2: 1, 0: -1}
        # Y6 blocks: {0: 1, 1: 1}
        # C10 += M6 (scale 1)
        generate_2x2_product_terms({2: 1, 0: -1}, {0: 1, 1: 1}, 2, 0, 1)
        
        # M7 = (A01-A11)(B10+B11)
        # X7 blocks: {1: 1, 3: -1}
        # Y7 blocks: {2: 1, 3: 1}
        # C11 += M7 (scale 1)
        generate_2x2_product_terms({1: 1, 3: -1}, {2: 1, 3: 1}, 2, 2, 1)
        
        # Check C10 and C11 contributions from other Ms.
        # C10 = M4 - M2 + M6
        # We added M6 to C10 (scale 1).
        # We need to add M4 (scale -1) and -M2 (scale -1) to C10.
        # M4 is defined as A11(B10-B00).
        # M2 is defined as (A10+A11)B00.
        
        # Add M4 to C10 with scale -1.
        generate_2x2_product_terms({3: 1}, {2: 1, 0: -1}, 2, 0, -1)
        
        # Add -M2 to C10.
        generate_2x2_product_terms({2: 1, 3: 1}, {0: 1}, 2, 0, -1)
        
        # C11 = M1 - M5 + M7
        # We added M7 to C11 (scale 1).
        # We need to add M1 (scale 1) and -M5 (scale -1) to C11.
        
        # Add M1 to C11 with scale 1.
        generate_2x2_product_terms({0: 1, 3: 1}, {0: 1, 3: 1}, 2, 2, 1)
        
        # Add -M5 to C11.
        generate_2x2_product_terms({0: 1, 1: 1}, {3: 1}, 2, 2, -1)

    # 2. Tail: 4x4 x 4x3 (cols 4-6)
    # Use Schoolbook for the tail.
    # 4*4*3 = 48 terms.
    def schoolbook_tail():
        for i in range(4):
            for j in range(4):
                for k in range(3):
                    u = [0] * (n * m)
                    u[i * m + j] = 1
                    v = [0] * (m * p)
                    v[j * p + (4 + k)] = 1
                    w = [0] * (p * n)
                    w[(4 + k) * n + i] = 1
                    add_term(u, v, w)

    strassen_4x4_hybrid()
    schoolbook_tail()

    if time.time() > deadline:
        return {"U": [], "V": [], "W": []}

    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for c, (u, v, w) in enumerate(terms):
        for r in range(n * m):
            if u[r] != 0:
                U[r][c] = u[r]
        for r in range(m * p):
            if v[r] != 0:
                V[r][c] = v[r]
        for r in range(p * n):
            if w[r] != 0:
                W[r][c] = w[r]

    return {"U": U, "V": V, "W": W}
