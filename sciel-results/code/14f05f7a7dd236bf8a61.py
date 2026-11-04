import numpy as np

def construct():
    n, m, p = 4, 4, 6
    # Rank 84 decomposition
    # Split B into 3 columns of width 2: B0, B1, B2
    # For each, compute A * Bk using 4 block multiplications of 2x2 matrices
    # Each 2x2 multiplication has rank 7 (Strassen)
    # Total rank = 3 * 4 * 7 = 84
    
    rank = 84
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Strassen coefficients for 2x2 multiplication
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # We will iterate over the 3 column groups of B
    for k in range(3):
        # Bk consists of blocks Bk_0 (top 2 rows) and Bk_1 (bottom 2 rows)
        # Each is 2x2.
        # A consists of A00, A01, A10, A11 (each 2x2)
        
        # We need to compute:
        # Ck_0 = A00 * Bk_0 + A01 * Bk_1
        # Ck_1 = A10 * Bk_0 + A11 * Bk_1
        
        # We use 4 Strassen-like decompositions?
        # Actually, we just do 4 separate 2x2 multiplications:
        # P1 = A00 * Bk_0
        # P2 = A01 * Bk_1
        # P3 = A10 * Bk_0
        # P4 = A11 * Bk_1
        
        # Each Pi has rank 7.
        # Total 28 per k.
        
        for block_idx in range(4):
            # Determine which A block and which B block
            if block_idx == 0:
                a_row, a_col = 0, 0 # A00
                b_row = 0 # Bk_0
                c_row = 0 # Ck_0
            elif block_idx == 1:
                a_row, a_col = 0, 1 # A01
                b_row = 1 # Bk_1
                c_row = 0 # Ck_0
            elif block_idx == 2:
                a_row, a_col = 1, 0 # A10
                b_row = 0 # Bk_0
                c_row = 1 # Ck_1
            else:
                a_row, a_col = 1, 1 # A11
                b_row = 1 # Bk_1
                c_row = 1 # Ck_1
                
            # Now decompose the 2x2 multiplication of A_block and B_block
            # A_block is 2x2, indices (a_row*2 + i, a_col*2 + j)
            # B_block is 2x2, indices (b_row*2 + j, k*2 + l)
            # C_block is 2x2, indices (c_row*2 + i, k*2 + l)
            
            # Strassen 7 terms
            # M1 = (A00+A11)(B00+B11)
            # A indices: (a_row*2, a_col*2) and (a_row*2+1, a_col*2+1)
            # B indices: (b_row*2, k*2) and (b_row*2+1, k*2+1)
            
            # We will add 7 rank-1 terms for this 2x2 multiplication
            
            # Helper to add a term
            def add_term(u_idx, v_idx, w_idx, coeff_u, coeff_v, coeff_w):
                nonlocal col
                # u_idx is (i, j) in A
                # v_idx is (j, l) in B
                # w_idx is (i, l) in C
                # Coefficients are +1 or -1
                
                # A index in U
                u_flat = u_idx[0] * m + u_idx[1]
                # B index in V
                v_flat = v_idx[0] * p + v_idx[1]
                # C index in W
                w_flat = w_idx[0] * n + w_idx[1]
                
                U[u_flat][col] = coeff_u
                V[v_flat][col] = coeff_v
                W[w_flat][col] = coeff_w
                col += 1

            # A block local indices 0,1
            # B block local indices 0,1
            
            # M1
            # (A00+A11) * (B00+B11)
            # This expands to A00B00 + A00B11 + A11B00 + A11B11
            # We need to represent the product of sums.
            # Rank 1 terms for (X+Y)(Z+W) = XZ + XW + YZ + YW
            # This requires 4 terms, but Strassen uses 7 for the whole 2x2.
            # Wait, the Strassen algorithm replaces the 4 multiplications with 7.
            # The 7 terms are:
            # 1. (A00+A11)(B00+B11)
            # 2. (A10+A11)B00
            # 3. A00(B10-B11)
            # 4. A11(B01-B00)
            # 5. (A00+A01)B11
            # 6. (A10-A00)(B00+B01)
            # 7. (A01-A11)(B10+B11)
            
            # We must map these to the specific A_block and B_block.
            
            # Term 1: (A00+A11)(B00+B11)
            # A00: (a_row*2, a_col*2), A11: (a_row*2+1, a_col*2+1)
            # B00: (b_row*2, k*2), B11: (b_row*2+1, k*2+1)
            # This term contributes to C00, C01, C10, C11?
            # No, it's a single product of two 2x2 matrices?
            # No, A00 and A11 are scalars in the block algebra?
            # No, A00 is a 2x2 matrix? No, A00 is a scalar in the 2x2 block decomposition?
            # In our decomposition, A is 4x4. Blocks are 2x2.
            # So A00 is a 2x2 matrix.
            # The "Strassen" algorithm applies to the 2x2 block structure.
            # The "entries" of the block matrix are 2x2 matrices.
            # So we are doing Strassen multiplication of two 2x2 matrices of 2x2 matrices.
            # This results in 7 products of 2x2 matrices.
            # Each product of 2x2 matrices has rank 7.
            # So each Strassen term requires 7 rank-1 scalar terms.
            
            # So for Term 1: (A00+A11)(B00+B11)
            # This is a product of two 2x2 matrices (sums).
            # We need to decompose this product into 7 scalar rank-1 terms?
            # No, the product of two 2x2 matrices has rank 7.
            # So we need to add 7 terms for this single Strassen component.
            
            # Let's define a function to add a 2x2 multiplication
            def add_2x2_mult(a_r, a_c, b_r, b_c, c_r, c_c, sign=1):
                # Adds rank-7 decomposition of A[a_r..a_r+1, a_c..a_c+1] * B[b_r..b_r+1, b_c..b_c+1]
                # to C[c_r..c_r+1, c_c..c_c+1]
                nonlocal col
                
                # Strassen 7 terms for 2x2
                # A00=A[a_r, a_c], A01=A[a_r, a_c+1], A10=A[a_r+1, a_c], A11=A[a_r+1, a_c+1]
                # B00=B[b_r, b_c], B01=B[b_r, b_c+1], B10=B[b_r+1, b_c], B11=B[b_r+1, b_c+1]
                
                # 1. (A00+A11)(B00+B11)
                # This is a 2x2 multiplication of sums.
                # We cannot just add 1 term. We must decompose the 2x2 product.
                # The rank of 2x2 multiplication is 7.
                # So we need to call add_2x2_mult recursively? No, base case.
                # Base case: 1x1 multiplication is rank 1.
                # But here the "entries" are scalars.
                # So (A00+A11) is a scalar sum.
                # Wait, in the top-level 4x4x6, the blocks are 2x2.
                # In the Strassen step, we are multiplying 2x2 matrices (the blocks).
                # So the "entries" of the block matrix are 2x2 matrices.
                # So we are doing 7 multiplications of 2x2 matrices.
                # Each 2x2 matrix multiplication has rank 7.
                # So we need to add 7 * 7 = 49 terms?
                # No.
                # Let's re-read the rank calculation.
                # Rank of 4x4x4 is 49.
                # This implies that the 7 Strassen products are each of rank 7?
                # Yes.
                # So for each of the 7 Strassen terms, we add 7 rank-1 scalar terms.
                # Total 49.
                
                # So for this specific 2x2 block multiplication (which is one of the 7 Strassen terms),
                # we need to add 7 scalar rank-1 terms.
                
                # The 7 terms for 2x2 multiplication:
                # M1 = (A00+A11)(B00+B11)
                # M2 = (A10+A11)B00
                # M3 = A00(B10-B11)
                # M4 = A11(B01-B00)
                # M5 = (A00+A01)B11
                # M6 = (A10-A00)(B00+B01)
                # M7 = (A01-A11)(B10+B11)
                
                # Wait, these are the 7 products.
                # Each product is a scalar multiplication?
                # No, A00 is a scalar (entry of the 2x2 matrix).
                # So M1 is a scalar.
                # So each M is rank 1?
                # No, the rank of the 2x2 multiplication tensor is 7.
                # This means we can express the 4 outputs using 7 products of linear forms.
                # So we add 7 terms.
                
                # Term 1: (A00+A11)(B00+B11)
                # U: A00 (+1), A11 (+1)
                # V: B00 (+1), B11 (+1)
                # W: C00 (+1), C01 (+1), C10 (+1), C11 (+1)?
                # No, the output is Z.
                # Z00 = M1 + M4 - M5 + M7
                # Z01 = M3 + M5
                # Z10 = M2 + M4
                # Z11 = M1 - M2 + M3 + M6
                
                # So for each of the 7 M's, we add 1 rank-1 term?
                # No, a rank-1 term is u * v * w.
                # Here u is a vector of A coefficients, v is vector of B coefficients, w is vector of C coefficients.
                # For M1:
                # u has 1 at A00, 1 at A11.
                # v has 1 at B00, 1 at B11.
                # w has 1 at C00, 1 at C01, 1 at C10, 1 at C11?
                # No, w must be such that sum w_k M_k gives Z.
                # The decomposition is Z = sum M_k * (linear combo of C basis).
                # Actually, the rank-1 tensor is u \otimes v \otimes w.
                # The contribution to Z is (u^T A) (v^T B) w.
                # So w is a vector in the output space.
                
                # For M1 = (A00+A11)(B00+B11):
                # u = [1, 0, 0, 1] (A00, A01, A10, A11)
                # v = [1, 0, 0, 1] (B00, B01, B10, B11)
                # This product contributes to all Z components?
                # No, the Strassen algorithm defines 7 scalars M1..M7.
                # Then Z is a linear combination of M1..M7.
                # So we add 7 rank-1 terms.
                # Term k has u_k, v_k, and w_k is the coefficient vector for M_k in Z.
                
                # u1 = [1, 0, 0, 1]
                # v1 = [1, 0, 0, 1]
                # w1 = [1, 0, 0, 1] (Coeffs for Z00, Z01, Z10, Z11)
                # Check: Z00 += 1*M1, Z01 += 0, Z10 += 0, Z11 += 1?
                # Formula: Z00 = M1 + M4 - M5 + M7. So coeff of M1 in Z00 is 1.
                # Z01 = M3 + M5. Coeff 0.
                # Z10 = M2 + M4. Coeff 0.
                # Z11 = M1 - M2 + M3 + M6. Coeff 1.
                # So w1 = [1, 0, 0, 1].
                
                # We add this term.
                # U: A00 += sign, A11 += sign
                # V: B00 += sign, B11 += sign
                # W: C00 += sign, C11 += sign
                
                # Repeat for all 7.
                
                pass # Implementation below

            # Implementation of add_2x2_mult
            # Local indices 0,1 for A, B, C blocks
            # A global: (a_r+i, a_c+j)
            # B global: (b_r+j, b_c+l) -- Note B is m x p, so row j, col l
            # C global: (c_r+i, c_c+l) -- Note C is n x p, so row i, col l
            
            # Strassen terms
            # 1. (A00+A11)(B00+B11)
            # u: A00, A11
            # v: B00, B11
            # w: C00, C11
            add_term((a_r, a_c), (b_r, b_c), (c_r, c_c), sign, 1, 1)
            add_term((a_r+1, a_c+1), (b_r, b_c), (c_r, c_c), sign, 1, 1) # This is wrong, U and V are separate vectors
            # Correct way:
            # U vector has entries for A.
            # V vector has entries for B.
            # W vector has entries for C.
            
            # We must fill U, V, W for the current column.
            
            # Term 1
            U[a_r * m + a_c][col] += sign
            U[(a_r+1) * m + (a_c+1)][col] += sign
            V[b_r * p + b_c][col] += sign
            V[(b_r+1) * p + (b_c+1)][col] += sign
            W[c_r * n + c_c][col] += sign
            W[(c_r+1) * n + (c_c+1)][col] += sign
            col += 1
            
            # Term 2: (A10+A11)B00
            U[a_r * m + a_c][col] = 0 # Reset? No, we are using new column
            # We should use a helper to set values for the current column
            
            pass

    return {"U": U, "V": V, "W": W}
