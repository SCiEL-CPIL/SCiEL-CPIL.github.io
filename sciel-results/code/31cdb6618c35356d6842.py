import numpy as np

def strassen_2x2_decomp():
    # Returns U, V, W for 2x2 matrix multiplication with rank 7
    # Indices for A: 0,1,2,3 (a,b,c,d)
    # Indices for B: 0,1,2,3 (e,f,g,h)
    # Indices for C: 0,1,2,3 (c00, c01, c10, c11)
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (a+d)(e+h)
    # a: U[0,0]=1, d: U[3,0]=1
    # e: V[0,0]=1, h: V[3,0]=1
    # c00: W[0,0]=1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1
    
    # M2 = (c+d)e
    # c: U[2,1]=1, d: U[3,1]=1
    # e: V[0,1]=1
    # c10: W[2,1]=1
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1
    
    # M3 = a(f-h)
    # a: U[0,2]=1
    # f: V[1,2]=1, h: V[3,2]=-1
    # c01: W[1,2]=1
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1
    
    # M4 = d(g-e)
    # d: U[3,3]=1
    # g: V[2,3]=1, e: V[0,3]=-1
    # c00: W[0,3]=1, c10: W[2,3]=1
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5 = (a+b)h
    # a: U[0,4]=1, b: U[1,4]=1
    # h: V[3,4]=1
    # c00: W[0,4]=-1, c01: W[1,4]=1
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # M6 = (c-a)(e+f)
    # c: U[2,5]=1, a: U[0,5]=-1
    # e: V[0,5]=1, f: V[1,5]=1
    # c11: W[3,5]=1
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (b-d)(g+h)
    # b: U[1,6]=1, d: U[3,6]=-1
    # g: V[2,6]=1, h: V[3,6]=1
    # c00: W[0,6]=1, c11: W[3,6]=-1 (Wait, C11 = M1 - M2 + M3 + M6. C00 = M1 + M4 - M5 + M7)
    # Let's re-verify C00 and C11 coefficients.
    # C00 = M1 + M4 - M5 + M7
    # C11 = M1 - M2 + M3 + M6
    
    # M7 contributes to C00 (+1) and C11 (-1)?
    # Standard Strassen:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # So M7 is only in C00?
    # Wait, M7 = (b-d)(g+h).
    # (b-d)(g+h) = bg + bh - dg - dh.
    # C00 = ae + bg.
    # C11 = ch + dg.
    # M1 = ae + ah + de + dh
    # M4 = dg - de
    # M5 = ah + bh
    # M7 = bg + bh - dg - dh
    # M1 + M4 - M5 + M7 = (ae+ah+de+dh) + (dg-de) - (ah+bh) + (bg+bh-dg-dh)
    # = ae + ah + de + dh + dg - de - ah - bh + bg + bh - dg - dh
    # = ae + bg. Correct.
    
    # M7 coefficient in C00 is 1.
    # M7 coefficient in C11 is 0.
    
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    # W[3][6] remains 0.
    
    return U, V, W

def construct():
    # Get 2x2 decomposition
    U2, V2, W2 = strassen_2x2_decomp()
    
    # Build 4x4 decomposition using 7 Strassen products
    # A is 4x4, split into 2x2 blocks A00, A01, A10, A11
    # Each block is 2x2, so indices 0-3 for A00, 4-7 for A01, 8-11 for A10, 12-15 for A11
    # Similarly for B and C.
    
    # Rank is 7 * 7 = 49
    rank_4x4 = 49
    U4 = [[0]*rank_4x4 for _ in range(16)]
    V4 = [[0]*rank_4x4 for _ in range(16)]
    W4 = [[0]*rank_4x4 for _ in range(16)]
    
    # Strassen products P1..P7
    # P1 = (A00 + A11)(B00 + B11)
    # P2 = (A10 + A11) B00
    # P3 = A00 (B01 - B11)
    # P4 = A11 (B10 - B00)
    # P5 = (A00 + A01) B11
    # P6 = (A10 - A00) (B00 + B01)
    # P7 = (A01 - A11) (B10 + B11)
    
    # For each product k (0..6), we use the 2x2 decomposition.
    # The 2x2 decomposition uses U2, V2, W2.
    # We need to map the 2x2 indices to the 4x4 indices.
    
    # Block offsets in 4x4 indices (row-major 4x4 -> 0..15)
    # A00: 0-3, A01: 4-7, A10: 8-11, A11: 12-15
    # B00: 0-3, B01: 4-7, B10: 8-11, B11: 12-15
    # C00: 0-3, C01: 4-7, C10: 8-11, C11: 12-15
    
    def add_decomp(U_target, V_target, W_target, col_offset, 
                   A_blocks, B_blocks, C_blocks, signs_A, signs_B, signs_C):
        # A_blocks: list of 2x2 block indices (e.g., [0, 1] for A00, A11)
        # signs_A: coefficients for those blocks
        # We combine them into a single 2x2 "super" matrix multiplication.
        # The U2 matrix has rows for the 4 elements of the 2x2 matrix.
        # We need to sum the contributions from the blocks.
        
        for c in range(7):
            col = col_offset + c
            # U contributions
            for local_idx, val in enumerate(U2[:, c]):
                if val != 0:
                    # local_idx 0..3 corresponds to a,b,c,d of the 2x2 matrix
                    # We need to map this to the 4x4 indices.
                    # The 2x2 matrix is formed by linear combination of blocks.
                    # This is tricky because U2 is for a single 2x2 matrix.
                    # Here, the "input" to the 2x2 multiplication is a linear combination of blocks.
                    # So U_target[row, col] += val * sum(sign * U_block_element)
                    
                    # Actually, U2[local_idx, c] is the coefficient for the element local_idx of the 2x2 matrix A'.
                    # A' is a linear combination of A blocks.
                    # A'_{local_idx} = sum_{b} signs_A[b] * A_{block_b, local_idx}
                    
                    # So U_target[A_{block_b, local_idx}, col] += val * signs_A[b]
                    
                    pass # Placeholder

    # This approach is complex. Let's use a simpler direct construction.
    # We know the rank is 49.
    # We can just hardcode the 49-rank decomposition or generate it carefully.
    
    # Alternative: Use the 98-rank solution (2x 49-rank) which is easier to verify.
    # If 98 is better than 128, it's a win.
    
    # Let's try to construct the 98-rank solution.
    # It's just two copies of the 49-rank solution.
    
    # First, we need a valid 49-rank solution.
    # I will implement the Strassen-based 49-rank solution properly.
    
    # The 2x2 Strassen decomposition U2, V2, W2 is correct.
    # For the 4x4 case, we have 7 "virtual" 2x2 multiplications.
    # Each virtual multiplication uses 7 rank-1 terms.
    # Total 49 terms.
    
    # Let's define the 7 virtual multiplications.
    # Each is defined by:
    # A_virtual = sum_{b} coeff_A_b * A_block_b
    # B_virtual = sum_{b} coeff_B_b * B_block_b
    # C_virtual = sum_{b} coeff_C_b * C_block_b
    
    # The rank-1 terms for this virtual multiplication are:
    # For each c in 0..6:
    #   U_term = sum_{b} coeff_A_b * U2[:, c] mapped to A_block_b
    #   V_term = sum_{b} coeff_B_b * V2[:, c] mapped to B_block_b
    #   W_term = sum_{b} coeff_C_b * W2[:, c] mapped to C_block_b
    
    # Let's code this.
    
    U4 = [[0]*49 for _ in range(16)]
    V4 = [[0]*49 for _ in range(16)]
    W4 = [[0]*49 for _ in range(16)]
    
    # Define the 7 Strassen products
    # Each product is a dict of block: coeff
    products = [
        # P1
        {'A': {0: 1, 3: 1}, 'B': {0: 1, 3: 1}, 'C': {0: 1, 3: -1, 2: 1, 1: 1}}, # C00=M1+M4-M5+M7, C11=M1-M2+M3+M6. Wait.
        # The C coefficients depend on which C block is being computed.
        # The virtual product P1 contributes to C00, C01, C10, C11.
        # Actually, P1 is just a matrix.
        # P1 = (A00+A11)(B00+B11).
        # This P1 is used in C00 and C11.
        # C00 += P1. C11 += P1.
        # So for the term corresponding to P1, the W part should add to C00 and C11.
        
        # Let's re-evaluate.
        # The decomposition is a sum of rank-1 tensors.
        # Each rank-1 tensor is U_c * V_c * W_c.
        # For a specific Strassen product P_k, we decompose it into 7 rank-1 tensors.
        # Each of these 7 tensors has U, V, W parts.
        # The U part is derived from A's blocks.
        # The V part is derived from B's blocks.
        # The W part is derived from C's blocks.
        
        # For P1:
        # A_part = A00 + A11
        # B_part = B00 + B11
        # C_part = C00 + C11 (since P1 adds to both)
        
        # So for each of the 7 sub-terms of P1:
        # U_sub = U2[:, c] mapped to A00 and A11 with coeff 1.
        # V_sub = V2[:, c] mapped to B00 and B11 with coeff 1.
        # W_sub = W2[:, c] mapped to C00 and C11 with coeff 1.
        
        # This seems correct.
    ]
    
    # Let's define the products with their C contributions.
    # P1: A00+A11, B00+B11, C00+1, C11+1
    # P2: A10+A11, B00, C10+1
    # P3: A00, B01-B11, C01+1
    # P4: A11, B10-B00, C00+1, C10+1
    # P5: A00+A01, B11, C00-1, C01+1
    # P6: A10-A00, B00+B01, C11+1
    # P7: A01-A11, B10+B11, C00+1
    
    # Wait, check C11 for P7.
    # C11 = M1 - M2 + M3 + M6.
    # P7 is M7. M7 is not in C11.
    # So P7 contributes to C00 only.
    
    # Check C00 for P7.
    # C00 = M1 + M4 - M5 + M7.
    # Yes, P7 contributes to C00.
    
    # Check C10 for P4.
    # C10 = M2 + M4.
    # P4 is M4.
    # Yes, P4 contributes to C10.
    # Also C00 = M1 + M4 - M5 + M7.
    # So P4 contributes to C00.
    
    # Correct mapping:
    # P1: A={0:1, 3:1}, B={0:1, 3:1}, C={0:1, 3:1}
    # P2: A={2:1, 3:1}, B={0:1}, C={2:1}
    # P3: A={0:1}, B={1:1, 3:-1}, C={1:1}
    # P4: A={3:1}, B={2:1, 0:-1}, C={0:1, 2:1}
    # P5: A={0:1, 1:1}, B={3:1}, C={0:-1, 1:1}
    # P6: A={2:1, 0:-1}, B={0:1, 1:1}, C={3:1}
    # P7: A={1:1, 3:-1}, B={2:1, 3:1}, C={0:1}
    
    prod_defs = [
        ({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1}),
        ({2:1, 3:1}, {0:1}, {2:1}),
        ({0:1}, {1:1, 3:-1}, {1:1}),
        ({3:1}, {2:1, 0:-1}, {0:1, 2:1}),
        ({0:1, 1:1}, {3:1}, {0:-1, 1:1}),
        ({2:1, 0:-1}, {0:1, 1:1}, {3:1}),
        ({1:1, 3:-1}, {2:1, 3:1}, {0:1})
    ]
    
    for k, (A_coeffs, B_coeffs, C_coeffs) in enumerate(prod_defs):
        col_base = k * 7
        for c in range(7):
            col = col_base + c
            # U
            for local_idx in range(4):
                val = U2[local_idx][c]
                if val != 0:
                    for block, coeff in A_coeffs.items():
                        # local_idx 0..3 in 2x2 block -> 0..3 in 4x4 block
                        # Block 0 is rows 0-1, cols 0-1 -> indices 0-3
                        # Block 1 is rows 0-1, cols 2-3 -> indices 4-7
                        # Block 2 is rows 2-3, cols 0-1 -> indices 8-11
                        # Block 3 is rows 2-3, cols 2-3 -> indices 12-15
                        row_in_4x4 = block * 4 + local_idx
                        U4[row_in_4x4][col] += val * coeff
                        
            # V
            for local_idx in range(4):
                val = V2[local_idx][c]
                if val != 0:
                    for block, coeff in B_coeffs.items():
                        row_in_4x4 = block * 4 + local_idx
                        V4[row_in_4x4][col] += val * coeff
                        
            # W
            for local_idx in range(4):
                val = W2[local_idx][c]
                if val != 0:
                    for block, coeff in C_coeffs.items():
                        row_in_4x4 = block * 4 + local_idx
                        W4[row_in_4x4][col] += val * coeff
                        
    # Now we have U4, V4, W4 for 4x4x4.
    # Construct 4x4x8.
    # U is 16x98.
    # V is 32x98.
    # W is 32x98.
    
    U = [[0]*98 for _ in range(16)]
    V = [[0]*98 for _ in range(32)]
    W = [[0]*98 for _ in range(32)]
    
    # First 49 columns for B_L (cols 0-3 of B, 0-3 of C)
    for i in range(16):
        for c in range(49):
            U[i][c] = U4[i][c]
            V[i][c] = V4[i][c] # B_L rows 0-15
            W[i][c] = W4[i][c] # C_L rows 0-15
            
    # Next 49 columns for B_R (cols 4-7 of B, 4-7 of C)
    for i in range(16):
        for c in range(49):
            U[i][c+49] = U4[i][c]
            V[i+16][c+49] = V4[i][c] # B_R rows 16-31
            W[i+16][c+49] = W4[i][c] # C_R rows 16-31
            
    return {"U": U, "V": V, "W": W}
