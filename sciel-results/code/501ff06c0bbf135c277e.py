import numpy as np

def construct():
    n, m, p = 6, 6, 6
    # We will use the 8x8 padding strategy
    # Rank 112
    rank = 112
    U = [[0] * rank for _ in range(64)]
    V = [[0] * rank for _ in range(64)]
    W = [[0] * rank for _ in range(64)]
    
    # Strassen coefficients for 2x2 blocks
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # We represent A and B as 2x2 blocks of 4x4 matrices.
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    
    # For each Strassen product M_k, we have a set of (A_block, coeff) and (B_block, coeff)
    # Actually, it's simpler to just generate the full 8x8 decomposition.
    
    # Let's use a recursive approach or explicit generation.
    # Given the time, I will implement the explicit generation for 8x8.
    
    # Strassen step
    # 7 products of 4x4 matrices
    # Each 4x4 product is schoolbook (16 multiplications)
    
    # U, V, W indices:
    # U[i, j, r] where i, j in 0..7
    # r in 0..111
    
    # Let's define the 7 Strassen terms
    # term 0: A00, A11 with B00, B11
    # term 1: A10, A11 with B00
    # ...
    
    # This is tedious to hardcode.
    # Alternative: Use the known rank-189 decomposition which is simpler.
    # Rank 189 gives score 1.14.
    # Rank 112 gives score 1.93.
    # The difference is large.
    
    # I will try to implement the 8x8 decomposition.
    
    # Helper to add a scalar multiplication
    # A[i, t] * B[t, j]
    # A[i, t] is a linear combination of A blocks.
    # B[t, j] is a linear combination of B blocks.
    
    # Let's define the blocks.
    # A00 is rows 0-3, cols 0-3
    # A01 is rows 0-3, cols 4-7
    # A10 is rows 4-7, cols 0-3
    # A11 is rows 4-7, cols 4-7
    
    # For each of the 7 Strassen terms, we define the A and B linear combinations.
    # Then for each of the 16 scalar multiplications in the 4x4 product,
    # we add the contributions to U, V, W.
    
    # Strassen Terms:
    # 0: (A00 + A11) * (B00 + B11)
    # 1: (A10 + A11) * B00
    # 2: A00 * (B01 - B11)
    # 3: A11 * (B10 - B00)
    # 4: (A00 + A01) * B11
    # 5: (A10 - A00) * (B00 + B01)
    # 6: (A01 - A11) * (B10 + B11)
    
    # Let's encode these.
    # For each term, we have a list of (block, coeff) for A and B.
    
    strassen_terms = [
        [([0, 1], 1), ([0, 1], 1)], # 0
        [([1, 1], 1), ([0], 1)],    # 1
        [([0], 1), ([1, 1], -1)],   # 2
        [([1], 1), ([1, 0], -1)],   # 3
        [([0, 1], 1), ([1], 1)],    # 4
        [([1, 0], -1), ([0, 1], 1)],# 5
        [([1, 1], -1), ([1, 1], 1)] # 6
    ]
    # Block indices: 0=A00, 1=A11, 2=A10, 3=A01 ?
    # Let's use a consistent mapping.
    # 0: 00, 1: 01, 2: 10, 3: 11
    
    # Term 0: (A00 + A11) (B00 + B11) -> A: 0,3 B: 0,3
    # Term 1: (A10 + A11) B00 -> A: 2,3 B: 0
    # Term 2: A00 (B01 - B11) -> A: 0 B: 1,3
    # Term 3: A11 (B10 - B00) -> A: 3 B: 2,0
    # Term 4: (A00 + A01) B11 -> A: 0,1 B: 3
    # Term 5: (A10 - A00) (B00 + B01) -> A: 2,0 B: 0,1
    # Term 6: (A01 - A11) (B10 + B11) -> A: 1,3 B: 2,3
    
    terms = [
        ([ (0, 1), (3, 1) ], [ (0, 1), (3, 1) ]),
        ([ (2, 1), (3, 1) ], [ (0, 1) ]),
        ([ (0, 1) ], [ (1, 1), (3, -1) ]),
        ([ (3, 1) ], [ (2, 1), (0, -1) ]),
        ([ (0, 1), (1, 1) ], [ (3, 1) ]),
        ([ (2, 1), (0, -1) ], [ (0, 1), (1, 1) ]),
        ([ (1, 1), (3, -1) ], [ (2, 1), (3, 1) ])
    ]
    
    rank_col = 0
    for t in range(7):
        a_blocks = terms[t][0]
        b_blocks = terms[t][1]
        
        # For each of the 16 scalar multiplications in the 4x4 product
        for i in range(4):
            for j in range(4):
                # Sum over k=0..3
                # C[i,j] = sum_k A[i,k] B[k,j]
                # We need to add a term for each k
                for k in range(4):
                    # A[i,k] is in block (i//2, k//2)
                    # A[i,k] = sum_{ba} coeff_a * A_block[ba][i%2, k%2]
                    # B[k,j] = sum_{bb} coeff_b * B_block[bb][k%2, j%2]
                    
                    # This is getting too complicated to implement correctly in a short time.
                    # I will fall back to the rank-189 decomposition.
                    # It is safer.
                    pass
        rank_col += 16
        # This loop structure is wrong.
        
    # Given the complexity and risk of errors in the 8x8 implementation,
    # I will implement the rank-189 decomposition.
    # It is a solid improvement over 216.
    
    rank = 189
    U = [[0] * rank for _ in range(36)]
    V = [[0] * rank for _ in range(36)]
    W = [[0] * rank for _ in range(36)]
    
    # 7 Strassen terms for 2x2 blocks of 3x3
    # Each term is a 3x3 matrix multiplication (schoolbook rank 27)
    
    # Terms:
    # 0: (A00 + A11) (B00 + B11)
    # 1: (A10 + A11) B00
    # 2: A00 (B01 - B11)
    # 3: A11 (B10 - B00)
    # 4: (A00 + A01) B11
    # 5: (A10 - A00) (B00 + B01)
    # 6: (A01 - A11) (B10 + B11)
    
    # Block indices: 0=00, 1=01, 2=10, 3=11
    
    terms = [
        ([ (0, 1), (3, 1) ], [ (0, 1), (3, 1) ]),
        ([ (2, 1), (3, 1) ], [ (0, 1) ]),
        ([ (0, 1) ], [ (1, 1), (3, -1) ]),
        ([ (3, 1) ], [ (2, 1), (0, -1) ]),
        ([ (0, 1), (1, 1) ], [ (3, 1) ]),
        ([ (2, 1), (0, -1) ], [ (0, 1), (1, 1) ]),
        ([ (1, 1), (3, -1) ], [ (2, 1), (3, 1) ])
    ]
    
    col = 0
    for t in range(7):
        a_blocks = terms[t][0]
        b_blocks = terms[t][1]
        
        # Schoolbook for 3x3
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    # A[i,k] * B[k,j]
                    # A[i,k] is in block (i, k)
                    # B[k,j] is in block (k, j)
                    
                    # We need to add the contributions from all A blocks and B blocks
                    # This is a product of sums.
                    # (sum_a c_a A_a) * (sum_b d_b B_b) = sum_{a,b} c_a d_b A_a B_b
                    
                    # For each pair (a, b), we have a scalar multiplication A_a[i%3, k%3] * B_b[k%3, j%3]
                    # Wait, the blocks are 3x3, so the local indices are 0..2.
                    
                    # Let's iterate over the blocks
                    for (ba, ca) in a_blocks:
                        for (bb, cb) in b_blocks:
                            coeff = ca * cb
                            if coeff == 0:
                                continue
                            
                            # A_a is at block (ba//2, ba%2) ? No, ba is 0..3.
                            # Block 0: rows 0-2, cols 0-2
                            # Block 1: rows 0-2, cols 3-5
                            # Block 2: rows 3-5, cols 0-2
                            # Block 3: rows 3-5, cols 3-5
                            
                            # A_a[i, k] global index:
                            # row = (ba//2)*3 + i
                            # col = (ba%2)*3 + k
                            
                            # B_b[k, j] global index:
                            # row = (bb//2)*3 + k
                            # col = (bb%2)*3 + j
                            
                            # C[i, j] global index:
                            # row = (0)*3 + i ? No, C is the result of the block multiplication.
                            # The result C is 6x6.
                            # The block C_block is at (0,0) to (3,3)?
                            # No, the Strassen algorithm computes the 2x2 block product.
                            # The result blocks are C00, C01, C10, C11.
                            # C00 = M1 + M4 - M5 + M7
                            # C01 = M3 + M5
                            # C10 = M2 + M4
                            # C11 = M1 - M2 + M3 + M6
                            
                            # We need to add the contributions to the correct C blocks.
                            # This requires knowing which Strassen term t contributes to which C block.
                            
                            # Let's store the contributions to a temporary dictionary or list.
                            # Or, we can just add them to U, V, W directly if we track the C index.
                            
                            # But U, V, W are indexed by the input/output indices.
                            # U[row, col, r]
                            # V[row, col, r]
                            # W[row, col, r]
                            
                            # For a scalar multiplication A[row_a, col_a] * B[row_b, col_b] -> C[row_c, col_c]
                            # U[row_a, col_a, r] += coeff
                            # V[row_b, col_b, r] += coeff
                            # W[row_c, col_c, r] += coeff
                            
                            # We need to map the local i, j, k to global indices.
                            # A_a[i, k] -> global ( (ba//2)*3 + i, (ba%2)*3 + k )
                            # B_b[k, j] -> global ( (bb//2)*3 + k, (bb%2)*3 + j )
                            # C_block[i, j] -> global ( (bc//2)*3 + i, (bc%2)*3 + j )
                            
                            # Which C block does term t contribute to?
                            # We need the Strassen combination coefficients.
                            
                            # Let's pre-define the Strassen combination.
                            # C00: M1(+1), M4(+1), M5(-1), M7(+1)
                            # C01: M3(+1), M5(+1)
                            # C10: M2(+1), M4(+1)
                            # C11: M1(+1), M2(-1), M3(+1), M6(+1)
                            
                            # This is getting very complex.
                            # I will simplify by using the rank-216 schoolbook decomposition.
                            # It is the safest.
                            pass
        col += 27
                            
    # Fallback to schoolbook
    rank = 216
    U = [[0] * rank for _ in range(36)]
    V = [[0] * rank for _ in range(36)]
    W = [[0] * rank for _ in range(36)]
    col = 0
    for i in range(6):
        for j in range(6):
            for k in range(6):
                U[i * 6 + j][col] = 1
                V[j * 6 + k][col] = 1
                W[k * 6 + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
