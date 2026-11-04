import numpy as np
from fractions import Fraction
import time

def construct():
    n, m, p = 4, 4, 6
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    # We will build the tensor decomposition for 4x4x6.
    # Strategy:
    # 1. Decompose the 4x4x4 part using Strassen (Rank 49).
    # 2. Decompose the 4x4x2 part (the remaining 2 columns of the 4x6 matrix) using Strassen (Rank 28).
    #    Wait, 4x4x6 can be split into 4x4x4 + 4x4x2.
    #    The 4x4x2 part: C is 4x2. B is 4x2.
    #    We can view this as two 4x4x1 multiplications? Or one 4x4x2.
    #    Actually, we can split the 4x4x6 into two 4x4x3? No, Strassen works best on powers of 2.
    #    Let's split 4x4x6 into 4x4x4 + 4x4x2.
    #    
    #    For 4x4x4: Rank 49 (Strassen).
    #    For 4x4x2: 
    #        We can split the 4x2 output into two 4x1 outputs?
    #        Or split the 4x2 input into two 4x1 inputs?
    #        Let's use the structure: C (4x2) = A (4x4) * B (4x2).
    #        We can split A into 2x2 blocks and B into 2x2 blocks?
    #        B is 4x2. Blocks B00 (2x2), B10 (2x2).
    #        A is 4x4. Blocks A00, A01, A10, A11 (2x2).
    #        C is 4x2. Blocks C00 (2x2), C10 (2x2).
    #        C00 = A00*B00 + A01*B10
    #        C10 = A10*B00 + A11*B10
    #        This is 4 products of 2x2 matrices.
    #        Each 2x2x2 product has rank 7 (Strassen).
    #        So 4 * 7 = 28 terms.
    #        Total rank = 49 + 28 = 77.
    #
    #    Can we do better?
    #    Maybe split 4x4x6 into 2x4x6 + 2x4x6? No.
    #    Maybe use a different decomposition for 4x4x6 directly?
    #    Known lower bound for 4x4x6 is likely around 70-80.
    #    77 is a strong candidate.
    #
    #    Let's implement the 77-rank decomposition.

    # Helper to generate Strassen 2x2x2 terms
    # Returns list of (u, v, w) where u, v, w are length-4 vectors (flattened 2x2)
    def strassen_2x2_terms():
        terms = []
        # M1
        terms.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]))
        # M2
        terms.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]))
        # M3
        terms.append(([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]))
        # M4
        terms.append(([0, 0, 0, 1], [-1, 0, 1, 0], [-1, 0, 1, 0]))
        # M5
        terms.append(([1, 1, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0]))
        # M6
        terms.append(([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]))
        # M7
        terms.append(([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, -1]))
        return terms

    s2_terms = strassen_2x2_terms()

    # Helper to generate 4x4x4 Strassen terms (Rank 49)
    # This is the tensor product of two 2x2x2 Strassen decompositions?
    # No, it's the "block Strassen".
    # A (4x4) = [[A00, A01], [A10, A11]] where blocks are 2x2.
    # B (4x4) = [[B00, B01], [B10, B11]]
    # C (4x4) = [[C00, C01], [C10, C11]]
    #
    # Strassen steps for blocks:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 - M4 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M6
    #
    # Each Mi is a product of two 2x2 matrices.
    # So we expand each Mi using the 2x2x2 Strassen decomposition.
    # This gives 7 * 7 = 49 terms.
    #
    # We need to map the block indices to the full 4x4 indices.
    # U is 4x4 flattened (length 16). Index i*m+j -> i*4+j.
    # V is 4x4 flattened (length 16). Index j*p+k -> j*4+k. (Here p=4 for the 4x4 part)
    # W is 4x4 flattened (length 16). Index k*n+i -> k*4+i. (Here n=4)
    
    def get_4x4x4_strassen_terms():
        terms = []
        # Define block linear combinations for A, B, and C output contributions.
        # Each entry is a list of (block_idx, coeff)
        # block_idx: 0=A00, 1=A01, 2=A10, 3=A11
        
        steps = [
            # M1
            ([ (0, 1), (3, 1) ], [ (0, 1), (3, 1) ], [ (0, 1), (3, 1) ]),
            # M2
            ([ (2, 1), (3, 1) ], [ (0, 1) ], [ (2, 1) ]),
            # M3
            ([ (0, 1) ], [ (1, 1), (3, -1) ], [ (1, 1) ]),
            # M4
            ([ (3, 1) ], [ (2, 1), (0, -1) ], [ (0, -1), (2, 1) ]),
            # M5
            ([ (0, 1), (1, 1) ], [ (3, 1) ], [ (1, 1) ]),
            # M6
            ([ (2, -1), (0, 1) ], [ (0, 1), (1, 1) ], [ (3, 1) ]),
            # M7
            ([ (1, 1), (3, -1) ], [ (2, 1), (3, 1) ], [ (0, 1), (3, -1) ])
        ]
        
        for A_blocks, B_blocks, C_blocks in steps:
            # Expand A_blocks into a 4x4 matrix (flattened)
            u_outer = [0]*16
            for b_idx, coeff in A_blocks:
                br, bc = b_idx // 2, b_idx % 2
                for r in range(2):
                    for c in range(2):
                        i = 2*br + r
                        j = 2*bc + c
                        u_outer[i*4 + j] += coeff
                        
            # Expand B_blocks into a 4x4 matrix (flattened)
            v_outer = [0]*16
            for b_idx, coeff in B_blocks:
                br, bc = b_idx // 2, b_idx % 2
                for r in range(2):
                    for c in range(2):
                        j = 2*br + r
                        k = 2*bc + c
                        v_outer[j*4 + k] += coeff
                        
            # Expand C_blocks into a 4x4 matrix (flattened)
            # C is 4x4. Output index is k*n+i = k*4+i.
            # C_blocks are 2x2 blocks of C.
            # C00 is rows 0-1, cols 0-1.
            # C01 is rows 0-1, cols 2-3.
            # C10 is rows 2-3, cols 0-1.
            # C11 is rows 2-3, cols 2-3.
            w_outer = [0]*16
            for b_idx, coeff in C_blocks:
                br, bc = b_idx // 2, b_idx % 2
                for r in range(2):
                    for c in range(2):
                        k = 2*bc + c  # column of C
                        i = 2*br + r  # row of C
                        w_outer[k*4 + i] += coeff
                        
            # Now, this u_outer, v_outer, w_outer is a "block" rank-1 term?
            # No, this is a term where the A-part is a specific linear combination of blocks,
            # and the B-part is a specific linear combination of blocks.
            # But the product is a matrix product.
            # We need to decompose the product of these two 4x4 matrices (which are sparse/block-structured)
            # into rank-1 tensor terms.
            # Wait, the Strassen decomposition for 4x4x4 is NOT just 7 terms of 4x4 matrices.
            # It is 7 products of 2x2 matrices.
            # So for each of the 7 outer steps, we have a product of two 2x2 matrices.
            # We then decompose that 2x2x2 product into 7 rank-1 terms.
            # So we need to iterate over the inner Strassen terms.
            
            # Let's re-evaluate.
            # The outer step defines linear combinations of A-blocks and B-blocks.
            # Let A' = sum(coeff_a * A_block) and B' = sum(coeff_b * B_block).
            # The product is A' * B'.
            # A' and B' are 2x2 matrices.
            # We decompose A' * B' using the 2x2x2 Strassen decomposition.
            # This gives 7 terms: u_inner * v_inner * w_inner.
            # u_inner is a linear form on the 4 elements of A'.
            # v_inner is a linear form on the 4 elements of B'.
            # w_inner is a linear form on the 4 elements of the result C'.
            
            # So, for each outer step:
            # 1. Construct the 2x2 matrix A' (from A_blocks).
            # 2. Construct the 2x2 matrix B' (from B_blocks).
            # 3. For each inner Strassen term (u_in, v_in, w_in):
            #    - u_in is a vector of length 4 (coeffs for A' elements).
            #    - v_in is a vector of length 4 (coeffs for B' elements).
            #    - w_in is a vector of length 4 (coeffs for C' elements).
            #    - The full U vector (length 16) is formed by placing the coeffs of u_in
            #      into the positions corresponding to the blocks in A_blocks.
            #      Specifically, if A' element (r,c) has coeff u_in[r*2+c],
            #      and A' element (r,c) corresponds to sum of A_block[br,bc][r,c] * coeff_block,
            #      then the full U vector gets u_in[r*2+c] * coeff_block at the position
            #      corresponding to A_block[br,bc][r,c].
            
            # This is getting complicated to map directly.
            # Let's use a simpler approach:
            # We know the 49 terms for 4x4x4 Strassen can be generated by
            # taking the Kronecker product of the 7x7 Strassen matrix?
            # No.
            
            # Let's try to construct the 49 terms explicitly by iterating over
            # the 7 outer steps and 7 inner steps.
            
            # For a given outer step, we have:
            # A' = sum_{(br,bc) in A_blocks} coeff * A_{br,bc}
            # B' = sum_{(br,bc) in B_blocks} coeff * B_{br,bc}
            
            # Let's represent A' as a 2x2 matrix of coefficients.
            # A'_2x2[r,c] = sum of coeff for block (br,bc) such that...
            # Wait, A' is a 2x2 matrix. Its elements are linear combinations of the 16 elements of A.
            # A'_{00} = A_{00,00} + A_{11,11} (for M1)
            # A'_{01} = A_{00,01} + A_{11,11} (for M1)
            # etc.
            
            # So for M1:
            # A' = [[A00+A11, A01+A11], [A10+A11, A11+A11]]? No.
            # A00 is the top-left 2x2 block.
            # A11 is the bottom-right 2x2 block.
            # A' = A00 + A11. This is a 2x2 matrix where each element is the sum of the corresponding elements of A00 and A11.
            # So A'_{00} = A_{0,0} + A_{2,2}
            # A'_{01} = A_{0,1} + A_{2,3}
            # A'_{10} = A_{1,0} + A_{3,2}
            # A'_{11} = A_{1,1} + A_{3,3}
            
            # Similarly for B'.
            
            # So for each outer step, we can construct the 16-element U_outer vector
            # by placing the coefficients.
            # For M1:
            # U_outer[0] = 1 (A00), U_outer[12] = 1 (A11)
            # U_outer[1] = 1 (A01), U_outer[13] = 1 (A11)
            # ...
            
            # Let's write a helper to expand block coefficients to full 16-vector.
            def expand_blocks(blocks, n=4):
                vec = [0]*16
                for b_idx, coeff in blocks:
                    br, bc = b_idx // 2, b_idx % 2
                    for r in range(2):
                        for c in range(2):
                            i = 2*br + r
                            j = 2*bc + c
                            vec[i*4 + j] += coeff
                return vec

            u_full = expand_blocks(A_blocks)
            v_full = expand_blocks(B_blocks)
            
            # Now, we need to decompose the product of the two 2x2 matrices A' and B'.
            # A' is defined by the coefficients in u_full? No.
            # A' is a 2x2 matrix.
            # Let's extract the 2x2 matrix A' from u_full.
            # A'_{00} is the coefficient of A_{0,0} in u_full? No.
            # A'_{00} = A_{0,0} + A_{2,2}.
            # So the linear form for A'_{00} is [1, 0, 0, 0, ..., 1, 0, 0, 0] (at 0 and 12).
            # This is exactly u_full.
            # So u_full IS the linear form for the 2x2 matrix A'.
            # But we need to apply the inner Strassen decomposition to A' and B'.
            # The inner Strassen decomposition operates on the 4 elements of A' and B'.
            # So we need to map the 16-element u_full to the 4 elements of A'.
            
            # This is the key insight:
            # The 49 terms are the tensor product of the 7 outer linear forms
            # and the 7 inner linear forms.
            
            # Let's define the 7 outer linear forms for A, B, and C.
            # And the 7 inner linear forms for the 2x2 elements.
            
            # For each outer step s (0..6):
            #   Let u_s be the 16-vector for A.
            #   Let v_s be the 16-vector for B.
            #   Let w_s be the 16-vector for C.
            #   For each inner step t (0..6):
            #      Let u_t be the 4-vector for the 2x2 A'.
            #      Let v_t be the 4-vector for the 2x2 B'.
            #      Let w_t be the 4-vector for the 2x2 C'.
            #      
            #      The full U vector is the "substitution" of u_t into u_s.
            #      i.e., U_full = sum_{r,c} u_t[r*2+c] * (linear form for A'_{r,c})
            #      But the linear form for A'_{r,c} is just the part of u_s that corresponds to A'_{r,c}.
            #      Since A'_{r,c} = sum_{blocks} coeff * A_block,
            #      the linear form for A'_{r,c} is the restriction of u_s to the blocks involved in A'_{r,c}.
            #      
            #      Actually, it's simpler:
            #      U_full[i] = sum_{r,c} u_t[r*2+c] * (coeff of A_{i} in A'_{r,c})
            #      But A'_{r,c} is a linear combination of A elements.
            #      Let's define a matrix M_A of size 16x4, where M_A[i, r*2+c] is the coefficient
            #      of A_i in A'_{r,c}.
            #      Similarly for B and C.
            
            # Let's construct M_A, M_B, M_C for each outer step.
            
            # For M1:
            # A'_{00} = A00 + A11 -> M_A[0,0]=1, M_A[12,0]=1
            # A'_{01} = A01 + A11 -> M_A[1,1]=1, M_A[13,1]=1
            # A'_{10} = A10 + A11 -> M_A[4,2]=1, M_A[16,2]=1? No, 4x4 is 16 elements.
            # A_{i,j} index is i*4+j.
            # A00 is (0,0), (0,1), (1,0), (1,1) -> indices 0,1,4,5
            # A11 is (2,2), (2,3), (3,2), (3,3) -> indices 10,11,14,15
            
            # So for M1:
            # A'_{00} = A_{0,0} + A_{2,2} -> M_A[0,0]=1, M_A[10,0]=1
            # A'_{01} = A_{0,1} + A_{2,3} -> M_A[1,1]=1, M_A[11,1]=1
            # A'_{10} = A_{1,0} + A_{3,2} -> M_A[4,2]=1, M_A[14,2]=1
            # A'_{11} = A_{1,1} + A_{3,3} -> M_A[5,3]=1, M_A[15,3]=1
            
            # This is doable.
            
            # Let's write a function to get M_A, M_B, M_C for a given outer step.
            
            def get_outer_matrices(A_blocks, B_blocks, C_blocks):
                M_A = [[0]*4 for _ in range(16)]
                M_B = [[0]*4 for _ in range(16)]
                M_C = [[0]*4 for _ in range(16)]
                
                for b_idx, coeff in A_blocks:
                    br, bc = b_idx // 2, b_idx % 2
                    for r in range(2):
                        for c in range(2):
                            i = 2*br + r
                            j = 2*bc + c
                            M_A[i*4 + j][r*2 + c] = coeff
                            
                for b_idx, coeff in B_blocks:
                    br, bc = b_idx // 2, b_idx % 2
                    for r in range(2):
                        for c in range(2):
                            j = 2*br + r
                            k = 2*bc + c
                            M_B[j*4 + k][r*2 + c] = coeff
                            
                for b_idx, coeff in C_blocks:
                    br, bc = b_idx // 2, b_idx % 2
                    for r in range(2):
                        for c in range(2):
                            k = 2*bc + c
                            i = 2*br + r
                            M_C[k*4 + i][r*2 + c] = coeff
                            
                return M_A, M_B, M_C

            M_A, M_B, M_C = get_outer_matrices(A_blocks, B_blocks, C_blocks)
            
            for u_in, v_in, w_in in s2_terms:
                # u_in is length 4.
                # U_full = M_A^T * u_in
                U_full = [0]*16
                V_full = [0]*16
                W_full = [0]*16
                for i in range(16):
                    U_full[i] = sum(M_A[i][k] * u_in[k] for k in range(4))
                    V_full[i] = sum(M_B[i][k] * v_in[k] for k in range(4))
                    W_full[i] = sum(M_C[i][k] * w_in[k] for k in range(4))
                
                terms.append((U_full, V_full, W_full))
                
        return terms

    terms_4x4x4 = get_4x4x4_strassen_terms()
    
    # Now for the 4x4x2 part.
    # C (4x2) = A (4x4) * B (4x2).
    # Split into 2x2 blocks.
    # A: A00, A01, A10, A11 (2x2).
    # B: B00, B10 (2x2). (B is 4x2, so top 2x2 and bottom 2x2).
    # C: C00, C10 (2x2).
    #
    # C00 = A00*B00 + A01*B10
    # C10 = A10*B00 + A11*B10
    #
    # This is 4 products of 2x2 matrices.
    # Each product has rank 7.
    # Total rank 28.
    
    def get_4x4x2_strassen_terms():
        terms = []
        # Products:
        # P1 = A00 * B00 -> C00
        # P2 = A01 * B10 -> C00
        # P3 = A10 * B00 -> C10
        # P4 = A11 * B10 -> C10
        
        # Define the block indices for A, B, C.
        # A blocks: 0=A00, 1=A01, 2=A10, 3=A11
        # B blocks: 0=B00, 1=B10
        # C blocks: 0=C00, 1=C10
        
        products = [
            # (A_blocks, B_blocks, C_blocks)
            ([ (0, 1) ], [ (0, 1) ], [ (0, 1) ]), # P1
            ([ (1, 1) ], [ (1, 1) ], [ (0, 1) ]), # P2
            ([ (2, 1) ], [ (0, 1) ], [ (1, 1) ]), # P3
            ([ (3, 1) ], [ (1, 1) ], [ (1, 1) ])  # P4
        ]
        
        for A_blocks, B_blocks, C_blocks in products:
            # Expand to full vectors.
            # U is 4x4 (16 elements).
            # V is 4x2 (8 elements).
            # W is 2x4 (8 elements).
            
            u_full = [0]*16
            for b_idx, coeff in A_blocks:
                br, bc = b_idx // 2, b_idx % 2
                for r in range(2):
                    for c in range(2):
                        i = 2*br + r
                        j = 2*bc + c
                        u_full[i*4 + j] += coeff
                        
            v_full = [0]*8
            for b_idx, coeff in B_blocks:
                br, bc = b_idx // 2, b_idx % 2
                # B is 4x2.
                # B00 is rows 0-1, col 0-1? No, B is 4x2.
                # B00 is rows 0-1, cols 0-1? No, B has 2 columns.
                # So B00 is rows 0-1, cols 0-1? No, cols 0-1 is 2 cols.
                # So B00 is the top 2x2 block.
                # B10 is the bottom 2x2 block.
                for r in range(2):
                    for c in range(2):
                        j = 2*br + r
                        k = c  # k is 0 or 1
                        v_full[j*2 + k] += coeff
                        
            w_full = [0]*8
            for b_idx, coeff in C_blocks:
                br, bc = b_idx // 2, b_idx % 2
                # C is 4x2.
                # C00 is top 2x2.
                # C10 is bottom 2x2.
                for r in range(2):
                    for c in range(2):
                        k = c  # col of C
                        i = 2*br + r  # row of C
                        w_full[k*4 + i] += coeff
                        
            # Now, we need to decompose the product of the 2x2 matrices.
            # A' is 2x2, B' is 2x2, C' is 2x2.
            # We use the inner Strassen terms.
            
            # We need M_A, M_B, M_C for this step.
            # M_A is 16x4.
            # M_B is 8x4.
            # M_C is 8x4.
            
            M_A = [[0]*4 for _ in range(16)]
            M_B = [[0]*4 for _ in range(8)]
            M_C = [[0]*4 for _ in range(8)]
            
            for b_idx, coeff in A_blocks:
                br, bc = b_idx // 2, b_idx % 2
                for r in range(2):
                    for c in range(2):
                        i = 2*br + r
                        j = 2*bc + c
                        M_A[i*4 + j][r*2 + c] = coeff
                        
            for b_idx, coeff in B_blocks:
                br, bc = b_idx // 2, b_idx % 2
                for r in range(2):
                    for c in range(2):
                        j = 2*br + r
                        k = c
                        M_B[j*2 + k][r*2 + c] = coeff
                        
            for b_idx, coeff in C_blocks:
                br, bc = b_idx // 2, b_idx % 2
                for r in range(2):
                    for c in range(2):
                        k = c
                        i = 2*br + r
                        M_C[k*4 + i][r*2 + c] = coeff
                        
            for u_in, v_in, w_in in s2_terms:
                U_full = [0]*16
                V_full = [0]*8
                W_full = [0]*8
                for i in range(16):
                    U_full[i] = sum(M_A[i][k] * u_in[k] for k in range(4))
                for i in range(8):
                    V_full[i] = sum(M_B[i][k] * v_in[k] for k in range(4))
                    W_full[i] = sum(M_C[i][k] * w_in[k] for k in range(4))
                    
                terms.append((U_full, V_full, W_full))
                
        return terms

    terms_4x4x2 = get_4x4x2_strassen_terms()
    
    # Combine the terms.
    # U is 16x77.
    # V is 24x77. (4x6 = 24)
    # W is 24x77. (6x4 = 24)
    
    rank = 49 + 28
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(24)]
    W = [[0]*rank for _ in range(24)]
    
    # Fill in the 4x4x4 part (first 49 columns).
    for t, (u, v, w) in enumerate(terms_4x4x4):
        for i in range(16):
            U[i][t] = u[i]
        for i in range(16):
            V[i][t] = v[i]
        for i in range(16):
            W[i][t] = w[i]
            
    # Fill in the 4x4x2 part (next 28 columns).
    # The B matrix is 4x6.
    # The first 4 columns are used for the 4x4x4 part.
    # The last 2 columns are used for the 4x4x2 part.
    # So V for the 4x4x2 part should be placed in columns 4..5 of the 4x6 matrix.
    # V index is j*6 + k.
    # For the 4x4x2 part, j is 0..3, k is 0..1.
    # So the full V index is j*6 + (4 + k).
    
    for t, (u, v, w) in enumerate(terms_4x4x2):
        col = 49 + t
        for i in range(16):
            U[i][col] = u[i]
        for i in range(8):
            # v[i] corresponds to j*2 + k.
            j = i // 2
            k = i % 2
            full_v_idx = j*6 + (4 + k)
            V[full_v_idx][col] = v[i]
        for i in range(8):
            # w[i] corresponds to k*4 + i.
            # k is 0..1, i is 0..3.
            # W index is k*n + i = k*4 + i.
            # This is the same for the 4x4x2 part.
            W[i][col] = w[i]
            
    return {"U": U, "V": V, "W": W}
