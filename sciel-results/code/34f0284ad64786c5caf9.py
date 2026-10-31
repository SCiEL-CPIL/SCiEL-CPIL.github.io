"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    # Rank 49 decomposition using recursive Strassen for 4x4 matrices.
    # This leverages the block structure of 4x4 matrices into 2x2 blocks,
    # applying Strassen's algorithm (rank 7) at both the outer (4x4) and inner (2x2) levels.
    
    # Time guard
    start_time = time.time()
    timeout = 150
    
    n, m, p = DIMENSIONS
    
    # The tensor indices for 4x4 multiplication are:
    # A is 4x4 (16 elements), B is 4x4 (16 elements), C is 4x4 (16 elements)
    # U shape: (16, rank)
    # V shape: (16, rank)
    # W shape: (16, rank)
    
    # We will build the rank-49 decomposition.
    # The rank-49 decomposition is the tensor product of two rank-7 Strassen decompositions.
    
    # First, define the rank-7 Strassen decomposition for 2x2 matrices.
    # Indices for 2x2 A: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # Indices for 2x2 B: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # Indices for 2x2 C: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    
    # Strassen products M1..M7:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # C reconstruction:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Let's define the vectors for A, B, C for each M.
    # A_vec has 4 components, B_vec has 4 components, C_vec has 4 components.
    
    # M1: A=[1,0,0,1], B=[1,0,0,1], C=[1,0,0,1]
    # M2: A=[0,0,1,1], B=[1,0,0,0], C=[0,0,1,0]
    # M3: A=[1,0,0,0], B=[0,1,0,-1], C=[0,1,0,0]
    # M4: A=[0,0,0,1], B=[0,0,1,-1], C=[1,0,0,0]
    # M5: A=[1,1,0,0], B=[0,0,0,1], C=[-1,0,0,0]
    # M6: A=[0,-1,1,0], B=[1,1,0,0], C=[0,0,0,1]
    # M7: A=[0,1,0,-1], B=[0,0,1,1], C=[0,0,0,1] # Wait, C11 has M7, C00 has M7.
    # Let's re-verify C11: M1 - M2 + M3 + M6. No M7?
    # Standard Strassen:
    # C11 = M1 - M2 + M3 + M6
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    
    # Let's check M7 in C00. Yes.
    # So for M7, C_vec is [1, 0, 0, 0]? No, M7 contributes to C00.
    # M7 = (A01 - A11) * (B10 + B11)
    # A_vec = [0, 1, 0, -1]
    # B_vec = [0, 0, 1, 1]
    # C contribution to C00 is +1. So C_vec = [1, 0, 0, 0].
    
    # Let's build the list of (A_vec, B_vec, C_vec) for the 2x2 case.
    S2x2 = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]), # M1 -> C00, C11
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]), # M2 -> C10, C11(-1)
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]), # M3 -> C01, C11
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 0, 0]), # M4 -> C00, C10
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 0, 0]), # M5 -> C00(-1), C01
        ([0, -1, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]), # M6 -> C11
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])  # M7 -> C00
    ]
    
    # Verify 2x2 Strassen correctness quickly in code to be safe?
    # Let's just trust the standard formulas, but I will double check M6 and M7.
    # M6 = (A10 - A00)(B00 + B01).
    # A10 is index 2. A00 is index 0. So A=[0,-1,1,0]. Correct.
    # B00 is index 0. B01 is index 1. So B=[1,1,0,0]. Correct.
    # C11 = M1 - M2 + M3 + M6.
    # M1 contributes 1 to C11.
    # M2 contributes -1 to C11? No, M2 is (A10+A11)B00. In C11 formula it is -M2.
    # Wait, my C_vec for M2 was [0,0,1,0]. That means it contributes to C10.
    # But in C11 = ... - M2 ... it should contribute to C11.
    # Let's re-read standard Strassen:
    # M1 = (A11+A22)(B11+B22) -> C11, C22
    # M2 = (A21+A22)B11 -> C21, C22(-1)
    # M3 = A11(B21-B11) -> C12, C22
    # M4 = A22(B22-B12) -> C11, C21
    # M5 = (A11+A12)B22 -> C11(-1), C12
    # M6 = (A21-A11)(B11+B12) -> C22
    # M7 = (A12-A22)(B21+B22) -> C11
    
    # Mapping to 0-based indices:
    # 11->0, 12->1, 21->2, 22->3.
    
    # M1: A=[1,0,0,1], B=[1,0,0,1]. C11(0) and C22(3). C=[1,0,0,1]. Correct.
    # M2: A=[0,0,1,1], B=[1,0,0,0]. C21(2) and C22(3). C=[0,0,1,-1]. 
    #      My previous C was [0,0,1,0]. This is WRONG. M2 contributes to C22 with -1.
    # M3: A=[1,0,0,0], B=[0,1,0,-1]. C12(1) and C22(3). C=[0,1,0,1].
    #      My previous C was [0,1,0,0]. WRONG.
    # M4: A=[0,0,0,1], B=[0,0,1,-1]. C11(0) and C21(2). C=[1,0,1,0].
    #      My previous C was [1,0,0,0]. WRONG.
    # M5: A=[1,1,0,0], B=[0,0,0,1]. C11(0) and C12(1). C=[-1,1,0,0].
    #      My previous C was [-1,0,0,0]. WRONG.
    # M6: A=[0,-1,1,0], B=[1,1,0,0]. C22(3). C=[0,0,0,1]. Correct.
    # M7: A=[0,1,0,-1], B=[0,0,1,1]. C11(0). C=[1,0,0,0]. Correct.
    
    # Corrected S2x2:
    S2x2 = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]), # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]), # M2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]), # M3
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]), # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]), # M5
        ([0, -1, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]), # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])  # M7
    ]
    
    # Now, for 4x4 matrices, we decompose A, B, C into four 2x2 blocks.
    # A blocks: A00 (indices 0-3), A01 (4-7), A10 (8-11), A11 (12-15).
    # B blocks: B00 (0-3), B01 (4-7), B10 (8-11), B11 (12-15).
    # C blocks: C00 (0-3), C01 (4-7), C10 (8-11), C11 (12-15).
    
    # The 4x4 multiplication can be viewed as a 2x2 block multiplication:
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    
    # We apply Strassen to this 2x2 block structure.
    # The "elements" of the 2x2 matrices are themselves 2x2 matrices.
    # So we apply Strassen at the block level (Outer Strassen) and element level (Inner Strassen).
    
    # Outer Strassen steps (operating on 2x2 blocks):
    # Let X = A00, Y = A01, Z = A10, W = A11
    # Let P = B00, Q = B01, R = B10, S = B11
    
    # M1_outer = (X + W) * (P + S)
    # M2_outer = (Z + W) * P
    # M3_outer = X * (Q - S)
    # M4_outer = W * (R - P)
    # M5_outer = (X + Y) * S
    # M6_outer = (Z - X) * (P + Q)
    # M7_outer = (Y - W) * (R + S)
    
    # C00 = M1_outer + M4_outer - M5_outer + M7_outer
    # C01 = M3_outer + M5_outer
    # C10 = M2_outer + M4_outer
    # C11 = M1_outer - M2_outer + M3_outer + M6_outer
    
    # Each M_outer is a product of two 2x2 matrices (blocks).
    # We decompose each M_outer using Inner Strassen (S2x2).
    
    # Total rank = 7 (outer) * 7 (inner) = 49.
    
    rank = 49
    U = np.zeros((16, rank))
    V = np.zeros((16, rank))
    W_mat = np.zeros((16, rank)) # Using W_mat to avoid conflict with variable W above
    
    # Helper to map 2x2 block index and local 2x2 index to global 4x4 index
    # Block 0 (00): 0-3
    # Block 1 (01): 4-7
    # Block 2 (10): 8-11
    # Block 3 (11): 12-15
    
    # Outer Strassen definitions:
    # Each outer term is defined by:
    # A_blocks: dict mapping block_index (0,1,2,3) to a vector of coefficients for the 2x2 elements
    # B_blocks: same
    # C_blocks: same, for the reconstruction
    
    # We define the outer terms M1..M7.
    # For each outer term, we have linear combinations of A blocks and B blocks.
    # e.g., M1_A = X + W. So Block 0 has coeff 1, Block 3 has coeff 1.
    
    outer_terms = []
    
    # M1: (X+W) * (P+S)
    # A: Block 0 (1), Block 3 (1). Others 0.
    # B: Block 0 (1), Block 3 (1). Others 0.
    # C: Used in C00 (+1), C11 (+1).
    outer_terms.append({
        'A': {0: 1, 3: 1}, 'B': {0: 1, 3: 1},
        'C': {0: 1, 3: 1} # C00 and C11
    })
    
    # M2: (Z+W) * P
    # A: Block 2 (1), Block 3 (1).
    # B: Block 0 (1).
    # C: Used in C10 (+1), C11 (-1).
    outer_terms.append({
        'A': {2: 1, 3: 1}, 'B': {0: 1},
        'C': {2: 1, 3: -1}
    })
    
    # M3: X * (Q-S)
    # A: Block 0 (1).
    # B: Block 1 (1), Block 3 (-1).
    # C: Used in C01 (+1), C11 (+1).
    outer_terms.append({
        'A': {0: 1}, 'B': {1: 1, 3: -1},
        'C': {1: 1, 3: 1}
    })
    
    # M4: W * (R-P)
    # A: Block 3 (1).
    # B: Block 2 (1), Block 0 (-1).
    # C: Used in C00 (+1), C10 (+1).
    outer_terms.append({
        'A': {3: 1}, 'B': {2: 1, 0: -1},
        'C': {0: 1, 2: 1}
    })
    
    # M5: (X+Y) * S
    # A: Block 0 (1), Block 1 (1).
    # B: Block 3 (1).
    # C: Used in C00 (-1), C01 (+1).
    outer_terms.append({
        'A': {0: 1, 1: 1}, 'B': {3: 1},
        'C': {0: -1, 1: 1}
    })
    
    # M6: (Z-X) * (P+Q)
    # A: Block 2 (1), Block 0 (-1).
    # B: Block 0 (1), Block 1 (1).
    # C: Used in C11 (+1).
    outer_terms.append({
        'A': {2: 1, 0: -1}, 'B': {0: 1, 1: 1},
        'C': {3: 1}
    })
    
    # M7: (Y-W) * (R+S)
    # A: Block 1 (1), Block 3 (-1).
    # B: Block 2 (1), Block 3 (1).
    # C: Used in C00 (+1).
    outer_terms.append({
        'A': {1: 1, 3: -1}, 'B': {2: 1, 3: 1},
        'C': {0: 1}
    })
    
    # Now iterate through outer terms and inner Strassen terms to build U, V, W.
    col = 0
    for ot in outer_terms:
        A_blocks = ot['A']
        B_blocks = ot['B']
        C_blocks = ot['C']
        
        # For this outer term, we need to compute the product of two 2x2 matrices:
        # A_mat = sum(coeff * A_block)
        # B_mat = sum(coeff * B_block)
        # But wait, Strassen works on the elements.
        # The "A_mat" for the outer Strassen is a 2x2 matrix of 2x2 blocks.
        # Actually, no. The outer Strassen computes M_outer = A_outer * B_outer.
        # A_outer is a 2x2 matrix where elements are 2x2 matrices.
        # We decompose A_outer and B_outer into linear combinations of blocks.
        # Then M_outer is a linear combination of products of blocks.
        # Each product of blocks (2x2 * 2x2) is decomposed using Inner Strassen.
        
        # So for each pair of blocks (ib, jb) that appear in A_outer and B_outer:
        # The contribution is coeff_A * coeff_B * (A_block_ib * B_block_jb).
        
        # Let's list the active blocks for A and B in this outer term.
        active_A_blocks = list(A_blocks.keys())
        active_B_blocks = list(B_blocks.keys())
        
        # For each combination of active A block and active B block:
        for ib in active_A_blocks:
            for jb in active_B_blocks:
                coeff_a = A_blocks[ib]
                coeff_b = B_blocks[jb]
                
                # We need to decompose A_ib * B_jb using Inner Strassen.
                # A_ib is a 2x2 matrix at global indices: base_a + 0..3
                # base_a = ib * 4
                # B_jb is a 2x2 matrix at global indices: base_b + 0..3
                # base_b = jb * 4
                
                base_a = ib * 4
                base_b = jb * 4
                
                # Use S2x2 to get the 7 rank-1 terms for this 2x2*2x2 product.
                for a_vec, b_vec, c_vec in S2x2:
                    # a_vec has 4 entries corresponding to local indices 0..3 of A_ib.
                    # b_vec has 4 entries corresponding to local indices 0..3 of B_jb.
                    # c_vec has 4 entries corresponding to local indices 0..3 of the result block.
                    
                    # We need to map the result block back to the global C matrix.
                    # The result of A_ib * B_jb is a 2x2 matrix.
                    # It contributes to C with a coefficient determined by C_blocks.
                    # Wait, C_blocks applies to the OUTER blocks.
                    # The outer term M_outer contributes to C_outer blocks.
                    # Specifically, C_outer[k] += coeff_c * M_outer.
                    # And M_outer is the sum of A_ib * B_jb terms.
                    # So the contribution to C_block_k is coeff_c * coeff_a * coeff_b * (A_ib * B_jb).
                    
                    # But which C block does the result of A_ib * B_jb go into?
                    # The outer Strassen formula gives:
                    # C00 = M1 + M4 - M5 + M7
                    # C01 = M3 + M5
                    # C10 = M2 + M4
                    # C11 = M1 - M2 + M3 + M6
                    
                    # So for a specific outer term M_outer_idx, we know which C blocks it affects.
                    # But here we are iterating over outer terms defined by their A and B block combinations.
                    # We lost the index of the outer term in the loop structure above?
                    # No, I defined outer_terms as a list of dicts. I should iterate with index.
                    
                    pass # Placeholder, will restructure loop
    
    # Restructure: Iterate over outer term index
    col = 0
    for t_idx, ot in enumerate(outer_terms):
        A_blocks = ot['A']
        B_blocks = ot['B']
        C_blocks = ot['C']
        
        # Determine which C blocks are affected by this outer term.
        # C_blocks is a dict mapping C_block_index to coefficient.
        
        # For each pair of A and B blocks involved:
        for ib in A_blocks.keys():
            for jb in B_blocks.keys():
                coeff_a = A_blocks[ib]
                coeff_b = B_blocks[jb]
                
                base_a = ib * 4
                base_b = jb * 4
                
                # Decompose A_ib * B_jb using Inner Strassen
                for a_vec, b_vec, c_vec in S2x2:
                    # a_vec, b_vec, c_vec are lists of 4 ints.
                    
                    # We need to add to U, V, W at column col.
                    # U[base_a + i] += coeff_a * a_vec[i]
                    # V[base_b + j] += coeff_b * b_vec[j]
                    
                    # For W:
                    # The result of this rank-1 term is a 2x2 matrix.
                    # It contributes to the C matrix.
                    # But which part of C?
                    # The outer term M_outer contributes to specific C blocks.
                    # The product A_ib * B_jb is a component of M_outer.
                    # So the result 2x2 matrix is placed at... where?
                    
                    # Wait, M_outer is a 2x2 matrix of 2x2 blocks? No.
                    # M_outer is the result of multiplying two 2x2 matrices (which are blocks of A and B).
                    # The result is a 2x2 matrix.
                    # This 2x2 matrix is ONE OF THE OUTER STRASSEN TERMS (e.g. M1).
                    # The outer Strassen formula says C00 (which is a 2x2 block of C) = M1 + ...
                    # So M1 is a 2x2 matrix.
                    # Therefore, the result of A_ib * B_jb is added to the 2x2 matrix M_outer.
                    # And M_outer is added to the appropriate C blocks.
                    
                    # So, if C_blocks says C_block_k gets coefficient c_k from this outer term,
                    # then the result 2x2 matrix is added to C_block_k with coefficient c_k.
                    
                    # But which C block k?
                    # C_blocks is a dict like {0: 1, 3: 1}.
                    # This means the outer term contributes to C_block_0 and C_block_3.
                    # But the result of A_ib * B_jb is a single 2x2 matrix.
                    # It is added to M_outer.
                    # And M_outer is added to ALL C blocks specified in C_blocks?
                    # Yes! Because M_outer is a single 2x2 matrix value.
                    # C00 (block) = M1 (block) + ...
                    # C11 (block) = M1 (block) + ...
                    # So the same 2x2 matrix M1 is added to C00 and C11.
                    
                    # Therefore, for each c_block in C_blocks:
                    for c_block_idx, coeff_c in C_blocks.items():
                        base_c = c_block_idx * 4
                        # c_vec has 4 entries corresponding to local indices 0..3 of the result 2x2 matrix.
                        # This result 2x2 matrix is placed at C_block_idx.
                        
                        # So W[base_c + k] += coeff_a * coeff_b * coeff_c * c_vec[k]
                        
                        for k in range(4):
                            U[base_a + k, col] += coeff_a * a_vec[k]
                            V[base_b + k, col] += coeff_b * b_vec[k]
                            W_mat[base_c + k, col] += coeff_a * coeff_b * coeff_c * c_vec[k]
                    
                    col += 1
                    
                    if time.time() - start_time > timeout:
                        # Return current best (which might be incomplete, but better than nothing? 
                        # Actually, if we break early, the rank is not 49, and it's likely invalid.
                        # It's better to just let it run or fail fast.
                        # Given the small size, it should finish in < 1s.
                        pass

    # Ensure exact integer/half-integer entries.
    # The entries are sums of products of integers, so they are integers.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W_mat.tolist()}
# EVOLVE-BLOCK-END
