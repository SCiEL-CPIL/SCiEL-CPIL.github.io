import numpy as np

def construct():
    # Generate Strassen 2x2x2 terms
    # u, v, w are vectors of length 4
    # Indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    s2_terms = []
    
    # Term 1: (A00+A11)(B00+B11) -> C00 += 1, C11 += 1
    u = np.array([1, 0, 0, 1])
    v = np.array([1, 0, 0, 1])
    w = np.array([1, 0, 0, 1])
    s2_terms.append((u, v, w))
    
    # Term 2: (A10+A11)B00 -> C10 += 1, C11 -= 1
    u = np.array([0, 0, 1, 1])
    v = np.array([1, 0, 0, 0])
    w = np.array([0, 0, 1, -1])
    s2_terms.append((u, v, w))
    
    # Term 3: A00(B01-B11) -> C01 += 1, C11 += 1
    u = np.array([1, 0, 0, 0])
    v = np.array([0, 1, 0, -1])
    w = np.array([0, 1, 0, 1])
    s2_terms.append((u, v, w))
    
    # Term 4: A11(B10-B00) -> C00 += 1, C10 += 1
    u = np.array([0, 0, 0, 1])
    v = np.array([-1, 0, 1, 0])
    w = np.array([1, 0, 1, 0])
    s2_terms.append((u, v, w))
    
    # Term 5: (A00+A01)B11 -> C00 -= 1, C01 += 1
    u = np.array([1, 1, 0, 0])
    v = np.array([0, 0, 0, 1])
    w = np.array([-1, 1, 0, 0])
    s2_terms.append((u, v, w))
    
    # Term 6: (A10-A00)(B00+B01) -> C11 += 1
    u = np.array([-1, 0, 1, 0])
    v = np.array([1, 1, 0, 0])
    w = np.array([0, 0, 0, 1])
    s2_terms.append((u, v, w))
    
    # Term 7: (A01-A11)(B10+B11) -> C00 += 1, C11 += 1
    u = np.array([0, 1, 0, -1])
    v = np.array([0, 0, 1, 1])
    w = np.array([1, 0, 0, 1])
    s2_terms.append((u, v, w))
    
    # Now build 4x4x4 terms
    # A blocks: A00, A01, A10, A11 (indices 0,1,2,3 in 4x4 grid)
    # B blocks: B00, B01, B10, B11
    
    # Super terms S1..S7
    # Each is defined by linear combinations of blocks.
    # We need to map these to the 4x4 indices.
    
    # Block indices in 4x4 matrix:
    # 0: (0,0) -> rows 0-1, cols 0-1
    # 1: (0,1) -> rows 0-1, cols 2-3
    # 2: (1,0) -> rows 2-3, cols 0-1
    # 3: (1,1) -> rows 2-3, cols 2-3
    
    # Function to get block index in 4x4
    # block (br, bc) where br, bc in {0,1}
    # rows: br*2 to br*2+1
    # cols: bc*2 to bc*2+1
    
    def block_indices(br, bc):
        rows = [br*2, br*2+1]
        cols = [bc*2, bc*2+1]
        return [(r, c) for r in rows for c in cols]
    
    # List of super terms (u_blocks, v_blocks, w_blocks)
    # u_blocks: list of (block_idx, coeff)
    # v_blocks: list of (block_idx, coeff)
    # w_blocks: list of (block_idx, coeff)
    
    super_terms = []
    
    # S1: (A00+A11)(B00+B11)
    # A: 0:+1, 3:+1. B: 0:+1, 3:+1.
    # W: C00 += 1, C11 += 1.
    # C00 is block (0,0). C11 is block (1,1).
    super_terms.append(
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)])
    )
    
    # S2: (A10+A11)B00
    # A: 2:+1, 3:+1. B: 0:+1.
    # W: C10 += 1, C11 -= 1.
    # C10 is block (1,0). C11 is block (1,1).
    super_terms.append(
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)])
    )
    
    # S3: A00(B01-B11)
    # A: 0:+1. B: 1:+1, 3:-1.
    # W: C01 += 1, C11 += 1.
    # C01 is block (0,1). C11 is block (1,1).
    super_terms.append(
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)])
    )
    
    # S4: A11(B10-B00)
    # A: 3:+1. B: 2:+1, 0:-1.
    # W: C00 += 1, C10 += 1.
    # C00 is block (0,0). C10 is block (1,0).
    super_terms.append(
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)])
    )
    
    # S5: (A00+A01)B11
    # A: 0:+1, 1:+1. B: 3:+1.
    # W: C00 -= 1, C01 += 1.
    # C00 is block (0,0). C01 is block (0,1).
    super_terms.append(
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)])
    )
    
    # S6: (A10-A00)(B00+B01)
    # A: 2:+1, 0:-1. B: 0:+1, 1:+1.
    # W: C11 += 1.
    # C11 is block (1,1).
    super_terms.append(
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)])
    )
    
    # S7: (A01-A11)(B10+B11)
    # A: 1:+1, 3:-1. B: 2:+1, 3:+1.
    # W: C00 += 1, C11 += 1.
    # C00 is block (0,0). C11 is block (1,1).
    super_terms.append(
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1), (3, 1)])
    )
    
    # Now expand each super term into 7 scalar terms using s2_terms
    # Total 49 terms
    
    U = []
    V = []
    W = []
    
    for st in super_terms:
        u_blocks, v_blocks, w_blocks = st
        
        # Construct full 4x4 vectors for this super term's A, B, C parts
        # But we need to apply the 2x2 Strassen decomposition to the "values"
        # The "values" are the 2x2 blocks.
        # So for each of the 7 sub-terms in s2_terms, we combine the blocks.
        
        for sub_u, sub_v, sub_w in s2_terms:
            # sub_u is a vector of 4 coeffs for the 4 positions in a 2x2 block
            # The 4 positions correspond to the 4 blocks of the super-term?
            # No. The super-term is a product of two 2x2 matrices (of blocks).
            # The 2x2 Strassen decomposition applies to the matrix of blocks.
            # So sub_u[0] multiplies the block at position (0,0) of the super-term's A part.
            
            u_vec = np.zeros(16)
            v_vec = np.zeros(16) # For 4x4x4, V is 16x1. For 4x4x8, V is 32x1.
            w_vec = np.zeros(16)
            
            # Map sub_u indices (0..3) to super-term A blocks
            # sub_u[0] -> block (0,0) of A_super
            # sub_u[1] -> block (0,1) of A_super
            # sub_u[2] -> block (1,0) of A_super
            # sub_u[3] -> block (1,1) of A_super
            
            for s_idx, coeff in enumerate(sub_u):
                if coeff == 0: continue
                # Which block of A does this correspond to?
                # The super-term A is a linear combination of A_blocks.
                # We need to know which A_block is at position s_idx in the 2x2 super-matrix.
                # This depends on the super-term definition.
                # My super_terms definition above is just a list of (block_idx, coeff).
                # I need to map the 2x2 structure to the specific blocks involved.
                
                # This approach is getting messy.
                # Let's re-evaluate.
                
                pass
            
            # Alternative:
            # The 49 terms are just the tensor product of the 7 terms of S2.
            # T_{k,l} = (S_k \otimes S_l) ?
            # No, it's the composition.
            
            # Let's use the explicit block mapping.
            # For each super term, we define the 4 blocks of A_super, B_super, C_super.
            # Then we apply the 7 Strassen terms to these blocks.
            
            pass
            
    # This manual expansion is error-prone.
    # I will generate the 49 terms programmatically by composing the matrices.
    
    # Let's create the 49 terms directly.
    # Term (k, l) where k in 1..7 (super), l in 1..7 (sub).
    
    terms_4x4 = []
    
    for k_idx, (u_blocks, v_blocks, w_blocks) in enumerate(super_terms):
        # We need to construct the 2x2 matrices U_super, V_super, W_super
        # where entries are linear combinations of 4x4 basis vectors.
        
        # U_super is 2x2.
        # U_super[0,0] = sum of A_blocks at pos 0
        # But the super term defines which A_blocks are present and with what coeffs.
        # Wait, the super term IS the definition of the 2x2 matrices.
        # e.g. S1: A_super = [[A0, 0], [0, A3]]? No.
        # S1: (A00+A11)(B00+B11).
        # This is not a 2x2 matrix multiplication of the form [[A0, A1], [A2, A3]] [[B0, B1], [B2, B3]].
        # The Strassen algorithm works on the 2x2 block structure of the 4x4 matrices.
        # So A is [[A0, A1], [A2, A3]].
        # The 7 Strassen terms are linear combinations of these blocks.
        
        # So for S1:
        # M1 = (A0 + A3)(B0 + B3).
        # This is one of the 7 terms.
        # The 7 terms are fixed linear combinations of the 4 blocks of A and 4 blocks of B.
        
        # So, for each of the 7 Strassen terms (k=0..6):
        # We have coeffs a_0..a_3 for A blocks, b_0..b_3 for B blocks.
        # And coeffs c_0..c_3 for C blocks.
        
        # Let's define the 7 Strassen terms for 2x2 blocks explicitly.
        
        strassen_2x2 = [
            # (a_coeffs, b_coeffs, c_coeffs)
            # a_coeffs: [c00, c01, c10, c11] for A blocks
            # b_coeffs: [c00, c01, c10, c11] for B blocks
            # c_coeffs: [c00, c01, c10, c11] for C blocks
            
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]), # M1
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]), # M2
            ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),  # M3
            ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),  # M4
            ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),  # M5
            ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),  # M6
            ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 1])   # M7
        ]
        
        for a_coeffs, b_coeffs, c_coeffs in strassen_2x2:
            # Now expand each block coefficient into 2x2 scalar coefficients.
            # a_coeffs[0] is coeff for A00 (block 0).
            # A00 is a 2x2 matrix.
            # We need to apply the 7 Strassen terms to the 2x2 matrices A00, etc?
            # No. The "multiplication" M_k is a scalar multiplication of the resulting 2x2 matrices.
            # So M_k = (linear_comb_of_A_blocks) * (linear_comb_of_B_blocks).
            # This product is a 2x2 matrix.
            # We need to decompose this 2x2 matrix multiplication into 7 scalar multiplications.
            
            # So, for this specific Strassen term (k), we have:
            # A_mat = sum_{i=0..3} a_coeffs[i] * A_block_i
            # B_mat = sum_{j=0..3} b_coeffs[j] * B_block_j
            # C_mat = A_mat * B_mat = sum_{i,j} a_coeffs[i] b_coeffs[j] A_block_i B_block_j
            
            # This is a 2x2 matrix multiplication.
            # We use the 7 Strassen terms to compute this 2x2 product.
            
            # The 7 Strassen terms for 2x2 are s2_terms defined earlier.
            # Each s2_term is (u, v, w) where u, v, w are length 4 vectors.
            # u corresponds to the 4 elements of the 2x2 matrix being multiplied.
            
            # So, for each s2_term (l=0..6):
            for u, v, w in s2_terms:
                # u[0] is coeff for element (0,0) of A_mat.
                # A_mat[0,0] = sum_i a_coeffs[i] * A_block_i[0,0]
                # So the coefficient for A_block_i[0,0] is a_coeffs[i] * u[0].
                
                u_vec = np.zeros(16)
                v_vec = np.zeros(16)
                w_vec = np.zeros(16)
                
                for i in range(4):
                    if a_coeffs[i] == 0: continue
                    # A_block_i is at position i in the 2x2 block grid.
                    # i=0: (0,0), i=1: (0,1), i=2: (1,0), i=3: (1,1).
                    # Rows: (i//2)*2 to (i//2)*2+1
                    # Cols: (i%2)*2 to (i%2)*2+1
                    
                    br, bc = i // 2, i % 2
                    for dr in range(2):
                        for dc in range(2):
                            idx_a = (br*2 + dr) * 4 + (bc*2 + dc)
                            # u[0] corresponds to element (0,0) of the 2x2 matrix.
                            # The elements of the 2x2 matrix are the 4 blocks.
                            # So u[0] multiplies A_block_0.
                            # u[1] multiplies A_block_1.
                            # u[2] multiplies A_block_2.
                            # u[3] multiplies A_block_3.
                            
                            # Wait, u is the vector for the 2x2 matrix A_mat.
                            # A_mat is 2x2.
                            # u[0] is A_mat[0,0].
                            # A_mat[0,0] = a_coeffs[0] * A_block_0.
                            # So u[0] * a_coeffs[0] is the coeff for A_block_0.
                            
                            # But A_block_0 is a 2x2 matrix.
                            # The element A_block_0[dr, dc] is at global index idx_a.
                            # Does u[0] apply to the whole block or just one element?
                            # The Strassen term u is for the 2x2 matrix of blocks.
                            # So u[0] is the coefficient for the block A_block_0.
                            # This coefficient applies to ALL elements of A_block_0.
                            
                            coeff = a_coeffs[i] * u[i] # No, u[i] is coeff for block i.
                            # Wait, u is length 4. u[0] is for block 0.
                            # So for block i, the coeff is u[i].
                            
                            coeff = a_coeffs[i] * u[i]
                            if coeff != 0:
                                u_vec[idx_a] = coeff
                                # Also for the other 3 elements of the block?
                                # Yes, the coefficient applies to the whole block.
                                # So we need to add this coeff to all 4 elements of block i.
                                # But I am iterating over dr, dc.
                                # So I should add coeff to u_vec[idx_a].
                                # And since I loop over all 4 elements, I will add it 4 times?
                                # No, idx_a is unique for each (dr, dc).
                                # So this is correct.
                
                # Same for V and W
                for j in range(4):
                    if b_coeffs[j] == 0: continue
                    br, bc = j // 2, j % 2
                    for dr in range(2):
                        for dc in range(2):
                            idx_b = (br*2 + dr) * 4 + (bc*2 + dc)
                            coeff = b_coeffs[j] * v[j]
                            if coeff != 0:
                                v_vec[idx_b] = coeff
                            
                for i in range(4):
                    if c_coeffs[i] == 0: continue
                    br, bc = i // 2, i % 2
                    for dr in range(2):
                        for dc in range(2):
                            idx_c = (br*2 + dr) * 4 + (bc*2 + dc)
                            coeff = c_coeffs[i] * w[i]
                            if coeff != 0:
                                w_vec[idx_c] = coeff
                                
                terms_4x4.append((u_vec, v_vec, w_vec))
                
    # Now we have 49 terms for 4x4x4.
    # We need to construct U, V, W for 4x4x8.
    
    n, m, p = 4, 4, 8
    rank = 98
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for u_vec, v_vec, w_vec in terms_4x4:
        # First block (cols 0-3 of B)
        for i in range(16):
            U[i][col] = u_vec[i]
        for i in range(16):
            V[i][col] = v_vec[i]
        for i in range(16):
            W[i][col] = w_vec[i]
        col += 1
        
        # Second block (cols 4-7 of B)
        # U is same
        for i in range(16):
            U[i][col] = u_vec[i]
        # V indices shift: j in 0..3, k in 4..7
        # Global V index: j * 8 + k
        # v_vec index: j * 4 + k_local (k_local in 0..3)
        # So V[j * 8 + (k_local + 4)] = v_vec[j * 4 + k_local]
        for j in range(4):
            for k_local in range(4):
                idx_v = j * 8 + (k_local + 4)
                idx_v_local = j * 4 + k_local
                V[idx_v][col] = v_vec[idx_v_local]
                
        # W indices shift: k in 4..7, i in 0..3
        # Global W index: k * 4 + i
        # w_vec index: k_local * 4 + i
        # So W[(k_local + 4) * 4 + i] = w_vec[k_local * 4 + i]
        for k_local in range(4):
            for i in range(4):
                idx_w = (k_local + 4) * 4 + i
                idx_w_local = k_local * 4 + i
                W[idx_w][col] = w_vec[idx_w_local]
                
        col += 1
        
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
