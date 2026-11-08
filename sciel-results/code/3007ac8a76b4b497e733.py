"""Matrix multiplication 4,4,9 rank-minimal decomposition.

Uses a block decomposition with Strassen:
  - Split B (4x9) into B1 (4x4), B2 (4x4), B3 (4x1)
  - A*B1 and A*B2 use Strassen for 4x4*4x4 (rank 49 each)
  - A*B3 uses schoolbook for 4x4*4x1 (rank 16)
  - Total rank: 49 + 49 + 16 = 114

Strassen for 2x2*2x2 (rank 7):
  m1 = (a+d)(e+h)
  m2 = (c+d)e
  m3 = a(f-h)
  m4 = d(g-e)
  m5 = (a+b)h
  m6 = (c-a)(e+f)
  m7 = (b-d)(g+h)

  c11 = m1 + m4 - m5 + m7
  c12 = m3 + m5
  c21 = m2 + m4
  c22 = m1 - m2 + m3 + m6
"""

DIMENSIONS = [4, 4, 9]

def construct():
    """Return a valid tensor decomposition for 4x4 * 4x9 matrix multiplication.
    
    Uses Strassen-based block decomposition with rank 114:
    - Two 4x4*4x4 Strassen blocks: 49 + 49 = 98
    - One 4x4*4x1 schoolbook: 16
    - Total: 114
    """
    n, m, p = DIMENSIONS  # n=4, m=4, p=9
    
    # We'll build U, V, W matrices
    # U has shape (n*m, rank) = (16, rank)
    # V has shape (m*p, rank) = (36, rank)  
    # W has shape (p*n, rank) = (36, rank)
    
    # Start with empty lists
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_vec, v_vec, w_vec):
        """Add a rank-1 term to the decomposition."""
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
    
    # Strassen decomposition for 2x2 * 2x2
    # Input: A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # Output: C = A*B = [[c11,c12],[c21,c22]]
    # 
    # For the tensor decomposition, each "multiplication" m_i is a bilinear form
    # that multiplies a linear combination of A entries by a linear combination of B entries
    # and adds it to specific C entries.
    #
    # We need to express each m_i as a product of:
    #   (sum of alpha_ij * A_ij) * (sum of beta_jk * B_jk)
    # and then m_i contributes to C entries via coefficients.
    
    # Strassen's 7 multiplications for 2x2:
    # m1 = (a+d)(e+h) -> contributes to c11 (+1), c22 (+1)
    # m2 = (c+d)e    -> contributes to c21 (+1), c22 (-1)
    # m3 = a(f-h)    -> contributes to c12 (+1), c22 (+1)
    # m4 = d(g-e)    -> contributes to c11 (+1), c21 (+1)
    # m5 = (a+b)h    -> contributes to c11 (-1), c12 (+1)
    # m6 = (c-a)(e+f) -> contributes to c22 (+1)
    # m7 = (b-d)(g+h) -> contributes to c11 (+1)
    
    # For each m_i, we have:
    # m_i = (sum over (i,j) of u_i * A_ij) * (sum over (j,k) of v_k * B_jk)
    # and m_i * w_c contributes to C_c
    
    # Let's define the Strassen terms for 2x2:
    # Each term has:
    #   - u: coefficients for A entries [a, b, c, d] (indices 0,1,2,3)
    #   - v: coefficients for B entries [e, f, g, h] (indices 0,1,2,3)
    #   - w: coefficients for C entries [c11, c12, c21, c22] (indices 0,1,2,3)
    
    strassen_2x2_terms = [
        # m1 = (a+d)(e+h), contributes +1 to c11, +1 to c22
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # m2 = (c+d)e, contributes +1 to c21, -1 to c22
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # m3 = a(f-h), contributes +1 to c12, +1 to c22
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        # m4 = d(g-e), contributes +1 to c11, +1 to c21
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        # m5 = (a+b)h, contributes -1 to c11, +1 to c12
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # m6 = (c-a)(e+f), contributes +1 to c22
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # m7 = (b-d)(g+h), contributes +1 to c11
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    def strassen_2x2_block(A_indices, B_indices, C_indices):
        """
        Add Strassen terms for a 2x2*2x2 block.
        
        A_indices: mapping from local (i,j) to global U index
                   local indices: 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1)
        B_indices: mapping from local (j,k) to global V index
                   local indices: 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1)
        C_indices: mapping from local (i,k) to global W index
                   local indices: 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1)
        """
        for u_coeff, v_coeff, w_coeff in strassen_2x2_terms:
            u_vec = [0] * 16
            v_vec = [0] * 36
            w_vec = [0] * 36
            
            for local_idx, coeff in u_coeff.items():
                gi, gj = A_indices[local_idx]
                u_vec[gi * m + gj] = coeff
            
            for local_idx, coeff in v_coeff.items():
                gj, gk = B_indices[local_idx]
                v_vec[gj * p + gk] = coeff
            
            for local_idx, coeff in w_coeff.items():
                gi, gk = C_indices[local_idx]
                w_vec[gk * n + gi] = coeff
            
            add_term(u_vec, v_vec, w_vec)
    
    # Now for 4x4 * 4x4, we view it as 2x2 blocks of 2x2 submatrices
    # A = [[A11, A12], [A21, A22]] where each block is 2x2
    # B = [[B11, B12], [B21, B22]] where each block is 2x2
    
    # The 7 Strassen multiplications at the block level:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # Each Mi is a 2x2*2x2 product, decomposed with Strassen (rank 7)
    # Total: 7 * 7 = 49 terms
    
    def strassen_4x4_block(A_base_i, A_base_j, B_base_j, B_base_k, C_base_i, C_base_k):
        """
        Add Strassen terms for a 4x4*4x4 block.
        
        A is at rows A_base_i..A_base_i+3, cols A_base_j..A_base_j+3
        B is at rows B_base_j..B_base_j+3, cols B_base_k..B_base_k+3
        C is at rows C_base_i..C_base_i+3, cols C_base_k..C_base_k+3
        """
        # Define block index mappings
        # A blocks: A11 at (0,0), A12 at (0,2), A21 at (2,0), A22 at (2,2)
        # Each block is 2x2, so local indices within block:
        # (0,0), (0,1), (1,0), (1,1)
        
        def get_A_indices(block_row, block_col):
            """Get global indices for a 2x2 A block at (block_row, block_col)."""
            indices = {}
            for li in range(2):
                for lj in range(2):
                    local_idx = li * 2 + lj
                    gi = A_base_i + block_row * 2 + li
                    gj = A_base_j + block_col * 2 + lj
                    indices[local_idx] = (gi, gj)
            return indices
        
        def get_B_indices(block_row, block_col):
            """Get global indices for a 2x2 B block at (block_row, block_col)."""
            indices = {}
            for lj in range(2):
                for lk in range(2):
                    local_idx = lj * 2 + lk
                    gj = B_base_j + block_row * 2 + lj
                    gk = B_base_k + block_col * 2 + lk
                    indices[local_idx] = (gj, gk)
            return indices
        
        def get_C_indices(block_row, block_col):
            """Get global indices for a 2x2 C block at (block_row, block_col)."""
            indices = {}
            for li in range(2):
                for lk in range(2):
                    local_idx = li * 2 + lk
                    gi = C_base_i + block_row * 2 + li
                    gk = C_base_k + block_col * 2 + lk
                    indices[local_idx] = (gi, gk)
            return indices
        
        # Define the 7 Strassen block multiplications
        # Each is a sum of products of A-blocks and B-blocks
        # We need to express each as (linear combo of A entries) * (linear combo of B entries)
        
        # M1 = (A11+A22)(B11+B22)
        # This is a sum of 4 products: A11*B11, A11*B22, A22*B11, A22*B22
        # Each is a 2x2*2x2 product
        # We can decompose this using the fact that (X+Y)(Z+W) = XZ + XW + YZ + YW
        # But we want to use Strassen structure.
        
        # Actually, for the tensor decomposition, we need each "multiplication" to be
        # a single bilinear term: (sum u_i * A_i) * (sum v_j * B_j)
        
        # For M1 = (A11+A22)(B11+B22), this is not a single bilinear term in the
        # original A and B entries. We need to expand it.
        
        # Let me reconsider. The Strassen algorithm at the block level computes:
        # M1 = (A11+A22)(B11+B22)
        # This requires computing A11+A22 (a 2x2 matrix) and B11+B22 (a 2x2 matrix),
        # then multiplying them. The multiplication of two 2x2 matrices has rank 7.
        
        # So M1 contributes 7 rank-1 terms, where each term is:
        #   (linear combo of entries in A11+A22) * (linear combo of entries in B11+B22)
        
        # A11+A22 has entries:
        #   (A11[0,0] + A22[0,0]), (A11[0,1] + A22[0,1])
        #   (A11[1,0] + A22[1,0]), (A11[1,1] + A22[1,1])
        
        # So for Strassen on (A11+A22)*(B11+B22):
        # The "a" entry is A11[0,0] + A22[0,0]
        # The "b" entry is A11[0,1] + A22[0,1]
        # etc.
        
        # This means for each Strassen term in the inner 2x2 multiplication,
        # we need to sum the contributions from A11 and A22 (or B11 and B22).
        
        # Let me define a helper that adds Strassen terms where the A and B
        # entries are sums of contributions from multiple blocks.
        
        def add_strassen_with_sums(A_block_list, B_block_list, C_block_list, C_coeff=1):
            """
            Add Strassen terms for sum(A_blocks) * sum(B_blocks) contributing to sum(C_blocks) * C_coeff.
            
            A_block_list: list of block positions (row, col) in A
            B_block_list: list of block positions (row, col) in B
            C_block_list: list of block positions (row, col) in C
            """
            for u_coeff, v_coeff, w_coeff in strassen_2x2_terms:
                u_vec = [0] * 16
                v_vec = [0] * 36
                w_vec = [0] * 36
                
                # For each local A index in u_coeff, add contributions from all A blocks
                for local_idx, coeff in u_coeff.items():
                    li = local_idx // 2
                    lj = local_idx % 2
                    for (br, bc) in A_block_list:
                        gi = A_base_i + br * 2 + li
                        gj = A_base_j + bc * 2 + lj
                        u_vec[gi * m + gj] += coeff
                
                # For each local B index in v_coeff, add contributions from all B blocks
                for local_idx, coeff in v_coeff.items():
                    lj = local_idx // 2
                    lk = local_idx % 2
                    for (br, bc) in B_block_list:
                        gj = B_base_j + br * 2 + lj
                        gk = B_base_k + bc * 2 + lk
                        v_vec[gj * p + gk] += coeff
                
                # For each local C index in w_coeff, add contributions to all C blocks
                for local_idx, coeff in w_coeff.items():
                    li = local_idx // 2
                    lk = local_idx % 2
                    for (br, bc) in C_block_list:
                        gi = C_base_i + br * 2 + li
                        gk = C_base_k + bc * 2 + lk
                        w_vec[gk * n + gi] += coeff * C_coeff
                
                add_term(u_vec, v_vec, w_vec)
        
        # M1 = (A11+A22)(B11+B22), contributes +1 to C11, +1 to C22
        add_strassen_with_sums(
            [(0,0), (1,1)],  # A11, A22
            [(0,0), (1,1)],  # B11, B22
            [(0,0), (1,1)],  # C11, C22
            C_coeff=1
        )
        
        # M2 = (A21+A22)B11, contributes +1 to C21, -1 to C22
        add_strassen_with_sums(
            [(1,0), (1,1)],  # A21, A22
            [(0,0)],         # B11
            [(1,0), (1,1)],  # C21, C22
            C_coeff=1
        )
        # But M2 contributes -1 to C22, so we need to handle the sign
        # Actually, the C_coeff should be per C block. Let me fix this.
        
        # Let me redo this more carefully.
        # I'll clear the terms added so far and restart with a cleaner approach.
        
        # Actually, let me just directly implement each of the 7 Strassen multiplications
        # with proper signs.
        
        # For now, let me just add the schoolbook for the 4x4*4x4 part to ensure correctness,
        # and we can optimize later.
        
        # Schoolbook for 4x4*4x4: 16 terms
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    u_vec = [0] * 16
                    v_vec = [0] * 36
                    w_vec = [0] * 36
                    u_vec[(A_base_i + i) * m + (A_base_j + j)] = 1
                    v_vec[(B_base_j + j) * p + (B_base_k + k)] = 1
                    w_vec[(C_base_k + k) * n + (C_base_i + i)] = 1
                    add_term(u_vec, v_vec, w_vec)
    
    # For simplicity and correctness, let's use schoolbook for now
    # and verify the structure is correct.
    
    # Actually, let me implement the Strassen properly.
    
    # Clear previous terms
    U_cols.clear()
    V_cols.clear()
    W_cols.clear()
    
    # Part 1: A * B1 where B1 is columns 0-3 of B (4x4)
    # Part 2: A * B2 where B2 is columns 4-7 of B (4x4)
    # Part 3: A * B3 where B3 is column 8 of B (4x1)
    
    # For Parts 1 and 2, use Strassen for 4x4*4x4 (rank 49 each)
    # For Part 3, use schoolbook for 4x4*4x1 (rank 16)
    
    # Let me implement Strassen for 4x4*4x4 properly.
    
    def strassen_4x4(A_base_i, A_base_j, B_base_j, B_base_k, C_base_i, C_base_k):
        """Add Strassen decomposition for 4x4 * 4x4. Produces exactly 49 rank-1 terms."""
        
        def get_A_entry(br, bc, li, lj):
            gi = A_base_i + br * 2 + li
            gj = A_base_j + bc * 2 + lj
            return gi * m + gj
        
        def get_B_entry(br, bc, lj, lk):
            gj = B_base_j + br * 2 + lj
            gk = B_base_k + bc * 2 + lk
            return gj * p + gk
        
        def get_C_entry(br, bc, li, lk):
            gi = C_base_i + br * 2 + li
            gk = C_base_k + bc * 2 + lk
            return gk * n + gi
        
        # Strassen 2x2 meta-terms: (A_coeffs, B_coeffs, C_contribs)
        # A_coeffs: (li,lj) -> coeff for A' entries
        # B_coeffs: (lj,lk) -> coeff for B' entries  
        # C_contribs: (li,lk) -> coeff for C' entries
        strassen_meta_terms = [
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
            ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
            ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
            ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
            ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
            ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
            ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
        ]
        
        def add_block_multiplication(A_side, B_side, C_side):
            """Add 7 Strassen terms for (sum A_side blocks) * (sum B_side blocks) -> C_side."""
            for A_coeffs, B_coeffs, C_contribs in strassen_meta_terms:
                u_vec = [0] * 16
                v_vec = [0] * 36
                w_vec = [0] * 36
                
                for (li, lj), meta_coeff in A_coeffs.items():
                    for (br, bc, block_coeff) in A_side:
                        gi = A_base_i + br * 2 + li
                        gj = A_base_j + bc * 2 + lj
                        u_vec[gi * m + gj] += meta_coeff * block_coeff
                
                for (lj, lk), meta_coeff in B_coeffs.items():
                    for (br, bc, block_coeff) in B_side:
                        gj = B_base_j + br * 2 + lj
                        gk = B_base_k + bc * 2 + lk
                        v_vec[gj * p + gk] += meta_coeff * block_coeff
                
                for (li, lk), meta_coeff in C_contribs.items():
                    for (br, bc), block_coeff in C_side.items():
                        gi = C_base_i + br * 2 + li
                        gk = C_base_k + bc * 2 + lk
                        w_vec[gk * n + gi] += meta_coeff * block_coeff
                
                add_term(u_vec, v_vec, w_vec)
        
        # 7 block-level Strassen multiplications for 4x4*4x4
        add_block_multiplication(
            A_side=[(0,0,1), (1,1,1)],
            B_side=[(0,0,1), (1,1,1)],
            C_side={(0,0): 1, (1,1): 1}
        )
        add_block_multiplication(
            A_side=[(1,0,1), (1,1,1)],
            B_side=[(0,0,1)],
            C_side={(1,0): 1, (1,1): -1}
        )
        add_block_multiplication(
            A_side=[(0,0,1)],
            B_side=[(0,1,1), (1,1,-1)],
            C_side={(0,1): 1, (1,1): 1}
        )
        add_block_multiplication(
            A_side=[(1,1,1)],
            B_side=[(1,0,1), (0,0,-1)],
            C_side={(0,0): 1, (1,0): 1}
        )
        add_block_multiplication(
            A_side=[(0,0,1), (0,1,1)],
            B_side=[(1,1,1)],
            C_side={(0,0): -1, (0,1): 1}
        )
        add_block_multiplication(
            A_side=[(1,0,1), (0,0,-1)],
            B_side=[(0,0,1), (0,1,1)],
            C_side={(1,1): 1}
        )
        add_block_multiplication(
            A_side=[(0,1,1), (1,1,-1)],
            B_side=[(1,0,1), (1,1,1)],
            C_side={(0,0): 1}
        )
    
    # Now call strassen_4x4 for the two 4x4*4x4 parts.
    
    # Part 1: A * B1 where B1 is columns 0-3
    strassen_4x4(
        A_base_i=0, A_base_j=0,
        B_base_j=0, B_base_k=0,
        C_base_i=0, C_base_k=0
    )
    
    # Part 2: A * B2 where B2 is columns 4-7
    strassen_4x4(
        A_base_i=0, A_base_j=0,
        B_base_j=0, B_base_k=4,
        C_base_i=0, C_base_k=4
    )
    
    # Part 3: A * B3 where B3 is column 8 (4x1)
    # Schoolbook: 16 terms
    for i in range(4):
        for j in range(4):
            u_vec = [0] * 16
            v_vec = [0] * 36
            w_vec = [0] * 36
            u_vec[i * m + j] = 1
            v_vec[j * p + 8] = 1
            w_vec[8 * n + i] = 1
            add_term(u_vec, v_vec, w_vec)
    
    # Build U, V, W matrices
    rank = len(U_cols)
    U = [[U_cols[col][row] for col in range(rank)] for row in range(16)]
    V = [[V_cols[col][row] for col in range(rank)] for row in range(36)]
    W = [[W_cols[col][row] for col in range(rank)] for row in range(36)]
    
    return {"U": U, "V": V, "W": W}
