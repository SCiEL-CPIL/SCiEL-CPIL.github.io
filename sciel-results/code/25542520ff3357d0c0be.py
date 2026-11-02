import numpy as np

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS  # n=4, m=4, p=7
    
    # Rank calculation:
    # 4x4x4 (Strassen) rank = 49
    # 4x4x3 (Schoolbook) rank = 48
    # Total rank = 97
    rank = 97
    
    # Initialize factor matrices
    # U: (n*m, rank) = (16, 97)
    # V: (m*p, rank) = (28, 97)
    # W: (p*n, rank) = (28, 97)
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    # Helper to add a term to the decomposition
    # a_idx: index in U (i*m + j)
    # b_idx: index in V (j*p + k)
    # c_idx: index in W (k*n + i)
    # val: coefficient (should be +/- 1 for Strassen, 1 for Schoolbook)
    def add_term(col, a_idx, b_idx, c_idx, val):
        U[a_idx, col] += val
        V[b_idx, col] += val
        W[c_idx, col] += val

    # -------------------------------------------------------------------------
    # PART 1: Strassen Decomposition for 4x4 x 4x4 -> 4x4
    # -------------------------------------------------------------------------
    # We split A (4x4), B_left (4x4), C_left (4x4) into 2x2 blocks.
    # A = [[A11, A12], [A21, A22]]
    # B = [[B11, B12], [B21, B22]]
    # C = [[C11, C12], [C21, C22]]
    
    # Strassen Formulas:
    # M1 = (A11 + A22) (B11 + B22)
    # M2 = (A21 + A22) B11
    # M3 = A11 (B21 - B22)
    # M4 = A22 (B12 - B11)
    # M5 = (A11 + A12) B22
    # M6 = (A21 - A11) (B11 + B12)
    # M7 = (A12 - A22) (B21 + B22)
    
    # C11 = M1 - M4 + M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We apply this logic recursively to the 2x2 blocks, but since we need explicit
    # vectors for the 1x1 elements, we will generate the 49 terms manually.
    # Each 2x2 Strassen multiplication involves 7 scalar multiplications.
    # There are 4 "outer" 2x2 multiplications? No, Strassen replaces 8 multiplications
    # of 2x2 blocks (which are 4x4 multiplications) with 7 multiplications of 2x2 blocks.
    # Wait, standard Strassen for 4x4:
    # Split 4x4 into four 2x2.
    # The "multiplication" of two 2x2 matrices can be done in 7 scalar multiplications
    # of their components? No, Strassen is for matrices of size N.
    # If we apply Strassen to 4x4 matrices, we treat them as 2x2 blocks of size 2x2.
    # The recursive step is: M_k = (linear_comb_A) * (linear_comb_B).
    # The product of two 2x2 matrices requires rank 7 (Strassen) or 4 (Schoolbook).
    # If we use Strassen for the top level, we have 7 terms. Each term is a product
    # of two 2x2 matrices.
    # If we compute each 2x2 product using Strassen, each contributes 7 rank.
    # Total rank = 7 * 7 = 49.
    # If we compute each 2x2 product using Schoolbook, each contributes 4 rank.
    # Total rank = 7 * 4 = 28? No, that's not right.
    
    # Let's stick to the simplest correct path:
    # Decompose 4x4x4 into 16 scalar multiplications? No, that's schoolbook rank 64.
    # Decompose using Strassen:
    # The tensor M_444 has rank 49 using Strassen.
    # We need to explicitly define the 49 vectors.
    
    # Let's define the Strassen vectors for 2x2x2 first, then lift them.
    # Actually, it is safer to just list the 49 terms for 4x4x4 directly.
    # A 4x4 matrix can be seen as a 2x2 matrix of 2x2 blocks.
    # Let A = [[A00, A01], [A10, A11]] where Aij are 2x2.
    # Let B = [[B00, B01], [B10, B11]] where Bij are 2x2.
    # C = A B.
    
    # Strassen for C = AB:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11(B01 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # C00 = M1 - M4 + M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Now, each M_k is a product of two 2x2 matrices.
    # To get the scalar decomposition, we must decompose each 2x2x2 product.
    # We will use the standard Strassen decomposition for the 2x2x2 sub-problems.
    # This results in 7 * 7 = 49 scalar multiplications.
    
    # Let's define the 2x2 Strassen terms explicitly for a generic 2x2 product X*Y.
    # X = [[x00, x01], [x10, x11]]
    # Y = [[y00, y01], [y10, y11]]
    # Z = X Y
    # S1 = (x00 + x11)(y00 + y11)
    # S2 = (x10 + x11)y00
    # S3 = x00(y10 - y11)
    # S4 = x11(y01 - y00)
    # S5 = (x00 + x01)y11
    # S6 = (x10 - x00)(y00 + y01)
    # S7 = (x01 - x11)(y10 + y11)
    # Z00 = S1 - S4 + S5 + S7
    # Z01 = S3 + S5
    # Z10 = S2 + S4
    # Z11 = S1 - S2 + S3 + S6
    
    # We will iterate through the 7 top-level Strassen terms (M1..M7).
    # For each term, we identify the linear combination of A blocks (X) and B blocks (Y).
    # Then we decompose the 2x2 product X*Y into 7 scalar terms.
    
    # Indices in 4x4 matrix A (row r, col c) -> index r*4+c
    # Block A00: (0,0)-(1,1) -> idx 0,1,4,5
    # Block A01: (0,2)-(1,3) -> idx 2,3,6,7
    # Block A10: (2,0)-(3,1) -> idx 8,9,12,13
    # Block A11: (2,2)-(3,3) -> idx 10,11,14,15
    
    # Same for B.
    
    # Define block indices
    def block_indices(block_row, block_col):
        """Returns the 4 linear indices for a 2x2 block in a 4x4 matrix."""
        r_start = block_row * 2
        c_start = block_col * 2
        idxs = []
        for dr in range(2):
            for dc in range(2):
                idxs.append((r_start + dr) * 4 + (c_start + dc))
        return idxs
    
    A_blocks = {
        (0,0): block_indices(0,0), (0,1): block_indices(0,1),
        (1,0): block_indices(1,0), (1,1): block_indices(1,1)
    }
    B_blocks = {
        (0,0): block_indices(0,0), (0,1): block_indices(0,1),
        (1,0): block_indices(1,0), (1,1): block_indices(1,1)
    }
    
    # The 7 top-level Strassen linear combinations for A and B
    # Format: { (block_row, block_col): coeff }
    top_level_terms = [
        # M1
        ( {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1} ),
        # M2
        ( {(1,0): 1, (1,1): 1}, {(0,0): 1} ),
        # M3
        ( {(0,0): 1}, {(1,0): 1, (1,1): -1} ),
        # M4
        ( {(1,1): 1}, {(0,1): 1, (0,0): -1} ),
        # M5
        ( {(0,0): 1, (0,1): 1}, {(1,1): 1} ),
        # M6
        ( {(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1} ),
        # M7
        ( {(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1} )
    ]
    
    # The 7 bottom-level Strassen linear combinations for 2x2 matrices
    # X is formed from A blocks, Y from B blocks.
    # We need to map the "virtual" 2x2 entries of X and Y to the actual 4x4 indices.
    # If A_block is e.g., A00, its elements are A[0,0], A[0,1], A[1,0], A[1,1].
    # In the 2x2 sub-problem, X00 corresponds to A[0,0] (if coeff is 1), etc.
    
    # Bottom level Strassen terms for 2x2 product X*Y:
    # S1: (X00+X11)(Y00+Y11)
    # S2: (X10+X11)Y00
    # S3: X00(Y10-Y11)
    # S4: X11(Y01-Y00)
    # S5: (X00+X01)Y11
    # S6: (X10-X00)(Y00+Y01)
    # S7: (X01-X11)(Y10+Y11)
    
    # Mapping of 2x2 sub-indices to 4x4 global indices given a top-level term's A/B block coefficients.
    # This is complex because the "coefficients" from the top level apply to the WHOLE block.
    # So if Top Term A has A00:1, then X00 (local) gets coeff 1 * A[0,0], X01 gets 1*A[0,1], etc.
    
    # Let's define the 7 bottom-level terms as linear combinations of local X and Y entries.
    # Local X: [[x00, x01], [x10, x11]]
    # Local Y: [[y00, y01], [y10, y11]]
    
    bottom_level_specs = [
        # (X coeffs, Y coeffs) -> Z contribution
        # S1
        ( {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1} ),
        # S2
        ( {(1,0): 1, (1,1): 1}, {(0,0): 1} ),
        # S3
        ( {(0,0): 1}, {(1,0): 1, (1,1): -1} ),
        # S4
        ( {(1,1): 1}, {(0,1): 1, (0,0): -1} ),
        # S5
        ( {(0,0): 1, (0,1): 1}, {(1,1): 1} ),
        # S6
        ( {(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1} ),
        # S7
        ( {(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1} )
    ]
    
    # Z contributions for each S_k to C blocks
    # C00 += S1 - S4 + S5 + S7
    # C01 += S3 + S5
    # C10 += S2 + S4
    # C11 += S1 - S2 + S3 + S6
    
    # We need to map the "Result Block" (e.g., C00) and "Local Result Entry" (e.g., Z00)
    # to the global C index.
    
    # For each of the 7 top-level terms (M_k):
    # We have specific A and B block linear combinations.
    # For each of the 7 bottom-level terms (S_j):
    # We have specific X and Y local entry linear combinations.
    # The scalar term is (Sum of A contributions) * (Sum of B contributions).
    # The contribution to C is the product of these sums?
    # NO. The decomposition is linear.
    # Term = (Vector A) * (Vector B) * (Vector C).
    # Vector A is the sum of the actual A elements involved.
    # Vector B is the sum of the actual B elements involved.
    # Vector C is the sum of the C elements affected.
    
    # Let's construct the 49 terms.
    col = 0
    for t_idx, (top_A, top_B) in enumerate(top_level_terms):
        for b_idx, (bot_X, bot_Y) in enumerate(bottom_level_specs):
            # Determine the global A elements involved and their coefficients
            # top_A is { (br, bc): coeff }
            # bot_X is { (lr, lc): coeff } (local row, local col)
            
            # The "virtual" 2x2 matrix X is formed by taking the blocks from A
            # specified by top_A and scaling them.
            # Then bot_X picks specific entries from this virtual X.
            
            # Wait, this logic is slightly flawed.
            # Strassen is: M_k = (Linear_A_k) * (Linear_B_k).
            # Here Linear_A_k is a 2x2 matrix formed from 2x2 blocks of A.
            # Let's say Linear_A_k = c1*A00 + c2*A01 + ...
            # Then the product M_k = (c1*A00 + ...) * (d1*B00 + ...)
            # This is NOT equal to (c1*A00)*(d1*B00) + ...
            # It IS a bilinear form.
            # The expansion is:
            # M_k = sum_{blocks i} sum_{blocks j} c_i d_j (A_i B_j)
            # This would lead to 16 terms per M_k, which is not Strassen.
            
            # Correction:
            # Strassen defines M_k as the product of two 2x2 MATRICES.
            # Let X_k be the 2x2 matrix resulting from the linear combination of A blocks.
            # Let Y_k be the 2x2 matrix resulting from the linear combination of B blocks.
            # M_k = X_k Y_k.
            # To compute M_k (a 2x2 matrix) using rank 7, we decompose X_k Y_k
            # into 7 scalar multiplications of the ENTRIES of X_k and Y_k.
            
            # So, for each top-level term k:
            # 1. Construct the coefficients for each global A element.
            #    X_k is 2x2. Its entry (r, c) is a linear combination of A blocks.
            #    Specifically, X_k[r, c] = sum_{blocks in top_A} coeff * A_block[r, c]?
            #    No. The "blocks" in Strassen are the 2x2 submatrices.
            #    X_k IS a 2x2 matrix. Its entries are 2x2 matrices themselves?
            #    No, that's recursive Strassen.
            
            # Let's step back.
            # We are decomposing 4x4x4.
            # We split A into four 2x2 blocks: A00, A01, A10, A11.
            # We define 7 matrices M1..M7.
            # M1 = (A00+A11)(B00+B11). Note that A00+A11 is a 2x2 matrix.
            # So M1 is a 2x2 matrix product of two 2x2 matrices.
            # To get the scalar rank, we must decompose the product of two 2x2 matrices.
            # Let U = A00+A11 (2x2). Let V = B00+B11 (2x2).
            # M1 = U V.
            # We can decompose U V using 7 scalar multiplications of the elements of U and V.
            # Let U = [[u00, u01], [u10, u11]] and V = [[v00, v01], [v10, v11]].
            # The 7 terms are:
            # s1 = (u00+u11)(v00+v11)
            # ...
            # Each s_j is a scalar product of a linear combo of u's and linear combo of v's.
            # u_ij is a linear combo of A elements.
            # v_ij is a linear combo of B elements.
            
            # So, for each of the 7 top-level terms, we have a specific U (from A) and V (from B).
            # We then apply the 7-term Strassen decomposition to the product U V.
            # This yields 7 * 7 = 49 scalar terms.
            
            # Let's implement this.
            
            # 1. Identify U and V for the current top-level term.
            # U is a 2x2 matrix. Each entry U[r,c] is a linear combination of 4 A elements.
            # Actually, U IS the sum of specific A blocks.
            # Example: M1 uses U = A00 + A11.
            # A00 is A[0:2, 0:2]. A11 is A[2:4, 2:4].
            # So U[0,0] = A[0,0] + A[2,2]
            # U[0,1] = A[0,1] + A[2,3]
            # U[1,0] = A[1,0] + A[3,2]
            # U[1,1] = A[1,1] + A[3,3]
            
            # So, for a given top-level term, we can determine the coefficients
            # of each global A element in each local U entry.
            
            # Let's create a function to get the coefficients of U and V entries.
            def get_block_coeffs(top_spec, matrix_type):
                """
                Returns a 2x2 array of dicts.
                array[r][c] is a dict { global_A_index: coeff }
                """
                # top_spec is { (block_r, block_c): coeff }
                coeffs = [[{} for _ in range(2)] for _ in range(2)]
                for (br, bc), c in top_spec.items():
                    # Get the 4 indices of this block
                    block_idxs = block_indices(br, bc)
                    # Map local (0,0)->idx0, (0,1)->idx1, (1,0)->idx2, (1,1)->idx3
                    local_map = [(0,0), (0,1), (1,0), (1,1)]
                    for lr, lc in local_map:
                        idx_in_block = lr * 2 + lc
                        global_idx = block_idxs[idx_in_block]
                        # Add to the corresponding local entry in U/V
                        coeffs[lr][lc][global_idx] = coeffs[lr][lc].get(global_idx, 0) + c
                return coeffs

            U_coeffs = get_block_coeffs(top_A, 'A')
            V_coeffs = get_block_coeffs(top_B, 'B')
            
            # Now, apply the 7 bottom-level Strassen terms to U and V.
            # For each bottom-level term j:
            for b_idx, (bot_X, bot_Y) in enumerate(bottom_level_specs):
                # bot_X is { (lr, lc): coeff } for U entries.
                # bot_Y is { (lr, lc): coeff } for V entries.
                
                # Construct the linear combination for A
                # A_comb = { global_A_idx: total_coeff }
                A_comb = {}
                for (lr, lc), c_x in bot_X.items():
                    # c_x is the coefficient of U[lr][lc] in this scalar term
                    # U[lr][lc] is defined by U_coeffs[lr][lc]
                    for ga_idx, c_u in U_coeffs[lr][lc].items():
                        total_c = c_x * c_u
                        A_comb[ga_idx] = A_comb.get(ga_idx, 0) + total_c
                
                # Construct the linear combination for B
                B_comb = {}
                for (lr, lc), c_y in bot_Y.items():
                    for gb_idx, c_v in V_coeffs[lr][lc].items():
                        total_c = c_y * c_v
                        B_comb[gb_idx] = B_comb.get(gb_idx, 0) + total_c
                
                # Now we have the scalar term: (Sum A) * (Sum B)
                # We need to determine which C entries this contributes to.
                # The term M_k (top level) contributes to specific C blocks.
                # The scalar term s_j (bottom level) contributes to specific entries of M_k.
                
                # Top-level contributions:
                # M1 -> C00 (+1), C11 (+1)
                # M2 -> C10 (+1), C11 (-1)
                # M3 -> C01 (+1), C11 (+1)
                # M4 -> C00 (-1), C10 (+1)
                # M5 -> C00 (+1), C01 (+1)
                # M6 -> C11 (+1)
                # M7 -> C00 (+1)
                
                top_contribs = {
                    0: [(0,0,1), (1,1,1)],   # M1
                    1: [(1,0,1), (1,1,-1)],   # M2
                    2: [(0,1,1), (1,1,1)],    # M3
                    3: [(0,0,-1), (1,0,1)],   # M4
                    4: [(0,0,1), (0,1,1)],    # M5
                    5: [(1,1,1)],             # M6
                    6: [(0,0,1)]              # M7
                }
                
                # Bottom-level contributions to the 2x2 matrix M_k
                # S1 -> M00 (+1), M11 (+1)
                # S2 -> M10 (+1), M11 (-1)
                # S3 -> M01 (+1), M11 (+1)
                # S4 -> M00 (-1), M10 (+1)
                # S5 -> M00 (+1), M01 (+1)
                # S6 -> M11 (+1)
                # S7 -> M00 (+1)
                
                bottom_contribs = {
                    0: [(0,0,1), (1,1,1)],
                    1: [(1,0,1), (1,1,-1)],
                    2: [(0,1,1), (1,1,1)],
                    3: [(0,0,-1), (1,0,1)],
                    4: [(0,0,1), (0,1,1)],
                    5: [(1,1,1)],
                    6: [(0,0,1)]
                }
                
                # Combine contributions
                # The total contribution to C is the product of the top-level block coeff
                # and the bottom-level entry coeff.
                
                for (c_br, c_bc, c_top) in top_contribs[t_idx]:
                    for (c_lr, c_lc, c_bot) in bottom_contribs[b_idx]:
                        # Global C index
                        # Block c_br, c_bc. Local entry c_lr, c_lc.
                        # c_global = (c_br*2 + c_lr)*4 + (c_bc*2 + c_lc)
                        c_global = (c_br * 2 + c_lr) * 4 + (c_bc * 2 + c_lc)
                        
                        # Total coefficient for this C entry in this scalar term
                        c_total = c_top * c_bot
                        
                        # Add to the term
                        # A part
                        for ga_idx, a_c in A_comb.items():
                            U[ga_idx, col] += a_c * c_total  # Wait, U holds A coeffs.
                            # Actually, U[i, col] is the coefficient for A[i] in this rank-1 term.
                            # The term is (Sum A_comb[i] * A[i]) * (Sum B_comb[j] * B[j]) -> C_contrib
                            # So U[ga_idx, col] should be a_c.
                            # But we are adding to the column.
                            # Let's just store the coeffs directly.
                            pass # We will fill U, V, W after the loop or inside.
                        
                        # Let's fill U, V, W directly.
                        # U[ga_idx, col] = coeff_A
                        # V[gb_idx, col] = coeff_B
                        # W[c_idx, col] = coeff_C
                        
                        # Since we are inside the loop for a specific scalar term,
                        # we can add the contributions.
                        
                        for ga_idx, a_c in A_comb.items():
                            U[ga_idx, col] += a_c
                        for gb_idx, b_c in B_comb.items():
                            V[gb_idx, col] += b_c
                        W[c_global, col] += c_total
                        
                col += 1
    # End of Strassen 4x4x4 generation (col should be 49)
    
    # -------------------------------------------------------------------------
    # PART 2: Schoolbook for 4x4 x 4x3
    # -------------------------------------------------------------------------
    # B_right is columns 4, 5, 6 of B (0-indexed: 4,5,6).
    # A is 4x4. C_right is 4x3.
    # Rank 48.
    
    offset = 49
    for i in range(n):
        for j in range(m):
            for k in range(3): # k is 0,1,2 corresponding to global cols 4,5,6
                global_k = k + 4
                col = offset + i * 4 + j * 3 + k
                # U index: i*m + j
                U[i * m + j, col] = 1
                # V index: j*p + global_k
                V[j * p + global_k, col] = 1
                # W index: global_k*n + i
                W[global_k * n + i, col] = 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

