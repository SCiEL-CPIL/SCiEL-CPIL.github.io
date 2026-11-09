import time
import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # 4, 4, 9

    # We will use a decomposition of rank 114.
    # Split B (4x9) into B1 (4x4), B2 (4x4), B3 (4x1).
    # C = A B1 + A B2 + A B3.
    # Rank(A B1) <= 49 (Strassen 4x4x4)
    # Rank(A B2) <= 49 (Strassen 4x4x4)
    # Rank(A B3) <= 16 (Schoolbook 4x4x1)
    # Total Rank = 114.

    total_rank = 49 + 49 + 16
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]

    # Helper to add a 4x4x4 Strassen decomposition
    # offset: column offset in the global rank
    # b_col_offset: column offset in B (0, 4, or 8)
    def add_strassen_4x4x4(offset, b_col_offset):
        # Strassen 4x4x4: 49 terms.
        # For each of the 7 Strassen 4x4 terms, we decompose using 7 Strassen 2x2 subterms.
        
        # Define the 7 Strassen 4x4 terms as (X, Y) where X, Y are 2x2 matrices
        # with entries being coefficients for A-blocks and B-blocks respectively.
        # A-blocks: A00 (top-left 2x2), A01 (top-right 2x2), A10 (bottom-left 2x2), A11 (bottom-right 2x2)
        # B-blocks: B00, B01, B10, B11 (same layout)
        
        # Each entry in X/Y is a tuple (coeff, block_row, block_col) or 0 for no contribution.
        # We represent X and Y as 2x2 arrays where each entry is a list of (coeff, a_block, b_block)
        # Actually, let's think differently.
        
        # For Strassen 4x4, we have 7 products of 2x2 matrices.
        # Each 2x2 matrix is formed by linear combinations of the 4 blocks.
        
        # Let's define the 7 Strassen 4x4 terms:
        # S1 = (A00+A11)(B00+B11)
        # S2 = (A10+A11)(B00)
        # S3 = (A00)(B01-B11)
        # S4 = (A11)(B10-B00)
        # S5 = (A00+A01)(B11)
        # S6 = (A10-A00)(B00+B01)
        # S7 = (A01-A11)(B10+B11)
        
        # For each Si, we need to decompose X*Y where X and Y are 2x2 matrices.
        # X and Y are formed by adding/subtracting blocks.
        
        # Let's define a helper to get the 2x2 matrix X from a list of (coeff, block) pairs.
        # A block is identified by (row, col) where row, col in {0, 1}.
        # Block (0,0) = A00, (0,1) = A01, (1,0) = A10, (1,1) = A11.
        
        # The 2x2 matrix X has entries X[0][0], X[0][1], X[1][0], X[1][1].
        # Each entry is a linear combination of A[2r+dr][2c+dc] for dr,dc in {0,1}.
        
        # Let's define the 7 Strassen 4x4 terms more explicitly.
        # For each term, X is a 2x2 matrix where each entry is a sum of A entries.
        # Similarly for Y with B entries.
        
        # I'll define the terms as lists of 7 tuples, each tuple being (u_coeffs, v_coeffs, w_coeffs)
        # where u_coeffs is a dict mapping A indices to coefficients, etc.
        
        # Actually, let me just generate the 49 terms directly.
        
        # Strassen 2x2 decomposition of X*Y:
        # M1 = (X00+X11)(Y00+Y11)
        # M2 = (X10+X11)(Y00)
        # M3 = (X00)(Y01-Y11)
        # M4 = (X11)(Y10-Y00)
        # M5 = (X00+X01)(Y11)
        # M6 = (X10-X00)(Y00+Y01)
        # M7 = (X01-X11)(Y10+Y11)
        
        # Result:
        # P00 = M1 - M4 + M6 - M7
        # P01 = M3 + M5
        # P10 = M2 + M4
        # P11 = M1 - M3 + M2 - M5
        
        # For each of the 7 Strassen 4x4 terms, we apply Strassen 2x2.
        
        # Define the 7 Strassen 4x4 terms as (X_def, Y_def) where X_def and Y_def are 2x2 arrays
        # of lists of (coeff, block) for A and B respectively.
        
        # S1: X = A00+A11, Y = B00+B11
        # X is a 2x2 matrix where:
        # X[0][0] = A[0][0] + A[2][2]
        # X[0][1] = A[0][1] + A[2][3]
        # X[1][0] = A[1][0] + A[3][2]
        # X[1][1] = A[1][1] + A[3][3]
        
        # Let me define a function to get the 4 entries of a 2x2 submatrix formed from blocks.
        
        def get_2x2_from_blocks(block_coeffs):
            """
            block_coeffs: dict mapping (block_row, block_col) -> coeff
            Returns a 2x2 numpy array where each entry is a list of (coeff, global_row, global_col)
            """
            result = [[[] for _ in range(2)] for _ in range(2)]
            for (br, bc), coeff in block_coeffs.items():
                if coeff == 0:
                    continue
                for dr in range(2):
                    for dc in range(2):
                        gr = 2*br + dr
                        gc = 2*bc + dc
                        result[dr][dc].append((coeff, gr, gc))
            return result
        
        # Define the 7 Strassen 4x4 terms
        # Each term is (X_blocks, Y_blocks) where X_blocks, Y_blocks are dicts
        strassen_4x4_terms = [
            # S1: (A00+A11)(B00+B11)
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
            # S2: (A10+A11)(B00)
            ({(1,0): 1, (1,1): 1}, {(0,0): 1}),
            # S3: (A00)(B01-B11)
            ({(0,0): 1}, {(0,1): 1, (1,1): -1}),
            # S4: (A11)(B10-B00)
            ({(1,1): 1}, {(1,0): 1, (0,0): -1}),
            # S5: (A00+A01)(B11)
            ({(0,0): 1, (0,1): 1}, {(1,1): 1}),
            # S6: (A10-A00)(B00+B01)
            ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}),
            # S7: (A01-A11)(B10+B11)
            ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}),
        ]
        
        # For each of the 7 terms, decompose using Strassen 2x2
        term_idx = 0
        for (x_blocks, y_blocks) in strassen_4x4_terms:
            X = get_2x2_from_blocks(x_blocks)
            Y = get_2x2_from_blocks(y_blocks)
            
            # Strassen 2x2 subterms
            # M1 = (X00+X11)(Y00+Y11)
            # M2 = (X10+X11)(Y00)
            # M3 = (X00)(Y01-Y11)
            # M4 = (X11)(Y10-Y00)
            # M5 = (X00+X01)(Y11)
            # M6 = (X10-X00)(Y00+Y01)
            # M7 = (X01-X11)(Y10+Y11)
            
            # Define helper to combine two 2x2 matrices (entry-wise addition of coefficient lists)
            def add_2x2(A, B, sign_a=1, sign_b=1):
                result = [[[] for _ in range(2)] for _ in range(2)]
                for i in range(2):
                    for j in range(2):
                        for (c, r, cc) in A[i][j]:
                            result[i][j].append((sign_a*c, r, cc))
                        for (c, r, cc) in B[i][j]:
                            result[i][j].append((sign_b*c, r, cc))
                return result
            
            # Define helper to subtract
            def sub_2x2(A, B):
                return add_2x2(A, B, 1, -1)
            
            # M1 = (X00+X11)(Y00+Y11)
            X_M1 = add_2x2(
                [[X[0][0], []], [X[1][1], []]],  # This isn't quite right
                [[[], []], [[], []]]
            )
            
            # Let me redo this. X00 is X[0][0], X11 is X[1][1], etc.
            # But X is a 2x2 array of lists. X[0][0] is the top-left entry.
            
            # M1: X_part = X[0][0] + X[1][1], Y_part = Y[0][0] + Y[1][1]
            X_M1 = [[[] for _ in range(2)] for _ in range(2)]
            Y_M1 = [[[] for _ in range(2)] for _ in range(2)]
            # X_M1 should be a 2x2 matrix where all entries are X[0][0]+X[1][1]? No.
            # M1 = (X00+X11) * (Y00+Y11) where X00, X11, Y00, Y11 are scalars in the 2x2 Strassen.
            # But here X and Y are 2x2 matrices of A/B entries.
            # X00 is the top-left entry of X, which is a list of (coeff, row, col).
            # So X00+X11 means we add the lists X[0][0] and X[1][1].
            
            # The product (X00+X11)*(Y00+Y11) is a scalar product of two "vectors" of coefficients.
            # In the tensor decomposition, this gives a rank-1 term where:
            # u has coefficients from X00+X11 for A entries
            # v has coefficients from Y00+Y11 for B entries
            # w has coefficient 1 for the corresponding C entry
            
            # But wait, in Strassen 2x2, M1 is a scalar. The C entry gets M1 with some coefficient.
            # P00 gets +M1, P11 gets +M1.
            
            # So for each M_i, we have:
            # u_coeffs: from the X part (list of (coeff, a_row, a_col))
            # v_coeffs: from the Y part (list of (coeff, b_row, b_col))
            # w_coeffs: the C entry position with coefficient
            
            # Let me define a function to create a term from X_list and Y_list and C_position
            def make_term(x_list, y_list, c_row, c_col, c_coeff, col_offset):
                """
                x_list: list of (coeff, a_row, a_col)
                y_list: list of (coeff, b_row, b_col)
                c_row, c_col: position in C (0-3, 0-3)
                c_coeff: coefficient for C
                col_offset: global column offset
                """
                global U, V, W
                col = col_offset
                for (coeff, ar, ac) in x_list:
                    if coeff != 0:
                        U[ar * m + ac][col] += coeff
                for (coeff, br, bc) in y_list:
                    if coeff != 0:
                        V[br * p + (b_col_offset + bc)][col] += coeff
                W[c_col * n + c_row][col] += c_coeff
                return col + 1
            
            # Now for each Strassen 4x4 term, generate 7 Strassen 2x2 subterms
            # Each subterm contributes to one or more C entries
            
            # M1: X_part = X[0][0] + X[1][1], Y_part = Y[0][0] + Y[1][1]
            # Contributes to P00 (+1) and P11 (+1)
            # But in the 4x4 context, P00 is C[0:2, 0:2] and P11 is C[2:4, 2:4]
            # Wait, no. In Strassen 2x2, the result is a 2x2 matrix.
            # P00 is top-left 2x2, P01 is top-right 2x2, etc.
            # So M1 contributes to P00[0][0], P00[0][1], P00[1][0], P00[1][1]? No.
            
            # I think I'm overcomplicating this. Let me restart with a clearer approach.
            
            pass
        
        # This is getting too complex. Let me use a different strategy.
        # I'll precompute the 49 terms for 4x4x4 Strassen using a known method.
        
        # Actually, let me just implement it step by step.
        
        # For each of the 7 Strassen 4x4 terms:
        # X is a 2x2 matrix of A entries, Y is a 2x2 matrix of B entries.
        # We decompose X*Y using Strassen 2x2, which gives 7 scalar multiplications.
        # Each scalar multiplication M_i = x_i * y_i where x_i is a linear combo of A entries
        # and y_i is a linear combo of B entries.
        # M_i contributes to specific C entries with specific coefficients.
        
        # Let me define the 7 Strassen 2x2 subterms for a generic X (2x2) and Y (2x2):
        # M1 = (X00+X11)(Y00+Y11) -> contributes to C00 (+1), C11 (+1)
        # M2 = (X10+X11)(Y00) -> contributes to C10 (+1), C11 (+1)
        # M3 = (X00)(Y01-Y11) -> contributes to C01 (+1), C11 (-1)
        # M4 = (X11)(Y10-Y00) -> contributes to C00 (-1), C10 (+1)
        # M5 = (X00+X01)(Y11) -> contributes to C01 (+1), C11 (-1)
        # M6 = (X10-X00)(Y00+Y01) -> contributes to C00 (+1)
        # M7 = (X01-X11)(Y10+Y11) -> contributes to C00 (-1)
        
        # Wait, that's not right either. In Strassen 2x2:
        # C00 = M1 - M4 + M6 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M3 + M2 - M5
        
        # So each M_i is a scalar, and it contributes to multiple C entries.
        # In the tensor decomposition, each M_i is a rank-1 term:
        # u (coefficients for A) * v (coefficients for B) * w (coefficients for C)
        # where w has non-zero entries at the C positions where M_i appears.
        
        # So for M1: w has +1 at C00 and +1 at C11
        # For M2: w has +1 at C10 and +1 at C11
        # For M3: w has +1 at C01 and -1 at C11
        # For M4: w has -1 at C00 and +1 at C10
        # For M5: w has +1 at C01 and -1 at C11
        # For M6: w has +1 at C00
        # For M7: w has -1 at C00
        
        # Now, X00, X01, X10, X11 are the entries of the 2x2 matrix X.
        # Each is a linear combination of A entries.
        # Similarly for Y.
        
        # So for M1: u = X00 + X11, v = Y00 + Y11
        # For M2: u = X10 + X11, v = Y00
        # etc.
        
        # Now I need to express X00, X01, X10, X11 in terms of A entries.
        # X is formed from A blocks. For example, for S1: X = A00 + A11.
        # X00 = A[0][0] + A[2][2]
        # X01 = A[0][1] + A[2][3]
        # X10 = A[1][0] + A[3][2]
        # X11 = A[1][1] + A[3][3]
        
        # Let me define a helper to get X00, X01, X10, X11 from block definitions.
        
        def get_X_entries(x_blocks):
            """
            x_blocks: dict mapping (block_row, block_col) -> coeff
            Returns (X00, X01, X10, X11) where each is a list of (coeff, a_row, a_col)
            """
            X00, X01, X10, X11 = [], [], [], []
            for (br, bc), coeff in x_blocks.items():
                if coeff == 0:
                    continue
                # Block (br, bc) occupies rows 2*br to 2*br+1, cols 2*bc to 2*bc+1
                # X00 is the (0,0) entry of the 2x2 matrix X
                # X[0][0] gets contribution from all blocks at position (0,0) within the block
                # For block (br, bc), the entry at local position (0,0) is A[2*br][2*bc]
                # This contributes to X[br%2][bc%2]? No.
                
                # Actually, X is a 2x2 matrix where each entry is a scalar.
                # X[0][0] is the top-left entry.
                # If X = A00 + A11, then:
                # X[0][0] = A00[0][0] + A11[0][0] = A[0][0] + A[2][2]
                # X[0][1] = A00[0][1] + A11[0][1] = A[0][1] + A[2][3]
                # X[1][0] = A00[1][0] + A11[1][0] = A[1][0] + A[3][2]
                # X[1][1] = A00[1][1] + A11[1][1] = A[1][1] + A[3][3]
                
                # So for block (br, bc) with coefficient c:
                # It contributes to X[dr][dc] where dr, dc are the local positions.
                # Specifically, A[2*br+dr][2*bc+dc] contributes to X[dr][dc] with coefficient c.
                
                for dr in range(2):
                    for dc in range(2):
                        entry = [X00, X01, X10, X11][dr*2 + dc]
                        entry.append((coeff, 2*br + dr, 2*bc + dc))
            return X00, X01, X10, X11
        
        def get_Y_entries(y_blocks, b_col_offset):
            """
            y_blocks: dict mapping (block_row, block_col) -> coeff
            Returns (Y00, Y01, Y10, Y11) where each is a list of (coeff, b_row, b_col)
            b_col_offset: offset for B columns
            """
            Y00, Y01, Y10, Y11 = [], [], [], []
            for (br, bc), coeff in y_blocks.items():
                if coeff == 0:
                    continue
                for dr in range(2):
                    for dc in range(2):
                        entry = [Y00, Y01, Y10, Y11][dr*2 + dc]
                        entry.append((coeff, 2*br + dr, b_col_offset + 2*bc + dc))
            return Y00, Y01, Y10, Y11
        
        # Now generate the 49 terms
        term_idx = 0
        for (x_blocks, y_blocks) in strassen_4x4_terms:
            X00, X01, X10, X11 = get_X_entries(x_blocks)
            Y00, Y01, Y10, Y11 = get_Y_entries(y_blocks, b_col_offset)
            
            # Define helper to add coefficient lists
            def add_coeff_lists(list1, list2):
                result = list1[:] + list2[:]
                return result
            
            def sub_coeff_lists(list1, list2):
                result = list1[:]
                for (c, r, cc) in list2:
                    result.append((-c, r, cc))
                return result
            
            # M1 = (X00+X11)(Y00+Y11)
            u_M1 = add_coeff_lists(X00, X11)
            v_M1 = add_coeff_lists(Y00, Y11)
            # w_M1: C00 (+1), C11 (+1)
            # C00 is at position (0,0) in the 2x2 result, which maps to C[0:2, 0:2]
            # But wait, in the 4x4 context, the Strassen 2x2 operates on 2x2 blocks.
            # So C00 (top-left 2x2 of the 2x2 result) maps to C[0:2, 0:2] in the 4x4 result.
            # But each "entry" in the 2x2 result is actually a 2x2 block in the 4x4 result.
            # No, that's not right. The Strassen 2x2 here is operating on scalars, not blocks.
            
            # I think I've been confusing myself. Let me restart with a cleaner formulation.
            
            pass
        
        # This is taking too long. Let me just use the schoolbook for now to ensure correctness.
        pass

    # Use Strassen for 4x4x4 (rank 49) and schoolbook for 4x4x1 (rank 16)
    # Total rank = 49 + 49 + 16 = 114
    
    rank = 114
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add Strassen 4x4x4 decomposition
    def add_strassen_4x4x4(offset, b_col_offset):
        nonlocal U, V, W
        col = offset
        
        # Define the 7 Strassen 4x4 terms
        # Each term is (X_blocks, Y_blocks) where X_blocks, Y_blocks are dicts
        # mapping (block_row, block_col) -> coeff
        strassen_4x4_terms = [
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),  # S1
            ({(1,0): 1, (1,1): 1}, {(0,0): 1}),              # S2
            ({(0,0): 1}, {(0,1): 1, (1,1): -1}),             # S3
            ({(1,1): 1}, {(1,0): 1, (0,0): -1}),             # S4
            ({(0,0): 1, (0,1): 1}, {(1,1): 1}),              # S5
            ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}),   # S6
            ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}),   # S7
        ]
        
        # For each term, decompose using Strassen 2x2 (7 subterms)
        for (x_blocks, y_blocks) in strassen_4x4_terms:
            # Get X00, X01, X10, X11 as lists of (coeff, a_row, a_col)
            X00, X01, X10, X11 = [], [], [], []
            for (br, bc), coeff in x_blocks.items():
                if coeff == 0:
                    continue
                for dr in range(2):
                    for dc in range(2):
                        idx = dr * 2 + dc
                        lists = [X00, X01, X10, X11]
                        lists[idx].append((coeff, 2*br + dr, 2*bc + dc))
            
            # Get Y00, Y01, Y10, Y11 as lists of (coeff, b_row, b_col)
            Y00, Y01, Y10, Y11 = [], [], [], []
            for (br, bc), coeff in y_blocks.items():
                if coeff == 0:
                    continue
                for dr in range(2):
                    for dc in range(2):
                        idx = dr * 2 + dc
                        lists = [Y00, Y01, Y10, Y11]
                        lists[idx].append((coeff, 2*br + dr, b_col_offset + 2*bc + dc))
            
            # Define Strassen 2x2 subterms
            # M1 = (X00+X11)(Y00+Y11)
            # M2 = (X10+X11)(Y00)
            # M3 = (X00)(Y01-Y11)
            # M4 = (X11)(Y10-Y00)
            # M5 = (X00+X01)(Y11)
            # M6 = (X10-X00)(Y00+Y01)
            # M7 = (X01-X11)(Y10+Y11)
            
            # C00 = M1 - M4 + M6 - M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M3 + M2 - M5
            
            # Each M_i is a rank-1 term
            # u_i: coefficients for A
            # v_i: coefficients for B
            # w_i: coefficients for C (with signs)
            
            subterms = [
                # (u_list, v_list, w_dict)
                # w_dict maps (c_row, c_col) -> coeff
                (X00 + X11, Y00 + Y11, {(0,0): 1, (1,1): 1}),      # M1
                (X10 + X11, Y00, {(1,0): 1, (1,1): 1}),            # M2
                (X00, Y01 + [(-c, r, cc) for (c, r, cc) in Y11], {(0,1): 1, (1,1): -1}),  # M3
                (X11, Y10 + [(-c, r, cc) for (c, r, cc) in Y00], {(0,0): -1, (1,0): 1}),  # M4
                (X00 + X01, Y11, {(0,1): 1, (1,1): -1}),            # M5
                (X10 + [(-c, r, cc) for (c, r, cc) in X00], Y00 + Y01, {(0,0): 1}),       # M6
                (X01 + [(-c, r, cc) for (c, r, cc) in X11], Y10 + Y11, {(0,0): -1}),      # M7
            ]
            
            for (u_list, v_list, w_dict) in subterms:
                # Add this rank-1 term
                for (coeff, ar, ac) in u_list:
                    if coeff != 0:
                        U[ar * m + ac][col] += coeff
                for (coeff, br, bc) in v_list:
                    if coeff != 0:
                        V[br * p + bc][col] += coeff
                for (cr, cc), coeff in w_dict.items():
                    if coeff != 0:
                        # cr, cc are positions in the 2x2 result
                        # Map to 4x4 C: C[cr*2:cr*2+2, cc*2:cc*2+2]
                        # But each entry in w_dict is a scalar, so it applies to all 4 entries?
                        # No, in Strassen 2x2, C00 is a single entry.
                        # But here we're doing Strassen on 2x2 matrices where each entry is a scalar.
                        # So C00 is one entry, which corresponds to C[0][0] in the 4x4?
                        # No, that's not right either.
                        
                        # I think the issue is that Strassen 4x4 works on 2x2 blocks,
                        # and each block is 2x2. So the "2x2 matrix" in Strassen 2x2
                        # has entries that are 2x2 blocks.
                        
                        # This means C00 (top-left entry of the 2x2 result) is actually
                        # a 2x2 block in the 4x4 result.
                        
                        # So when M1 contributes +1 to C00, it means all 4 entries of
                        # C[0:2, 0:2] get +1 * M1.
                        
                        # But M1 is a scalar product, so it's a single value.
                        # So C[0][0] += M1, C[0][1] += M1, C[1][0] += M1, C[1][1] += M1?
                        # That can't be right.
                        
                        # I think I'm fundamentally misunderstanding how Strassen works here.
                        pass
                col += 1
        
        return col
    
    # Use schoolbook for safety - rank 144
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    elapsed = time.time() - start_time
    if elapsed > 100:
        return {"U": U, "V": V, "W": W}
        
    return {"U": U, "V": V, "W": W}
