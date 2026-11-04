import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We assume n=m=p=4 for this specific optimization.
    if n != 4 or m != 4 or p != 4:
        # Fallback to schoolbook if dimensions are not 4x4x4
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    # Rank 49 decomposition for 4x4x4 using Recursive Strassen
    rank = 49
    
    # Initialize U, V, W as dictionaries for easier sparse construction, then convert to lists
    # U shape: (16, 49)
    # V shape: (16, 49)
    # W shape: (16, 49)
    # Indices:
    # A indices: i*4 + j for i,j in 0..3
    # B indices: j*4 + k for j,k in 0..3
    # C indices: k*4 + i for k,i in 0..3 (Trace convention W[k*n+i, r])
    
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(16)]
    
    # Helper to add to a linear form
    def add_to_vec(vec, idx, val, col):
        vec[idx][col] += val

    # Define the 7 Strassen formulas for 2x2 matrices.
    # Returns list of 7 tuples: (coeffs for X, coeffs for Y, coeffs for Z)
    # X, Y are 2x2 matrices with entries as linear forms.
    # We represent a 2x2 matrix as a list of 4 linear forms [x11, x12, x21, x22]
    # Each linear form is a dict {index: coeff}
    
    # Actually, we can just generate the 49 columns directly.
    
    # Top-level blocks of A (indices in 0..15)
    # A11: 0,1,4,5
    # A12: 2,3,6,7
    # A21: 8,9,12,13
    # A22: 10,11,14,15
    
    # Top-level blocks of B (indices in 0..15)
    # B11: 0,1,4,5
    # B12: 2,3,6,7
    # B21: 8,9,12,13
    # B22: 10,11,14,15
    
    # We define the 7 top-level terms.
    # Each term is a product X * Y of two 2x2 matrices.
    # X and Y are linear combinations of A and B blocks.
    
    # Let's define a function to get the linear form for a block sum/diff
    # Block indices:
    # 0: TL, 1: TR, 2: BL, 3: BR
    
    def get_block_indices(block_type, matrix_type):
        # matrix_type: 'A' or 'B'
        # Returns list of 4 indices for the 2x2 block
        # A: row-major 4x4
        # B: row-major 4x4
        # C: col-major? No, C[k, i] = sum A[i, j] B[j, k].
        # Output W index is k*4 + i.
        
        # For A (input 1):
        # A11 (rows 0,1; cols 0,1): 0, 1, 4, 5
        # A12 (rows 0,1; cols 2,3): 2, 3, 6, 7
        # A21 (rows 2,3; cols 0,1): 8, 9, 12, 13
        # A22 (rows 2,3; cols 2,3): 10, 11, 14, 15
        
        # For B (input 2):
        # B11 (rows 0,1; cols 0,1): 0, 1, 4, 5
        # B12 (rows 0,1; cols 2,3): 2, 3, 6, 7
        # B21 (rows 2,3; cols 0,1): 8, 9, 12, 13
        # B22 (rows 2,3; cols 2,3): 10, 11, 14, 15
        
        if block_type == 0: # TL
            return [0, 1, 4, 5]
        elif block_type == 1: # TR
            return [2, 3, 6, 7]
        elif block_type == 2: # BL
            return [8, 9, 12, 13]
        elif block_type == 3: # BR
            return [10, 11, 14, 15]

    # Linear form representation: list of (index, coeff)
    # A 2x2 matrix of linear forms: list of 4 linear forms [x11, x12, x21, x22]
    
    def linear_combination(blocks, coeffs):
        # blocks: list of block types (0,1,2,3)
        # coeffs: list of coefficients for each block
        # Returns a 2x2 matrix of linear forms
        # Each entry in the 2x2 matrix is a sum of indices from the blocks
        # Note: The block indices correspond to the 4 entries of the 2x2 submatrix.
        # So if we take block 0 (TL), its entries are at positions (0,0), (0,1), (1,0), (1,1) of the 2x2 matrix.
        
        result = [[[] for _ in range(2)] for _ in range(2)]
        for b, c in zip(blocks, coeffs):
            if c == 0: continue
            inds = get_block_indices(b, 'A') # Indices are same for A and B in terms of layout
            # Map block indices to 2x2 positions
            # Block indices [0, 1, 4, 5] correspond to positions:
            # 0 -> (0,0)
            # 1 -> (0,1)
            # 4 -> (1,0)
            # 5 -> (1,1)
            pos_map = {0: (0,0), 1: (0,1), 4: (1,0), 5: (1,1)}
            # Wait, get_block_indices returns [0, 1, 4, 5] for TL.
            # These are the global indices.
            # The local positions in the 2x2 block are fixed.
            # Local (0,0) is global index base + 0
            # Local (0,1) is global index base + 1
            # Local (1,0) is global index base + 4
            # Local (1,1) is global index base + 5
            
            # So for a given block type, we know the 4 global indices.
            # We add coeff * global_index to the corresponding local position.
            
            # Let's just iterate through the 4 global indices and map them to local (r, c)
            # For block 0 (TL): global 0->(0,0), 1->(0,1), 4->(1,0), 5->(1,1)
            # For block 1 (TR): global 2->(0,0), 3->(0,1), 6->(1,0), 7->(1,1)
            # For block 2 (BL): global 8->(0,0), 9->(0,1), 12->(1,0), 13->(1,1)
            # For block 3 (BR): global 10->(0,0), 11->(0,1), 14->(1,0), 15->(1,1)
            
            # We can derive local (r,c) from global index:
            # row = global // 4
            # col = global % 4
            # local_r = row % 2
            # local_c = col % 2
            
            for g_idx in inds:
                r = g_idx // 4
                c = g_idx % 4
                lr = r % 2
                lc = c % 2
                result[lr][lc].append((g_idx, c))
                
        return result

    # The 7 top-level terms (X, Y)
    # X and Y are 2x2 matrices of linear forms
    # Term 1: X = A11 + A22, Y = B11 + B22
    # Term 2: X = A21 + A22, Y = B11
    # Term 3: X = A11, Y = B12 - B22
    # Term 4: X = A22, Y = B21 - B11
    # Term 5: X = A11 + A12, Y = B22
    # Term 6: X = A21 - A11, Y = B11 + B12
    # Term 7: X = A12 - A22, Y = B21 + B22
    
    terms = [
        ([0, 3], [1, 1], [0, 3], [1, 1]), # M1: X(A11+A22), Y(B11+B22)
        ([2, 3], [1, 1], [0], [1]),       # M2: X(A21+A22), Y(B11)
        ([0], [1], [1, 3], [1, -1]),      # M3: X(A11), Y(B12-B22)
        ([3], [1], [2, 0], [1, -1]),      # M4: X(A22), Y(B21-B11)
        ([0, 1], [1, 1], [3], [1]),       # M5: X(A11+A12), Y(B22)
        ([2, 0], [1, -1], [0, 1], [1, 1]),# M6: X(A21-A11), Y(B11+B12)
        ([1, 3], [1, -1], [2, 3], [1, 1]) # M7: X(A12-A22), Y(B21+B22)
    ]
    
    # Output combination for C
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Each Mk is a 2x2 matrix.
    # C is 4x4.
    # C11 is top-left 2x2 of C.
    # C12 is top-right 2x2 of C.
    # C21 is bottom-left 2x2 of C.
    # C22 is bottom-right 2x2 of C.
    
    # We need to map the local (r, c) of Mk to the global (i, k) of C.
    # C index is k*4 + i.
    # C11 (local r, c) -> global i = r, k = c? 
    # Wait, C = A B.
    # C[i, k] = sum_j A[i, j] B[j, k].
    # The block structure of C:
    # C11 (rows 0,1; cols 0,1): i in 0,1; k in 0,1.
    # C12 (rows 0,1; cols 2,3): i in 0,1; k in 2,3.
    # C21 (rows 2,3; cols 0,1): i in 2,3; k in 0,1.
    # C22 (rows 2,3; cols 2,3): i in 2,3; k in 2,3.
    
    # In the Mk matrix (2x2), the entry at (lr, lc) corresponds to:
    # If Mk is C11: i = lr, k = lc. Global C index = k*4 + i = lc*4 + lr.
    # If Mk is C12: i = lr, k = 2+lc. Global C index = (2+lc)*4 + lr.
    # If Mk is C21: i = 2+lr, k = lc. Global C index = lc*4 + (2+lr).
    # If Mk is C22: i = 2+lr, k = 2+lc. Global C index = (2+lc)*4 + (2+lr).

    # The 7 bottom-level Strassen formulas for a 2x2 matrix product Z = X Y:
    # P1 = (x11 + x22)(y11 + y22) -> Z11 += 1, Z22 += 1
    # P2 = (x21 + x22)y11 -> Z21 += 1, Z22 -= 1
    # P3 = x11(y12 - y22) -> Z12 += 1, Z22 += 1
    # P4 = x22(y21 - y11) -> Z11 += 1, Z21 += 1
    # P5 = (x11 + x12)y22 -> Z11 -= 1, Z12 += 1
    # P6 = (x21 - x11)(y11 + y12) -> Z22 += 1
    # P7 = (x12 - x22)(y21 + y22) -> Z11 += 1
    
    # Note: The standard Strassen output is:
    # Z11 = P1 + P4 - P5 + P7
    # Z12 = P3 + P5
    # Z21 = P2 + P4
    # Z22 = P1 - P2 + P3 + P6
    
    # Let's verify P2:
    # Z21 = P2 + P4.
    # P2 = (x21+x22)y11.
    # Z21 = x21y11 + x22y21.
    # P4 = x22(y21-y11) = x22y21 - x22y11.
    # P2 + P4 = x21y11 + x22y21 - x22y11 + x22y11 = x21y11 + x22y21. Correct.
    
    # P3: Z12 = P3 + P5.
    # P3 = x11(y12-y22) = x11y12 - x11y22.
    # P5 = (x11+x12)y22 = x11y22 + x12y22.
    # P3 + P5 = x11y12 + x12y22. Correct.
    
    # P6: Z22 = P1 - P2 + P3 + P6.
    # P1 = (x11+x22)(y11+y22) = x11y11 + x11y22 + x22y11 + x22y22.
    # -P2 = -(x21y11 + x22y11) = -x21y11 - x22y11.
    # P3 = x11y12 - x11y22.
    # P6 = (x21-x11)(y11+y12) = x21y11 + x21y12 - x11y11 - x11y12.
    # Sum:
    # x11y11 (1 - 1 = 0)
    # x11y22 (1 - 1 = 0)
    # x22y11 (1 - 1 = 0)
    # x22y22 (1)
    # -x21y11 + x21y11 = 0
    # x11y12 - x11y12 = 0
    # x21y12
    # Wait, Z22 should be x21y12 + x22y22.
    # My sum gives x22y22 + x21y12. Correct.
    
    # So the contributions of P1..P7 to Z11..Z22 are:
    # P1: Z11(+1), Z22(+1)
    # P2: Z21(+1), Z22(-1)
    # P3: Z12(+1), Z22(+1)
    # P4: Z11(+1), Z21(+1)
    # P5: Z11(-1), Z12(+1)
    # P6: Z22(+1)
    # P7: Z11(+1)
    
    # Now we iterate through the 7 top-level terms.
    col = 0
    for t_idx in range(7):
        x_blocks, x_coeffs, y_blocks, y_coeffs = terms[t_idx]
        X = linear_combination(x_blocks, x_coeffs)
        Y = linear_combination(y_blocks, y_coeffs)
        
        # X and Y are 2x2 lists of lists of (idx, coeff)
        # We extract the 4 linear forms for X: x11, x12, x21, x22
        x11 = X[0][0]
        x12 = X[0][1]
        x21 = X[1][0]
        x22 = X[1][1]
        
        y11 = Y[0][0]
        y12 = Y[0][1]
        y21 = Y[1][0]
        y22 = Y[1][1]
        
        # Define the 7 P terms as pairs of linear forms (L_A, L_B)
        # L_A is a linear form in A, L_B is a linear form in B.
        
        def combine_lf(lf1, lf2, sign1=1, sign2=1):
            # Combines two linear forms with signs
            res = {}
            for idx, c in lf1:
                res[idx] = res.get(idx, 0) + c * sign1
            for idx, c in lf2:
                res[idx] = res.get(idx, 0) + c * sign2
            # Remove zeros
            return [(k, v) for k, v in res.items() if v != 0]
        
        def add_lf(lf1, lf2):
            return combine_lf(lf1, lf2, 1, 1)
            
        def sub_lf(lf1, lf2):
            return combine_lf(lf1, lf2, 1, -1)
        
        # P1 = (x11 + x22)(y11 + y22)
        p1_a = add_lf(x11, x22)
        p1_b = add_lf(y11, y22)
        
        # P2 = (x21 + x22)y11
        p2_a = add_lf(x21, x22)
        p2_b = y11
        
        # P3 = x11(y12 - y22)
        p3_a = x11
        p3_b = sub_lf(y12, y22)
        
        # P4 = x22(y21 - y11)
        p4_a = x22
        p4_b = sub_lf(y21, y11)
        
        # P5 = (x11 + x12)y22
        p5_a = add_lf(x11, x12)
        p5_b = y22
        
        # P6 = (x21 - x11)(y11 + y12)
        p6_a = sub_lf(x21, x11)
        p6_b = add_lf(y11, y12)
        
        # P7 = (x12 - x22)(y21 + y22)
        p7_a = sub_lf(x12, x22)
        p7_b = add_lf(y21, y22)
        
        p_terms = [
            (p1_a, p1_b, [(0,0,1), (1,1,1)]), # (Z11, +1), (Z22, +1)
            (p2_a, p2_b, [(1,0,1), (1,1,-1)]),# (Z21, +1), (Z22, -1)
            (p3_a, p3_b, [(0,1,1), (1,1,1)]), # (Z12, +1), (Z22, +1)
            (p4_a, p4_b, [(0,0,1), (1,0,1)]), # (Z11, +1), (Z21, +1)
            (p5_a, p5_b, [(0,0,-1), (0,1,1)]),# (Z11, -1), (Z12, +1)
            (p6_a, p6_b, [(1,1,1)]),          # (Z22, +1)
            (p7_a, p7_b, [(0,0,1)])           # (Z11, +1)
        ]
        
        for p_a, p_b, z_contribs in p_terms:
            # Add to U and V
            for idx, coeff in p_a:
                U[idx][col] += coeff
            for idx, coeff in p_b:
                V[idx][col] += coeff
            
            # Add to W
            # Determine which C block this term belongs to
            # t_idx 0 -> M1 -> C11, C22
            # t_idx 1 -> M2 -> C21, C22
            # t_idx 2 -> M3 -> C12, C22
            # t_idx 3 -> M4 -> C11, C21
            # t_idx 4 -> M5 -> C11, C12
            # t_idx 5 -> M6 -> C22
            # t_idx 6 -> M7 -> C11
            
            # We need to map the local (lr, lc) of Z to global C index.
            # And apply the sign from the top-level combination.
            
            # Top-level signs:
            # C11 = M1 + M4 - M5 + M7
            # C12 = M3 + M5
            # C21 = M2 + M4
            # C22 = M1 - M2 + M3 + M6
            
            # For each z_contrib (lr, lc, sign_z):
            # We need to find which C block (C11, C12, C21, C22) the term t_idx contributes to, and with what top-level sign.
            
            # t_idx 0 (M1): C11 (+1), C22 (+1)
            # t_idx 1 (M2): C21 (+1), C22 (-1)
            # t_idx 2 (M3): C12 (+1), C22 (+1)
            # t_idx 3 (M4): C11 (+1), C21 (+1)
            # t_idx 4 (M5): C11 (-1), C12 (+1)
            # t_idx 5 (M6): C22 (+1)
            # t_idx 6 (M7): C11 (+1)
            
            block_map = {
                0: [(0, 1), (3, 1)], # M1: C11(+1), C22(+1)
                1: [(2, 1), (3, -1)],# M2: C21(+1), C22(-1)
                2: [(1, 1), (3, 1)], # M3: C12(+1), C22(+1)
                3: [(0, 1), (2, 1)], # M4: C11(+1), C21(+1)
                4: [(0, -1), (1, 1)],# M5: C11(-1), C12(+1)
                5: [(3, 1)],         # M6: C22(+1)
                6: [(0, 1)]          # M7: C11(+1)
            }
            
            for c_block, top_sign in block_map[t_idx]:
                # c_block: 0=C11, 1=C12, 2=C21, 3=C22
                for lr, lc, sign_z in z_contribs:
                    # Determine global C index
                    if c_block == 0: # C11
                        i = lr
                        k = lc
                    elif c_block == 1: # C12
                        i = lr
                        k = 2 + lc
                    elif c_block == 2: # C21
                        i = 2 + lr
                        k = lc
                    else: # C22
                        i = 2 + lr
                        k = 2 + lc
                    
                    c_idx = k * 4 + i
                    total_sign = top_sign * sign_z
                    W[c_idx][col] += total_sign
                    
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
