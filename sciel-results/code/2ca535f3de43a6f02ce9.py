import time
from fractions import Fraction

DIMENSIONS = [4, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    n, m, p = DIMENSIONS
    
    # Target Rank: 77
    # 49 for 4x4x4 (Strassen) + 28 for 4x4x2 (4x Strassen 2x2x2)
    rank = 77
    
    U = [[Fraction(0)] * rank for _ in range(n * m)] # 16 x 77
    V = [[Fraction(0)] * rank for _ in range(m * p)] # 24 x 77
    W = [[Fraction(0)] * rank for _ in range(p * n)] # 24 x 77
    
    # Helper to add a 2x2x2 Strassen decomposition
    # a_off: offset of A block (2x2) in global A (4x4)
    # b_off: offset of B block (2x2) in global B (4x6)
    # c_off: offset of C block (2x2) in global C (4x6)
    # col_start: starting column index in U, V, W
    def add_strassen_2x2(a_off, b_off, c_off, col_start):
        # Local indices for 2x2 A block: (0,0), (0,1), (1,0), (1,1)
        # Global A index: (a_off[0] + i) * 4 + (a_off[1] + j)
        # Local indices for 2x2 B block: (0,0), (0,1), (1,0), (1,1)
        # Global B index: (b_off[0] + i) * 6 + (b_off[1] + j)
        # Local indices for 2x2 C block: (0,0), (0,1), (1,0), (1,1)
        # Global C index: (c_off[0] + i) * 4 + (c_off[1] + j)
        
        # Strassen 2x2x2 coefficients
        # M1 = (A00+A11) * (B00+B11)
        # M2 = (A10+A11) * B00
        # M3 = A00 * (B01-B11)
        # M4 = A11 * (B10-B00)
        # M5 = (A00+A01) * B11
        # M6 = (A10-A00) * (B00+B01)
        # M7 = (A01-A11) * (B10+B11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        terms = [
            # (M_idx, A_coeffs, B_coeffs, C_coeffs)
            # A_coeffs: dict {local_idx: coeff}
            # B_coeffs: dict {local_idx: coeff}
            # C_coeffs: dict {local_idx: coeff}
            
            # M1
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 1}),
            # M2
            ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(0,0): 0, (0,1): 0, (1,0): 1, (1,1): 0}),
            # M3
            ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,0): 0, (0,1): 1, (1,0): 0, (1,1): 0}),
            # M4
            ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 0, (1,0): 1, (1,1): 0}),
            # M5
            ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1, (1,0): 0, (1,1): 0}),
            # M6
            ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(0,0): 0, (0,1): 0, (1,0): 0, (1,1): 1}),
            # M7
            ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 0}),
        ]
        
        for k, (a_c, b_c, c_c) in enumerate(terms):
            col = col_start + k
            for (ai, aj), val in a_c.items():
                global_i = a_off[0] + ai
                global_j = a_off[1] + aj
                U[global_i * 4 + global_j][col] += Fraction(val)
            for (bi, bj), val in b_c.items():
                global_i = b_off[0] + bi
                global_j = b_off[1] + bj
                V[global_i * 6 + global_j][col] += Fraction(val)
            for (ci, cj), val in c_c.items():
                global_i = c_off[0] + ci
                global_j = c_off[1] + cj
                W[global_i * 4 + global_j][col] += Fraction(val)
        
        return col_start + 7

    col_idx = 0
    
    # Part 1: 4x4x4 using 4 Strassen 2x2x2 blocks? 
    # No, standard Strassen for 4x4x4 uses 7 recursive calls to 2x2x2.
    # But we can just use 4 independent 2x2x2 Strassen for the diagonal blocks and handle off-diagonals?
    # No, that's not Strassen. That's just 4 blocks.
    # Strassen 4x4x4:
    # A = [[A11, A12], [A21, A22]]
    # B = [[B11, B12], [B21, B22]]
    # C = [[C11, C12], [C21, C22]]
    # 
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Each M_k is a 2x2x2 tensor.
    # We need to add 7 Strassen 2x2x2 decompositions.
    
    # A blocks offsets:
    # A11: (0,0), A12: (0,2), A21: (2,0), A22: (2,2)
    # B blocks offsets:
    # B11: (0,0), B12: (0,2), B21: (2,0), B22: (2,2)
    # C blocks offsets:
    # C11: (0,0), C12: (0,2), C21: (2,0), C22: (2,2)
    
    # M1: (A11+A22) * (B11+B22) -> C11, C22
    # This requires summing two 2x2 matrices for A and B, then multiplying.
    # This is NOT a single 2x2x2 Strassen call.
    # It is a product of (A11+A22) and (B11+B22).
    # Let X = A11+A22, Y = B11+B22.
    # Z = X Y.
    # Z is a 2x2 matrix.
    # We need to decompose Z = X Y into rank-1 terms.
    # X and Y are linear combinations of A and B blocks.
    # We can use the standard Strassen 2x2x2 decomposition on X and Y?
    # Yes, because the multiplication tensor is the same.
    # The inputs are just different linear combinations.
    
    # So for each M_k, we identify the linear combinations of A and B blocks.
    # Then we apply the 7-term Strassen decomposition to those combinations.
    
    # M1: A_part = A11 + A22, B_part = B11 + B22
    # A11 is at (0,0), A22 is at (2,2).
    # So A_part has 1 at A11 and 1 at A22.
    # B_part has 1 at B11 and 1 at B22.
    # Result contributes to C11 (coeff 1) and C22 (coeff 1).
    
    def add_strassen_combo(a_combos, b_combos, c_combos, col_start):
        # a_combos: list of (offset, coeff)
        # b_combos: list of (offset, coeff)
        # c_combos: list of (offset, coeff)
        
        # Build the effective 2x2 A and B matrices for this term
        # We need to add the 7 Strassen terms for the product of these combined matrices.
        
        # The 7 Strassen terms for 2x2x2 are fixed.
        # We just scale the U, V, W vectors by the coefficients.
        
        # However, the "A" in the Strassen formula is the combined matrix.
        # The local indices (0,0), (0,1), etc. refer to the 2x2 block.
        
        # We will iterate over the 7 standard Strassen terms.
        # For each term, we update U, V, W.
        
        terms = [
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 1}),
            ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(0,0): 0, (0,1): 0, (1,0): 1, (1,1): 0}),
            ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,0): 0, (0,1): 1, (1,0): 0, (1,1): 0}),
            ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 0, (1,0): 1, (1,1): 0}),
            ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1, (1,0): 0, (1,1): 0}),
            ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(0,0): 0, (0,1): 0, (1,0): 0, (1,1): 1}),
            ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 0}),
        ]
        
        for k, (a_c, b_c, c_c) in enumerate(terms):
            col = col_start + k
            # Update U
            for (ai, aj), val in a_c.items():
                # val is the coefficient in the Strassen term
                # We need to apply this to all A blocks in a_combos
                for a_off, a_coeff in a_combos:
                    global_i = a_off[0] + ai
                    global_j = a_off[1] + aj
                    U[global_i * 4 + global_j][col] += Fraction(val * a_coeff)
            
            # Update V
            for (bi, bj), val in b_c.items():
                for b_off, b_coeff in b_combos:
                    global_i = b_off[0] + bi
                    global_j = b_off[1] + bj
                    V[global_i * 6 + global_j][col] += Fraction(val * b_coeff)
            
            # Update W
            for (ci, cj), val in c_c.items():
                for c_off, c_coeff in c_combos:
                    global_i = c_off[0] + ci
                    global_j = c_off[1] + cj
                    W[global_i * 4 + global_j][col] += Fraction(val * c_coeff)
        
        return col_start + 7

    # M1: (A11+A22)(B11+B22) -> C11 + C22
    a1 = [((0,0), 1), ((2,2), 1)]
    b1 = [((0,0), 1), ((2,2), 1)]
    c1 = [((0,0), 1), ((2,2), 1)]
    col_idx = add_strassen_combo(a1, b1, c1, col_idx)
    
    # M2: (A21+A22)B11 -> C21
    a2 = [((2,0), 1), ((2,2), 1)]
    b2 = [((0,0), 1)]
    c2 = [((2,0), 1)]
    col_idx = add_strassen_combo(a2, b2, c2, col_idx)
    
    # M3: A11(B12-B22) -> C12
    a3 = [((0,0), 1)]
    b3 = [((0,2), 1), ((2,2), -1)]
    c3 = [((0,2), 1)]
    col_idx = add_strassen_combo(a3, b3, c3, col_idx)
    
    # M4: A22(B21-B11) -> C11 + C21
    a4 = [((2,2), 1)]
    b4 = [((2,0), 1), ((0,0), -1)]
    c4 = [((0,0), 1), ((2,0), 1)]
    col_idx = add_strassen_combo(a4, b4, c4, col_idx)
    
    # M5: (A11+A12)B22 -> C11 - C12
    a5 = [((0,0), 1), ((0,2), 1)]
    b5 = [((2,2), 1)]
    c5 = [((0,0), -1), ((0,2), 1)]
    col_idx = add_strassen_combo(a5, b5, c5, col_idx)
    
    # M6: (A21-A11)(B11+B12) -> C22
    a6 = [((2,0), 1), ((0,0), -1)]
    b6 = [((0,0), 1), ((0,2), 1)]
    c6 = [((2,2), 1)]
    col_idx = add_strassen_combo(a6, b6, c6, col_idx)
    
    # M7: (A12-A22)(B21+B22) -> C11
    a7 = [((0,2), 1), ((2,2), -1)]
    b7 = [((2,0), 1), ((2,2), 1)]
    c7 = [((0,0), 1)]
    col_idx = add_strassen_combo(a7, b7, c7, col_idx)
    
    # Part 2: 4x4x2
    # B2 is columns 4,5.
    # B_top = B[0:2, 4:6] (rows 0-1, cols 4-5)
    # B_bot = B[2:4, 4:6] (rows 2-3, cols 4-5)
    # Offsets for B_top: (0,4), B_bot: (2,4)
    # Offsets for C: C_top = C[0:2, 4:6] (offset (0,4)), C_bot = C[2:4, 4:6] (offset (2,4))
    
    # P1 = A11 B_top -> C_top
    a_p1 = [((0,0), 1)]
    b_p1 = [((0,4), 1)]
    c_p1 = [((0,4), 1)]
    col_idx = add_strassen_combo(a_p1, b_p1, c_p1, col_idx)
    
    # P2 = A12 B_bot -> C_top
    a_p2 = [((0,2), 1)]
    b_p2 = [((2,4), 1)]
    c_p2 = [((0,4), 1)]
    col_idx = add_strassen_combo(a_p2, b_p2, c_p2, col_idx)
    
    # P3 = A21 B_top -> C_bot
    a_p3 = [((2,0), 1)]
    b_p3 = [((0,4), 1)]
    c_p3 = [((2,4), 1)]
    col_idx = add_strassen_combo(a_p3, b_p3, c_p3, col_idx)
    
    # P4 = A22 B_bot -> C_bot
    a_p4 = [((2,2), 1)]
    b_p4 = [((2,4), 1)]
    c_p4 = [((2,4), 1)]
    col_idx = add_strassen_combo(a_p4, b_p4, c_p4, col_idx)
    
    # col_idx should be 77
    assert col_idx == 77, f"Expected 77, got {col_idx}"
    
    def to_numeric(mat):
        out = []
        for row in mat:
            new_row = []
            for x in row:
                if x == int(x):
                    new_row.append(int(x))
                else:
                    new_row.append(float(x))
            out.append(new_row)
        return out
    
    U_out = to_numeric(U)
    V_out = to_numeric(V)
    W_out = to_numeric(W)
    
    return {"U": U_out, "V": V_out, "W": W_out}

