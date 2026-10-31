"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    # Strassen-based rank-49 decomposition for 4x4 matrix multiplication.
    # Uses nested Strassen: outer 2x2 blocks (7 terms), each block multiplied
    # via inner Strassen (7 terms) = 49 total rank-1 tensors.
    import time
    start_time = time.time()
    
    n, m, p = DIMENSIONS
    rank = 49
    
    U = [[0] * rank for _ in range(n * m)]  # 16 x 49
    V = [[0] * rank for _ in range(m * p)]  # 16 x 49
    W = [[0] * rank for _ in range(p * n)]  # 16 x 49
    
    # Strassen 2x2 multiplication formulas:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B10 - B11)
    # M4 = A11 * (B01 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B10)
    # M7 = (A01 - A11) * (B01 + B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Outer Strassen for 4x4 (viewed as 2x2 of 2x2 blocks):
    # Let A = [[A00, A01], [A10, A11]] where each Ai is 2x2
    # Let B = [[B00, B01], [B10, B11]] where each Bj is 2x2
    # C = A*B
    # 
    # Outer Strassen gives:
    # P1 = (A00 + A11) * (B00 + B11)
    # P2 = (A10 + A11) * B00
    # P3 = A00 * (B10 - B11)
    # P4 = A11 * (B01 - B00)
    # P5 = (A00 + A01) * B11
    # P6 = (A10 - A00) * (B00 + B10)
    # P7 = (A01 - A11) * (B01 + B11)
    #
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    #
    # Each Pi is a product of two 2x2 matrices, so we apply inner Strassen to each.
    
    # Indexing: element (i,j) of a 4x4 matrix is at row i*4+j
    # Block (bi,bj) of 2x2 blocks contains elements at positions (bi*2+r, bj*2+c) for r,c in {0,1}
    
    # For each outer term (7 terms), for each inner Strassen term (7 terms),
    # we get one rank-1 tensor.
    
    # Let's define the outer Strassen factors:
    # For each outer term t in 0..6, we have:
    # U_outer[t] = linear combination of A elements
    # V_outer[t] = linear combination of B elements
    # 
    # Then for each inner Strassen term s in 0..6, we have:
    # U_inner[s] = linear combination of U_outer elements (within the 2x2 block structure)
    # V_inner[s] = linear combination of V_outer elements
    
    # Let me define the 2x2 Strassen patterns.
    # For a 2x2 matrix multiplication C = A*B:
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    
    strassen_patterns = [
        # (u_coefficients, v_coefficients, w_coefficients)
        # Each is a dict from (row, col) to coefficient
        # M1 = (A00 + A11) * (B00 + B11)
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1},
         {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 0}),
        # M2 = (A10 + A11) * B00
        ({(1,0): 1, (1,1): 1}, {(0,0): 1},
         {(0,0): 0, (0,1): 0, (1,0): 1, (1,1): 0}),
        # M3 = A00 * (B10 - B11)
        ({(0,0): 1}, {(1,0): 1, (1,1): -1},
         {(0,0): 0, (0,1): 1, (1,0): 0, (1,1): 0}),
        # M4 = A11 * (B01 - B00)
        ({(1,1): 1}, {(0,1): 1, (0,0): -1},
         {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 0}),
        # M5 = (A00 + A01) * B11
        ({(0,0): 1, (0,1): 1}, {(1,1): 1},
         {(0,0): -1, (0,1): 0, (1,0): 0, (1,1): 0}),
        # M6 = (A10 - A00) * (B00 + B10)
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1},
         {(0,0): 0, (0,1): 0, (1,0): 0, (1,1): 1}),
        # M7 = (A01 - A11) * (B01 + B11)
        ({(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1},
         {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 0}),
    ]
    
    # Outer Strassen patterns for 4x4:
    # P1 = (A00 + A11) * (B00 + B11)
    # P2 = (A10 + A11) * B00
    # P3 = A00 * (B10 - B11)
    # P4 = A11 * (B01 - B00)
    # P5 = (A00 + A01) * B11
    # P6 = (A10 - A00) * (B00 + B10)
    # P7 = (A01 - A11) * (B01 + B11)
    
    # For each outer term, the "A block" and "B block" are 2x2 matrices.
    # The outer term multiplies two 2x2 matrices, which we decompose via inner Strassen.
    
    # Outer term definitions:
    # Each outer term has:
    # - A_block: which 2x2 block(s) of A to use, with coefficients
    # - B_block: which 2x2 block(s) of B to use, with coefficients
    
    # Let's represent blocks by (row_block, col_block) where row_block, col_block in {0,1}
    # A_block (rb, cb) contains elements at global positions (rb*2+r, cb*2+c) for r,c in {0,1}
    
    outer_patterns = [
        # P1 = (A00 + A11) * (B00 + B11)
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # P2 = (A10 + A11) * B00
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}),
        # P3 = A00 * (B10 - B11)
        ({(0,0): 1}, {(1,0): 1, (1,1): -1}),
        # P4 = A11 * (B01 - B00)
        ({(1,1): 1}, {(0,1): 1, (0,0): -1}),
        # P5 = (A00 + A01) * B11
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # P6 = (A10 - A00) * (B00 + B10)
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # P7 = (A01 - A11) * (B01 + B11)
        ({(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
    ]
    
    # For each outer term ot in 0..6, for each inner Strassen term st in 0..6:
    # We construct one rank-1 tensor.
    
    for ot in range(7):
        a_block_pattern, b_block_pattern = outer_patterns[ot]
        
        for st in range(7):
            u_idx = ot * 7 + st
            u_inner, v_inner, w_inner = strassen_patterns[st]
            
            # U vector: for each element in A (16 elements), compute coefficient
            for (ar, ac), a_coeff in a_block_pattern.items():
                for (ir, ic), inner_coeff in u_inner.items():
                    # Global position: block (ar, ac), inner position (ir, ic)
                    global_row = ar * 2 + ir
                    global_col = ac * 2 + ic
                    U[global_row * 4 + global_col][u_idx] += a_coeff * inner_coeff
            
            # V vector: for each element in B (16 elements), compute coefficient
            for (br, bc), b_coeff in b_block_pattern.items():
                for (ir, ic), inner_coeff in v_inner.items():
                    # Global position: block (br, bc), inner position (ir, ic)
                    global_row = br * 2 + ir
                    global_col = bc * 2 + ic
                    V[global_row * 4 + global_col][u_idx] += b_coeff * inner_coeff
            
            # W vector: for each element in C (16 elements), compute coefficient
            # The outer term contributes to specific C blocks based on the outer Strassen formula
            # C00 = P1 + P4 - P5 + P7
            # C01 = P3 + P5
            # C10 = P2 + P4
            # C11 = P1 - P2 + P3 + P6
            
            # For outer term ot, which C blocks does it contribute to?
            c_block_contrib = {
                0: {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 0},   # P1 -> C00
                1: {(0,0): 0, (0,1): 0, (1,0): 1, (1,1): 0},   # P2 -> C10
                2: {(0,0): 0, (0,1): 1, (1,0): 0, (1,1): 0},   # P3 -> C01
                3: {(0,0): 1, (0,1): 0, (1,0): 1, (1,1): 0},   # P4 -> C00, C10
                4: {(0,0): -1, (0,1): 1, (1,0): 0, (1,1): 0},  # P5 -> -C00, C01
                5: {(0,0): 0, (0,1): 0, (1,0): 0, (1,1): 1},   # P6 -> C11
                6: {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 0},   # P7 -> C00
            }
            
            # Wait, let me recheck:
            # C00 = P1 + P4 - P5 + P7
            # C01 = P3 + P5
            # C10 = P2 + P4
            # C11 = P1 - P2 + P3 + P6
            
            # So:
            # P1: C00 += 1, C11 += 1
            # P2: C10 += 1, C11 -= 1
            # P3: C01 += 1, C11 += 1
            # P4: C00 += 1, C10 += 1
            # P5: C00 -= 1, C01 += 1
            # P6: C11 += 1
            # P7: C00 += 1
            
            c_block_contrib = {
                0: {(0,0): 1, (1,1): 1},   # P1 -> C00 +1, C11 +1
                1: {(1,0): 1, (1,1): -1},  # P2 -> C10 +1, C11 -1
                2: {(0,1): 1, (1,1): 1},   # P3 -> C01 +1, C11 +1
                3: {(0,0): 1, (1,0): 1},   # P4 -> C00 +1, C10 +1
                4: {(0,0): -1, (0,1): 1},  # P5 -> C00 -1, C01 +1
                5: {(1,1): 1},             # P6 -> C11 +1
                6: {(0,0): 1},             # P7 -> C00 +1
            }
            
            for (cr, cc), c_coeff in c_block_contrib[ot].items():
                for (ir, ic), inner_coeff in w_inner.items():
                    # Global position: block (cr, cc), inner position (ir, ic)
                    global_row = cr * 2 + ir
                    global_col = cc * 2 + ic
                    W[global_row * 4 + global_col][u_idx] += c_coeff * inner_coeff
    
    # Convert to float if needed (entries should be integers)
    # The task says integer/half-integer entries, so integers are fine
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
