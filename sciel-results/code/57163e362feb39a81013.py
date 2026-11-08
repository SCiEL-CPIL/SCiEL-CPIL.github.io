import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # Rank 77 decomposition
    # Part 1: 4x4x4 (cols 0-3)
    # Part 2: 4x4x2 (cols 4-5)
    
    # We will build U, V, W as lists of lists or numpy arrays.
    # Since we don't know the exact rank until we finish, we can use lists and then convert.
    # But the rank is fixed at 77.
    rank = 77
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add Strassen 2x2x2 decomposition
    # A block: (i_a, j_a) -> global rows 2*i_a..2*i_a+1, cols 2*j_a..2*j_a+1
    # B block: (i_b, j_b) -> global rows 2*i_b..2*i_b+1, cols 2*j_b..2*j_b+1
    # C block: (i_c, j_c) -> global rows 2*i_c..2*i_c+1, cols 2*j_c..2*j_c+1
    # scale: multiplier for the contribution
    
    def add_strassen_2x2x2(i_a, j_a, i_b, j_b, i_c, j_c, scale):
        nonlocal col
        # Strassen steps for X (2x2), Y (2x2) -> Z (2x2)
        # M1 = (X00+X11)(Y00+Y11)
        # M2 = (X10+X11)Y00
        # M3 = X00(Y10-Y11)
        # M4 = X11(Y01-Y00)
        # M5 = (X00+X01)Y11
        # M6 = (X10-X00)(Y00+Y10)
        # M7 = (X01-X11)(Y11+Y01)
        
        # Z00 = M1 + M4 - M5 + M7
        # Z01 = M3 + M5
        # Z10 = M2 + M4
        # Z11 = M1 - M2 + M3 + M6
        
        steps = [
            # (coeffs_X, coeffs_Y, coeffs_Z, step_id)
            # coeffs are lists of (local_index, coeff)
            # Local indices: 00, 01, 10, 11
            # X: 0:00, 1:01, 2:10, 3:11
            # Y: 0:00, 1:01, 2:10, 3:11
            # Z: 0:00, 1:01, 2:10, 3:11
            
            # M1
            ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
            # M2
            ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
            # M3
            ([(0, 1)], [(2, 1), (3, -1)], [(1, 1), (3, 1)]),
            # M4
            ([(3, 1)], [(1, 1), (0, -1)], [(0, 1), (2, 1)]),
            # M5
            ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
            # M6
            ([(2, 1), (0, -1)], [(0, 1), (2, 1)], [(3, 1)]),
            # M7
            ([(1, 1), (3, -1)], [(3, 1), (1, 1)], [(0, 1), (3, 1)]),
        ]
        
        for coeffs_x, coeffs_y, coeffs_z in steps:
            # Add to U
            for loc_idx, coeff in coeffs_x:
                a, b = divmod(loc_idx, 2)
                gi = 2 * i_a + a
                gj = 2 * j_a + b
                u_idx = gi * m + gj
                U[u_idx][col] += scale * coeff
            
            # Add to V
            for loc_idx, coeff in coeffs_y:
                a, b = divmod(loc_idx, 2)
                gi = 2 * i_b + a
                gj = 2 * j_b + b
                v_idx = gi * p + gj
                V[v_idx][col] += scale * coeff
            
            # Add to W
            for loc_idx, coeff in coeffs_z:
                a, b = divmod(loc_idx, 2)
                gi = 2 * i_c + a
                gj = 2 * j_c + b
                # W index is k*n + i. Here gj is k, gi is i.
                w_idx = gj * n + gi
                W[w_idx][col] += scale * coeff
            
            col += 1

    # Part 1: 4x4x4
    # Strassen at 4x4 level
    # M1 = (A11+A22)(B11+B22)
    # A11: (0,0), A22: (1,1)
    # B11: (0,0), B22: (1,1)
    # C11: (0,0)
    
    # We need to distribute the linear combinations.
    # M1 = A11B11 + A11B22 + A22B11 + A22B22
    # This is 4 products of 2x2x2.
    # But Strassen combines them.
    # The "step" above handles the 2x2x2 internal structure.
    # We need to call add_strassen_2x2x2 for each of the 7 Strassen products of the 4x4 case.
    
    # Strassen 4x4:
    # P1 = (A11+A22)(B11+B22) -> 4 terms: (0,0)x(0,0), (0,0)x(1,1), (1,1)x(0,0), (1,1)x(1,1)
    # P2 = (A21+A22)B11 -> 2 terms: (1,0)x(0,0), (1,1)x(0,0)
    # P3 = A11(B21-B22) -> 2 terms: (0,0)x(1,0), (0,0)x(1,1) with -1
    # P4 = A22(B12-B11) -> 2 terms: (1,1)x(0,1), (1,1)x(0,0) with -1
    # P5 = (A11+A12)B22 -> 2 terms: (0,0)x(1,1), (0,1)x(1,1)
    # P6 = (A21-A11)(B11+B21) -> 4 terms
    # P7 = (A12-A22)(B22+B12) -> 4 terms
    
    # Total 4+2+2+2+2+4+4 = 20 calls to add_strassen_2x2x2?
    # No, each "product" in the 4x4 Strassen is a 2x2x2 multiplication.
    # So we call add_strassen_2x2x2 for each pair of blocks.
    
    # P1
    add_strassen_2x2x2(0, 0, 0, 0, 0, 0, 1)
    add_strassen_2x2x2(0, 0, 1, 1, 0, 0, 1)
    add_strassen_2x2x2(1, 1, 0, 0, 0, 0, 1)
    add_strassen_2x2x2(1, 1, 1, 1, 0, 0, 1)
    
    # P2
    add_strassen_2x2x2(1, 0, 0, 0, 1, 0, 1)
    add_strassen_2x2x2(1, 1, 0, 0, 1, 0, 1)
    
    # P3
    add_strassen_2x2x2(0, 0, 1, 0, 0, 1, 1)
    add_strassen_2x2x2(0, 0, 1, 1, 0, 1, -1)
    
    # P4
    add_strassen_2x2x2(1, 1, 0, 1, 1, 1, 1)
    add_strassen_2x2x2(1, 1, 0, 0, 1, 1, -1)
    
    # P5
    add_strassen_2x2x2(0, 0, 1, 1, 0, 1, 1)
    add_strassen_2x2x2(0, 1, 1, 1, 0, 1, 1)
    
    # P6
    add_strassen_2x2x2(1, 0, 0, 0, 1, 0, 1)
    add_strassen_2x2x2(1, 0, 1, 0, 1, 0, 1)
    add_strassen_2x2x2(0, 0, 0, 0, 1, 0, -1)
    add_strassen_2x2x2(0, 0, 1, 0, 1, 0, -1)
    
    # P7
    add_strassen_2x2x2(0, 1, 1, 1, 0, 1, 1)
    add_strassen_2x2x2(0, 1, 0, 1, 0, 1, 1)
    add_strassen_2x2x2(1, 1, 1, 1, 0, 1, -1)
    add_strassen_2x2x2(1, 1, 0, 1, 0, 1, -1)
    
    # Wait, I need to check the C block indices for the 4x4 Strassen.
    # C11 gets P1, P4, P5, P7
    # C12 gets P3, P5
    # C21 gets P2, P4
    # C22 gets P1, P2, P3, P6
    
    # My add_strassen_2x2x2 calls above assumed C block is (0,0) for all.
    # I need to specify the correct C block for each term.
    
    # Let's restart the 4x4 part with correct C indices.
    
    # Reset col to 0 and U, V, W to zero?
    # It's easier to just define the calls correctly in one go.
    
    # Let's clear and redo.
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    
    # P1 = (A11+A22)(B11+B22)
    # A11B11 -> C11
    add_strassen_2x2x2(0, 0, 0, 0, 0, 0, 1)
    # A11B22 -> C11
    add_strassen_2x2x2(0, 0, 1, 1, 0, 0, 1)
    # A22B11 -> C11
    add_strassen_2x2x2(1, 1, 0, 0, 0, 0, 1)
    # A22B22 -> C11
    add_strassen_2x2x2(1, 1, 1, 1, 0, 0, 1)
    
    # P2 = (A21+A22)B11
    # A21B11 -> C21
    add_strassen_2x2x2(1, 0, 0, 0, 1, 0, 1)
    # A22B11 -> C21
    add_strassen_2x2x2(1, 1, 0, 0, 1, 0, 1)
    
    # P3 = A11(B21-B22)
    # A11B21 -> C12
    add_strassen_2x2x2(0, 0, 1, 0, 0, 1, 1)
    # A11B22 -> C12
    add_strassen_2x2x2(0, 0, 1, 1, 0, 1, -1)
    
    # P4 = A22(B12-B11)
    # A22B12 -> C22
    add_strassen_2x2x2(1, 1, 0, 1, 1, 1, 1)
    # A22B11 -> C22
    add_strassen_2x2x2(1, 1, 0, 0, 1, 1, -1)
    
    # P5 = (A11+A12)B22
    # A11B22 -> C12
    add_strassen_2x2x2(0, 0, 1, 1, 0, 1, 1)
    # A12B22 -> C12
    add_strassen_2x2x2(0, 1, 1, 1, 0, 1, 1)
    
    # P6 = (A21-A11)(B11+B21)
    # A21B11 -> C22
    add_strassen_2x2x2(1, 0, 0, 0, 1, 1, 1)
    # A21B21 -> C22
    add_strassen_2x2x2(1, 0, 1, 0, 1, 1, 1)
    # A11B11 -> C22
    add_strassen_2x2x2(0, 0, 0, 0, 1, 1, -1)
    # A11B21 -> C22
    add_strassen_2x2x2(0, 0, 1, 0, 1, 1, -1)
    
    # P7 = (A12-A22)(B22+B12)
    # A12B22 -> C11
    add_strassen_2x2x2(0, 1, 1, 1, 0, 0, 1)
    # A12B12 -> C11
    add_strassen_2x2x2(0, 1, 0, 1, 0, 0, 1)
    # A22B22 -> C11
    add_strassen_2x2x2(1, 1, 1, 1, 0, 0, -1)
    # A22B12 -> C11
    add_strassen_2x2x2(1, 1, 0, 1, 0, 0, -1)
    
    # Part 2: 4x4x2 (cols 4-5 of B)
    # B blocks are at col 2 (global 4-5)
    # B1 (rows 0-1) is block (0, 2)
    # B2 (rows 2-3) is block (1, 2)
    
    # C1 (rows 0-1) is block (0, 2)
    # C2 (rows 2-3) is block (1, 2)
    
    # C1 = A11B1 + A12B2
    add_strassen_2x2x2(0, 0, 0, 2, 0, 2, 1)
    add_strassen_2x2x2(0, 1, 1, 2, 0, 2, 1)
    
    # C2 = A21B1 + A22B2
    add_strassen_2x2x2(1, 0, 0, 2, 1, 2, 1)
    add_strassen_2x2x2(1, 1, 1, 2, 1, 2, 1)
    
    # Total calls: 20 (for 4x4x4) + 4 (for 4x4x2) = 24 calls?
    # No, 4x4x4 had 4+2+2+2+2+4+4 = 20 calls.
    # 4x4x2 had 4 calls.
    # Total 24 calls to add_strassen_2x2x2.
    # Each call adds 7 terms.
    # Total rank = 24 * 7 = 168?
    # Wait. 4x4x4 rank is 49. 4x4x2 rank is 28. Total 77.
    # My count of calls was wrong.
    
    # Let's re-verify the 4x4x4 calls.
    # P1: 4 calls
    # P2: 2 calls
    # P3: 2 calls
    # P4: 2 calls
    # P5: 2 calls
    # P6: 4 calls
    # P7: 4 calls
    # Total 20 calls.
    # 20 * 7 = 140 terms.
    # But Strassen 4x4 is rank 49.
    # Why?
    # Because Strassen 4x4 uses 7 products of 2x2 matrices.
    # Each 2x2 product has rank 7.
    # So 7 * 7 = 49.
    # My decomposition above expanded the linear combinations into individual block products.
    # P1 = (A11+A22)(B11+B22) = A11B11 + A11B22 + A22B11 + A22B22.
    # This is 4 separate 2x2x2 multiplications.
    # But Strassen combines them into ONE 2x2x2 multiplication of the sums?
    # No, Strassen computes M1 = (A11+A22)(B11+B22).
    # This is a single matrix multiplication of two 2x2 matrices.
    # So it requires rank 7.
    # But my code calls add_strassen_2x2x2 for EACH term in the expansion.
    # That is wrong.
    
    # I should call add_strassen_2x2x2 ONCE for each Strassen step M1..M7.
    # But add_strassen_2x2x2 takes specific block indices.
    # For M1, the "A block" is A11+A22. This is not a single block.
    # So I cannot use add_strassen_2x2x2 directly for the 4x4 Strassen steps.
    
    # I need a more general helper or just expand it.
    # If I expand it, I get rank 140 for 4x4x4, which is worse than 96.
    
    # So I must implement the 4x4x4 Strassen correctly.
    # The 7 steps of Strassen for 4x4 are:
    # M1 = (A11+A22)(B11+B22)
    # ...
    # Each M is a 2x2 matrix product.
    # To compute M1, I need to compute the product of two 2x2 matrices X and Y.
    # X = A11+A22, Y = B11+B22.
    # The entries of X are sums of entries of A11 and A22.
    # The entries of Y are sums of entries of B11 and B22.
    # The product M1 is then added to C11 and C22.
    
    # So, for each of the 7 Strassen steps:
    # 1. Define the linear forms for X (from A) and Y (from B).
    # 2. Use the 7-term Strassen decomposition for 2x2x2 to generate the rank-1 terms.
    # 3. Map the result Z to the appropriate C blocks.
    
    # This is doable.
    # I will create a helper that takes:
    # - A list of (block_row, block_col, coeff) for X
    # - A list of (block_row, block_col, coeff) for Y
    # - A list of (block_row, block_col, coeff) for Z
    # And adds the 7 terms.
    
    # This is essentially what add_strassen_2x2x2 does, but with multiple blocks.
    
    # Let's refine add_strassen_2x2x2 to accept lists of blocks.
    
    # Re-initialize
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    
    def add_strassen_multi(a_blocks, b_blocks, c_blocks):
        nonlocal col
        # a_blocks: list of (i, j, coeff)
        # b_blocks: list of (i, j, coeff)
        # c_blocks: list of (i, j, coeff)
        
        # The 7 Strassen steps
        steps = [
            ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
            ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
            ([(0, 1)], [(2, 1), (3, -1)], [(1, 1), (3, 1)]),
            ([(3, 1)], [(1, 1), (0, -1)], [(0, 1), (2, 1)]),
            ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
            ([(2, 1), (0, -1)], [(0, 1), (2, 1)], [(3, 1)]),
            ([(1, 1), (3, -1)], [(3, 1), (1, 1)], [(0, 1), (3, 1)]),
        ]
        
        for coeffs_x, coeffs_y, coeffs_z in steps:
            # Build U vector
            for loc_idx, coeff_x in coeffs_x:
                a, b = divmod(loc_idx, 2)
                for bi, bj, bc in a_blocks:
                    gi = 2 * bi + a
                    gj = 2 * bj + b
                    u_idx = gi * m + gj
                    U[u_idx][col] += coeff_x * bc
            
            # Build V vector
            for loc_idx, coeff_y in coeffs_y:
                a, b = divmod(loc_idx, 2)
                for bi, bj, bc in b_blocks:
                    gi = 2 * bi + a
                    gj = 2 * bj + b
                    v_idx = gi * p + gj
                    V[v_idx][col] += coeff_y * bc
            
            # Build W vector
            for loc_idx, coeff_z in coeffs_z:
                a, b = divmod(loc_idx, 2)
                for bi, bj, bc in c_blocks:
                    gi = 2 * bi + a
                    gj = 2 * bj + b
                    w_idx = gj * n + gi
                    W[w_idx][col] += coeff_z * bc
            
            col += 1

    # 4x4x4 Part
    # M1 = (A11+A22)(B11+B22) -> C11, C22
    a_blocks = [(0, 0, 1), (1, 1, 1)]
    b_blocks = [(0, 0, 1), (1, 1, 1)]
    c_blocks = [(0, 0, 1), (1, 1, 1)]
    add_strassen_multi(a_blocks, b_blocks, c_blocks)
    
    # M2 = (A21+A22)B11 -> C21
    a_blocks = [(1, 0, 1), (1, 1, 1)]
    b_blocks = [(0, 0, 1)]
    c_blocks = [(1, 0, 1)]
    add_strassen_multi(a_blocks, b_blocks, c_blocks)
    
    # M3 = A11(B21-B22) -> C12
    a_blocks = [(0, 0, 1)]
    b_blocks = [(1, 0, 1), (1, 1, -1)]
    c_blocks = [(0, 1, 1)]
    add_strassen_multi(a_blocks, b_blocks, c_blocks)
    
    # M4 = A22(B12-B11) -> C22
    a_blocks = [(1, 1, 1)]
    b_blocks = [(0, 1, 1), (0, 0, -1)]
    c_blocks = [(1, 1, 1)]
    add_strassen_multi(a_blocks, b_blocks, c_blocks)
    
    # M5 = (A11+A12)B22 -> C12
    a_blocks = [(0, 0, 1), (0, 1, 1)]
    b_blocks = [(1, 1, 1)]
    c_blocks = [(0, 1, 1)]
    add_strassen_multi(a_blocks, b_blocks, c_blocks)
    
    # M6 = (A21-A11)(B11+B21) -> C22
    a_blocks = [(1, 0, 1), (0, 0, -1)]
    b_blocks = [(0, 0, 1), (1, 0, 1)]
    c_blocks = [(1, 1, 1)]
    add_strassen_multi(a_blocks, b_blocks, c_blocks)
    
    # M7 = (A12-A22)(B22+B12) -> C11
    a_blocks = [(0, 1, 1), (1, 1, -1)]
    b_blocks = [(1, 1, 1), (0, 1, 1)]
    c_blocks = [(0, 0, 1)]
    add_strassen_multi(a_blocks, b_blocks, c_blocks)
    
    # 4x4x2 Part
    # C1 = A11B1 + A12B2 -> C1 (block 0,2)
    a_blocks = [(0, 0, 1), (0, 1, 1)]
    b_blocks = [(0, 2, 1), (1, 2, 1)]
    c_blocks = [(0, 2, 1)]
    add_strassen_multi(a_blocks, b_blocks, c_blocks)
    
    # C2 = A21B1 + A22B2 -> C2 (block 1,2)
    a_blocks = [(1, 0, 1), (1, 1, 1)]
    b_blocks = [(0, 2, 1), (1, 2, 1)]
    c_blocks = [(1, 2, 1)]
    add_strassen_multi(a_blocks, b_blocks, c_blocks)
    
    # Total calls: 7 + 2 = 9 calls.
    # Each call adds 7 terms.
    # Total rank = 9 * 7 = 63.
    # Wait, 4x4x4 is 7 calls. 4x4x2 is 2 calls.
    # Total 9 calls.
    # Rank 63.
    # This is much better than 77.
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
