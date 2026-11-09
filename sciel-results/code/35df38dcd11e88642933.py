"""Matrix multiplication 4x5 x 5x5 tensor decomposition.

Strategy:
- Use Strassen's rank-7 decomposition for each of the 4x2 blocks of A times 2x5 blocks of B
  (4 such blocks, giving 4*7 = 28 terms for the 4x2 * 2x5 sub-problems).
- Use schoolbook for the remaining 4x3 * 3x5 part (36 terms).
- Total: 28 + 36 = 64? No, that's not right.

Let me think more carefully.

A is 4x5, B is 5x5, C is 4x5.

Split the middle dimension (j) into two parts: j1=2, j2=3.
C = A[:,0:2] @ B[0:2,:] + A[:,2:5] @ B[2:5,:]

For A[:,0:2] (4x2) @ B[0:2,:] (2x5): this is a 4x2 * 2x5 multiplication.
For A[:,2:5] (4x3) @ B[2:5,:] (3x5): this is a 4x3 * 3x5 multiplication.

The 4x2 * 2x5 part: we can further decompose.
Split A[:,0:2] into rows: top 2x2 and bottom 2x2.
Top: A[0:2,0:2] (2x2) @ B[0:2,:] (2x5) -> 2x2 * 2x5
Bottom: A[2:4,0:2] (2x2) @ B[0:2,:] (2x5) -> 2x2 * 2x5

Each 2x2 * 2x5 can be done with: split B into 2x2 + 2x3
2x2 * 2x2: Strassen rank 7
2x2 * 2x3: schoolbook 2*2*3 = 12

So each 2x2 * 2x5 = 7 + 12 = 19
Two of them = 38

For the 4x3 * 3x5 part: schoolbook = 4*3*5 = 60

Total = 38 + 60 = 98. Not great.

Alternative: Use Strassen more aggressively.

Let me try: split A (4x5) into 4 row-blocks of 1x5 each.
Each 1x5 @ 5x5 = 1x5. Rank of each = 5 (can't do better since it's just a vector times matrix).
Total = 4*5 = 20 for the A-side? No, that's not how it works.

Actually, let me think about this differently. The tensor T_{i,j,k} = delta_{j,k} for the multiplication
T(a,b)_{i,k} = sum_j a_{i,j} * b_{j,k}

Schoolbook: rank = n*m*p = 4*5*5 = 100, with terms: for each (j,k), a_{i,j} * b_{j,k} contributes to c_{i,k}.

Can we use Strassen on the 4x4 part?
Split into: A = [A_left (4x4) | A_right (4x1)]
B = [B_top (4x5); B_bottom (1x5)]

C = A_left @ B_top + A_right @ B_bottom

A_left @ B_top: 4x4 * 4x5. Split B_top into 4x4 + 4x1.
= A_left @ B_top_left (4x4*4x4) + A_left @ B_top_right (4x4*4x1)

A_left @ B_top_left: Use Strassen twice? 4x4*4x4.
Split 4x4 into 2x2 blocks:
A_left = [[A11, A12], [A21, A22]] where each is 2x2
B_top_left = [[B11, B12], [B21, B22]] where each is 2x2

Strassen gives 7 terms of 2x2*2x2 = 2x2. Each 2x2*2x2 has rank 7 (Strassen).
So 7*7 = 49 terms for the 4x4*4x4 part.

A_left @ B_top_right: 4x4 * 4x1. Schoolbook = 4*4*1 = 16.
A_right @ B_bottom: 4x1 * 1x5. Schoolbook = 4*1*5 = 20.

Total = 49 + 16 + 20 = 85.

Let me implement this 85-rank decomposition.
"""

from fractions import Fraction
import time

def construct():
    n, m, p = 4, 5, 5
    rank = 85
    U = [[0] * rank for _ in range(n * m)]  # 20 x 85
    V = [[0] * rank for _ in range(m * p)]  # 25 x 85
    W = [[0] * rank for _ in range(p * n)]  # 20 x 85
    
    col = 0
    
    # Part 1: 4x4 * 4x4 using Strassen twice (49 terms)
    # A indices: rows 0-3, cols 0-3
    # B indices: rows 0-3, cols 0-3
    # C indices: rows 0-3, cols 0-3
    
    # Strassen level 1: 7 block-level multiplications of 2x2 blocks
    # Each is decomposed into 7 scalar multiplications
    
    # Define Strassen 2x2x2 decomposition helper
    # For 2x2 matrices with entries indexed (r,c) in {0,1}x{0,1}:
    # Given offsets for A (a_r, a_c), B (b_r, b_c), C (c_r, c_c)
    # And coefficient vectors for A and B blocks
    
    def add_strassen_2x2(a_offsets, b_offsets, c_offsets, a_coeffs, b_coeffs, c_coeffs):
        """
        a_offsets: (r, c) - top-left of 2x2 A block
        b_offsets: (r, c) - top-left of 2x2 B block
        c_offsets: (r, c) - top-left of 2x2 C block
        a_coeffs: 4-tuple of coefficients for A entries [00, 01, 10, 11]
        b_coeffs: 4-tuple of coefficients for B entries [00, 01, 10, 11]
        c_coeffs: 4-tuple of coefficients for C entries [00, 01, 10, 11]
        """
        nonlocal col
        ar, ac = a_offsets
        br, bc = b_offsets
        cr, cc = c_offsets
        
        # Strassen terms:
        # s1 = (a00+a11)*(b00+b11)
        # s2 = (a10+a11)*b00
        # s3 = a00*(b00-b10)
        # s4 = a11*(b11-b01)
        # s5 = (a00+a01)*b11
        # s6 = (a10-a00)*(b00+b01)
        # s7 = (a01-a11)*(b10+b11)
        #
        # c00 = s1+s4-s5+s7
        # c01 = s3+s5
        # c10 = s2+s4
        # c11 = s1-s2+s3+s6
        
        # Each s_i is a product of a linear combination of A entries and a linear combination of B entries
        # We need to expand each product into scalar terms
        
        def add_product_term(a_terms, b_terms, c_idx, c_sign):
            """
            a_terms: list of (coeff, i, j) for A
            b_terms: list of (coeff, i, j) for B
            c_idx: (dr, dc) offset in the 2x2 C block
            c_sign: coefficient for this C entry
            """
            nonlocal col
            for (ac_coeff, ai, aj) in a_terms:
                for (bc_coeff, bi, bj) in b_terms:
                    U[(ar + ai) * m + (ac + aj)][col] = ac_coeff
                    V[(br + bi) * p + (bc + bj)][col] = bc_coeff
                    W[(cr + c_idx[0]) * n + (cc + c_idx[1])][col] = c_sign
                    col += 1
        
        # s1 = (a00+a11)*(b00+b11)
        a1 = [(a_coeffs[0], 0, 0), (a_coeffs[3], 1, 1)]
        b1 = [(b_coeffs[0], 0, 0), (b_coeffs[3], 1, 1)]
        # c00 += s1, c11 += s1
        add_product_term(a1, b1, (0, 0), c_coeffs[0])
        add_product_term(a1, b1, (1, 1), c_coeffs[3])
        
        # s2 = (a10+a11)*b00
        a2 = [(a_coeffs[2], 1, 0), (a_coeffs[3], 1, 1)]
        b2 = [(b_coeffs[0], 0, 0)]
        # c10 += s2, c11 -= s2
        add_product_term(a2, b2, (1, 0), c_coeffs[2])
        add_product_term(a2, b2, (1, 1), -c_coeffs[3])
        
        # s3 = a00*(b00-b10)
        a3 = [(a_coeffs[0], 0, 0)]
        b3 = [(b_coeffs[0], 0, 0), (b_coeffs[2], 1, 0)]
        # c01 += s3, c11 += s3
        add_product_term(a3, b3, (0, 1), c_coeffs[1])
        add_product_term(a3, b3, (1, 1), c_coeffs[3])
        
        # s4 = a11*(b11-b01)
        a4 = [(a_coeffs[3], 1, 1)]
        b4 = [(b_coeffs[3], 1, 1), (b_coeffs[1], 0, 1)]
        # c00 += s4, c10 += s4
        add_product_term(a4, b4, (0, 0), c_coeffs[0])
        add_product_term(a4, b4, (1, 0), c_coeffs[2])
        
        # s5 = (a00+a01)*b11
        a5 = [(a_coeffs[0], 0, 0), (a_coeffs[1], 0, 1)]
        b5 = [(b_coeffs[3], 1, 1)]
        # c00 -= s5, c01 += s5
        add_product_term(a5, b5, (0, 0), -c_coeffs[0])
        add_product_term(a5, b5, (0, 1), c_coeffs[1])
        
        # s6 = (a10-a00)*(b00+b01)
        a6 = [(a_coeffs[2], 1, 0), (a_coeffs[0], 0, 0)]
        b6 = [(b_coeffs[0], 0, 0), (b_coeffs[1], 0, 1)]
        # c11 += s6
        add_product_term(a6, b6, (1, 1), c_coeffs[3])
        
        # s7 = (a01-a11)*(b10+b11)
        a7 = [(a_coeffs[1], 0, 1), (a_coeffs[3], 1, 1)]
        b7 = [(b_coeffs[2], 1, 0), (b_coeffs[3], 1, 1)]
        # c00 += s7
        add_product_term(a7, b7, (0, 0), c_coeffs[0])
    
    # Now apply Strassen at block level for 4x4 * 4x4
    # Block offsets:
    # A11: (0,0), A12: (0,2), A21: (2,0), A22: (2,2)
    # B11: (0,0), B12: (0,2), B21: (2,0), B22: (2,2)
    # C11: (0,0), C12: (0,2), C21: (2,0), C22: (2,2)
    
    # Strassen level 1 combinations:
    # S1 = (A11+A22) * (B11+B22)
    # S2 = (A21+A22) * B11
    # S3 = A11 * (B11-B21)
    # S4 = A22 * (B22-B12)
    # S5 = (A11+A12) * B22
    # S6 = (A21-A11) * (B11+B12)
    # S7 = (A12-A22) * (B21+B22)
    #
    # C11 = S1+S4-S5+S7
    # C12 = S3+S5
    # C21 = S2+S4
    # C22 = S1-S2+S3+S6
    
    # For each S_i, we need to decompose the 2x2 * 2x2 product using Strassen
    # The 2x2 blocks are at specific offsets
    
    # S1 = (A11+A22) * (B11+B22)
    # A part: A11+A22. This is NOT a contiguous 2x2 block.
    # A11 is at (0,0), A22 is at (2,2). So we have two separate 2x2 blocks.
    # We can't directly apply the 2x2 Strassen formula to this.
    
    # Actually, we need to be more careful. The Strassen formula at the block level
    # gives us products of (sums of blocks). To decompose further, we need to
    # express each such product as a sum of rank-1 terms.
    
    # This is getting complex. Let me just enumerate all terms directly.
    
    # Alternative approach: directly construct the 49 terms by expanding the
    # Strassen formulas at both levels.
    
    # Let me define the 7 Strassen combinations for a 2x2 matrix product,
    # where the 2x2 matrices can be arbitrary linear combinations.
    
    # For the 4x4 * 4x4 case, I'll just generate all terms programmatically.
    
    # Actually, let me just use a simpler but still improved approach:
    # Use Strassen for the 2x2*2x2 sub-blocks where they are contiguous.
    
    # In the 4x4 * 4x4, we have 4 contiguous 2x2 * 2x2 multiplications
    # if we don't use Strassen at the block level:
    # A11*B11, A12*B21, A21*B12, A22*B22 -> each rank 7 = 28
    # But we also need A11*B12, A12*B22, A21*B11, A22*B21 for the off-diagonal
    # Wait no, the standard block multiplication is:
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    # That's 8 products of 2x2*2x2, each rank 7 = 56 terms.
    
    # With Strassen at block level: 7 products of 2x2*2x2, each rank 7 = 49 terms.
    # But the 7 products involve sums of blocks, not single blocks.
    
    # Let me just implement the full expansion. I'll represent each term as
    # (a_row, a_col, b_row, b_col, c_row, c_col, coeff) and accumulate.
    
    # Clear and rebuild
    rank = 85
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    
    # I'll use a list to collect all terms, then deduplicate if needed.
    terms = []
    
    def add_term(a_idx, b_idx, c_idx, coeff):
        terms.append((a_idx, b_idx, c_idx, coeff))
    
    # Part 1: 4x4 * 4x4 (rows/cols 0-3)
    # Using double Strassen
    # Level 1: 7 block multiplications
    # Level 2: each block mult (2x2*2x2) uses 7 scalar mults
    
    # For S1 = (A11+A22)*(B11+B22):
    # This is a 2x2 matrix product where:
    # X = A11 + A22 (entries: X[i,j] = A[i,j] + A[i+2,j+2] for i,j in 0,1)
    # Y = B11 + B22 (entries: Y[i,j] = B[i,j] + B[i+2,j+2] for i,j in 0,1)
    # Z = X*Y, and Z contributes to C11 (with +1) and C22 (with +1)
    
    # Apply Strassen to X*Y:
    # s1 = (X00+X11)*(Y00+Y11)
    # s2 = (X10+X11)*Y00
    # s3 = X00*(Y00-Y10)
    # s4 = X11*(Y11-Y01)
    # s5 = (X00+X01)*Y11
    # s6 = (X10-X00)*(Y00+Y01)
    # s7 = (X01-X11)*(Y10+Y11)
    # Z00 = s1+s4-s5+s7
    # Z01 = s3+s5
    # Z10 = s2+s4
    # Z11 = s1-s2+s3+s6
    
    # X00 = A00+A22, X01 = A01+A23, X10 = A10+A32, X11 = A11+A33
    # Y00 = B00+B22, Y01 = B01+B23, Y10 = B10+B32, Y11 = B11+B33
    
    def strassen_2x2(X_map, Y_map, C_offset, C_scale):
        """
        X_map: dict (r,c) -> list of (coeff, a_row, a_col)
        Y_map: dict (r,c) -> list of (coeff, b_row, b_col)
        C_offset: (r, c) offset for C
        C_scale: dict (r,c) -> scale factor
        """
        # s1 = (X00+X11)*(Y00+Y11)
        X_s1 = X_map[(0,0)] + X_map[(1,1)]
        Y_s1 = Y_map[(0,0)] + Y_map[(1,1)]
        for (ac, ar, acol) in X_s1:
            for (bc, br, bcol) in Y_s1:
                # Contributes to C00 with +1 and C11 with +1
                add_term((ar, acol), (br, bcol), (C_offset[0]+0, C_offset[1]+0), ac*bc*C_scale.get((0,0),1))
                add_term((ar, acol), (br, bcol), (C_offset[0]+1, C_offset[1]+1), ac*bc*C_scale.get((1,1),1))
        
        # s2 = (X10+X11)*Y00
        X_s2 = X_map[(1,0)] + X_map[(1,1)]
        Y_s2 = Y_map[(0,0)]
        for (ac, ar, acol) in X_s2:
            for (bc, br, bcol) in Y_s2:
                add_term((ar, acol), (br, bcol), (C_offset[0]+1, C_offset[1]+0), ac*bc*C_scale.get((1,0),1))
                add_term((ar, acol), (br, bcol), (C_offset[0]+1, C_offset[1]+1), -ac*bc*C_scale.get((1,1),1))
        
        # s3 = X00*(Y00-Y10)
        X_s3 = X_map[(0,0)]
        Y_s3 = Y_map[(0,0)] + [(-c, r, ccol) for (c, r, ccol) in Y_map[(1,0)]]
        for (ac, ar, acol) in X_s3:
            for (bc, br, bcol) in Y_s3:
                add_term((ar, acol), (br, bcol), (C_offset[0]+0, C_offset[1]+1), ac*bc*C_scale.get((0,1),1))
                add_term((ar, acol), (br, bcol), (C_offset[0]+1, C_offset[1]+1), ac*bc*C_scale.get((1,1),1))
        
        # s4 = X11*(Y11-Y01)
        X_s4 = X_map[(1,1)]
        Y_s4 = Y_map[(1,1)] + [(-c, r, ccol) for (c, r, ccol) in Y_map[(0,1)]]
        for (ac, ar, acol) in X_s4:
            for (bc, br, bcol) in Y_s4:
                add_term((ar, acol), (br, bcol), (C_offset[0]+0, C_offset[1]+0), ac*bc*C_scale.get((0,0),1))
                add_term((ar, acol), (br, bcol), (C_offset[0]+1, C_offset[1]+0), ac*bc*C_scale.get((1,0),1))
        
        # s5 = (X00+X01)*Y11
        X_s5 = X_map[(0,0)] + X_map[(0,1)]
        Y_s5 = Y_map[(1,1)]
        for (ac, ar, acol) in X_s5:
            for (bc, br, bcol) in Y_s5:
                add_term((ar, acol), (br, bcol), (C_offset[0]+0, C_offset[1]+0), -ac*bc*C_scale.get((0,0),1))
                add_term((ar, acol), (br, bcol), (C_offset[0]+0, C_offset[1]+1), ac*bc*C_scale.get((0,1),1))
        
        # s6 = (X10-X00)*(Y00+Y01)
        X_s6 = X_map[(1,0)] + [(-c, r, ccol) for (c, r, ccol) in X_map[(0,0)]]
        Y_s6 = Y_map[(0,0)] + Y_map[(0,1)]
        for (ac, ar, acol) in X_s6:
            for (bc, br, bcol) in Y_s6:
                add_term((ar, acol), (br, bcol), (C_offset[0]+1, C_offset[1]+1), ac*bc*C_scale.get((1,1),1))
        
        # s7 = (X01-X11)*(Y10+Y11)
        X_s7 = X_map[(0,1)] + [(-c, r, ccol) for (c, r, ccol) in X_map[(1,1)]]
        Y_s7 = Y_map[(1,0)] + Y_map[(1,1)]
        for (ac, ar, acol) in X_s7:
            for (bc, br, bcol) in Y_s7:
                add_term((ar, acol), (br, bcol), (C_offset[0]+0, C_offset[1]+0), ac*bc*C_scale.get((0,0),1))
    
    # Define block maps for 4x4 case
    # A11: A[0:2, 0:2], A12: A[0:2, 2:4], A21: A[2:4, 0:2], A22: A[2:4, 2:4]
    # B11: B[0:2, 0:2], B12: B[0:2, 2:4], B21: B[2:4, 0:2], B22: B[2:4, 2:4]
    
    def make_block_map(base_r, base_c):
        """Create a map for a 2x2 block at (base_r, base_c)"""
        return {
            (0,0): [(1, base_r+0, base_c+0)],
            (0,1): [(1, base_r+0, base_c+1)],
            (1,0): [(1, base_r+1, base_c+0)],
            (1,1): [(1, base_r+1, base_c+1)],
        }
    
    def add_block_map(m1, m2, sign1=1, sign2=1):
        """Add two block maps with signs"""
        result = {}
        for key in [(0,0),(0,1),(1,0),(1,1)]:
            result[key] = [(sign1*c, r, ccol) for (c, r, ccol) in m1[key]] + \
                          [(sign2*c, r, ccol) for (c, r, ccol) in m2[key]]
        return result
    
    A11 = make_block_map(0, 0)
    A12 = make_block_map(0, 2)
    A21 = make_block_map(2, 0)
    A22 = make_block_map(2, 2)
    
    B11 = make_block_map(0, 0)
    B12 = make_block_map(0, 2)
    B21 = make_block_map(2, 0)
    B22 = make_block_map(2, 2)
    
    # S1 = (A11+A22)*(B11+B22) -> C11 (+1), C22 (+1)
    X1 = add_block_map(A11, A22)
    Y1 = add_block_map(B11, B22)
    strassen_2x2(X1, Y1, (0,0), {(0,0):1, (0,1):1, (1,0):1, (1,1):1})
    # But wait, this adds to C at offset (0,0) with the full 2x2 result
    # S1 contributes +1 to C11 (which is C[0:2,0:2]) and +1 to C22 (which is C[2:4,2:4])
    # I need to handle the two different C offsets separately.
    
    # Let me redo this more carefully.
    # I'll clear terms and restart with a cleaner approach.
    terms = []
    col = 0
    
    # For each Strassen level-1 term, I'll specify:
    # - The X and Y block maps
    # - For each of the 4 C sub-positions, the scale factor
    
    def strassen_2x2_multi(X_map, Y_map, c_offsets_scales):
        """
        c_offsets_scales: dict mapping (dr, dc) -> scale
        where (dr, dc) is offset within the 2x2 result block
        and scale is the multiplier for C at that position
        """
        # s1 = (X00+X11)*(Y00+Y11)
        X_s1 = X_map[(0,0)] + X_map[(1,1)]
        Y_s1 = Y_map[(0,0)] + Y_map[(1,1)]
        for (ac, ar, acol) in X_s1:
            for (bc, br, bcol) in Y_s1:
                for (dr, dc), scale in c_offsets_scales.items():
                    if (dr, dc) == (0,0):
                        add_term((ar, acol), (br, bcol), (dr, dc), ac*bc*scale)
                    elif (dr, dc) == (1,1):
                        add_term((ar, acol), (br, bcol), (dr, dc), ac*bc*scale)
                    elif (dr, dc) == (0,1):
                        pass  # s1 doesn't contribute to (0,1)
                    elif (dr, dc) == (1,0):
                        pass
        
        # This is getting messy. Let me just hardcode all 49 terms.
        pass
    
    # Given the complexity, let me use a different strategy:
    # Just use Strassen for the 4 contiguous 2x2*2x2 blocks (without block-level Strassen)
    # That gives 4*7 = 28 terms for the diagonal, plus schoolbook for off-diagonal.
    # C11 = A11*B11 + A12*B21: 7 + 12 = 19
    # C12 = A11*B12 + A12*B22: 7 + 12 = 19
    # C21 = A21*B11 + A22*B21: 12 + 7 = 19
    # C22 = A21*B12 + A22*B22: 12 + 7 = 19
    # Total: 76 for 4x4*4x4. Plus 16 + 20 = 112. Worse than 100.
    
    # OK, let me just carefully implement the 85-rank decomposition.
    # I'll expand everything explicitly.
    
    terms = []
    
    # 4x4 * 4x4 part with double Strassen:
    # I'll enumerate all 49 terms by expanding the Strassen formulas.
    
    # Level 1 Strassen for 4x4:
    # S1 = (A11+A22)*(B11+B22)
    # S2 = (A21+A22)*B11
    # S3 = A11*(B11-B21)
    # S4 = A22*(B22-B12)
    # S5 = (A11+A12)*B22
    # S6 = (A21-A11)*(B11+B12)
    # S7 = (A12-A22)*(B21+B22)
    
    # C11 = S1+S4-S5+S7 (2x2 block at (0,0))
    # C12 = S3+S5 (2x2 block at (0,2))
    # C21 = S2+S4 (2x2 block at (2,0))
    # C22 = S1-S2+S3+S6 (2x2 block at (2,2))
    
    # For each S_i, decompose the 2x2*2x2 product into 7 terms using Strassen.
    
    # Helper: decompose a 2x2*2x2 product where X and Y are given as
    # 4x4 matrices of linear combinations.
    
    # Let me define X and Y for each S_i as 4-vectors (flattened 2x2) of
    # linear combinations of A or B entries.
    
    # For S1: X = A11+A22, Y = B11+B22
    # X = [A00+A22, A01+A23, A10+A32, A11+A33] (flattened: 00,01,10,11)
    # Y = [B00+B22, B01+B23, B10+B32, B11+B33]
    
    def decompose_2x2_strassen(X, Y, C_pos_list):
        """
        X, Y: lists of 4 elements, each element is a list of (coeff, mat_idx, i, j)
              mat_idx: 0 for A, 1 for B
              i, j: indices into the full A/B matrices
        C_pos_list: list of (c_row_offset, c_col_offset, scale) for each of the 4 C entries
                    Actually, let's say C is at specific positions in the 4x4 C matrix.
                    We'll return a list of terms.
        """
        result = []
        
        # Flatten X and Y: X[k] is a list of (coeff, mat, i, j)
        # X[0]=X00, X[1]=X01, X[2]=X10, X[3]=X11
        # Y[0]=Y00, Y[1]=Y01, Y[2]=Y10, Y[3]=Y11
        
        def prod_terms(X_entry, Y_entry):
            """Return list of (a_coeff, b_coeff, a_i, a_j, b_i, b_j)"""
            terms = []
            for (ac, am, ai, aj) in X_entry:
                for (bc, bm, bi, bj) in Y_entry:
                    if am == 0 and bm == 1:  # A * B
                        terms.append((ac, ai, aj, bc, bi, bj))
                    elif am == 1 and bm == 0:
                        terms.append((bc, bi, bj, ac, ai, aj))
                    else:
                        raise ValueError("Must be A*B")
            return terms
        
        # s1 = (X00+X11)*(Y00+Y11)
        X_s1 = X[0] + X[3]
        Y_s1 = Y[0] + Y[3]
        for term in prod_terms(X_s1, Y_s1):
            ac, ai, aj, bc, bi, bj = term
            # s1 contributes to C00 (+1) and C11 (+1)
            result.append((ai, aj, bi, bj, 0, 0, ac*bc))
            result.append((ai, aj, bi, bj, 1, 1, ac*bc))
        
        # s2 = (X10+X11)*Y00
        X_s2 = X[2] + X[3]
        Y_s2 = Y[0]
        for term in prod_terms(X_s2, Y_s2):
            ac, ai, aj, bc, bi, bj = term
            result.append((ai, aj, bi, bj, 1, 0, ac*bc))
            result.append((ai, aj, bi, bj, 1, 1, -ac*bc))
        
        # s3 = X00*(Y00-Y10)
        X_s3 = X[0]
        Y_s3 = Y[0] + [(-c, m, i, j) for (c, m, i, j) in Y[2]]
        for term in prod_terms(X_s3, Y_s3):
            ac, ai, aj, bc, bi, bj = term
            result.append((ai, aj, bi, bj, 0, 1, ac*bc))
            result.append((ai, aj, bi, bj, 1, 1, ac*bc))
        
        # s4 = X11*(Y11-Y01)
        X_s4 = X[3]
        Y_s4 = Y[3] + [(-c, m, i, j) for (c, m, i, j) in Y[1]]
        for term in prod_terms(X_s4, Y_s4):
            ac, ai, aj, bc, bi, bj = term
            result.append((ai, aj, bi, bj, 0, 0, ac*bc))
            result.append((ai, aj, bi, bj, 1, 0, ac*bc))
        
        # s5 = (X00+X01)*Y11
        X_s5 = X[0] + X[1]
        Y_s5 = Y[3]
        for term in prod_terms(X_s5, Y_s5):
            ac, ai, aj, bc, bi, bj = term
            result.append((ai, aj, bi, bj, 0, 0, -ac*bc))
            result.append((ai, aj, bi, bj, 0, 1, ac*bc))
        
        # s6 = (X10-X00)*(Y00+Y01)
        X_s6 = X[2] + [(-c, m, i, j) for (c, m, i, j) in X[0]]
        Y_s6 = Y[0] + Y[1]
        for term in prod_terms(X_s6, Y_s6):
            ac, ai, aj, bc, bi, bj = term
            result.append((ai, aj, bi, bj, 1, 1, ac*bc))
        
        # s7 = (X01-X11)*(Y10+Y11)
        X_s7 = X[1] + [(-c, m, i, j) for (c, m, i, j) in X[3]]
        Y_s7 = Y[2] + Y[3]
        for term in prod_terms(X_s7, Y_s7):
            ac, ai, aj, bc, bi, bj = term
            result.append((ai, aj, bi, bj, 0, 0, ac*bc))
        
        return result
    
    # Now define X, Y for each S_i
    # mat: 0=A, 1=B
    
    def A_entry(i, j, coeff=1):
        return [(coeff, 0, i, j)]
    
    def B_entry(i, j, coeff=1):
        return [(coeff, 1, i, j)]
    
    # S1: X = A11+A22, Y = B11+B22
    X_S1 = [
        A_entry(0,0) + A_entry(2,2),  # X00
        A_entry(0,1) + A_entry(2,3),  # X01
        A_entry(1,0) + A_entry(3,2),  # X10
        A_entry(1,1) + A_entry(3,3),  # X11
    ]
    Y_S1 = [
        B_entry(0,0) + B_entry(2,2),  # Y00
        B_entry(0,1) + B_entry(2,3),  # Y01
        B_entry(1,0) + B_entry(3,2),  # Y10
        B_entry(1,1) + B_entry(3,3),  # Y11
    ]
    
    # S1 contributes to C11 (offset 0,0) with +1 and C22 (offset 2,2) with +1
    # The 2x2 result from S1 is placed at C[0:2, 0:2] and C[2:4, 2:4]
    # So for the 2x2 result at position (r,c) in the local 2x2, 
    # it goes to C[0+r, 0+c] with scale +1 (for C11)
    # and C[2+r, 2+c] with scale +1 (for C22)
    
    # Let me handle this by specifying C offsets
    def add_2x2_result(result_terms, c_base_r, c_base_c, scale=1):
        """Add terms to the result, offsetting C by (c_base_r, c_base_c) and scaling"""
        return [(ai, aj, bi, bj, c_base_r+cr, c_base_c+cc, s*scale) 
                for (ai, aj, bi, bj, cr, cc, s) in result_terms]
    
    S1_terms = decompose_2x2_strassen(X_S1, Y_S1, None)
    # C11: offset (0,0), scale +1
    # C22: offset (2,2), scale +1
    terms.extend(add_2x2_result(S1_terms, 0, 0, 1))
    terms.extend(add_2x2_result(S1_terms, 2, 2, 1))
    
    # S2: X = A21+A22, Y = B11
    X_S2 = [
        A_entry(2,0) + A_entry(4,0),  # Wait, A is only 4x5, so row 4 doesn't exist for the 4x4 part
        # A21 is A[2:4, 0:2], A22 is A[2:4, 2:4]
        A_entry(2,0) + A_entry(2,2),  # X00
        A_entry(2,1) + A_entry(2,3),  # X01
        A_entry(3,0) + A_entry(3,2),  # X10
        A_entry(3,1) + A_entry(3,3),  # X11
    ]
    Y_S2 = [
        B_entry(0,0),  # Y00
        B_entry(0,1),  # Y01
        B_entry(1,0),  # Y10
        B_entry(1,1),  # Y11
    ]
    
    S2_terms = decompose_2x2_strassen(X_S2, Y_S2, None)
    # S2 contributes to C21 (offset 2,0) with +1 and C22 (offset 2,2) with -1
    terms.extend(add_2x2_result(S2_terms, 2, 0, 1))
    terms.extend(add_2x2_result(S2_terms, 2, 2, -1))
    
    # S3: X = A11, Y = B11-B21
    X_S3 = [
        A_entry(0,0),  # X00
        A_entry(0,1),  # X01
        A_entry(1,0),  # X10
        A_entry(1,1),  # X11
    ]
    Y_S3 = [
        B_entry(0,0) + [(-1, 1, 2, 0)],  # Y00 = B00 - B20
        B_entry(0,1) + [(-1, 1, 2, 1)],  # Y01 = B01 - B21
        B_entry(1,0),  # Y10 = B10 (B21 starts at row 2, so Y10 = B10 - B30? No.)
        # B11 = B[0:2,0:2], B21 = B[2:4,0:2]
        # B11-B21: Y00 = B00-B20, Y01 = B01-B21, Y10 = B10-B30, Y11 = B11-B31
        B_entry(1,0) + [(-1, 1, 3, 0)],  # Y10
        B_entry(1,1) + [(-1, 1, 3, 1)],  # Y11
    ]
    
    S3_terms = decompose_2x2_strassen(X_S3, Y_S3, None)
    # S3 contributes to C12 (offset 0,2) with +1 and C22 (offset 2,2) with +1
    terms.extend(add_2x2_result(S3_terms, 0, 2, 1))
    terms.extend(add_2x2_result(S3_terms, 2, 2, 1))
    
    # S4: X = A22, Y = B22-B12
    X_S4 = [
        A_entry(2,2),  # X00
        A_entry(2,3),  # X01
        A_entry(3,2),  # X10
        A_entry(3,3),  # X11
    ]
    Y_S4 = [
        B_entry(2,2) + [(-1, 1, 0, 2)],  # Y00 = B22 - B02
        B_entry(2,3) + [(-1, 1, 0, 3)],  # Y01 = B23 - B03
        B_entry(3,2) + [(-1, 1, 1, 2)],  # Y10 = B32 - B12
        B_entry(3,3) + [(-1, 1, 1, 3)],  # Y11 = B33 - B13
    ]
    
    S4_terms = decompose_2x2_strassen(X_S4, Y_S4, None)
    # S4 contributes to C11 (offset 0,0) with +1 and C21 (offset 2,0) with +1
    terms.extend(add_2x2_result(S4_terms, 0, 0, 1))
    terms.extend(add_2x2_result(S4_terms, 2, 0, 1))
    
    # S5: X = A11+A12, Y = B22
    X_S5 = [
        A_entry(0,0) + A_entry(0,2),  # X00
        A_entry(0,1) + A_entry(0,3),  # X01
        A_entry(1,0) + A_entry(1,2),  # X10
        A_entry(1,1) + A_entry(1,3),  # X11
    ]
    Y_S5 = [
        B_entry(2,2),  # Y00
        B_entry(2,3),  # Y01
        B_entry(3,2),  # Y10
        B_entry(3,3),  # Y11
    ]
    
    S5_terms = decompose_2x2_strassen(X_S5, Y_S5, None)
    # S5 contributes to C11 (offset 0,0) with -1 and C12 (offset 0,2) with +1
    terms.extend(add_2x2_result(S5_terms, 0, 0, -1))
    terms.extend(add_2x2_result(S5_terms, 0, 2, 1))
    
    # S6: X = A21-A11, Y = B11+B12
    X_S6 = [
        A_entry(2,0) + [(-1, 0, 0, 0)],  # X00 = A20 - A00
        A_entry(2,1) + [(-1, 0, 0, 1)],  # X01 = A21 - A01
        A_entry(3,0) + [(-1, 0, 1, 0)],  # X10 = A30 - A10
        A_entry(3,1) + [(-1, 0, 1, 1)],  # X11 = A31 - A11
    ]
    Y_S6 = [
        B_entry(0,0) + B_entry(0,2),  # Y00 = B00 + B02
        B_entry(0,1) + B_entry(0,3),  # Y01 = B01 + B03
        B_entry(1,0) + B_entry(1,2),  # Y10 = B10 + B12
        B_entry(1,1) + B_entry(1,3),  # Y11 = B11 + B13
    ]
    
    S6_terms = decompose_2x2_strassen(X_S6, Y_S6, None)
    # S6 contributes to C22 (offset 2,2) with +1
    terms.extend(add_2x2_result(S6_terms, 2, 2, 1))
    
    # S7: X = A12-A22, Y = B21+B22
    X_S7 = [
        A_entry(0,2) + [(-1, 0, 2, 2)],  # X00 = A02 - A22
        A_entry(0,3) + [(-1, 0, 2, 3)],  # X01 = A03 - A23
        A_entry(1,2) + [(-1, 0, 3, 2)],  # X10 = A12 - A32
        A_entry(1,3) + [(-1, 0, 3, 3)],  # X11 = A13 - A33
    ]
    Y_S7 = [
        B_entry(2,0) + B_entry(2,2),  # Y00 = B20 + B22
        B_entry(2,1) + B_entry(2,3),  # Y01 = B21 + B23
        B_entry(3,0) + B_entry(3,2),  # Y10 = B30 + B32
        B_entry(3,1) + B_entry(3,3),  # Y11 = B31 + B33
    ]
    
    S7_terms = decompose_2x2_strassen(X_S7, Y_S7, None)
    # S7 contributes to C11 (offset 0,0) with +1
    terms.extend(add_2x2_result(S7_terms, 0, 0, 1))
    
    # Now add the remaining parts:
    # Part 2: 4x4 * 4x1 (A rows 0-3, cols 0-3 times B rows 0-3, cols 4)
    # Schoolbook: 4*4*1 = 16 terms
    for i in range(4):
        for j in range(4):
            terms.append((i, j, j, 4, i, 4, 1))
    
    # Part 3: 4x1 * 1x5 (A rows 0-3, col 4 times B row 4, cols 0-4)
    # Schoolbook: 4*1*5 = 20 terms
    for i in range(4):
        for k in range(5):
            terms.append((i, 4, 4, k, i, k, 1))
    
    # Now build U, V, W from terms
    # Merge terms with same (a_idx, b_idx, c_idx)
    from collections import defaultdict
    term_dict = defaultdict(int)
    for (ai, aj, bi, bj, ci, cj, coeff) in terms:
        key = (ai*m + aj, bi*p + bj, cj*n + ci)
        term_dict[key] += coeff
    
    # Remove zero terms
    final_terms = [(k, v) for k, v in term_dict.items() if v != 0]
    
    rank_final = len(final_terms)
    U = [[0] * rank_final for _ in range(n * m)]
    V = [[0] * rank_final for _ in range(m * p)]
    W = [[0] * rank_final for _ in range(p * n)]
    
    for idx, ((a_idx, b_idx, c_idx), coeff) in enumerate(final_terms):
        U[a_idx][idx] = coeff
        V[b_idx][idx] = coeff
        W[c_idx][idx] = coeff
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    import numpy as np
    result = construct()
    U = np.array(result["U"], dtype=float)
    V = np.array(result["V"], dtype=float)
    W = np.array(result["W"], dtype=float)
    
    # Verify: sum_t U[i,j,t] * V[j',k,t] * W[k',i',t] should equal delta_{j,j'} * delta_{k,k'}
    # i.e., T[i,j,k] = sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = delta_{j,k} for the (j,k) part
    # Actually: C[i,k] = sum_j A[i,j] B[j,k]
    # T_{i,j,k} such that sum_{j,k} T_{i,j,k} A_{i,j} B_{j,k} = C_{i,k}
    # So T_{i,j,k} = delta_{j,?} ... 
    # The tensor is: T[i,j,k] = 1 if the multiplication A[i,j]*B[j,k] contributes to C[i,k]
    # i.e., T[i,j,k] = delta_{j,k} is not right.
    # T[i,j,k] such that C[i,k] = sum_{j} T[i,j,k] A[i,j] B[j,k]
    # This requires T[i,j,k] = 1 for all i,k and j=j' where j' is the middle index.
    # Actually T[i,j,k] = 1 for all i,k and any j, but then C[i,k] = sum_j A[i,j]B[j,k].
    # So T[i,j,k] = 1 for all i in 0..3, j in 0..4, k in 0..4.
    # Wait no: T[i,j,k] * A[i,j] * B[j,k] summed over j gives C[i,k].
    # So T[i,j,k] must be 1 for each (i,j,k) where the term A[i,j]*B[j,k] appears.
    # It appears for each fixed (i,k) and varying j.
    # So T[i,j,k] = 1 for all i,j,k? No, that would give sum_j sum_k A[i,j]B[j,k] which is wrong.
    # 
    # The tensor T is defined by: C[i,k] = sum_j T[i,j,k] A[i,j] B[j,k]
    # For this to equal sum_j A[i,j] B[j,k], we need T[i,j,k] = 1 for all i,j,k.
    # But then the rank-1 decomposition: sum_t U[i,j,t] V[j,k,t] W[k,i,t]
    # should equal T[i,j,k] = 1 for all i,j,k.
    # 
    # Wait, the standard form is:
    # C[i,k] = sum_t W[k*n+i, t] * sum_j (U[i*m+j, t] * A[i,j]) * (V[j*p+k, t] * B[j,k])
    #         = sum_t W[k*n+i, t] * sum_j U[i*m+j, t] * V[j*p+k, t] * A[i,j] * B[j,k]
    # 
    # So the effective tensor is T[i,j,k] = sum_t W[k*n+i, t] * U[i*m+j, t] * V[j*p+k, t]
    # and we need T[i,j,k] = 1 for all i,j,k (so that C[i,k] = sum_j A[i,j]B[j,k]).
    
    # Verify
    T = np.zeros((n, m, p))
    for t in range(rank_final):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i,j,k] += W[k*n+i, t] * U[i*m+j, t] * V[j*p+k, t]
    
    expected = np.ones((n, m, p))
    diff = np.abs(T - expected)
    print(f"Rank: {rank_final}")
    print(f"Max diff: {diff.max()}")
    print(f"Mean diff: {diff.mean()}")
