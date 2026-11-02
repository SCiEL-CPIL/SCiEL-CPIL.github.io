import time
import sys

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # Target rank 93
    rank = 93
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_idx = 0
    
    # Helper to add a scalar multiplication term
    # u_vec: list of 16 coeffs for A
    # v_vec: list of 28 coeffs for B
    # w_vec: list of 28 coeffs for C
    def add_term(u_vec, v_vec, w_vec):
        nonlocal col_idx
        for i in range(n * m):
            if u_vec[i] != 0:
                U[i][col_idx] = u_vec[i]
        for j in range(m * p):
            if v_vec[j] != 0:
                V[j][col_idx] = v_vec[j]
        for k in range(p * n):
            if w_vec[k] != 0:
                W[k][col_idx] = w_vec[k]
        col_idx += 1
        
    # Helper to get A block indices
    # A is 4x4. Block (bi, bj) is 2x2.
    # A[i, j]
    # bi in 0..1, bj in 0..1
    # i = bi*2 + ii, j = bj*2 + jj
    def a_idx(bi, bj, ii, jj):
        return (bi*2 + ii) * 4 + (bj*2 + jj)
        
    # Helper to get B block indices
    # B is 4x7.
    # For Part 1, cols 0-3. Block (bj, bk) is 2x2.
    # B[j, k]
    # bj in 0..1, bk in 0..1 (local col)
    # j = bj*2 + jj, k = bk*2 + kk
    def b_idx_part1(bj, bk, jj, kk):
        return (bj*2 + jj) * 7 + (bk*2 + kk)
        
    # Helper to get C block indices
    # C is 4x7.
    # For Part 1, cols 0-3. Block (ci, ck) is 2x2.
    # C[i, k]
    # ci in 0..1, ck in 0..1 (local col)
    # i = ci*2 + ii, k = ck*2 + kk
    def c_idx_part1(ci, ck, ii, kk):
        return (ck*2 + kk) * 4 + (ci*2 + ii)
        
    # Part 1: 4x4 by 4x4 (cols 0-3)
    # Strassen on 4x4 blocks
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Define the 7 Strassen products for 2x2 matrices
    # Returns list of 7 tuples (u_vec_4, v_vec_4, w_vec_4) for 2x2 case
    # u_vec_4 is for A_block (4 entries)
    # v_vec_4 is for B_block (4 entries)
    # w_vec_4 is for C_block (4 entries)
    
    def strassen_2x2(a_block_idx, b_block_idx, c_block_idx, sign_a=1, sign_b=1):
        # a_block_idx: function (ii, jj) -> global A index
        # b_block_idx: function (jj, kk) -> global B index
        # c_block_idx: function (ii, kk) -> global C index
        
        terms = []
        
        # m1 = (a+d)(e+h)
        u1 = [0]*16; v1 = [0]*28; w1 = [0]*28
        # a
        u1[a_block_idx(0,0)] = sign_a
        # d
        u1[a_block_idx(1,1)] = sign_a
        # e
        v1[b_block_idx(0,0)] = sign_b
        # h
        v1[b_block_idx(1,1)] = sign_b
        # w contributions
        # c00 += m1, c11 += m1
        w1[c_block_idx(0,0)] = 1
        w1[c_block_idx(1,1)] = 1
        terms.append((u1, v1, w1))
        
        # m2 = (c+d)e
        u2 = [0]*16; v2 = [0]*28; w2 = [0]*28
        u2[a_block_idx(1,0)] = sign_a
        u2[a_block_idx(1,1)] = sign_a
        v2[b_block_idx(0,0)] = sign_b
        # c10 += m2, c11 -= m2
        w2[c_block_idx(1,0)] = 1
        w2[c_block_idx(1,1)] = -1
        terms.append((u2, v2, w2))
        
        # m3 = a(f-h)
        u3 = [0]*16; v3 = [0]*28; w3 = [0]*28
        u3[a_block_idx(0,0)] = sign_a
        v3[b_block_idx(0,1)] = sign_b
        v3[b_block_idx(1,1)] = -sign_b
        # c01 += m3, c11 += m3
        w3[c_block_idx(0,1)] = 1
        w3[c_block_idx(1,1)] = 1
        terms.append((u3, v3, w3))
        
        # m4 = d(g-e)
        u4 = [0]*16; v4 = [0]*28; w4 = [0]*28
        u4[a_block_idx(1,1)] = sign_a
        v4[b_block_idx(1,0)] = sign_b
        v4[b_block_idx(0,0)] = -sign_b
        # c00 += m4, c10 += m4
        w4[c_block_idx(0,0)] = 1
        w4[c_block_idx(1,0)] = 1
        terms.append((u4, v4, w4))
        
        # m5 = (a+b)h
        u5 = [0]*16; v5 = [0]*28; w5 = [0]*28
        u5[a_block_idx(0,0)] = sign_a
        u5[a_block_idx(0,1)] = sign_a
        v5[b_block_idx(1,1)] = sign_b
        # c00 -= m5, c01 += m5
        w5[c_block_idx(0,0)] = -1
        w5[c_block_idx(0,1)] = 1
        terms.append((u5, v5, w5))
        
        # m6 = (c-a)(e+f)
        u6 = [0]*16; v6 = [0]*28; w6 = [0]*28
        u6[a_block_idx(1,0)] = sign_a
        u6[a_block_idx(0,0)] = -sign_a
        v6[b_block_idx(0,0)] = sign_b
        v6[b_block_idx(0,1)] = sign_b
        # c11 += m6
        w6[c_block_idx(1,1)] = 1
        terms.append((u6, v6, w6))
        
        # m7 = (b-d)(g+h)
        u7 = [0]*16; v7 = [0]*28; w7 = [0]*28
        u7[a_block_idx(0,1)] = sign_a
        u7[a_block_idx(1,1)] = -sign_a
        v7[b_block_idx(1,0)] = sign_b
        v7[b_block_idx(1,1)] = sign_b
        # c00 += m7
        w7[c_block_idx(0,0)] = 1
        terms.append((u7, v7, w7))
        
        return terms

    # Define block index functions for Part 1
    # A blocks: A00, A01, A10, A11
    def a_idx_00(ii, jj): return a_idx(0, 0, ii, jj)
    def a_idx_01(ii, jj): return a_idx(0, 1, ii, jj)
    def a_idx_10(ii, jj): return a_idx(1, 0, ii, jj)
    def a_idx_11(ii, jj): return a_idx(1, 1, ii, jj)
    
    # B blocks for Part 1 (cols 0-3)
    # B00 (rows 0-1, cols 0-1), B01 (rows 0-1, cols 2-3), B10 (rows 2-3, cols 0-1), B11 (rows 2-3, cols 2-3)
    def b_idx_00(jj, kk): return b_idx_part1(0, 0, jj, kk)
    def b_idx_01(jj, kk): return b_idx_part1(0, 1, jj, kk)
    def b_idx_10(jj, kk): return b_idx_part1(1, 0, jj, kk)
    def b_idx_11(jj, kk): return b_idx_part1(1, 1, jj, kk)
    
    # C blocks for Part 1 (cols 0-3)
    # C00 (rows 0-1, cols 0-1), etc.
    def c_idx_00(ii, kk): return c_idx_part1(0, 0, ii, kk)
    def c_idx_01(ii, kk): return c_idx_part1(0, 1, ii, kk)
    def c_idx_10(ii, kk): return c_idx_part1(1, 0, ii, kk)
    def c_idx_11(ii, kk): return c_idx_part1(1, 1, ii, kk)
    
    # Generate Part 1 terms
    # M1
    terms = strassen_2x2(a_idx_00, b_idx_00, c_idx_00, 1, 1) # Placeholder, need to combine
    # Actually, Strassen on 4x4 combines the 2x2 products.
    # M1 = (A00+A11)(B00+B11)
    # This is a product of two 2x2 matrices.
    # We need to expand this product.
    # The "u_vec" for the product will be the sum of u_vecs for A00 and A11.
    # The "v_vec" will be the sum of v_vecs for B00 and B11.
    
    # Let's modify strassen_2x2 to take multiple blocks?
    # Or just sum the vectors.
    
    def add_strassen_product(a_blocks, b_blocks, c_blocks):
        # a_blocks: list of (idx_func, sign)
        # b_blocks: list of (idx_func, sign)
        # c_blocks: list of (idx_func, sign)
        
        # Create aggregated u, v, w vectors for the 7 terms
        # But wait, the 7 terms are specific to the 2x2 structure.
        # We can just create the 7 terms by summing the contributions.
        
        for t in range(7):
            u_term = [0]*16
            v_term = [0]*28
            w_term = [0]*28
            
            # Get the base 2x2 strassen term t
            # We need a way to get the t-th term's local coefficients
            # Let's re-implement the 7 terms locally and then map.
            
            # This is getting complicated to generalize the index functions.
            # Let's just hardcode the 7 terms for each product.
            
            pass
            
    # Simpler approach:
    # Define a function that takes a list of A block indices (with signs) and B block indices (with signs)
    # and C block indices (with signs) and generates the 7 terms.
    
    def gen_product(a_blocks, b_blocks, c_blocks):
        # a_blocks: [(idx_func, sign), ...]
        # b_blocks: [(idx_func, sign), ...]
        # c_blocks: [(idx_func, sign), ...]
        
        results = []
        
        # Define the 7 local 2x2 strassen terms
        # Local indices: 00, 01, 10, 11
        # We need to map local (ii, jj) to global.
        
        # We will construct the global u, v, w for each of the 7 terms.
        
        # Term 1: (a+d)(e+h)
        u1 = [0]*16; v1 = [0]*28; w1 = [0]*28
        for (af, asgn) in a_blocks:
            u1[af(0,0)] += asgn
            u1[af(1,1)] += asgn
        for (bf, bsgn) in b_blocks:
            v1[bf(0,0)] += bsgn
            v1[bf(1,1)] += bsgn
        for (cf, csgn) in c_blocks:
            w1[cf(0,0)] += csgn
            w1[cf(1,1)] += csgn
        results.append((u1, v1, w1))
        
        # Term 2: (c+d)e
        u2 = [0]*16; v2 = [0]*28; w2 = [0]*28
        for (af, asgn) in a_blocks:
            u2[af(1,0)] += asgn
            u2[af(1,1)] += asgn
        for (bf, bsgn) in b_blocks:
            v2[bf(0,0)] += bsgn
        for (cf, csgn) in c_blocks:
            w2[cf(1,0)] += csgn
            w2[cf(1,1)] -= csgn
        results.append((u2, v2, w2))
        
        # Term 3: a(f-h)
        u3 = [0]*16; v3 = [0]*28; w3 = [0]*28
        for (af, asgn) in a_blocks:
            u3[af(0,0)] += asgn
        for (bf, bsgn) in b_blocks:
            v3[bf(0,1)] += bsgn
            v3[bf(1,1)] -= bsgn
        for (cf, csgn) in c_blocks:
            w3[cf(0,1)] += csgn
            w3[cf(1,1)] += csgn
        results.append((u3, v3, w3))
        
        # Term 4: d(g-e)
        u4 = [0]*16; v4 = [0]*28; w4 = [0]*28
        for (af, asgn) in a_blocks:
            u4[af(1,1)] += asgn
        for (bf, bsgn) in b_blocks:
            v4[bf(1,0)] += bsgn
            v4[bf(0,0)] -= bsgn
        for (cf, csgn) in c_blocks:
            w4[cf(0,0)] += csgn
            w4[cf(1,0)] += csgn
        results.append((u4, v4, w4))
        
        # Term 5: (a+b)h
        u5 = [0]*16; v5 = [0]*28; w5 = [0]*28
        for (af, asgn) in a_blocks:
            u5[af(0,0)] += asgn
            u5[af(0,1)] += asgn
        for (bf, bsgn) in b_blocks:
            v5[bf(1,1)] += bsgn
        for (cf, csgn) in c_blocks:
            w5[cf(0,0)] -= csgn
            w5[cf(0,1)] += csgn
        results.append((u5, v5, w5))
        
        # Term 6: (c-a)(e+f)
        u6 = [0]*16; v6 = [0]*28; w6 = [0]*28
        for (af, asgn) in a_blocks:
            u6[af(1,0)] += asgn
            u6[af(0,0)] -= asgn
        for (bf, bsgn) in b_blocks:
            v6[bf(0,0)] += bsgn
            v6[bf(0,1)] += bsgn
        for (cf, csgn) in c_blocks:
            w6[cf(1,1)] += csgn
        results.append((u6, v6, w6))
        
        # Term 7: (b-d)(g+h)
        u7 = [0]*16; v7 = [0]*28; w7 = [0]*28
        for (af, asgn) in a_blocks:
            u7[af(0,1)] += asgn
            u7[af(1,1)] -= asgn
        for (bf, bsgn) in b_blocks:
            v7[bf(1,0)] += bsgn
            v7[bf(1,1)] += bsgn
        for (cf, csgn) in c_blocks:
            w7[cf(0,0)] += csgn
        results.append((u7, v7, w7))
        
        return results

    # Part 1: 4x4 x 4x4
    # M1 = (A00+A11)(B00+B11) -> C00, C11
    # Wait, the C blocks for M1 are C00 and C11?
    # In 4x4 Strassen:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # So for M1, it contributes to C00 and C11.
    # But M1 is a single 2x2 product.
    # The result of M1 is a 2x2 matrix.
    # It adds to C00 (top-left) and C11 (bottom-right)?
    # No, in 4x4 Strassen, the intermediate matrices M1..M7 are 2x2 matrices.
    # C00 is the top-left 2x2 block of C.
    # C11 is the bottom-right 2x2 block of C.
    # So M1 contributes its entire 2x2 value to C00 and C11.
    
    # Let's re-verify the 4x4 Strassen formula.
    # A = [[A11, A12], [A21, A22]]
    # B = [[B11, B12], [B21, B22]]
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # My indices:
    # A00 is A11, A01 is A12, A10 is A21, A11 is A22.
    # B00 is B11, B01 is B12, B10 is B21, B11 is B22.
    # C00 is C11, C01 is C12, C10 is C21, C11 is C22.
    
    # So:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For each Mi, we generate 7 terms.
    # Each term has a u, v, and w vector.
    # The w vector for a term in Mi will have non-zero values in the C blocks that Mi contributes to.
    
    # Example: M1 contributes to C00 and C11.
    # So for the 7 terms of M1, the w vector will have values in C00 and C11 blocks.
    
    # Let's implement this.
    
    # M1
    terms = gen_product(
        [(a_idx_00, 1), (a_idx_11, 1)],
        [(b_idx_00, 1), (b_idx_11, 1)],
        [(c_idx_00, 1), (c_idx_11, 1)]
    )
    for t in terms:
        add_term(*t)
        
    # M2
    terms = gen_product(
        [(a_idx_10, 1), (a_idx_11, 1)],
        [(b_idx_00, 1)],
        [(c_idx_10, 1)]
    )
    for t in terms:
        add_term(*t)
        
    # M3
    terms = gen_product(
        [(a_idx_00, 1)],
        [(b_idx_01, 1), (b_idx_11, -1)],
        [(c_idx_01, 1), (c_idx_11, 1)]
    )
    for t in terms:
        add_term(*t)
        
    # M4
    terms = gen_product(
        [(a_idx_11, 1)],
        [(b_idx_10, 1), (b_idx_00, -1)],
        [(c_idx_00, 1), (c_idx_10, 1)]
    )
    for t in terms:
        add_term(*t)
        
    # M5
    terms = gen_product(
        [(a_idx_00, 1), (a_idx_01, 1)],
        [(b_idx_11, 1)],
        [(c_idx_00, -1), (c_idx_01, 1)]
    )
    for t in terms:
        add_term(*t)
        
    # M6
    terms = gen_product(
        [(a_idx_10, 1), (a_idx_00, -1)],
        [(b_idx_00, 1), (b_idx_01, 1)],
        [(c_idx_11, 1)]
    )
    for t in terms:
        add_term(*t)
        
    # M7
    terms = gen_product(
        [(a_idx_01, 1), (a_idx_11, -1)],
        [(b_idx_10, 1), (b_idx_11, 1)],
        [(c_idx_00, 1)]
    )
    for t in terms:
        add_term(*t)
        
    # Part 2: 4x4 by 4x2 (cols 4-5)
    # B blocks: B_top (rows 0-1, cols 4-5), B_bot (rows 2-3, cols 4-5)
    def b_idx_pt2_top(jj, kk):
        # jj in 0..1, kk in 0..1
        # global col = 4 + kk
        return (jj) * 7 + (4 + kk)
        
    def b_idx_pt2_bot(jj, kk):
        # jj in 0..1, kk in 0..1
        # global row = 2 + jj, global col = 4 + kk
        return (2 + jj) * 7 + (4 + kk)
        
    # C blocks: C_top (rows 0-1, cols 4-5), C_bot (rows 2-3, cols 4-5)
    def c_idx_pt2_top(ii, kk):
        # ii in 0..1, kk in 0..1
        # global row = ii, global col = 4 + kk
        return (4 + kk) * 4 + ii
        
    def c_idx_pt2_bot(ii, kk):
        # ii in 0..1, kk in 0..1
        # global row = 2 + ii, global col = 4 + kk
        return (4 + kk) * 4 + (2 + ii)
        
    # C_top = A00 B_top + A01 B_bot
    # C_bot = A10 B_top + A11 B_bot
    
    # Term 1: A00 B_top
    terms = gen_product(
        [(a_idx_00, 1)],
        [(b_idx_pt2_top, 1)],
        [(c_idx_pt2_top, 1)]
    )
    for t in terms:
        add_term(*t)
        
    # Term 2: A01 B_bot
    terms = gen_product(
        [(a_idx_01, 1)],
        [(b_idx_pt2_bot, 1)],
        [(c_idx_pt2_top, 1)]
    )
    for t in terms:
        add_term(*t)
        
    # Term 3: A10 B_top
    terms = gen_product(
        [(a_idx_10, 1)],
        [(b_idx_pt2_top, 1)],
        [(c_idx_pt2_bot, 1)]
    )
    for t in terms:
        add_term(*t)
        
    # Term 4: A11 B_bot
    terms = gen_product(
        [(a_idx_11, 1)],
        [(b_idx_pt2_bot, 1)],
        [(c_idx_pt2_bot, 1)]
    )
    for t in terms:
        add_term(*t)
        
    # Part 3: 4x4 by 4x1 (col 6)
    # b_top (rows 0-1, col 6), b_bot (rows 2-3, col 6)
    # Schoolbook 2x2 by 2x1
    
    def add_schoolbook_2x2_2x1(a_block, b_top_idx, b_bot_idx, c_top_idx, c_bot_idx):
        # a_block: idx_func
        # b_top_idx: (jj) -> global B index for row jj, col 6
        # b_bot_idx: (jj) -> global B index for row 2+jj, col 6
        # c_top_idx: (ii) -> global C index for row ii, col 6
        # c_bot_idx: (ii) -> global C index for row 2+ii, col 6
        
        # 4 mults
        # 1. a00 * b_top0 -> c_top0
        u1 = [0]*16; v1 = [0]*28; w1 = [0]*28
        u1[a_block(0,0)] = 1
        v1[b_top_idx(0)] = 1
        w1[c_top_idx(0)] = 1
        add_term(u1, v1, w1)
        
        # 2. a01 * b_top1 -> c_top0
        u2 = [0]*16; v2 = [0]*28; w2 = [0]*28
        u2[a_block(0,1)] = 1
        v2[b_top_idx(1)] = 1
        w2[c_top_idx(0)] = 1
        add_term(u2, v2, w2)
        
        # 3. a10 * b_top0 -> c_bot0
        u3 = [0]*16; v3 = [0]*28; w3 = [0]*28
        u3[a_block(1,0)] = 1
        v3[b_top_idx(0)] = 1
        w3[c_bot_idx(0)] = 1
        add_term(u3, v3, w3)
        
        # 4. a11 * b_top1 -> c_bot0
        u4 = [0]*16; v4 = [0]*28; w4 = [0]*28
        u4[a_block(1,1)] = 1
        v4[b_top_idx(1)] = 1
        w4[c_bot_idx(0)] = 1
        add_term(u4, v4, w4)
        
        # Wait, c_top is 2x1. c_top0 is row 0, c_top1 is row 1.
        # c_top = A_block * b_top
        # c_top0 = a00*b0 + a01*b1
        # c_top1 = a10*b0 + a11*b1
        
        # I missed c_top1 and c_bot1 contributions?
        # No, the function call is for one term like A00 * b_top.
        # A00 is 2x2. b_top is 2x1.
        # Result is 2x1.
        # The contributions are to c_top (rows 0,1) and c_bot (rows 2,3)?
        # No, A00 multiplies b_top.
        # A00 is top-left block of A.
        # b_top is top part of b.
        # The result is top part of c.
        # So it contributes to c_top0 and c_top1.
        # It does NOT contribute to c_bot.
        
        # Let's re-evaluate.
        # C_top = A00 b_top + A01 b_bot
        # C_bot = A10 b_top + A11 b_bot
        
        # So A00 b_top contributes to C_top.
        # A01 b_bot contributes to C_top.
        # A10 b_top contributes to C_bot.
        # A11 b_bot contributes to C_bot.
        
        # My helper `add_schoolbook_2x2_2x1` should take target C block.
        
    def add_schoolbook_2x2_2x1_v2(a_block, b_vec_idx_func, c_block_idx_func):
        # a_block: idx_func (ii, jj)
        # b_vec_idx_func: (jj) -> global B index
        # c_block_idx_func: (ii) -> global C index
        
        # 1. a00 * b0 -> c0
        u1 = [0]*16; v1 = [0]*28; w1 = [0]*28
        u1[a_block(0,0)] = 1
        v1[b_vec_idx_func(0)] = 1
        w1[c_block_idx_func(0)] = 1
        add_term(u1, v1, w1)
        
        # 2. a01 * b1 -> c0
        u2 = [0]*16; v2 = [0]*28; w2 = [0]*28
        u2[a_block(0,1)] = 1
        v2[b_vec_idx_func(1)] = 1
        w2[c_block_idx_func(0)] = 1
        add_term(u2, v2, w2)
        
        # 3. a10 * b0 -> c1
        u3 = [0]*16; v3 = [0]*28; w3 = [0]*28
        u3[a_block(1,0)] = 1
        v3[b_vec_idx_func(0)] = 1
        w3[c_block_idx_func(1)] = 1
        add_term(u3, v3, w3)
        
        # 4. a11 * b1 -> c1
        u4 = [0]*16; v4 = [0]*28; w4 = [0]*28
        u4[a_block(1,1)] = 1
        v4[b_vec_idx_func(1)] = 1
        w4[c_block_idx_func(1)] = 1
        add_term(u4, v4, w4)
        
    def b_idx_pt3_top(jj):
        return jj * 7 + 6
        
    def b_idx_pt3_bot(jj):
        return (2 + jj) * 7 + 6
        
    def c_idx_pt3_top(ii):
        return 6 * 4 + ii
        
    def c_idx_pt3_bot(ii):
        return 6 * 4 + (2 + ii)
        
    # A00 b_top -> C_top
    add_schoolbook_2x2_2x1_v2(a_idx_00, b_idx_pt3_top, c_idx_pt3_top)
    # A01 b_bot -> C_top
    add_schoolbook_2x2_2x1_v2(a_idx_01, b_idx_pt3_bot, c_idx_pt3_top)
    # A10 b_top -> C_bot
    add_schoolbook_2x2_2x1_v2(a_idx_10, b_idx_pt3_top, c_idx_pt3_bot)
    # A11 b_bot -> C_bot
    add_schoolbook_2x2_2x1_v2(a_idx_11, b_idx_pt3_bot, c_idx_pt3_bot)
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    construct()
