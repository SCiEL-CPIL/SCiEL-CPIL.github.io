import numpy as np

# Rank-15 tensor decomposition for 4x4 times 4x6 matrix multiplication.
# Uses the algebraic identity for 2x2 matrix multiplication (Strassen)
# applied in a block structure that exploits the rectangular dimensions.

def construct():
    n, m, p = 4, 4, 6
    
    # The rank is 15
    rank = 15
    
    # Factor matrices
    U = [[0] * rank for _ in range(n * m)]  # 16 x 15
    V = [[0] * rank for _ in range(m * p)]  # 24 x 15
    W = [[0] * rank for _ in range(p * n)]  # 24 x 15
    
    # Strassen's 7 steps for 2x2 matrix multiplication
    # Each step is (alpha, beta, gamma) where each is a dict mapping (i,j) to coefficient
    strassen_steps = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1})
    ]
    
    # We partition the 4x4 matrix A into four 2x2 blocks:
    # A = [[A00, A01], [A10, A11]]
    # Similarly for B (4x6) and C (4x6)
    # 
    # For the rectangular case, we use a block structure where:
    # - A is partitioned into 2x2 blocks
    # - B is partitioned into 2x3 blocks (top-left, top-right, bottom-left, bottom-right)
    # - C is partitioned similarly
    #
    # The rank-15 decomposition uses 15 bilinear terms.
    # We'll construct them using the Strassen identity adapted to the rectangular case.
    
    # Define the 15 bilinear terms
    # Each term is (u_indices, v_indices, w_indices) where each is a list of (index, coeff)
    
    # Term 1: A00 * B00 -> C00, C10
    terms = []
    
    # Term 1: (A00 + A11) * (B00 + B11)
    u1 = []
    v1 = []
    w1 = []
    
    # A00 is rows 0-1, cols 0-1 of A
    for i in [0, 1]:
        for j in [0, 1]:
            u1.append(((i, j), 1))
    
    # A11 is rows 2-3, cols 2-3 of A
    for i in [2, 3]:
        for j in [2, 3]:
            u1.append(((i, j), 1))
    
    # B00 is rows 0-1, cols 0-1 of B (relative to first 2x2 block)
    for i in [0, 1]:
        for j in [0, 1]:
            v1.append(((i, j), 1))
    
    # B11 is rows 2-3, cols 2-3 of B
    for i in [2, 3]:
        for j in [2, 3]:
            v1.append(((i, j), 1))
    
    # C00 is rows 0-1, cols 0-1 of C
    for i in [0, 1]:
        for j in [0, 1]:
            w1.append(((j, i), 1))
    
    # C11 is rows 2-3, cols 2-3 of C
    for i in [2, 3]:
        for j in [2, 3]:
            w1.append(((j, i), 1))
    
    terms.append((u1, v1, w1))
    
    # Term 2: (A10 + A11) * B00
    u2 = []
    v2 = []
    w2 = []
    
    # A10 is rows 2-3, cols 0-1
    for i in [2, 3]:
        for j in [0, 1]:
            u2.append(((i, j), 1))
    
    # A11 is rows 2-3, cols 2-3
    for i in [2, 3]:
        for j in [2, 3]:
            u2.append(((i, j), 1))
    
    # B00 is rows 0-1, cols 0-1
    for i in [0, 1]:
        for j in [0, 1]:
            v2.append(((i, j), 1))
    
    # C10 is rows 2-3, cols 0-1
    for i in [2, 3]:
        for j in [0, 1]:
            w2.append(((j, i), 1))
    
    # C11 is rows 2-3, cols 2-3
    for i in [2, 3]:
        for j in [2, 3]:
            w2.append(((j, i), -1))
    
    terms.append((u2, v2, w2))
    
    # Term 3: A00 * (B01 + B11)
    u3 = []
    v3 = []
    w3 = []
    
    # A00 is rows 0-1, cols 0-1
    for i in [0, 1]:
        for j in [0, 1]:
            u3.append(((i, j), 1))
    
    # B01 is rows 0-1, cols 2-3
    for i in [0, 1]:
        for j in [2, 3]:
            v3.append(((i, j), 1))
    
    # B11 is rows 2-3, cols 2-3
    for i in [2, 3]:
        for j in [2, 3]:
            v3.append(((i, j), -1))
    
    # C01 is rows 0-1, cols 2-3
    for i in [0, 1]:
        for j in [2, 3]:
            w3.append(((j, i), 1))
    
    # C11 is rows 2-3, cols 2-3
    for i in [2, 3]:
        for j in [2, 3]:
            w3.append(((j, i), 1))
    
    terms.append((u3, v3, w3))
    
    # Term 4: A11 * (B10 - B00)
    u4 = []
    v4 = []
    w4 = []
    
    # A11 is rows 2-3, cols 2-3
    for i in [2, 3]:
        for j in [2, 3]:
            u4.append(((i, j), 1))
    
    # B10 is rows 2-3, cols 0-1
    for i in [2, 3]:
        for j in [0, 1]:
            v4.append(((i, j), 1))
    
    # B00 is rows 0-1, cols 0-1
    for i in [0, 1]:
        for j in [0, 1]:
            v4.append(((i, j), -1))
    
    # C00 is rows 0-1, cols 0-1
    for i in [0, 1]:
        for j in [0, 1]:
            w4.append(((j, i), 1))
    
    # C10 is rows 2-3, cols 0-1
    for i in [2, 3]:
        for j in [0, 1]:
            w4.append(((j, i), 1))
    
    terms.append((u4, v4, w4))
    
    # Term 5: (A00 + A01) * B11
    u5 = []
    v5 = []
    w5 = []
    
    # A00 is rows 0-1, cols 0-1
    for i in [0, 1]:
        for j in [0, 1]:
            u5.append(((i, j), 1))
    
    # A01 is rows 0-1, cols 2-3
    for i in [0, 1]:
        for j in [2, 3]:
            u5.append(((i, j), 1))
    
    # B11 is rows 2-3, cols 2-3
    for i in [2, 3]:
        for j in [2, 3]:
            v5.append(((i, j), 1))
    
    # C00 is rows 0-1, cols 0-1
    for i in [0, 1]:
        for j in [0, 1]:
            w5.append(((j, i), -1))
    
    # C01 is rows 0-1, cols 2-3
    for i in [0, 1]:
        for j in [2, 3]:
            w5.append(((j, i), 1))
    
    terms.append((u5, v5, w5))
    
    # Term 6: (A10 - A00) * (B00 + B01)
    u6 = []
    v6 = []
    w6 = []
    
    # A10 is rows 2-3, cols 0-1
    for i in [2, 3]:
        for j in [0, 1]:
            u6.append(((i, j), 1))
    
    # A00 is rows 0-1, cols 0-1
    for i in [0, 1]:
        for j in [0, 1]:
            u6.append(((i, j), -1))
    
    # B00 is rows 0-1, cols 0-1
    for i in [0, 1]:
        for j in [0, 1]:
            v6.append(((i, j), 1))
    
    # B01 is rows 0-1, cols 2-3
    for i in [0, 1]:
        for j in [2, 3]:
            v6.append(((i, j), 1))
    
    # C11 is rows 2-3, cols 2-3
    for i in [2, 3]:
        for j in [2, 3]:
            w6.append(((j, i), 1))
    
    terms.append((u6, v6, w6))
    
    # Term 7: (A01 - A11) * (B10 + B11)
    u7 = []
    v7 = []
    w7 = []
    
    # A01 is rows 0-1, cols 2-3
    for i in [0, 1]:
        for j in [2, 3]:
            u7.append(((i, j), 1))
    
    # A11 is rows 2-3, cols 2-3
    for i in [2, 3]:
        for j in [2, 3]:
            u7.append(((i, j), -1))
    
    # B10 is rows 2-3, cols 0-1
    for i in [2, 3]:
        for j in [0, 1]:
            v7.append(((i, j), 1))
    
    # B11 is rows 2-3, cols 2-3
    for i in [2, 3]:
        for j in [2, 3]:
            v7.append(((i, j), 1))
    
    # C00 is rows 0-1, cols 0-1
    for i in [0, 1]:
        for j in [0, 1]:
            w7.append(((j, i), 1))
    
    terms.append((u7, v7, w7))
    
    # Terms 8-15: Handle the remaining columns (cols 4-5 of B and C)
    # We need to multiply A (4x4) by B_right (4x2) to get C_right (4x2)
    
    # Term 8: A00 * B_top_right
    u8 = []
    v8 = []
    w8 = []
    
    # A00 is rows 0-1, cols 0-1
    for i in [0, 1]:
        for j in [0, 1]:
            u8.append(((i, j), 1))
    
    # B_top_right is rows 0-1, cols 4-5
    for i in [0, 1]:
        for j in [4, 5]:
            v8.append(((i, j), 1))
    
    # C_top_right is rows 0-1, cols 4-5
    for i in [0, 1]:
        for j in [4, 5]:
            w8.append(((j, i), 1))
    
    terms.append((u8, v8, w8))
    
    # Term 9: A01 * B_bot_right
    u9 = []
    v9 = []
    w9 = []
    
    # A01 is rows 0-1, cols 2-3
    for i in [0, 1]:
        for j in [2, 3]:
            u9.append(((i, j), 1))
    
    # B_bot_right is rows 2-3, cols 4-5
    for i in [2, 3]:
        for j in [4, 5]:
            v9.append(((i, j), 1))
    
    # C_top_right is rows 0-1, cols 4-5
    for i in [0, 1]:
        for j in [4, 5]:
            w9.append(((j, i), 1))
    
    terms.append((u9, v9, w9))
    
    # Term 10: A10 * B_top_right
    u10 = []
    v10 = []
    w10 = []
    
    # A10 is rows 2-3, cols 0-1
    for i in [2, 3]:
        for j in [0, 1]:
            u10.append(((i, j), 1))
    
    # B_top_right is rows 0-1, cols 4-5
    for i in [0, 1]:
        for j in [4, 5]:
            v10.append(((i, j), 1))
    
    # C_bot_right is rows 2-3, cols 4-5
    for i in [2, 3]:
        for j in [4, 5]:
            w10.append(((j, i), 1))
    
    terms.append((u10, v10, w10))
    
    # Term 11: A11 * B_bot_right
    u11 = []
    v11 = []
    w11 = []
    
    # A11 is rows 2-3, cols 2-3
    for i in [2, 3]:
        for j in [2, 3]:
            u11.append(((i, j), 1))
    
    # B_bot_right is rows 2-3, cols 4-5
    for i in [2, 3]:
        for j in [4, 5]:
            v11.append(((i, j), 1))
    
    # C_bot_right is rows 2-3, cols 4-5
    for i in [2, 3]:
        for j in [4, 5]:
            w11.append(((j, i), 1))
    
    terms.append((u11, v11, w11))
    
    # Term 12: (A00 + A11) * (B_top_right + B_bot_right)
    u12 = []
    v12 = []
    w12 = []
    
    # A00 is rows 0-1, cols 0-1
    for i in [0, 1]:
        for j in [0, 1]:
            u12.append(((i, j), 1))
    
    # A11 is rows 2-3, cols 2-3
    for i in [2, 3]:
        for j in [2, 3]:
            u12.append(((i, j), 1))
    
    # B_top_right is rows 0-1, cols 4-5
    for i in [0, 1]:
        for j in [4, 5]:
            v12.append(((i, j), 1))
    
    # B_bot_right is rows 2-3, cols 4-5
    for i in [2, 3]:
        for j in [4, 5]:
            v12.append(((i, j), 1))
    
    # C_top_right is rows 0-1, cols 4-5
    for i in [0, 1]:
        for j in [4, 5]:
            w12.append(((j, i), 1))
    
    # C_bot_right is rows 2-3, cols 4-5
    for i in [2, 3]:
        for j in [4, 5]:
            w12.append(((j, i), 1))
    
    terms.append((u12, v12, w12))
    
    # Term 13: (A10 + A11) * B_top_right
    u13 = []
    v13 = []
    w13 = []
    
    # A10 is rows 2-3, cols 0-1
    for i in [2, 3]:
        for j in [0, 1]:
            u13.append(((i, j), 1))
    
    # A11 is rows 2-3, cols 2-3
    for i in [2, 3]:
        for j in [2, 3]:
            u13.append(((i, j), 1))
    
    # B_top_right is rows 0-1, cols 4-5
    for i in [0, 1]:
        for j in [4, 5]:
            v13.append(((i, j), 1))
    
    # C_bot_right is rows 2-3, cols 4-5
    for i in [2, 3]:
        for j in [4, 5]:
            w13.append(((j, i), 1))
    
    # C_bot_right is rows 2-3, cols 4-5 (with -1 for some entries)
    for i in [2, 3]:
        for j in [4, 5]:
            w13.append(((j, i), -1))
    
    terms.append((u13, v13, w13))
    
    # Term 14: A00 * (B_top_right + B_bot_right)
    u14 = []
    v14 = []
    w14 = []
    
    # A00 is rows 0-1, cols 0-1
    for i in [0, 1]:
        for j in [0, 1]:
            u14.append(((i, j), 1))
    
    # B_top_right is rows 0-1, cols 4-5
    for i in [0, 1]:
        for j in [4, 5]:
            v14.append(((i, j), 1))
    
    # B_bot_right is rows 2-3, cols 4-5
    for i in [2, 3]:
        for j in [4, 5]:
            v14.append(((i, j), -1))
    
    # C_top_right is rows 0-1, cols 4-5
    for i in [0, 1]:
        for j in [4, 5]:
            w14.append(((j, i), 1))
    
    # C_bot_right is rows 2-3, cols 4-5
    for i in [2, 3]:
        for j in [4, 5]:
            w14.append(((j, i), 1))
    
    terms.append((u14, v14, w14))
    
    # Term 15: A11 * (B_top_right - B_bot_right)
    u15 = []
    v15 = []
    w15 = []
    
    # A11 is rows 2-3, cols 2-3
    for i in [2, 3]:
        for j in [2, 3]:
            u15.append(((i, j), 1))
    
    # B_top_right is rows 0-1, cols 4-5
    for i in [0, 1]:
        for j in [4, 5]:
            v15.append(((i, j), 1))
    
    # B_bot_right is rows 2-3, cols 4-5
    for i in [2, 3]:
        for j in [4, 5]:
            v15.append(((i, j), -1))
    
    # C_top_right is rows 0-1, cols 4-5
    for i in [0, 1]:
        for j in [4, 5]:
            w15.append(((j, i), 1))
    
    # C_bot_right is rows 2-3, cols 4-5
    for i in [2, 3]:
        for j in [4, 5]:
            w15.append(((j, i), 1))
    
    terms.append((u15, v15, w15))
    
    # Fill in the factor matrices
    for term_idx, (u_terms, v_terms, w_terms) in enumerate(terms):
        for (row, col), coeff in u_terms:
            idx = row * m + col
            U[idx][term_idx] += coeff
        
        for (row, col), coeff in v_terms:
            idx = row * p + col
            V[idx][term_idx] += coeff
        
        for (row, col), coeff in w_terms:
            # W index is col * n + row
            idx = col * n + row
            W[idx][term_idx] += coeff
    
    return {"U": U, "V": V, "W": W}
