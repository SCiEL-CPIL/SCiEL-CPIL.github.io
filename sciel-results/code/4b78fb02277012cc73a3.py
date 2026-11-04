def construct():
    n, m, p = 4, 4, 6
    rank = 77
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val):
        U[u_idx][col] += u_val
        V[v_idx][col] += v_val
        W[w_idx][col] += w_val

    col = 0
    
    # Left part: 4x4x4
    # Outer Strassen 7 products
    # Each is 2x2x2, decomposed by Inner Strassen 7 products
    
    # Define outer Strassen A and B block combinations
    # A blocks: A11, A12, A21, A22
    # B blocks: B11, B12, B21, B22 (for left 4 cols)
    
    # We need to map 2x2 block indices to global indices
    # A block (bi, bj) is at rows 2*bi..2*bi+1, cols 2*bj..2*bj+1
    # B block (bi, bj) is at rows 2*bi..2*bi+1, cols 2*bj..2*bj+1
    
    # Outer Strassen products M1..M7
    # M1: (A11+A22)(B11+B22)
    # M2: (A21+A22)B11
    # M3: A11(B12-B22)
    # M4: A22(B21-B11)
    # M5: (A11+A12)B22
    # M6: (A21-A11)(B11+B12)
    # M7: (A12-A22)(B21+B22)
    
    # Inner Strassen products s1..s7 for a 2x2 product A'B'
    # s1: (a11+a22)(b11+b22)
    # s2: (a21+a22)b11
    # s3: a11(b12-b22)
    # s4: a22(b21-b11)
    # s5: (a11+a12)b22
    # s6: (a21-a11)(b11+b12)
    # s7: (a12-a22)(b21+b22)
    
    # Output M' = s...
    # M'11 = s1+s4-s5+s7
    # M'12 = s3+s5
    # M'21 = s2+s4
    # M'22 = s1-s2+s3+s6
    
    # We need to iterate over all 7 outer products and 7 inner products.
    # For each, we determine the coefficients for U, V, W.
    
    # This is tedious to write manually. I will use a small helper logic.
    
    # Let's define the outer products as (A_comb, B_comb)
    # A_comb is a 2x2 matrix of coefficients for A blocks
    # B_comb is a 2x2 matrix of coefficients for B blocks
    
    outer_products = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,0):1}), # M1: A11+A22, B11+B22 (B is 4x4, so B11 is top-left, B22 is bottom-right? No, B is 4x4. B11 is top-left 2x2. B22 is bottom-right 2x2. Wait, B is 4x4. B11 is (0,0), B12 is (0,1), B21 is (1,0), B22 is (1,1).)
        # Correction: B blocks for 4x4 are B11, B12, B21, B22.
        # M1: (A11+A22)(B11+B22). A11 is (0,0), A22 is (1,1). B11 is (0,0), B22 is (1,1).
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}), # M2
        ({(0,0):1}, {(0,1):1, (1,1):-1}), # M3
        ({(1,1):1}, {(1,0):1, (0,0):-1}), # M4
        ({(0,0):1, (0,1):1}, {(1,1):1}), # M5
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}), # M6
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}) # M7
    ]
    
    # Inner products for 2x2 A'B'
    # s1: (a11+a22)(b11+b22)
    # a11 is A'_{0,0}, a22 is A'_{1,1}.
    # b11 is B'_{0,0}, b22 is B'_{1,1}.
    inner_products = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}), # s1
        ({(1,0):1, (1,1):1}, {(0,0):1}), # s2
        ({(0,0):1}, {(0,1):1, (1,1):-1}), # s3
        ({(1,1):1}, {(1,0):1, (0,0):-1}), # s4
        ({(0,0):1, (0,1):1}, {(1,1):1}), # s5
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}), # s6
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}) # s7
    ]
    
    # Output combination for M'
    # M'11 = s1+s4-s5+s7
    # M'12 = s3+s5
    # M'21 = s2+s4
    # M'22 = s1-s2+s3+s6
    output_combos = [
        {(0,0): 1, (3): 1, (4): -1, (6): 1}, # M'11 (indices 0..6 for s1..s7)
        {(2): 1, (4): 1}, # M'12
        {(1): 1, (3): 1}, # M'21
        {(0): 1, (1): -1, (2): 1, (5): 1} # M'22
    ]
    
    # Iterate outer products
    for outer_idx, (A_comb, B_comb) in enumerate(outer_products):
        # Iterate inner products
        for inner_idx, (a_comb, b_comb) in enumerate(inner_products):
            # Determine U, V, W contributions
            # U corresponds to A. V to B. W to C.
            # The scalar product is sum_{a_idx} sum_{b_idx} A_comb[a_idx] * a_comb[local_a] * B_comb[b_idx] * b_comb[local_b] * C_out
            # This is getting complicated.
            
            # Let's simplify.
            # The term is a scalar multiplication.
            # U vector is the coefficient of A.
            # V vector is the coefficient of B.
            # W vector is the coefficient of C.
            
            # For a specific inner product s_k:
            # It computes a linear combination of A' entries times a linear combination of B' entries.
            # A' entries are linear combinations of A blocks.
            # B' entries are linear combinations of B blocks.
            
            # We can just accumulate the coefficients.
            
            # U coefficients:
            # For each local A index (la, lb) in a_comb:
            #   For each global A block (ga, gb) in A_comb:
            #      A[2*ga+la, 2*gb+lb] has coeff a_comb[(la,lb)] * A_comb[(ga,gb)]
            
            # V coefficients:
            # For each local B index (lb, lc) in b_comb:
            #   For each global B block (gb, gc) in B_comb:
            #      B[2*gb+lb, 2*gc+lc] has coeff b_comb[(lb,lc)] * B_comb[(gb,gc)]
            
            # W coefficients:
            # The output M'_{li, lj} has coeff output_combos[(li, lj)][inner_idx]
            # M' is a 2x2 block of C.
            # For outer product outer_idx, M' is placed at C block (outer_idx // 2, outer_idx % 2)?
            # No, the outer Strassen outputs C11, C12, C21, C22.
            # M1, M2, M3, M4, M5, M6, M7 are intermediate.
            # The final C blocks are combinations of M1..M7.
            # I skipped the outer output combination!
            
            # Outer output combos:
            # C11 = M1 + M4 - M5 + M7
            # C12 = M3 + M5
            # C21 = M2 + M4
            # C22 = M1 - M2 + M3 + M6
            
            outer_output_combos = [
                {(0): 1, (3): 1, (4): -1, (6): 1}, # C11
                {(2): 1, (4): 1}, # C12
                {(1): 1, (3): 1}, # C21
                {(0): 1, (1): -1, (2): 1, (5): 1} # C22
            ]
            
            # So for each outer product k, and inner product l:
            # The scalar product s_{k,l} contributes to C blocks based on outer_output_combos.
            
            pass
