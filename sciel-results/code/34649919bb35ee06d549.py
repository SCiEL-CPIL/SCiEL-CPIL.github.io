def construct():
    n, m, p = 2, 4, 8
    rank = 56  # 8 products * 7 Strassen terms
    
    U = [[0] * rank for _ in range(n * m)]  # 8 x 56
    V = [[0] * rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * rank for _ in range(p * n)]  # 16 x 56
    
    # Strassen terms for 2x2x2: (U_coeffs, V_coeffs, W_coeffs)
    # U_coeffs: [X00, X01, X10, X11]
    # V_coeffs: [Y00, Y01, Y10, Y11]
    # W_coeffs: [Z00, Z01, Z10, Z11]
    strassen_terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),   # m1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),   # m2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),   # m3
        ([0, 0, 0, 1], [0, -1, 1, 0], [1, 0, 1, 0]),   # m4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),   # m5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),   # m6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),   # m7
    ]
    
    # A block U indices: A1 = [0,1,4,5], A2 = [2,3,6,7]
    A_block_indices = {
        1: [0, 1, 4, 5],  # A1: (i=0,j=0), (i=0,j=1), (i=1,j=0), (i=1,j=1)
        2: [2, 3, 6, 7],  # A2: (i=0,j=2), (i=0,j=3), (i=1,j=2), (i=1,j=3)
    }
    
    # B block V indices
    B_block_indices = {
        (1,1): [0, 1, 8, 9],    # B11
        (1,2): [2, 3, 10, 11],  # B12
        (1,3): [4, 5, 12, 13],  # B13
        (1,4): [6, 7, 14, 15],  # B14
        (2,1): [16, 17, 24, 25],# B21
        (2,2): [18, 19, 26, 27],# B22
        (2,3): [20, 21, 28, 29],# B23
        (2,4): [22, 23, 30, 31],# B24
    }
    
    # C block W indices
    C_block_indices = {
        1: [0, 1, 2, 3],   # C1: (k=0,i=0), (k=0,i=1), (k=1,i=0), (k=1,i=1)
        2: [4, 5, 6, 7],   # C2
        3: [8, 9, 10, 11], # C3
        4: [12, 13, 14, 15],# C4
    }
    
    # Mapping: (A_block, B_block) -> C_block
    # A1*B11 -> C1, A2*B21 -> C1
    # A1*B12 -> C2, A2*B22 -> C2
    # A1*B13 -> C3, A2*B23 -> C3
    # A1*B14 -> C4, A2*B24 -> C4
    product_to_C = {
        (1, (1,1)): 1,
        (2, (2,1)): 1,
        (1, (1,2)): 2,
        (2, (2,2)): 2,
        (1, (1,3)): 3,
        (2, (2,3)): 3,
        (1, (1,4)): 4,
        (2, (2,4)): 4,
    }
    
    col = 0
    for a_block in [1, 2]:
        for b_block in [(1,1), (1,2), (1,3), (1,4), (2,1), (2,2), (2,3), (2,4)]:
            if a_block == 1 and b_block[0] == 2:
                continue  # A1 only multiplies with B1*
            if a_block == 2 and b_block[0] == 1:
                continue  # A2 only multiplies with B2*
            
            c_block = product_to_C[(a_block, b_block)]
            u_indices = A_block_indices[a_block]
            v_indices = B_block_indices[b_block]
            w_indices = C_block_indices[c_block]
            
            for u_coeffs, v_coeffs, w_coeffs in strassen_terms:
                for idx, coeff in zip(u_indices, u_coeffs):
                    U[idx][col] += coeff
                for idx, coeff in zip(v_indices, v_coeffs):
                    V[idx][col] += coeff
                for idx, coeff in zip(w_indices, w_coeffs):
                    W[idx][col] += coeff
                col += 1
    
    return {"U": U, "V": V, "W": W}
