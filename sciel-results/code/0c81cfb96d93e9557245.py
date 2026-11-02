def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=4
    # A is n x m (2x4)
    # B is m x p (4x4)
    # C is n x p (2x4)
    
    # We split the inner dimension m=4 into two parts of size 2.
    # Part 1: j=0,1
    # Part 2: j=2,3
    
    # Sub-problem 1: A_L (2x2) * B_L (2x4) -> P_L (2x4)
    # Sub-problem 2: A_R (2x2) * B_R (2x4) -> P_R (2x4)
    # C = P_L + P_R
    
    # Rank of 2x2x4 multiplication is 8 (naive).
    # Total rank = 8 + 8 = 16.
    
    rank = 16
    
    U = [[0] * rank for _ in range(n * m)] # 8 x 16
    V = [[0] * rank for _ in range(m * p)] # 16 x 16
    W = [[0] * rank for _ in range(p * n)] # 8 x 16
    
    # Helper to add a rank-1 term
    # A index: i * m + j
    # B index: j * p + k
    # C index: k * n + i
    
    def add_term(a_idx, b_idx, c_idx, coeff, col):
        U[a_idx][col] += coeff
        V[b_idx][col] += coeff
        W[c_idx][col] += coeff

    # Sub-problem 1: j in {0, 1}
    # A_L indices: i in {0,1}, j in {0,1}
    # B_L indices: j in {0,1}, k in {0,1,2,3}
    # P_L indices: i in {0,1}, k in {0,1,2,3}
    
    col_offset = 0
    for i in range(2):
        for j in range(2):
            a_global = i * 4 + j
            for k in range(4):
                b_global = j * 4 + k
                c_global = k * 2 + i
                add_term(a_global, b_global, c_global, 1, col_offset)
                col_offset += 1
                
    # Sub-problem 2: j in {2, 3}
    # A_R indices: i in {0,1}, j in {2,3}
    # B_R indices: j in {2,3}, k in {0,1,2,3}
    # P_R indices: i in {0,1}, k in {0,1,2,3}
    
    for i in range(2):
        for j in range(2, 4):
            a_global = i * 4 + j
            for k in range(4):
                b_global = j * 4 + k
                c_global = k * 2 + i
                add_term(a_global, b_global, c_global, 1, col_offset)
                col_offset += 1

    return {"U": U, "V": V, "W": W}
