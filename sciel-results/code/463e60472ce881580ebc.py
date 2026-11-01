import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    n, m, p = 2, 4, 4
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Indices mapping
    # A is 2x4. U index: i*m + j
    # B is 4x4. V index: j*p + k
    # C is 2x4. W index: k*n + i
    
    # Block definitions
    # A blocks: 1x2
    # A00: A[0,0], A[0,1] -> U[0], U[1]
    # A01: A[0,2], A[0,3] -> U[2], U[3]
    # A10: A[1,0], A[1,1] -> U[4], U[5]
    # A11: A[1,2], A[1,3] -> U[6], U[7]
    
    # B blocks: 2x2
    # B00: B[0,0], B[0,1], B[1,0], B[1,1] -> V[0], V[1], V[4], V[5]
    # B01: B[0,2], B[0,3], B[1,2], B[1,3] -> V[2], V[3], V[6], V[7]
    # B10: B[2,0], B[2,1], B[3,0], B[3,1] -> V[8], V[9], V[12], V[13]
    # B11: B[2,2], B[2,3], B[3,2], B[3,3] -> V[10], V[11], V[14], V[15]
    
    # C blocks: 1x2
    # C00: C[0,0], C[0,1] -> W[0], W[2]
    # C01: C[0,2], C[0,3] -> W[4], W[6]
    # C10: C[1,0], C[1,1] -> W[1], W[3]
    # C11: C[1,2], C[1,3] -> W[5], W[7]

    # Helper to add a rank-1 tensor
    # term: (u_idx, v_idx, w_idx, coeff)
    def add_tensor(u_idx, v_idx, w_idx, coeff, r):
        if coeff != 0:
            U[u_idx][r] += coeff
            V[v_idx][r] += 1
            W[w_idx][r] += 1

    # Strassen Steps
    # M1 = (A00+A01)(B10-B11) -> C01(+), C11(+)
    # M2 = (A10+A11)(B00+B11) -> C00(-), C01(+)
    # M3 = (A00-A10)(B00-B10) -> C10(+), C11(-)
    # M4 = (A10+A11)(B01-B11) -> C00(+), C10(+)
    # M5 = (A11-A01)(B10-B11) -> C11(+)
    # M6 = (A00-A10)(B00+B11) -> C00(+)
    # M7 = (A11-A01)(B00-B10) -> C00(+), C11(+)

    # Define the linear combinations for X and Y for each M
    # X is 1x2, so X[0], X[1]
    # Y is 2x2, so Y[0][0], Y[0][1], Y[1][0], Y[1][1]
    
    # A indices for blocks
    A00 = [(0, 1), (1, 1)] # (u_idx, coeff)
    A01 = [(2, 1), (3, 1)]
    A10 = [(4, 1), (5, 1)]
    A11 = [(6, 1), (7, 1)]
    
    # B indices for blocks
    # B block indices: (v_idx, coeff)
    # B00: V[0], V[1], V[4], V[5]
    B00 = [(0, 1), (1, 1), (4, 1), (5, 1)]
    B01 = [(2, 1), (3, 1), (6, 1), (7, 1)]
    B10 = [(8, 1), (9, 1), (12, 1), (13, 1)]
    B11 = [(10, 1), (11, 1), (14, 1), (15, 1)]

    # C indices for blocks
    # C block indices: (w_idx, coeff)
    # C00: W[0], W[2]
    C00 = [(0, 1), (2, 1)]
    C01 = [(4, 1), (6, 1)]
    C10 = [(1, 1), (3, 1)]
    C11 = [(5, 1), (7, 1)]

    # Function to compute linear combo of blocks
    def combine_blocks(blocks, signs):
        # blocks: list of block lists
        # signs: list of signs corresponding to blocks
        # Returns list of (idx, coeff)
        res = {}
        for b, s in zip(blocks, signs):
            for idx, c in b:
                res[idx] = res.get(idx, 0) + s * c
        return list(res.items())

    # M1 = (A00+A01)(B10-B11)
    X1_0 = combine_blocks([A00, A01], [1, 1])
    X1_1 = combine_blocks([A00, A01], [1, 1])
    Y1_00 = combine_blocks([B10, B11], [1, -1])
    Y1_01 = combine_blocks([B10, B11], [1, -1])
    Y1_10 = combine_blocks([B10, B11], [1, -1])
    Y1_11 = combine_blocks([B10, B11], [1, -1])
    
    # Note: X is 1x2. X[0] and X[1] are the two components.
    # In A00+A01, A00 is 1x2, A01 is 1x2. Sum is 1x2.
    # So X1_0 corresponds to col 0 of the result, X1_1 to col 1.
    # A00 = [A[0,0], A[0,1]]. A01 = [A[0,2], A[0,3]].
    # Sum = [A[0,0]+A[0,2], A[0,1]+A[0,3]].
    # So X1_0 should be A[0,0]+A[0,2] and X1_1 should be A[0,1]+A[0,3].
    
    # My combine_blocks mixes all indices. I need to handle the vector structure.
    # Let's redefine blocks as lists of 2 elements (since A blocks are 1x2).
    # A00 = [(0,1), (1,1)] -> elem 0, elem 1
    # A01 = [(2,1), (3,1)] -> elem 0, elem 1
    
    def combine_vectors(vecs, signs):
        # vecs: list of lists of (idx, coeff)
        # signs: list of signs
        # Returns list of (idx, coeff) for each component
        # Since vectors are length 2, we return [list0, list1]
        res = [{}, {}]
        for v, s in zip(vecs, signs):
            for i, (idx, c) in enumerate(v):
                res[i][idx] = res[i].get(idx, 0) + s * c
        return [list(r.items()) for r in res]

    def combine_matrices(mats, signs):
        # mats: list of lists of lists (2x2)
        # signs: list of signs
        # Returns 2x2 list of (idx, coeff)
        res = [[{}, {}], [{}, {}]]
        for m, s in zip(mats, signs):
            for r in range(2):
                for c in range(2):
                    idx, coeff = m[r][c]
                    res[r][c][idx] = res[r][c].get(idx, 0) + s * coeff
        return [[list(res[r][c].items()) for c in range(2)] for r in range(2)]

    # Redefine blocks with structure
    A00 = [[(0,1)], [(1,1)]] # 1x2 matrix: row 0, col 0; row 0, col 1
    A01 = [[(2,1)], [(3,1)]]
    A10 = [[(4,1)], [(5,1)]]
    A11 = [[(6,1)], [(7,1)]]
    
    B00 = [[(0,1), (1,1)], [(4,1), (5,1)]] # 2x2 matrix
    B01 = [[(2,1), (3,1)], [(6,1), (7,1)]]
    B10 = [[(8,1), (9,1)], [(12,1), (13,1)]]
    B11 = [[(10,1), (11,1)], [(14,1), (15,1)]]
    
    # C blocks are vectors (1x2)
    C00 = [[(0,1)], [(2,1)]]
    C01 = [[(4,1)], [(6,1)]]
    C10 = [[(1,1)], [(3,1)]]
    C11 = [[(5,1)], [(7,1)]]

    # M1 = (A00+A01)(B10-B11)
    X1 = combine_vectors([A00, A01], [1, 1])
    Y1 = combine_matrices([B10, B11], [1, -1])
    
    # Process M1
    # Z0 = X1[0] * Y1[0][0] + X1[1] * Y1[1][0]
    # Z1 = X1[0] * Y1[0][1] + X1[1] * Y1[1][1]
    
    # Contributions to C:
    # C01 += M1
    # C11 += M1
    
    # We need to iterate through the 4 terms of the scalar mult.
    # Term 1: X1[0] * Y1[0][0] -> Z0
    # Term 2: X1[1] * Y1[1][0] -> Z0
    # Term 3: X1[0] * Y1[0][1] -> Z1
    # Term 4: X1[1] * Y1[1][1] -> Z1
    
    # For each term, we have a vector X_part and a matrix element Y_part.
    # X_part is a list of (u_idx, coeff)
    # Y_part is a list of (v_idx, coeff)
    # We iterate over all pairs.
    
    # Target C blocks and their coeffs for M1:
    # C01: +1, C11: +1
    
    # Z0 maps to first element of C block.
    # Z1 maps to second element of C block.
    
    def process_strassen_term(X_part, Y_part, C_block, coeff, r):
        # X_part: list of (u_idx, c_u)
        # Y_part: list of (v_idx, c_v)
        # C_block: list of (w_idx, c_w) - actually just w_idx since c_w is 1
        # coeff: scalar
        # r: rank index
        for u_idx, c_u in X_part:
            for v_idx, c_v in Y_part:
                # Determine W index
                # If this term contributes to Z0, use C_block[0]
                # If Z1, use C_block[1]
                pass # This logic is inside the caller

    # Let's just write it out for clarity and speed, given the small size.
    
    r = 0
    # M1
    X1_0, X1_1 = X1
    Y1_00, Y1_01 = Y1[0]
    Y1_10, Y1_11 = Y1[1]
    
    # Z0 terms
    # X1_0 * Y1_00
    for u_idx, cu in X1_0:
        for v_idx, cv in Y1_00:
            add_tensor(u_idx, v_idx, C00[0][0][0], cu*cv*0, r) # C00 not affected by M1
            add_tensor(u_idx, v_idx, C01[0][0][0], cu*cv*1, r) # C01 +1
            add_tensor(u_idx, v_idx, C11[0][0][0], cu*cv*1, r) # C11 +1
    # X1_1 * Y1_10
    for u_idx, cu in X1_1:
        for v_idx, cv in Y1_10:
            add_tensor(u_idx, v_idx, C01[0][0][0], cu*cv*1, r)
            add_tensor(u_idx, v_idx, C11[0][0][0], cu*cv*1, r)
            
    # Z1 terms
    # X1_0 * Y1_01
    for u_idx, cu in X1_0:
        for v_idx, cv in Y1_01:
            add_tensor(u_idx, v_idx, C01[1][0][0], cu*cv*1, r)
            add_tensor(u_idx, v_idx, C11[1][0][0], cu*cv*1, r)
    # X1_1 * Y1_11
    for u_idx, cu in X1_1:
        for v_idx, cv in Y1_11:
            add_tensor(u_idx, v_idx, C01[1][0][0], cu*cv*1, r)
            add_tensor(u_idx, v_idx, C11[1][0][0], cu*cv*1, r)
            
    r += 1
    
    # M2 = (A10+A11)(B00+B11)
    # C00 -= M2, C01 += M2
    X2 = combine_vectors([A10, A11], [1, 1])
    Y2 = combine_matrices([B00, B11], [1, 1])
    X2_0, X2_1 = X2
    Y2_00, Y2_01 = Y2[0]
    Y2_10, Y2_11 = Y2[1]
    
    # Z0
    for u_idx, cu in X2_0:
        for v_idx, cv in Y2_00:
            add_tensor(u_idx, v_idx, C00[0][0][0], cu*cv*(-1), r)
            add_tensor(u_idx, v_idx, C01[0][0][0], cu*cv*1, r)
    for u_idx, cu in X2_1:
        for v_idx, cv in Y2_10:
            add_tensor(u_idx, v_idx, C00[0][0][0], cu*cv*(-1), r)
            add_tensor(u_idx, v_idx, C01[0][0][0], cu*cv*1, r)
    # Z1
    for u_idx, cu in X2_0:
        for v_idx, cv in Y2_01:
            add_tensor(u_idx, v_idx, C00[1][0][0], cu*cv*(-1), r)
            add_tensor(u_idx, v_idx, C01[1][0][0], cu*cv*1, r)
    for u_idx, cu in X2_1:
        for v_idx, cv in Y2_11:
            add_tensor(u_idx, v_idx, C00[1][0][0], cu*cv*(-1), r)
            add_tensor(u_idx, v_idx, C01[1][0][0], cu*cv*1, r)
            
    r += 1
    
    # M3 = (A00-A10)(B00-B10)
    # C10 += M3, C11 -= M3
    X3 = combine_vectors([A00, A10], [1, -1])
    Y3 = combine_matrices([B00, B10], [1, -1])
    X3_0, X3_1 = X3
    Y3_00, Y3_01 = Y3[0]
    Y3_10, Y3_11 = Y3[1]
    
    # Z0
    for u_idx, cu in X3_0:
        for v_idx, cv in Y3_00:
            add_tensor(u_idx, v_idx, C10[0][0][0], cu*cv*1, r)
            add_tensor(u_idx, v_idx, C11[0][0][0], cu*cv*(-1), r)
    for u_idx, cu in X3_1:
        for v_idx, cv in Y3_10:
            add_tensor(u_idx, v_idx, C10[0][0][0], cu*cv*1, r)
            add_tensor(u_idx, v_idx, C11[0][0][0], cu*cv*(-1), r)
    # Z1
    for u_idx, cu in X3_0:
        for v_idx, cv in Y3_01:
            add_tensor(u_idx, v_idx, C10[1][0][0], cu*cv*1, r)
            add_tensor(u_idx, v_idx, C11[1][0][0], cu*cv*(-1), r)
    for u_idx, cu in X3_1:
        for v_idx, cv in Y3_11:
            add_tensor(u_idx, v_idx, C10[1][0][0], cu*cv*1, r)
            add_tensor(u_idx, v_idx, C11[1][0][0], cu*cv*(-1), r)
            
    r += 1
    
    # M4 = (A10+A11)(B01-B11)
    # C00 += M4, C10 += M4
    X4 = combine_vectors([A10, A11], [1, 1])
    Y4 = combine_matrices([B01, B11], [1, -1])
    X4_0, X4_1 = X4
    Y4_00, Y4_01 = Y4[0]
    Y4_10, Y4_11 = Y4[1]
    
    # Z0
    for u_idx, cu in X4_0:
        for v_idx, cv in Y4_00:
            add_tensor(u_idx, v_idx, C00[0][0][0], cu*cv*1, r)
            add_tensor(u_idx, v_idx, C10[0][0][0], cu*cv*1, r)
    for u_idx, cu in X4_1:
        for v_idx, cv in Y4_10:
            add_tensor(u_idx, v_idx, C00[0][0][0], cu*cv*1, r)
            add_tensor(u_idx, v_idx, C10[0][0][0], cu*cv*1, r)
    # Z1
    for u_idx, cu in X4_0:
        for v_idx, cv in Y4_01:
            add_tensor(u_idx, v_idx, C00[1][0][0], cu*cv*1, r)
            add_tensor(u_idx, v_idx, C10[1][0][0], cu*cv*1, r)
    for u_idx, cu in X4_1:
        for v_idx, cv in Y4_11:
            add_tensor(u_idx, v_idx, C00[1][0][0], cu*cv*1, r)
            add_tensor(u_idx, v_idx, C10[1][0][0], cu*cv*1, r)
            
    r += 1
    
    # M5 = (A11-A01)(B10-B11)
    # C11 += M5
    X5 = combine_vectors([A11, A01], [1, -1])
    Y5 = combine_matrices([B10, B11], [1, -1])
    X5_0, X5_1 = X5
    Y5_00, Y5_01 = Y5[0]
    Y5_10, Y5_11 = Y5[1]
    
    # Z0
    for u_idx, cu in X5_0:
        for v_idx, cv in Y5_00:
            add_tensor(u_idx, v_idx, C11[0][0][0], cu*cv*1, r)
    for u_idx, cu in X5_1:
        for v_idx, cv in Y5_10:
            add_tensor(u_idx, v_idx, C11[0][0][0], cu*cv*1, r)
    # Z1
    for u_idx, cu in X5_0:
        for v_idx, cv in Y5_01:
            add_tensor(u_idx, v_idx, C11[1][0][0], cu*cv*1, r)
    for u_idx, cu in X5_1:
        for v_idx, cv in Y5_11:
            add_tensor(u_idx, v_idx, C11[1][0][0], cu*cv*1, r)
            
    r += 1
    
    # M6 = (A00-A10)(B00+B11)
    # C00 += M6
    X6 = combine_vectors([A00, A10], [1, -1])
    Y6 = combine_matrices([B00, B11], [1, 1])
    X6_0, X6_1 = X6
    Y6_00, Y6_01 = Y6[0]
    Y6_10, Y6_11 = Y6[1]
    
    # Z0
    for u_idx, cu in X6_0:
        for v_idx, cv in Y6_00:
            add_tensor(u_idx, v_idx, C00[0][0][0], cu*cv*1, r)
    for u_idx, cu in X6_1:
        for v_idx, cv in Y6_10:
            add_tensor(u_idx, v_idx, C00[0][0][0], cu*cv*1, r)
    # Z1
    for u_idx, cu in X6_0:
        for v_idx, cv in Y6_01:
            add_tensor(u_idx, v_idx, C00[1][0][0], cu*cv*1, r)
    for u_idx, cu in X6_1:
        for v_idx, cv in Y6_11:
            add_tensor(u_idx, v_idx, C00[1][0][0], cu*cv*1, r)
            
    r += 1
    
    # M7 = (A11-A01)(B00-B10)
    # C00 += M7, C11 += M7
    X7 = combine_vectors([A11, A01], [1, -1])
    Y7 = combine_matrices([B00, B10], [1, -1])
    X7_0, X7_1 = X7
    Y7_00, Y7_01 = Y7[0]
    Y7_10, Y7_11 = Y7[1]
    
    # Z0
    for u_idx, cu in X7_0:
        for v_idx, cv in Y7_00:
            add_tensor(u_idx, v_idx, C00[0][0][0], cu*cv*1, r)
            add_tensor(u_idx, v_idx, C11[0][0][0], cu*cv*1, r)
    for u_idx, cu in X7_1:
        for v_idx, cv in Y7_10:
            add_tensor(u_idx, v_idx, C00[0][0][0], cu*cv*1, r)
            add_tensor(u_idx, v_idx, C11[0][0][0], cu*cv*1, r)
    # Z1
    for u_idx, cu in X7_0:
        for v_idx, cv in Y7_01:
            add_tensor(u_idx, v_idx, C00[1][0][0], cu*cv*1, r)
            add_tensor(u_idx, v_idx, C11[1][0][0], cu*cv*1, r)
    for u_idx, cu in X7_1:
        for v_idx, cv in Y7_11:
            add_tensor(u_idx, v_idx, C00[1][0][0], cu*cv*1, r)
            add_tensor(u_idx, v_idx, C11[1][0][0], cu*cv*1, r)
            
    r += 1
    
    return {"U": U, "V": V, "W": W}
