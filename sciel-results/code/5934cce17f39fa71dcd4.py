def construct():
    n, m, p = 6, 6, 6
    
    # Laderman's rank-19 decomposition for 3x3 matrix multiplication
    # Each term: list of (a_idx, coeff) for A, (b_idx, coeff) for B, (c_idx, coeff) for C
    # a_idx = i*3+j (row-major), b_idx = j*3+k, c_idx = k*3+i (column-major for W)
    
    # The 19 bilinear forms for 3x3:
    # We define each term by its A-coefficients, B-coefficients, and C-coefficients
    
    laderman_terms = [
        # m0 = a00*b00
        {'a': [(0, 1)], 'b': [(0, 1)], 'c': [(0, 1)]},
        # m1 = a11*b11
        {'a': [(4, 1)], 'b': [(4, 1)], 'c': [(4, 1)]},
        # m2 = a22*b22
        {'a': [(8, 1)], 'b': [(8, 1)], 'c': [(8, 1)]},
        # m3 = a01*b10
        {'a': [(1, 1)], 'b': [(3, 1)], 'c': [(1, 1)]},
        # m4 = a12*b20
        {'a': [(2, 1)], 'b': [(6, 1)], 'c': [(2, 1)]},
        # m5 = a20*b01
        {'a': [(6, 1)], 'b': [(1, 1)], 'c': [(6, 1)]},
        # m6 = a02*b21
        {'a': [(2, 1)], 'b': [(7, 1)], 'c': [(7, 1)]},
        # m7 = a10*b02
        {'a': [(3, 1)], 'b': [(2, 1)], 'c': [(3, 1)]},
        # m8 = a21*b12
        {'a': [(7, 1)], 'b': [(5, 1)], 'c': [(5, 1)]},
        # m9 = (a00+a10)*(b00+b10)
        {'a': [(0, 1), (3, 1)], 'b': [(0, 1), (3, 1)], 'c': [(0, 1), (3, 1), (1, 1), (4, 1)]},
        # m10 = (a01+a11)*(b10+b11)
        {'a': [(1, 1), (4, 1)], 'b': [(3, 1), (4, 1)], 'c': [(1, 1), (4, 1), (0, 1), (3, 1)]},
        # m11 = (a02+a12)*(b20+b21)
        {'a': [(2, 1), (5, 1)], 'b': [(6, 1), (7, 1)], 'c': [(2, 1), (5, 1), (1, 1), (4, 1)]},
        # m12 = (a00+a20)*(b00+b20)
        {'a': [(0, 1), (6, 1)], 'b': [(0, 1), (6, 1)], 'c': [(0, 1), (6, 1), (1, 1), (7, 1)]},
        # m13 = (a01+a21)*(b10+b20)
        {'a': [(1, 1), (7, 1)], 'b': [(3, 1), (6, 1)], 'c': [(1, 1), (7, 1), (0, 1), (6, 1)]},
        # m14 = (a02+a22)*(b20+b22)
        {'a': [(2, 1), (8, 1)], 'b': [(6, 1), (8, 1)], 'c': [(2, 1), (8, 1), (1, 1), (7, 1)]},
        # m15 = (a10+a20)*(b01+b21)
        {'a': [(3, 1), (6, 1)], 'b': [(1, 1), (7, 1)], 'c': [(3, 1), (7, 1), (0, 1), (6, 1)]},
        # m16 = (a11+a21)*(b11+b21)
        {'a': [(4, 1), (7, 1)], 'b': [(4, 1), (7, 1)], 'c': [(4, 1), (7, 1), (3, 1), (6, 1)]},
        # m17 = (a12+a22)*(b21+b22)
        {'a': [(5, 1), (8, 1)], 'b': [(7, 1), (8, 1)], 'c': [(5, 1), (8, 1), (4, 1), (7, 1)]},
        # m18 = (a00+a01+a02)*(b00+b10+b20)
        {'a': [(0, 1), (1, 1), (2, 1)], 'b': [(0, 1), (3, 1), (6, 1)], 'c': [(0, 1), (1, 1), (2, 1)]},
    ]
    
    r3 = 19
    U33 = [[0] * r3 for _ in range(9)]
    V33 = [[0] * r3 for _ in range(9)]
    W33 = [[0] * r3 for _ in range(9)]
    
    for c_idx, term in enumerate(laderman_terms):
        for (a_idx, coeff) in term['a']:
            U33[a_idx][c_idx] = coeff
        for (b_idx, coeff) in term['b']:
            V33[b_idx][c_idx] = coeff
        for (c_entry, coeff) in term['c']:
            W33[c_entry][c_idx] = coeff
    
    # Verify: C[i,k] = sum_j A[i,j]*B[j,k]
    # C[i,k] = sum_c W33[k*3+i, c] * (sum_{i',j'} U33[i'*3+j', c] * A[i',j']) * (sum_{j',k'} V33[j'*3+k', c] * B[j',k'])
    # For this to equal sum_j A[i,j]*B[j,k], we need:
    # sum_c W33[k*3+i, c] * U33[i'*3+j', c] * V33[j'*3+k', c] = delta(i',i) * delta(j',k) ... 
    # Actually: sum_c W33[k*3+i, c] * U33[i'*3+j', c] * V33[j'*3+k', c] should give 1 if (i',j',k') = (i,j,k) for the sum over j
    # i.e., sum_c W33[k*3+i, c] * U33[i'*3+j', c] * V33[j'*3+k', c] = 1 if i'=i and k'=k, summed over j' with the constraint that j' appears in both...
    
    # Let me just verify numerically
    import numpy as np
    U33n = np.array(U33, dtype=float)
    V33n = np.array(V33, dtype=float)
    W33n = np.array(W33, dtype=float)
    
    # Check: for random A, B, compute C via tensor and via direct matmul
    np.random.seed(42)
    A_test = np.random.randn(3, 3)
    B_test = np.random.randn(3, 3)
    C_direct = A_test @ B_test
    
    C_tensor = np.zeros((3, 3))
    for c in range(r3):
        a_lin = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                a_lin[i, j] = U33n[i*3+j, c] * A_test[i, j]
        b_lin = np.zeros((3, 3))
        for j in range(3):
            for k in range(3):
                b_lin[j, k] = V33n[j*3+k, c] * B_test[j, k]
        prod_c = np.sum(a_lin) * np.sum(b_lin)  # This is wrong - it should be the product of the two linear forms
        
    # Actually, the bilinear form for term c is:
    # (sum_{i,j} U33[i*3+j, c] * A[i,j]) * (sum_{j,k} V33[j*3+k, c] * B[j,k])
    # And C[i,k] = sum_c W33[k*3+i, c] * bilinear_c
    
    C_tensor = np.zeros((3, 3))
    for c in range(r3):
        a_val = sum(U33n[i*3+j, c] * A_test[i, j] for i in range(3) for j in range(3))
        b_val = sum(V33n[j*3+k, c] * B_test[j, k] for j in range(3) for k in range(3))
        for i in range(3):
            for k in range(3):
                C_tensor[i, k] += W33n[k*3+i, c] * a_val * b_val
    
    if not np.allclose(C_tensor, C_direct, atol=1e-10):
        print("Laderman verification FAILED!")
        print("C_tensor:", C_tensor)
        print("C_direct:", C_direct)
        # Fall back to schoolbook
        r3 = 27
        U33 = [[0] * r3 for _ in range(9)]
        V33 = [[0] * r3 for _ in range(9)]
        W33 = [[0] * r3 for _ in range(9)]
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    U33[i*3+j][col] = 1
                    V33[j*3+k][col] = 1
                    W33[k*3+i][col] = 1
                    col += 1
    else:
        print("Laderman verification PASSED!")
    
    # Now build the full 6x6 decomposition using Strassen at the block level
    rank = 7 * r3
    U = [[0] * rank for _ in range(36)]
    V = [[0] * rank for _ in range(36)]
    W = [[0] * rank for _ in range(36)]
    
    strassen_terms = [
        {'a': {'A00': 1, 'A11': 1}, 'b': {'B00': 1, 'B11': 1}, 'c': {'C00': 1, 'C11': 1}},
        {'a': {'A10': 1, 'A11': 1}, 'b': {'B00': 1}, 'c': {'C10': 1, 'C11': -1}},
        {'a': {'A00': 1}, 'b': {'B01': 1, 'B11': -1}, 'c': {'C01': 1, 'C11': 1}},
        {'a': {'A11': 1}, 'b': {'B10': 1, 'B00': -1}, 'c': {'C00': 1, 'C10': 1}},
        {'a': {'A00': 1, 'A01': 1}, 'b': {'B11': 1}, 'c': {'C00': -1, 'C01': 1}},
        {'a': {'A10': 1, 'A00': -1}, 'b': {'B00': 1, 'B01': 1}, 'c': {'C11': 1}},
        {'a': {'A01': 1, 'A11': -1}, 'b': {'B10': 1, 'B11': 1}, 'c': {'C00': 1}},
    ]
    
    block_map = {
        'A00': (0, 0), 'A01': (0, 1), 'A10': (1, 0), 'A11': (1, 1),
        'B00': (0, 0), 'B01': (0, 1), 'B10': (1, 0), 'B11': (1, 1),
        'C00': (0, 0), 'C01': (0, 1), 'C10': (1, 0), 'C11': (1, 1),
    }
    
    for m_idx, term in enumerate(strassen_terms):
        base_col = m_idx * r3
        
        for local_col in range(r3):
            col = base_col + local_col
            
            for a_block, ca in term['a'].items():
                bi, bj = block_map[a_block]
                for i3 in range(3):
                    for j3 in range(3):
                        u_row = (bi*3+i3)*6 + (bj*3+j3)
                        U[u_row][col] += ca * U33[i3*3+j3][local_col]
            
            for b_block, cb in term['b'].items():
                bk, bl = block_map[b_block]
                for j3 in range(3):
                    for k3 in range(3):
                        v_row = (bk*3+j3)*6 + (bl*3+k3)
                        V[v_row][col] += cb * V33[j3*3+k3][local_col]
            
            for c_block, cc in term['c'].items():
                ci, ck = block_map[c_block]
                for i3 in range(3):
                    for k3 in range(3):
                        w_row = (ck*3+k3)*6 + (ci*3+i3)
                        W[w_row][col] += cc * W33[k3*3+i3][local_col]
    
    return {"U": U, "V": V, "W": W}
