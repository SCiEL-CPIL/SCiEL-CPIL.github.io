def construct():
    n, m, p = 4, 4, 8
    
    # Strassen 2x2x2 terms
    # Each term: (u_dict, v_dict, w_dict)
    # u_dict: {(i,j): coeff} for A
    # v_dict: {(j,k): coeff} for B
    # w_dict: {(i,k): coeff} for C
    strassen_2x2 = [
        # M1 = (A00+A11)(B00+B11)
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M2 = (A10+A11)B00
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        # M3 = A00(B01-B11)
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,0): -1, (1,1): 1}),
        # M4 = A11(B10-B00)
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(1,0): 1}),
        # M5 = (A00+A01)B11
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        # M6 = (A10-A00)(B00+B01)
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(0,0): 1}),
        # M7 = (A01-A11)(B10+B11)
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1, (0,1): 1}),
    ]
    
    # Top-level Strassen for 4x4x4
    # Each term: (a_blocks, b_blocks, c_blocks)
    # a_blocks: [(block_row, block_col, coeff), ...] for A
    # b_blocks: [(block_row, block_col, coeff), ...] for B
    # c_blocks: [(block_row, block_col, coeff), ...] for C
    top_level = [
        # M1 = (A00+A11)(B00+B11)
        ([((0,0), 1), ((1,1), 1)], [((0,0), 1), ((1,1), 1)], [((0,0), 1), ((1,1), 1)]),
        # M2 = (A10+A11)B00
        ([((1,0), 1), ((1,1), 1)], [((0,0), 1)], [((1,0), 1), ((1,1), -1)]),
        # M3 = A00(B01-B11)
        ([((0,0), 1)], [((0,1), 1), ((1,1), -1)], [((0,0), -1), ((1,1), 1)]),
        # M4 = A11(B10-B00)
        ([((1,1), 1)], [((1,0), 1), ((0,0), -1)], [((1,0), 1)]),
        # M5 = (A00+A01)B11
        ([((0,0), 1), ((0,1), 1)], [((1,1), 1)], [((0,0), -1), ((0,1), 1)]),
        # M6 = (A10-A00)(B00+B01)
        ([((1,0), 1), ((0,0), -1)], [((0,0), 1), ((0,1), 1)], [((0,0), 1)]),
        # M7 = (A01-A11)(B10+B11)
        ([((0,1), 1), ((1,1), -1)], [((1,0), 1), ((1,1), 1)], [((0,0), 1), ((0,1), 1)]),
    ]
    
    # Generate 4x4x4 terms
    all_terms = []
    for t_idx in range(7):
        a_blocks, b_blocks, c_blocks = top_level[t_idx]
        for s_idx in range(7):
            s_u, s_v, s_w = strassen_2x2[s_idx]
            
            u_global = {}
            for (r, c), coeff_s in s_u.items():
                for (u, v), coeff_a in a_blocks:
                    i = 2*u + r
                    j = 2*v + c
                    u_global[(i, j)] = u_global.get((i, j), 0) + coeff_s * coeff_a
            
            v_global = {}
            for (r, c), coeff_s in s_v.items():
                for (v, w), coeff_b in b_blocks:
                    j = 2*v + r
                    k = 2*w + c
                    v_global[(j, k)] = v_global.get((j, k), 0) + coeff_s * coeff_b
            
            w_global = {}
            for (r, c), coeff_s in s_w.items():
                for (u, w), coeff_c in c_blocks:
                    i = 2*u + r
                    k = 2*w + c
                    w_global[(i, k)] = w_global.get((i, k), 0) + coeff_s * coeff_c
            
            u_global = {k: v for k, v in u_global.items() if v != 0}
            v_global = {k: v for k, v in v_global.items() if v != 0}
            w_global = {k: v for k, v in w_global.items() if v != 0}
            
            if u_global and v_global and w_global:
                all_terms.append((u_global, v_global, w_global))
    
    # Create right half (columns 4-7)
    terms_right = []
    for u_dict, v_dict, w_dict in all_terms:
        v_new = {}
        for (j, k), coeff in v_dict.items():
            v_new[(j, k + 4)] = coeff
        w_new = {}
        for (i, k), coeff in w_dict.items():
            w_new[(i, k + 4)] = coeff
        terms_right.append((u_dict, v_new, w_new))
    
    total_terms = all_terms + terms_right
    rank = len(total_terms)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col_idx, (u_dict, v_dict, w_dict) in enumerate(total_terms):
        for (i, j), coeff in u_dict.items():
            U[i * m + j][col_idx] = coeff
        for (j, k), coeff in v_dict.items():
            V[j * p + k][col_idx] = coeff
        for (i, k), coeff in w_dict.items():
            W[k * n + i][col_idx] = coeff
    
    # Verify
    import numpy as np
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for l in range(p):
                T[i*m+j, j*p+l, l*n+i] = 1
    
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    
    result = np.zeros((n*m, m*p, p*n))
    for s in range(rank):
        result += np.einsum('a,b,c->abc', U_arr[:, s], V_arr[:, s], W_arr[:, s])
    
    if np.max(np.abs(result - T)) < 1e-10:
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for l in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + l][col] = 1
                    W[l * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
