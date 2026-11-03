def construct():
    n, m, p = 3, 4, 6
    terms = []  # Each term: (u_dict, v_dict, w_dict)
    
    def add_strassen_block(i1, i2, j1, j2, k1, k2):
        """Add Strassen decomposition for A[i1:i2+1, j1:j2+1] * B[j1:j2+1, k1:k2+1] -> C[i1:i2+1, k1:k2+1]"""
        a = (i1, j1)
        b = (i1, j2)
        c = (i2, j1)
        d = (i2, j2)
        e = (j1, k1)
        f = (j1, k2)
        g = (j2, k1)
        h = (j2, k2)
        
        C11 = (i1, k1)
        C12 = (i1, k2)
        C21 = (i2, k1)
        C22 = (i2, k2)
        
        # M1 = (A11+A22)(B11+B22)
        terms.append((
            {a: 1, d: 1},
            {e: 1, h: 1},
            {C11: 1, C22: 1}
        ))
        # M2 = (A21+A22)B11
        terms.append((
            {c: 1, d: 1},
            {e: 1},
            {C21: 1, C22: -1}
        ))
        # M3 = A11(B12-B22)
        terms.append((
            {a: 1},
            {f: 1, h: -1},
            {C12: 1, C22: 1}
        ))
        # M4 = A22(B21-B11)
        terms.append((
            {d: 1},
            {g: 1, e: -1},
            {C11: 1, C21: 1}
        ))
        # M5 = (A11+A12)B22
        terms.append((
            {a: 1, b: 1},
            {h: 1},
            {C11: -1, C12: 1}
        ))
        # M6 = (A21-A11)(B11+B12)
        terms.append((
            {c: 1, a: -1},
            {e: 1, f: 1},
            {C22: 1}
        ))
        # M7 = (A12-A22)(B21+B22)
        terms.append((
            {b: 1, d: -1},
            {g: 1, h: 1},
            {C11: 1}
        ))
    
    def add_schoolbook_term(i, j, k):
        terms.append((
            {(i, j): 1},
            {(j, k): 1},
            {(i, k): 1}
        ))
    
    # Part 1: Row 0 of A, schoolbook (1x4x6 = 24 terms)
    for j in range(4):
        for k in range(6):
            add_schoolbook_term(0, j, k)
    
    # Part 2: Rows 1-2 of A, using Strassen on 2x2x2 blocks
    # Split columns j into 2 groups of 2, columns k into 3 groups of 2
    for j_base in [0, 2]:
        for k_base in [0, 2, 4]:
            add_strassen_block(1, 2, j_base, j_base + 1, k_base, k_base + 1)
    
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col, (u_dict, v_dict, w_dict) in enumerate(terms):
        for (i, j), coeff in u_dict.items():
            U[i * m + j][col] = coeff
        for (j, k), coeff in v_dict.items():
            V[j * p + k][col] = coeff
        for (i, k), coeff in w_dict.items():
            W[k * n + i][col] = coeff
    
    return {"U": U, "V": V, "W": W}
