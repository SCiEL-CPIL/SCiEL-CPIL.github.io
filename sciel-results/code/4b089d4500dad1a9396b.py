def construct():
    n, m, p = 4, 4, 7
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_dict, v_dict, w_dict):
        u = [0] * (n * m)
        v = [0] * (m * p)
        w = [0] * (p * n)
        for (i, j), c in u_dict.items():
            u[i * m + j] = c
        for (j, k), c in v_dict.items():
            v[j * p + k] = c
        for (k, i), c in w_dict.items():
            w[k * n + i] = c
        U_cols.append(u)
        V_cols.append(v)
        W_cols.append(w)
    
    def add_strassen_term(a_map, b_map, c_map, c_coeff=1):
        for r in range(2):
            for s in range(2):
                for t in range(2):
                    a_terms = a_map.get((r, s), [])
                    b_terms = b_map.get((s, t), [])
                    c_terms = c_map.get((r, t), [])
                    if not a_terms or not b_terms or not c_terms:
                        continue
                    for (i, j, ca) in a_terms:
                        for (j2, k, cb) in b_terms:
                            if j != j2:
                                continue
                            c_coeff_found = 0
                            for (i2, k2, cc) in c_terms:
                                if i2 == i and k2 == k:
                                    c_coeff_found += cc
                                    break
                            if c_coeff_found == 0:
                                continue
                            coeff = ca * cb * c_coeff_found * c_coeff
                            if coeff == 0:
                                continue
                            add_term(
                                {(i, j): coeff},
                                {(j, k): coeff},
                                {(k, i): coeff}
                            )
    
    # M1 = (A11+A22)(B11+B22) -> C11
    a_map_m1 = {}
    b_map_m1 = {}
    c_map_m1 = {}
    for r in range(2):
        for s in range(2):
            a_map_m1[(r,s)] = [(r,s,1), (r+2,s+2,1)]
    for s in range(2):
        for t in range(2):
            b_map_m1[(s,t)] = [(s,t,1), (s+2,t+2,1)]
    for r in range(2):
        for t in range(2):
            c_map_m1[(r,t)] = [(r,t,1)]
    add_strassen_term(a_map_m1, b_map_m1, c_map_m1, c_coeff=1)
    
    # M2 = (A21+A22)B11 -> C21
    a_map_m2 = {}
    b_map_m2 = {}
    c_map_m2 = {}
    for r in range(2):
        for s in range(2):
            a_map_m2[(r,s)] = [(r+2,s,1), (r+2,s+2,1)]
    for s in range(2):
        for t in range(2):
            b_map_m2[(s,t)] = [(s,t,1)]
    for r in range(2):
        for t in range(2):
            c_map_m2[(r,t)] = [(r+2,t,1)]
    add_strassen_term(a_map_m2, b_map_m2, c_map_m2, c_coeff=1)
    
    # M3 = A11(B12-B22) -> C12
    a_map_m3 = {}
    b_map_m3 = {}
    c_map_m3 = {}
    for r in range(2):
        for s in range(2):
            a_map_m3[(r,s)] = [(r,s,1)]
    for s in range(2):
        for t in range(2):
            b_map_m3[(s,t)] = [(s,t+2,1), (s+2,t+2,-1)]
    for r in range(2):
        for t in range(2):
            c_map_m3[(r,t)] = [(r,t+2,1)]
    add_strassen_term(a_map_m3, b_map_m3, c_map_m3, c_coeff=1)
    
    # M4 = A22(B21-B11) -> C11 and C21
    a_map_m4 = {}
    b_map_m4 = {}
    c_map_m4_c11 = {}
    c_map_m4_c21 = {}
    for r in range(2):
        for s in range(2):
            a_map_m4[(r,s)] = [(r+2,s+2,1)]
    for s in range(2):
        for t in range(2):
            b_map_m4[(s,t)] = [(s+2,t,1), (s,t,-1)]
    for r in range(2):
        for t in range(2):
            c_map_m4_c11[(r,t)] = [(r,t,1)]
            c_map_m4_c21[(r,t)] = [(r+2,t,1)]
    add_strassen_term(a_map_m4, b_map_m4, c_map_m4_c11, c_coeff=1)
    add_strassen_term(a_map_m4, b_map_m4, c_map_m4_c21, c_coeff=1)
    
    # M5 = (A11+A12)B22 -> C11 (-1) and C12 (+1)
    a_map_m5 = {}
    b_map_m5 = {}
    c_map_m5_c11 = {}
    c_map_m5_c12 = {}
    for r in range(2):
        for s in range(2):
            a_map_m5[(r,s)] = [(r,s,1), (r,s+2,1)]
    for s in range(2):
        for t in range(2):
            b_map_m5[(s,t)] = [(s+2,t+2,1)]
    for r in range(2):
        for t in range(2):
            c_map_m5_c11[(r,t)] = [(r,t,1)]
            c_map_m5_c12[(r,t)] = [(r,t+2,1)]
    add_strassen_term(a_map_m5, b_map_m5, c_map_m5_c11, c_coeff=-1)
    add_strassen_term(a_map_m5, b_map_m5, c_map_m5_c12, c_coeff=1)
    
    # M6 = (A21-A11)(B11+B12) -> C22
    a_map_m6 = {}
    b_map_m6 = {}
    c_map_m6 = {}
    for r in range(2):
        for s in range(2):
            a_map_m6[(r,s)] = [(r+2,s,1), (r,s,-1)]
    for s in range(2):
        for t in range(2):
            b_map_m6[(s,t)] = [(s,t,1), (s,t+2,1)]
    for r in range(2):
        for t in range(2):
            c_map_m6[(r,t)] = [(r+2,t+2,1)]
    add_strassen_term(a_map_m6, b_map_m6, c_map_m6, c_coeff=1)
    
    # M7 = (A12-A22)(B21+B22) -> C11
    a_map_m7 = {}
    b_map_m7 = {}
    c_map_m7 = {}
    for r in range(2):
        for s in range(2):
            a_map_m7[(r,s)] = [(r,s+2,1), (r+2,s+2,-1)]
    for s in range(2):
        for t in range(2):
            b_map_m7[(s,t)] = [(s+2,t,1), (s+2,t+2,1)]
    for r in range(2):
        for t in range(2):
            c_map_m7[(r,t)] = [(r,t,1)]
    add_strassen_term(a_map_m7, b_map_m7, c_map_m7, c_coeff=1)
    
    # Part 2: 4x4 x 4x3 schoolbook
    for i in range(4):
        for j in range(4):
            for k in range(3):
                add_term(
                    {(i, j): 1},
                    {(j, 4+k): 1},
                    {(4+k, i): 1}
                )
    
    rank = len(U_cols)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col in range(rank):
        for row in range(n * m):
            U[row][col] = U_cols[col][row]
        for row in range(m * p):
            V[row][col] = V_cols[col][row]
        for row in range(p * n):
            W[row][col] = W_cols[col][row]
    
    return {"U": U, "V": V, "W": W}
