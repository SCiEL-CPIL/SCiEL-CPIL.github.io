DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # 1. Build Strassen decomposition for 4x4x4
    U_444, V_444, W_444 = _strassen_decomp(4, 4, 4)
    
    # 2. Build schoolbook decomposition for the 5th column (k=4)
    U_extra = []
    V_extra = []
    W_extra = []
    
    for i in range(n):
        for j in range(m):
            U_extra.append([1 if (idx == i * m + j) else 0 for idx in range(n * m)])
            V_extra.append([1 if (idx == j * p + 4) else 0 for idx in range(m * p)])
            W_extra.append([1 if (idx == 4 * n + i) else 0 for idx in range(p * n)])
            
    # Combine
    rank_444 = len(U_444[0])
    rank_extra = len(U_extra[0]) if U_extra else 0
    total_rank = rank_444 + rank_extra
    
    U = [row[:rank_444] + [0] * rank_extra for row in U_444]
    V = [row[:rank_444] + [0] * rank_extra for row in V_444]
    W = [row[:rank_444] + [0] * rank_extra for row in W_444]
    
    for r in range(rank_extra):
        for i in range(n * m):
            U[i].append(U_extra[r][i])
        for j in range(m * p):
            V[j].append(V_extra[r][j])
        for k in range(p * n):
            W[k].append(W_extra[r][k])
            
    return {"U": U, "V": V, "W": W}

def _strassen_decomp(n, m, p):
    if n == 1 and m == 1 and p == 1:
        return [[1]], [[1]], [[1]]
    
    n2, m2, p2 = n // 2, m // 2, p // 2
    U_base, V_base, W_base = _strassen_decomp(n2, m2, p2)
    rank_base = len(U_base[0])
    
    # Map block indices
    # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
    
    M_defs = [
        # (A_coeffs, B_coeffs, C_coeffs)
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}), # M1
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}), # M2
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}), # M3
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}), # M4
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}), # M5
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}), # M6
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}), # M7
    ]
    
    U_total = []
    V_total = []
    W_total = []
    
    for A_coeffs, B_coeffs, C_coeffs in M_defs:
        for r in range(rank_base):
            U_new = [0] * (n * m)
            V_new = [0] * (m * p)
            W_new = [0] * (p * n)
            
            for i_sub in range(n2):
                for j_sub in range(m2):
                    base_val = U_base[r][i_sub * m2 + j_sub]
                    if base_val == 0: continue
                    for (bi, bj), coeff in A_coeffs.items():
                        i_glob = bi * n2 + i_sub
                        j_glob = bj * m2 + j_sub
                        U_new[i_glob * m + j_glob] += base_val * coeff
                        
            for j_sub in range(m2):
                for k_sub in range(p2):
                    base_val = V_base[r][j_sub * p2 + k_sub]
                    if base_val == 0: continue
                    for (bj, bk), coeff in B_coeffs.items():
                        j_glob = bj * m2 + j_sub
                        k_glob = bk * p2 + k_sub
                        V_new[j_glob * p + k_glob] += base_val * coeff
                        
            for k_sub in range(p2):
                for i_sub in range(n2):
                    base_val = W_base[r][k_sub * n2 + i_sub]
                    if base_val == 0: continue
                    for (bk, bi), coeff in C_coeffs.items():
                        k_glob = bk * p2 + k_sub
                        i_glob = bi * n2 + i_sub
                        W_new[k_glob * n + i_glob] += base_val * coeff
                        
            U_total.append(U_new)
            V_total.append(V_new)
            W_total.append(W_new)
            
    return U_total, V_total, W_total
