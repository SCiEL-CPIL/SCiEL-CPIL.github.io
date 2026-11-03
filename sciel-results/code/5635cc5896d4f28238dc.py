import time

def construct():
    n, m, p = 4, 5, 5
    rank = 73
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a block
    def add_block(bU, bV, bW, i_off, k_off, j_off, r_off, b_n, b_m, b_p):
        # bU: b_n * b_m x b_rank
        # bV: b_m * b_p x b_rank
        # bW: b_p * b_n x b_rank
        b_rank = bU[0].__len__() if bU else 0
        
        for li in range(b_n):
            for lk in range(b_m):
                for r in range(b_rank):
                    U[(i_off + li) * m + (k_off + lk)][r_off + r] += bU[li * b_m + lk][r]
                    
        for lk in range(b_m):
            for lj in range(b_p):
                for r in range(b_rank):
                    V[(k_off + lk) * p + (j_off + lj)][r_off + r] += bV[lk * b_p + lj][r]
                    
        for lj in range(b_p):
            for li in range(b_n):
                for r in range(b_rank):
                    W[(j_off + lj) * n + (i_off + li)][r_off + r] += bW[lj * b_n + li][r]

    # Strassen 2x2x2
    def get_strassen():
        U_s = [[0]*7 for _ in range(4)]
        V_s = [[0]*7 for _ in range(4)]
        W_s = [[0]*7 for _ in range(4)]
        
        # M1
        U_s[0, 0] = 1; U_s[3, 0] = 1
        V_s[0, 0] = 1; V_s[3, 0] = 1
        W_s[0, 0] = 1; W_s[3, 0] = 1
        W_s[1, 0] = -1; W_s[2, 0] = -1
        
        # M2
        U_s[2, 1] = 1; U_s[3, 1] = 1
        V_s[0, 1] = 1
        W_s[2, 1] = 1; W_s[3, 1] = -1
        
        # M3
        U_s[0, 2] = 1
        V_s[2, 2] = 1; V_s[3, 2] = -1
        W_s[1, 2] = 1; W_s[3, 2] = 1
        
        # M4
        U_s[3, 3] = 1
        V_s[1, 3] = 1; V_s[0, 3] = -1
        W_s[0, 3] = 1; W_s[2, 3] = 1
        
        # M5
        U_s[0, 4] = 1; U_s[1, 4] = 1
        V_s[3, 4] = 1
        W_s[0, 4] = -1; W_s[1, 4] = 1
        
        # M6
        U_s[2, 5] = 1; U_s[0, 5] = -1
        V_s[0, 5] = 1; V_s[1, 5] = 1
        W_s[3, 5] = 1
        
        # M7
        U_s[1, 6] = 1; U_s[3, 6] = -1
        V_s[2, 6] = 1; V_s[3, 6] = 1
        W_s[0, 6] = 1; W_s[3, 6] = 1
        
        return U_s, V_s, W_s

    # Trivial 2x2x1 (Rank 4)
    def get_trivial_2x2x1():
        U_t = [[0]*4 for _ in range(4)]
        V_t = [[0]*4 for _ in range(2)] # 2x1 -> 2 entries
        W_t = [[0]*4 for _ in range(2)] # 1x2 -> 2 entries
        
        # C0 = A00 B00 + A01 B10
        U_t[0, 0] = 1; V_t[0, 0] = 1; W_t[0, 0] = 1
        U_t[1, 1] = 1; V_t[1, 1] = 1; W_t[0, 1] = 1
        
        # C1 = A10 B00 + A11 B10
        U_t[2, 2] = 1; V_t[0, 2] = 1; W_t[1, 2] = 1
        U_t[3, 3] = 1; V_t[1, 3] = 1; W_t[1, 3] = 1
        
        return U_t, V_t, W_t

    sU, sV, sW = get_strassen()
    tU, tV, tW = get_trivial_2x2x1()
    
    r_off = 0
    
    # Term 1: k in {0,1}
    # Subterm 1.1: i in {0,1}
    # 1.1.1: j in {0,1}
    add_block(sU, sV, sW, 0, 0, 0, r_off, 2, 2, 2)
    r_off += 7
    # 1.1.2: j in {2,3}
    add_block(sU, sV, sW, 0, 0, 2, r_off, 2, 2, 2)
    r_off += 7
    # 1.1.3: j in {4}
    add_block(tU, tV, tW, 0, 0, 4, r_off, 2, 2, 1)
    r_off += 4
    
    # Subterm 1.2: i in {2,3}
    # 1.2.1: j in {0,1}
    add_block(sU, sV, sW, 2, 0, 0, r_off, 2, 2, 2)
    r_off += 7
    # 1.2.2: j in {2,3}
    add_block(sU, sV, sW, 2, 0, 2, r_off, 2, 2, 2)
    r_off += 7
    # 1.2.3: j in {4}
    add_block(tU, tV, tW, 2, 0, 4, r_off, 2, 2, 1)
    r_off += 4
    
    # Term 2: k in {2,3}
    # Subterm 2.1: i in {0,1}
    # 2.1.1: j in {0,1}
    add_block(sU, sV, sW, 0, 2, 0, r_off, 2, 2, 2)
    r_off += 7
    # 2.1.2: j in {2,3}
    add_block(sU, sV, sW, 0, 2, 2, r_off, 2, 2, 2)
    r_off += 7
    # 2.1.3: j in {4}
    add_block(tU, tV, tW, 0, 2, 4, r_off, 2, 2, 1)
    r_off += 4
    
    # Subterm 2.2: i in {2,3}
    # 2.2.1: j in {0,1}
    add_block(sU, sV, sW, 2, 2, 0, r_off, 2, 2, 2)
    r_off += 7
    # 2.2.2: j in {2,3}
    add_block(sU, sV, sW, 2, 2, 2, r_off, 2, 2, 2)
    r_off += 7
    # 2.2.3: j in {4}
    add_block(tU, tV, tW, 2, 2, 4, r_off, 2, 2, 1)
    r_off += 4
    
    # Term 3: k in {4}
    # Rank 1
    for i in range(4):
        U[i * 5 + 4][r_off] = 1
    for j in range(5):
        V[4 * 5 + j][r_off] = 1
    for j in range(5):
        for i in range(4):
            W[j * 4 + i][r_off] = 1
    r_off += 1
    
    return {"U": U, "V": V, "W": W}

