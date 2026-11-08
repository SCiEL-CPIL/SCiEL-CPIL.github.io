import time
import numpy as np

DIMENSIONS = [3, 3, 7]

def construct():
    start_time = time.time()
    deadline = 150.0
    safety_margin = 5.0
    
    n, m, p = DIMENSIONS
    rank = 60
    
    U = [0.0] * (n * m * rank)
    V = [0.0] * (m * p * rank)
    W = [0.0] * (p * n * rank)
    
    col = 0
    
    def add_rank1_term(u_vec, v_vec, w_vec):
        nonlocal col
        if col >= rank:
            return
        # u_vec is length n*m
        # v_vec is length m*p
        # w_vec is length p*n
        for i in range(n * m):
            U[i * rank + col] += u_vec[i]
        for i in range(m * p):
            V[i * rank + col] += v_vec[i]
        for i in range(p * n):
            W[i * rank + col] += w_vec[i]
        col += 1

    def u_idx(i, j): return i * m + j
    def v_idx(i, j): return i * p + j
    def w_idx(i, j): return j * n + i

    def add_strassen(a_base_i, b_base_i, c_base_i, k_base):
        a00 = u_idx(a_base_i, 0)
        a01 = u_idx(a_base_i, 1)
        a10 = u_idx(a_base_i + 1, 0)
        a11 = u_idx(a_base_i + 1, 1)
        
        b00 = v_idx(b_base_i, k_base)
        b01 = v_idx(b_base_i, k_base + 1)
        b10 = v_idx(b_base_i + 1, k_base)
        b11 = v_idx(b_base_i + 1, k_base + 1)
        
        c00 = w_idx(c_base_i, k_base)
        c01 = w_idx(c_base_i, k_base + 1)
        c10 = w_idx(c_base_i + 1, k_base)
        c11 = w_idx(c_base_i + 1, k_base + 1)
        
        # m1: (a00+a11)(b00+b11)
        U1 = [0.0]*(n*m); V1 = [0.0]*(m*p); W1 = [0.0]*(p*n)
        U1[a00] = 1; U1[a11] = 1
        V1[b00] = 1; V1[b11] = 1
        W1[c00] = 1; W1[c11] = 1
        add_rank1_term(U1, V1, W1)
        
        # m2: (a10+a11)b00
        U2 = [0.0]*(n*m); V2 = [0.0]*(m*p); W2 = [0.0]*(p*n)
        U2[a10] = 1; U2[a11] = 1
        V2[b00] = 1
        W2[c10] = 1; W2[c11] = -1
        add_rank1_term(U2, V2, W2)
        
        # m3: a00(b01-b11)
        U3 = [0.0]*(n*m); V3 = [0.0]*(m*p); W3 = [0.0]*(p*n)
        U3[a00] = 1
        V3[b01] = 1; V3[b11] = -1
        W3[c01] = 1; W3[c11] = 1
        add_rank1_term(U3, V3, W3)
        
        # m4: a11(b10-b00)
        U4 = [0.0]*(n*m); V4 = [0.0]*(m*p); W4 = [0.0]*(p*n)
        U4[a11] = 1
        V4[b10] = 1; V4[b00] = -1
        W4[c00] = 1; W4[c10] = 1
        add_rank1_term(U4, V4, W4)
        
        # m5: (a00+a01)b11
        U5 = [0.0]*(n*m); V5 = [0.0]*(m*p); W5 = [0.0]*(p*n)
        U5[a00] = 1; U5[a01] = 1
        V5[b11] = 1
        W5[c00] = -1; W5[c01] = 1
        add_rank1_term(U5, V5, W5)
        
        # m6: (a10-a00)(b00+b01)
        U6 = [0.0]*(n*m); V6 = [0.0]*(m*p); W6 = [0.0]*(p*n)
        U6[a10] = 1; U6[a00] = -1
        V6[b00] = 1; V6[b01] = 1
        W6[c11] = 1
        add_rank1_term(U6, V6, W6)
        
        # m7: (a01-a11)(b10+b11)
        U7 = [0.0]*(n*m); V7 = [0.0]*(m*p); W7 = [0.0]*(p*n)
        U7[a01] = 1; U7[a11] = -1
        V7[b10] = 1; V7[b11] = 1
        W7[c00] = 1
        add_rank1_term(U7, V7, W7)

    def add_schoolbook_2x2x1(a_base_i, b_base_i, c_base_i, k_base):
        for i in range(2):
            for j in range(2):
                U_term = [0.0] * (n*m); V_term = [0.0] * (m*p); W_term = [0.0] * (p*n)
                U_term[u_idx(a_base_i + i, j)] = 1
                V_term[v_idx(b_base_i + j, k_base)] = 1
                W_term[w_idx(c_base_i + i, k_base)] = 1
                add_rank1_term(U_term, V_term, W_term)

    # 1. M(2,2,7) part
    add_strassen(0, 0, 0, 0)
    add_strassen(0, 0, 0, 2)
    add_strassen(0, 0, 0, 4)
    add_schoolbook_2x2x1(0, 0, 0, 6)
    
    # 2. M(1,2,7) part
    for i in range(1):
        for j in range(2):
            for k in range(7):
                U_term = [0.0] * (n*m); V_term = [0.0] * (m*p); W_term = [0.0] * (p*n)
                U_term[u_idx(2, j)] = 1
                V_term[v_idx(j, k)] = 1
                W_term[w_idx(2, k)] = 1
                add_rank1_term(U_term, V_term, W_term)

    # 3. M(3,1,7) part
    for i in range(3):
        for k in range(7):
            U_term = [0.0] * (n*m); V_term = [0.0] * (m*p); W_term = [0.0] * (p*n)
            U_term[u_idx(i, 2)] = 1
            V_term[v_idx(2, k)] = 1
            W_term[w_idx(i, k)] = 1
            add_rank1_term(U_term, V_term, W_term)

    # Convert to lists of lists for return
    U_out = [[U[i * rank + r] for r in range(rank)] for i in range(n * m)]
    V_out = [[V[i * rank + r] for r in range(rank)] for i in range(m * p)]
    W_out = [[W[i * rank + r] for r in range(rank)] for i in range(p * n)]

    return {"U": U_out, "V": V_out, "W": W_out}
