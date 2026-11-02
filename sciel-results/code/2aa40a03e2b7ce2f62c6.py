def construct():
    n, m, p = 2, 3, 8
    total_rank = 30
    U = [[0] * total_rank for _ in range(n * m)]  # 6 x 30
    V = [[0] * total_rank for _ in range(m * p)]  # 24 x 30
    W = [[0] * total_rank for _ in range(p * n)]  # 16 x 30

    # Part 1: M_{2,1,8} for j=0
    # r=0
    U[0][0] = 1  # i=0, j=0
    # U[3][0] = 0
    for k in range(8):
        V[k][0] = 1  # j=0, k=k
        W[2*k][0] = 1  # k=k, i=0
        W[2*k+1][0] = 0  # k=k, i=1
    
    # r=1
    # U[0][1] = 0
    U[3][1] = 1  # i=1, j=0
    for k in range(8):
        V[k][1] = 1
        W[2*k][1] = 0
        W[2*k+1][1] = 1

    # Part 2: M_{2,2,8} for j=1,2
    # This consists of 4 blocks of M_{2,2,2}
    
    # Define Strassen matrices for M_{2,2,2}
    # U_S: 4x7, V_S: 4x7, W_S: 4x7
    # Local indices: i in {0,1}, j' in {0,1}, k' in {0,1}
    # U_S[i*2+j', r]
    # V_S[j'*2+k', r]
    # W_S[k'*2+i, r]
    
    U_S = [[0]*7 for _ in range(4)]
    V_S = [[0]*7 for _ in range(4)]
    W_S = [[0]*7 for _ in range(4)]
    
    # r=0: m1 = (a+d)(e+h)
    U_S[0][0] = 1 # a (i=0, j'=0)
    U_S[3][0] = 1 # d (i=1, j'=1)
    V_S[0][0] = 1 # e (j'=0, k'=0)
    V_S[3][0] = 1 # h (j'=1, k'=1)
    W_S[0][0] = 1 # C_00 (k'=0, i=0)
    W_S[3][0] = 1 # C_11 (k'=1, i=1)
    
    # r=1: m2 = (c+d)e
    U_S[2][1] = 1 # c (i=1, j'=0)
    U_S[3][1] = 1 # d (i=1, j'=1)
    V_S[0][1] = 1 # e
    W_S[2][1] = 1 # C_10 (k'=0, i=1)
    W_S[3][1] = -1 # C_11
    
    # r=2: m3 = a(f-h)
    U_S[0][2] = 1 # a
    V_S[1][2] = 1 # f (j'=0, k'=1)
    V_S[3][2] = -1 # h
    W_S[1][2] = 1 # C_01 (k'=1, i=0)
    W_S[3][2] = 1 # C_11
    
    # r=3: m4 = d(g-e)
    U_S[3][3] = 1 # d
    V_S[2][3] = 1 # g (j'=1, k'=0)
    V_S[0][3] = -1 # e
    W_S[0][3] = 1 # C_00
    W_S[2][3] = 1 # C_10
    
    # r=4: m5 = (a+b)h
    U_S[0][4] = 1 # a
    U_S[1][4] = 1 # b (i=0, j'=1)
    V_S[3][4] = 1 # h
    W_S[0][4] = -1 # C_00
    W_S[1][4] = 1 # C_01
    
    # r=5: m6 = (c-a)(e+f)
    U_S[2][5] = 1 # c
    U_S[0][5] = -1 # a
    V_S[0][5] = 1 # e
    V_S[1][5] = 1 # f
    W_S[3][5] = 1 # C_11
    
    # r=6: m7 = (b-d)(g+h)
    U_S[1][6] = 1 # b
    U_S[3][6] = -1 # d
    V_S[2][6] = 1 # g
    V_S[3][6] = 1 # h
    W_S[0][6] = 1 # C_00

    # Embed 4 instances of M_{2,2,2}
    for t in range(4):
        k_off = 2 * t
        j_off = 1 # Global j starts at 1
        base_r = 2 + 7 * t
        
        for s in range(7):
            r = base_r + s
            
            # U mapping
            for i in range(2):
                for jp in range(2):
                    val = U_S[i*2+jp][s]
                    if val != 0:
                        global_u_idx = i * 3 + (j_off + jp)
                        U[global_u_idx][r] += val
                        
            # V mapping
            for jp in range(2):
                for kp in range(2):
                    val = V_S[jp*2+kp][s]
                    if val != 0:
                        global_v_idx = (j_off + jp) * 8 + (k_off + kp)
                        V[global_v_idx][r] += val
                        
            # W mapping
            for kp in range(2):
                for i in range(2):
                    val = W_S[kp*2+i][s]
                    if val != 0:
                        global_w_idx = (k_off + kp) * 2 + i
                        W[global_w_idx][r] += val
                        
    return {"U": U, "V": V, "W": W}
