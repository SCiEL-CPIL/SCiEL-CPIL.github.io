"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    # Rank 49 decomposition using tensor product of 2x2 Strassen
    n, m, p = DIMENSIONS
    rank = 49
    
    # 2x2 Strassen factors
    # U2[i*2+j, s], V2[j*2+k, s], W2[k*2+i, s]
    # s in 0..6
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # Form 0: (a+d)(e+h)
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[3][0] = 1
    
    # Form 1: (c+d)e
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[1][1] = 1 # C_10 -> i=1, k=0 -> W[0*2+1]
    
    # Form 2: a(f-h)
    U2[0][2] = 1
    V2[1][2] = 1; V2[3][2] = -1
    W2[1][2] = 1; W2[3][2] = 1 # C_01 (i=0,k=1->W[1]), C_11 (i=1,k=1->W[3])
    
    # Form 3: d(g-f)
    U2[3][3] = 1
    V2[2][3] = 1; V2[1][3] = -1
    W2[0][3] = 1; W2[1][3] = 1 # C_00 (i=0,k=0->W[0]), C_10 (i=1,k=0->W[1])
    
    # Form 4: (a+b)h
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    W2[0][4] = -1; W2[1][4] = 1 # -C_00, C_01
    
    # Form 5: (c-a)(e+f)
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[1][5] = 1
    W2[3][5] = 1 # C_11
    
    # Form 6: (b-d)(g+h)
    U2[1][6] = 1; U2[3][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    W2[0][6] = 1; W2[3][6] = 1 # C_00, C_11

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for i in range(n):
        for j in range(m):
            i_h, i_l = i // 2, i % 2
            j_h, j_l = j // 2, j % 2
            u_row = 4 * i + j
            for s1 in range(7):
                u1 = U2[2 * i_h + j_h][s1]
                if u1 == 0: continue
                for s2 in range(7):
                    u2 = U2[2 * i_l + j_l][s2]
                    if u2 == 0: continue
                    s = s1 * 7 + s2
                    U[u_row][s] = u1 * u2
    
    for j in range(m):
        for k in range(p):
            j_h, j_l = j // 2, j % 2
            k_h, k_l = k // 2, k % 2
            v_row = 4 * j + k
            for s1 in range(7):
                v1 = V2[2 * j_h + k_h][s1]
                if v1 == 0: continue
                for s2 in range(7):
                    v2 = V2[2 * j_l + k_l][s2]
                    if v2 == 0: continue
                    s = s1 * 7 + s2
                    V[v_row][s] = v1 * v2
    
    for k in range(p):
        for i in range(n):
            k_h, k_l = k // 2, k % 2
            i_h, i_l = i // 2, i % 2
            w_row = 4 * k + i
            for s1 in range(7):
                w1 = W2[2 * k_h + i_h][s1]
                if w1 == 0: continue
                for s2 in range(7):
                    w2 = W2[2 * k_l + i_l][s2]
                    if w2 == 0: continue
                    s = s1 * 7 + s2
                    W[w_row][s] = w1 * w2

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
