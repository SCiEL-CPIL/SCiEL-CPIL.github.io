DIMENSIONS = [4, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen decomposition for 2x2x2
    # Indices for 2x2: i,j in {0,1}. Index = 2*i + j
    # U2[s][2*i + j], V2[s][2*j + k], W2[s][2*k + i]
    
    # M1: (A00+A11)(B00+B11)
    u1 = [1, 0, 0, 1] # A00, A01, A10, A11
    v1 = [1, 0, 0, 1] # B00, B01, B10, B11
    w1 = [1, 0, 0, 1] # C00, C01, C10, C11 (index 2k+i: C00->0, C11->3)
    
    # M2: (A10+A11)B00
    u2 = [0, 0, 1, 1]
    v2 = [1, 0, 0, 0]
    w2 = [0, 0, 1, 0] # C10 (2*1+0=2)
    
    # M3: A00(B01-B11)
    u3 = [1, 0, 0, 0]
    v3 = [0, 1, 0, -1]
    w3 = [0, 1, 0, 0] # C01 (2*0+1=1)
    
    # M4: A11(B10-B00)
    u4 = [0, 0, 0, 1]
    v4 = [-1, 0, 1, 0]
    w4 = [1, 0, 1, 0] # C00 (0), C10 (2)
    
    # M5: (A00+A01)B11
    u5 = [1, 1, 0, 0]
    v5 = [0, 0, 0, 1]
    w5 = [-1, 1, 0, 0] # C00 (-1), C01 (1)
    
    # M6: (A10-A00)(B00+B01)
    u6 = [-1, 0, 1, 0]
    v6 = [1, 1, 0, 0]
    w6 = [0, 0, 0, 1] # C11 (3)
    
    # M7: (A01-A11)(B10+B11)
    u7 = [0, 1, 0, -1]
    v7 = [0, 0, 1, 1]
    w7 = [1, 0, 0, 0] # C00 (0)
    
    U2 = [u1, u2, u3, u4, u5, u6, u7]
    V2 = [v1, v2, v3, v4, v5, v6, v7]
    W2 = [w1, w2, w3, w4, w5, w6, w7]
    
    for s in range(7):
        for t in range(7):
            r = s * 7 + t
            for I in range(2):
                for J in range(2):
                    for K in range(2):
                        for ip in range(2):
                            for jp in range(2):
                                for kp in range(2):
                                    # U index: i*m + j. i=2I+ip, j=2J+jp
                                    i = 2 * I + ip
                                    j = 2 * J + jp
                                    u_idx = i * 4 + j
                                    
                                    # V index: j*p + k. j=2J+jp, k=2K+kp
                                    k = 2 * K + kp
                                    v_idx = j * 4 + k
                                    
                                    # W index: k*n + i. k=2K+kp, i=2I+ip
                                    w_idx = k * 4 + i
                                    
                                    # Values
                                    # U2[s] index for block (I,J) is 2I+J
                                    # U2[t] index for sub (ip,jp) is 2ip+jp
                                    val_u = U2[s][2 * I + J] * U2[t][2 * ip + jp]
                                    
                                    # V2[s] index for block (J,K) is 2J+K
                                    # V2[t] index for sub (jp,kp) is 2jp+kp
                                    val_v = V2[s][2 * J + K] * V2[t][2 * jp + kp]
                                    
                                    # W2[s] index for block (K,I) is 2K+I
                                    # W2[t] index for sub (kp,ip) is 2kp+ip
                                    val_w = W2[s][2 * K + I] * W2[t][2 * kp + ip]
                                    
                                    U[u_idx][r] = val_u
                                    V[v_idx][r] = val_v
                                    W[w_idx][r] = val_w
                                    
    return {"U": U, "V": V, "W": W}
