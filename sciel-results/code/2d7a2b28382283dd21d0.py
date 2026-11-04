import time

DIMENSIONS = [4, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin
    
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    # 1. Base 2x2x2 Strassen Decomposition
    # Rank 7
    # U, V, W are 4x7
    # Indices: i,j,k in {0,1}
    # U row: i*2+j
    # V row: j*2+k
    # W row: k*2+i
    
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # Term 0: (A00+A11)(B00+B11) -> C00+, C11+
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[3][0] = 1
    
    # Term 1: (A10+A11)B00 -> C10+, C11-
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[2][1] = 1; W2[3][1] = -1
    
    # Term 2: A00(B01-B11) -> C01+, C11+
    U2[0][2] = 1
    V2[1][2] = 1; V2[3][2] = -1
    W2[1][2] = 1; W2[3][2] = 1
    
    # Term 3: A11(B10-B00) -> C10+, C00-
    U2[3][3] = 1
    V2[2][3] = 1; V2[0][3] = -1
    W2[2][3] = 1; W2[0][3] = -1
    
    # Term 4: (A00+A01)B11 -> C00-, C01+
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    W2[0][4] = -1; W2[1][4] = 1
    
    # Term 5: (A10-A00)(B00+B01) -> C11+
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[1][5] = 1
    W2[3][5] = 1
    
    # Term 6: (A01-A11)(B10+B11) -> C00+
    U2[1][6] = 1; U2[3][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    W2[0][6] = 1
    
    # 2. Kronecker Product for 4x4x4
    rank_444 = 49
    U4 = [[0]*rank_444 for _ in range(16)]
    V4 = [[0]*rank_444 for _ in range(16)]
    W4 = [[0]*rank_444 for _ in range(16)]
    
    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2
            
            for i1 in range(2):
                for j1 in range(2):
                    # U part
                    u_val1 = U2[i1*2+j1][r1]
                    if u_val1 == 0:
                        continue
                    
                    for i2 in range(2):
                        for j2 in range(2):
                            u_val2 = U2[i2*2+j2][r2]
                            if u_val2 == 0:
                                continue
                            
                            val = u_val1 * u_val2
                            i = i1 * 2 + i2
                            j = j1 * 2 + j2
                            U4[i*4+j][r] = val
                            
                            # V part
                            # V index depends on j and k.
                            # We need to loop k1, k2
                            for k1 in range(2):
                                v_val_v1 = V2[j1*2+k1][r1]
                                if v_val_v1 == 0:
                                    continue
                                
                                for k2 in range(2):
                                    v_val_v2 = V2[j2*2+k2][r2]
                                    if v_val_v2 == 0:
                                        continue
                                    
                                    v_val = v_val_v1 * v_val_v2
                                    k = k1 * 2 + k2
                                    V4[j*4+k][r] += v_val
                                    
                                    # W part
                                    # W index depends on k and i.
                                    w_val_w1 = W2[k1*2+i1][r1]
                                    if w_val_w1 == 0:
                                        continue
                                    
                                    w_val_w2 = W2[k2*2+i2][r2]
                                    if w_val_w2 == 0:
                                        continue
                                    
                                    w_val = w_val_w1 * w_val_w2
                                    W4[k*4+i][r] += w_val

    if time.time() > deadline:
        # Return partial if timeout, though 49 is small and fast
        return {"U": U4, "V": V4, "W": W4}

    # 3. Trivial Decomposition for 5th Column (4x4x1 part)
    # C[:, 4] = A * B[:, 4]
    # Rank 16
    rank_441 = 16
    U_total = [[0]*(rank_444 + rank_441) for _ in range(16)]
    V_total = [[0]*(rank_444 + rank_441) for _ in range(20)]
    W_total = [[0]*(rank_444 + rank_441) for _ in range(20)]
    
    # Copy 4x4x4 part
    for i in range(16):
        for r in range(rank_444):
            U_total[i][r] = U4[i][r]
    
    for j in range(4):
        for k in range(4):
            for r in range(rank_444):
                V_total[j*5+k][r] = V4[j*4+k][r]
                
    for k in range(4):
        for i in range(4):
            for r in range(rank_444):
                W_total[k*4+i][r] = W4[k*4+i][r]
                
    # Add 4x4x1 part
    offset = rank_444
    for i in range(4):
        for j in range(4):
            r = offset + i * 4 + j
            U_total[i*4+j][r] = 1
            # B index: j, 4. Row in V: j*5 + 4
            V_total[j*5 + 4][r] = 1
            # C index: 4, i. Row in W: 4*4 + i
            W_total[4*4 + i][r] = 1

    return {"U": U_total, "V": V_total, "W": W_total}
