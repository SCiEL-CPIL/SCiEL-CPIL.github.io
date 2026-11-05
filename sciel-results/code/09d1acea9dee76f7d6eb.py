import time

def construct():
    start_time = time.time()
    
    # Strassen 2x2 decomposition (rank 7)
    # Rows indexed by i*2+j for i,j in {0,1}
    U_s = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1,  0, 1],
        [0, 1, 0, 0, 0,  1, 0],
        [1, 1, 0, 1, 0,  0, -1]
    ]
    
    V_s = [
        [1, 1,  0, -1, 0, 1, 0],
        [0, 0,  1,  0, 0, 0, 1],
        [0, 0,  0,  1, 0, 1, 0],
        [1, 0, -1,  0, 1, 0, 1]
    ]
    
    W_s = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1,  0, 0],
        [0, 1, 0, 1, 0,  0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    
    # Schoolbook 3x3 decomposition (rank 27)
    n3 = 3
    rank3 = 27
    U_3 = [[0]*rank3 for _ in range(9)]
    V_3 = [[0]*rank3 for _ in range(9)]
    W_3 = [[0]*rank3 for _ in range(9)]
    
    col = 0
    for i in range(n3):
        for j in range(n3):
            for k in range(n3):
                U_3[i*n3 + j][col] = 1
                V_3[j*n3 + k][col] = 1
                W_3[k*n3 + i][col] = 1
                col += 1
    
    # Tensor product: 6x6 = Strassen(2x2) x Schoolbook(3x3)
    rank6 = 7 * 27  # 189
    
    U_6 = [[0]*rank6 for _ in range(36)]
    V_6 = [[0]*rank6 for _ in range(36)]
    W_6 = [[0]*rank6 for _ in range(36)]
    
    # U_6: row = i*6+j, i=a*3+b, j=d*3+e
    # col = s*27+t
    # U_6[row, col] = U_s[a*2+d][s] * U_3[b*3+e][t]
    for a in range(2):
        for d in range(2):
            for b in range(3):
                for e in range(3):
                    i = a*3 + b
                    j = d*3 + e
                    row_u = i*6 + j
                    for s in range(7):
                        us_val = U_s[a*2+d][s]
                        if us_val == 0:
                            continue
                        for t in range(27):
                            u3_val = U_3[b*3+e][t]
                            if u3_val == 0:
                                continue
                            col6 = s*27 + t
                            U_6[row_u][col6] = us_val * u3_val
    
    # V_6: row = j*6+k, j=d*3+e, k=f*3+g
    # V_6[row, col] = V_s[d*2+f][s] * V_3[e*3+g][t]
    for d in range(2):
        for f in range(2):
            for e in range(3):
                for g in range(3):
                    j = d*3 + e
                    k = f*3 + g
                    row_v = j*6 + k
                    for s in range(7):
                        vs_val = V_s[d*2+f][s]
                        if vs_val == 0:
                            continue
                        for t in range(27):
                            v3_val = V_3[e*3+g][t]
                            if v3_val == 0:
                                continue
                            col6 = s*27 + t
                            V_6[row_v][col6] = vs_val * v3_val
    
    # W_6: row = k*6+i, k=f*3+g, i=a*3+b
    # W_6[row, col] = W_s[f*2+a][s] * W_3[g*3+b][t]
    for f in range(2):
        for a in range(2):
            for g in range(3):
                for b in range(3):
                    k = f*3 + g
                    i = a*3 + b
                    row_w = k*6 + i
                    for s in range(7):
                        ws_val = W_s[f*2+a][s]
                        if ws_val == 0:
                            continue
                        for t in range(27):
                            w3_val = W_3[g*3+b][t]
                            if w3_val == 0:
                                continue
                            col6 = s*27 + t
                            W_6[row_w][col6] = ws_val * w3_val
    
    return {"U": U_6, "V": V_6, "W": W_6}
