# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Strassen decomposition for 2x2
    # U2, V2, W2 are 4x7 matrices
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # M1 = (A00+A11)(B00+B11)
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[3][0] = 1
    
    # M2 = (A10+A11)B00
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[1][1] = 1; W2[3][1] = -1
    
    # M3 = A00(B01-B11)
    U2[0][2] = 1
    V2[1][2] = 1; V2[3][2] = -1
    W2[2][2] = 1; W2[3][2] = 1
    
    # M4 = A11(B10-B00)
    U2[3][3] = 1
    V2[2][3] = 1; V2[0][3] = -1
    W2[0][3] = 1; W2[1][3] = 1
    
    # M5 = (A00+A01)B11
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    W2[0][4] = -1; W2[2][4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[1][5] = 1
    W2[3][5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U2[1][6] = 1; U2[3][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    W2[0][6] = 1
    
    rank = 49
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # Construct Kronecker products
    for i in range(n):
        for j in range(m):
            i1 = i // 2
            i2 = i % 2
            j1 = j // 2
            j2 = j % 2
            row_u2_1 = i1 * 2 + j1
            row_u2_2 = i2 * 2 + j2
            for c1 in range(7):
                u1 = U2[row_u2_1][c1]
                if u1 == 0: continue
                for c2 in range(7):
                    u2 = U2[row_u2_2][c2]
                    if u2 == 0: continue
                    c = c1 * 7 + c2
                    U[i*m + j][c] = u1 * u2

    for j in range(m):
        for k in range(p):
            j1 = j // 2
            j2 = j % 2
            k1 = k // 2
            k2 = k % 2
            row_v2_1 = j1 * 2 + k1
            row_v2_2 = j2 * 2 + k2
            for c1 in range(7):
                v1 = V2[row_v2_1][c1]
                if v1 == 0: continue
                for c2 in range(7):
                    v2 = V2[row_v2_2][c2]
                    if v2 == 0: continue
                    c = c1 * 7 + c2
                    V[j*p + k][c] = v1 * v2

    for k in range(p):
        for i in range(n):
            k1 = k // 2
            k2 = k % 2
            i1 = i // 2
            i2 = i % 2
            row_w2_1 = k1 * 2 + i1
            row_w2_2 = k2 * 2 + i2
            for c1 in range(7):
                w1 = W2[row_w2_1][c1]
                if w1 == 0: continue
                for c2 in range(7):
                    w2 = W2[row_w2_2][c2]
                    if w2 == 0: continue
                    c = c1 * 7 + c2
                    W[k*n + i][c] = w1 * w2

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
