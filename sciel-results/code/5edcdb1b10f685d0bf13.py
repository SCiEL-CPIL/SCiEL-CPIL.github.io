import time

DIMENSIONS = [4, 4, 8]

def construct():
    start_time = time.time()
    
    # Strassen decomposition for 2x2x2x2, rank 7
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # Term 0: (a+d)(e+h) -> C11(+1), C22(+1)
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[3][0] = 1
    
    # Term 1: (c+d)e -> C21(+1), C22(-1)
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[1][1] = 1; W2[3][1] = -1
    
    # Term 2: a(f-h) -> C12(+1), C22(+1)
    U2[0][2] = 1
    V2[1][2] = 1; V2[3][2] = -1
    W2[2][2] = 1; W2[3][2] = 1
    
    # Term 3: d(g-e) -> C11(+1), C21(+1)
    U2[3][3] = 1
    V2[2][3] = 1; V2[0][3] = -1
    W2[0][3] = 1; W2[1][3] = 1
    
    # Term 4: (a+b)h -> C11(-1), C12(+1)
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    W2[0][4] = -1; W2[2][4] = 1
    
    # Term 5: (c-a)(e+f) -> C22(+1)
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[1][5] = 1
    W2[3][5] = 1
    
    # Term 6: (b-d)(g+h) -> C11(+1)
    U2[1][6] = 1; U2[3][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    W2[0][6] = 1
    
    # Build 4x4x4x4 decomposition, rank 49
    U4 = [[0]*49 for _ in range(16)]
    V4 = [[0]*49 for _ in range(16)]
    W4 = [[0]*49 for _ in range(16)]
    
    for c2 in range(7):
        for c1 in range(7):
            c = c2 * 7 + c1
            for i2 in range(2):
                for j2 in range(2):
                    for i1 in range(2):
                        for j1 in range(2):
                            i4 = i2*2 + i1
                            j4 = j2*2 + j1
                            U4[i4*4+j4][c] = U2[i2*2+j2][c2] * U2[i1*2+j1][c1]
            for j2 in range(2):
                for k2 in range(2):
                    for j1 in range(2):
                        for k1 in range(2):
                            j4 = j2*2 + j1
                            k4 = k2*2 + k1
                            V4[j4*4+k4][c] = V2[j2*2+k2][c2] * V2[j1*2+k1][c1]
            for k2 in range(2):
                for i2 in range(2):
                    for k1 in range(2):
                        for i1 in range(2):
                            k4 = k2*2 + k1
                            i4 = i2*2 + i1
                            W4[k4*4+i4][c] = W2[k2*2+i2][c2] * W2[k1*2+i1][c1]
    
    # Build 4x4x4x8 decomposition, rank 98
    U = [[0]*98 for _ in range(16)]
    V = [[0]*98 for _ in range(32)]
    W = [[0]*98 for _ in range(32)]
    
    for c in range(49):
        c2 = c + 49
        for i in range(16):
            U[i][c] = U4[i][c]
            U[i][c2] = U4[i][c]
        for j4 in range(16):
            j_row = j4 // 4
            j_col = j4 % 4
            V[j_row*8 + j_col][c] = V4[j4][c]
            V[j_row*8 + j_col + 4][c2] = V4[j4][c]
        for k4 in range(16):
            k_row = k4 // 4
            k_col = k4 % 4
            W[k4][c] = W4[k4][c]
            W[(k_row+4)*4 + k_col][c2] = W4[k4][c]
    
    return {"U": U, "V": V, "W": W}
