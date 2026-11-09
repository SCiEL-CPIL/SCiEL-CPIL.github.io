import numpy as np

def construct():
    n, m, p = 6, 6, 6
    
    # Strassen 2x2 factors
    # U1: 4x7, V1: 4x7, W1: 4x7
    U1 = [[0]*7 for _ in range(4)]
    V1 = [[0]*7 for _ in range(4)]
    W1 = [[0]*7 for _ in range(4)]
    
    # c=0
    U1[0][0] = 1; U1[3][0] = 1
    V1[0][0] = 1; V1[3][0] = 1
    W1[0][0] = 1; W1[3][0] = 1
    # c=1
    U1[2][1] = 1; U1[3][1] = 1
    V1[0][1] = 1
    W1[2][1] = 1; W1[3][1] = -1
    # c=2
    U1[0][2] = 1
    V1[1][2] = 1; V1[3][2] = -1
    W1[2][2] = 1; W1[3][2] = 1
    # c=3
    U1[3][3] = 1
    V1[2][3] = 1; V1[0][3] = -1
    W1[0][3] = 1; W1[2][3] = 1
    # c=4
    U1[0][4] = 1; U1[1][4] = 1
    V1[3][4] = 1
    W1[0][4] = -1; W1[2][4] = 1
    # c=5
    U1[2][5] = 1; U1[0][5] = -1
    V1[0][5] = 1; V1[1][5] = 1
    W1[3][5] = 1
    # c=6
    U1[1][6] = 1; U1[3][6] = -1
    V1[2][6] = 1; V1[3][6] = 1
    W1[0][6] = 1
    
    # Schoolbook 3x3 factors
    # U2: 9x27, V2: 9x27, W2: 9x27
    U2 = [[0]*27 for _ in range(9)]
    V2 = [[0]*27 for _ in range(9)]
    W2 = [[0]*27 for _ in range(9)]
    
    for i2 in range(3):
        for j2 in range(3):
            for k2 in range(3):
                c2 = i2 * 9 + j2 * 3 + k2
                U2[i2 * 3 + j2][c2] = 1
                V2[j2 * 3 + k2][c2] = 1
                W2[k2 * 3 + i2][c2] = 1
                
    # Kronecker Product Construction
    rank = 7 * 27
    U = [[0] * rank for _ in range(36)]
    V = [[0] * rank for _ in range(36)]
    W = [[0] * rank for _ in range(36)]
    
    for i1 in range(2):
        for i2 in range(3):
            i = i1 * 3 + i2
            for j1 in range(2):
                for j2 in range(3):
                    j = j1 * 3 + j2
                    row_u = i * 6 + j
                    for c1 in range(7):
                        val_u1 = U1[i1 * 2 + j1][c1]
                        if val_u1 == 0: continue
                        for c2 in range(27):
                            val_u2 = U2[i2 * 3 + j2][c2]
                            if val_u2 == 0: continue
                            c = c1 * 27 + c2
                            U[row_u][c] = val_u1 * val_u2
                            
    for j1 in range(2):
        for j2 in range(3):
            j = j1 * 3 + j2
            for k1 in range(2):
                for k2 in range(3):
                    k = k1 * 3 + k2
                    row_v = j * 6 + k
                    for c1 in range(7):
                        val_v1 = V1[j1 * 2 + k1][c1]
                        if val_v1 == 0: continue
                        for c2 in range(27):
                            val_v2 = V2[j2 * 3 + k2][c2]
                            if val_v2 == 0: continue
                            c = c1 * 27 + c2
                            V[row_v][c] = val_v1 * val_v2
                            
    for k1 in range(2):
        for k2 in range(3):
            k = k1 * 3 + k2
            for i1 in range(2):
                for i2 in range(3):
                    i = i1 * 3 + i2
                    row_w = k * 6 + i
                    for c1 in range(7):
                        val_w1 = W1[k1 * 2 + i1][c1]
                        if val_w1 == 0: continue
                        for c2 in range(27):
                            val_w2 = W2[k2 * 3 + i2][c2]
                            if val_w2 == 0: continue
                            c = c1 * 27 + c2
                            W[row_w][c] = val_w1 * val_w2

    return {"U": U, "V": V, "W": W}
