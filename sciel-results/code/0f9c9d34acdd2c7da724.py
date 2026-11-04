"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Strassen factors for 2x2
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # r=0: M1
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[3][0] = 1
    
    # r=1: M2
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[2][1] = 1; W2[3][1] = -1
    
    # r=2: M3
    U2[0][2] = 1
    V2[2][2] = 1; V2[3][2] = -1
    W2[1][2] = 1; W2[3][2] = 1
    
    # r=3: M4
    U2[3][3] = 1
    V2[1][3] = 1; V2[0][3] = -1
    W2[0][3] = 1; W2[2][3] = 1
    
    # r=4: M5
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    W2[0][4] = -1; W2[1][4] = 1
    
    # r=5: M6
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[2][5] = 1
    W2[3][5] = 1
    
    # r=6: M7
    U2[1][6] = 1; U2[3][6] = -1
    V2[3][6] = 1; V2[1][6] = 1
    W2[0][6] = 1

    # Schoolbook factors for 3x3
    rank3 = 27
    U3 = [[0]*rank3 for _ in range(9)]
    V3 = [[0]*rank3 for _ in range(9)]
    W3 = [[0]*rank3 for _ in range(9)]
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U3[i*3+j][col] = 1
                V3[j*3+k][col] = 1
                W3[k*3+i][col] = 1
                col += 1

    # Combine
    rank = rank3 * 7
    U = [[0]*rank for _ in range(36)]
    V = [[0]*rank for _ in range(36)]
    W = [[0]*rank for _ in range(36)]
    
    for r3 in range(rank3):
        for r2 in range(7):
            r = r3 * 7 + r2
            for i1 in range(3):
                for j1 in range(3):
                    for k1 in range(3):
                        u3_val = U3[i1*3+j1][r3]
                        v3_val = V3[j1*3+k1][r3]
                        w3_val = W3[k1*3+i1][r3]
                        
                        for i2 in range(2):
                            for j2 in range(2):
                                u2_val = U2[i2*2+j2][r2]
                                if u3_val != 0 and u2_val != 0:
                                    i = i1*2 + i2
                                    j = j1*2 + j2
                                    U[i*6+j][r] = u3_val * u2_val
                                    
                        for j2 in range(2):
                            for k2 in range(2):
                                v2_val = V2[j2*2+k2][r2]
                                if v3_val != 0 and v2_val != 0:
                                    j = j1*2 + j2
                                    k = k1*2 + k2
                                    V[j*6+k][r] = v3_val * v2_val
                                    
                        for k2 in range(2):
                            for i2 in range(2):
                                w2_val = W2[k2*2+i2][r2]
                                if w3_val != 0 and w2_val != 0:
                                    k = k1*2 + k2
                                    i = i1*2 + i2
                                    W[k*6+i][r] = w3_val * w2_val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
