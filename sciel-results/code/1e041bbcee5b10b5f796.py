"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    # Strassen decomposition for 2x2x2
    # U, V, W shapes (4, 7)
    U_222 = [[0]*7 for _ in range(4)]
    V_222 = [[0]*7 for _ in range(4)]
    W_222 = [[0]*7 for _ in range(4)]
    
    # r=0: M1 = A00(B00+B11)
    U_222[0][0] = 1
    V_222[0][0] = 1
    V_222[3][0] = 1
    W_222[0][0] = 1
    W_222[3][0] = -1
    
    # r=1: M2 = (A10+A11)B00
    U_222[2][1] = 1
    U_222[3][1] = 1
    V_222[0][1] = 1
    W_222[2][1] = 1
    W_222[3][1] = -1
    
    # r=2: M3 = (A00-A10)(B10+B11)
    U_222[0][2] = 1
    U_222[2][2] = -1
    V_222[2][2] = 1
    V_222[3][2] = 1
    W_222[1][2] = 1
    W_222[3][2] = 1
    
    # r=3: M4 = A11(B01-B10)
    U_222[3][3] = 1
    V_222[1][3] = -1
    V_222[2][3] = 1
    W_222[0][3] = 1
    W_222[2][3] = 1
    
    # r=4: M5 = (A00+A01)B11
    U_222[0][4] = 1
    U_222[1][4] = 1
    V_222[3][4] = 1
    W_222[0][4] = -1
    W_222[1][4] = 1
    
    # r=5: M6 = (A01-A11)(B00+B10)
    U_222[1][5] = 1
    U_222[3][5] = -1
    V_222[0][5] = 1
    V_222[2][5] = 1
    W_222[3][5] = 1
    
    # r=6: M7 = (A10-A00)(B01+B11)
    U_222[2][6] = -1
    U_222[0][6] = 1
    V_222[1][6] = 1
    V_222[3][6] = 1
    W_222[0][6] = 1

    # 2x2x4 tensor (Rank 14)
    # U_224 shape (4, 14)
    # V_224 shape (8, 14)
    # W_224 shape (8, 14)
    U_224 = [[0]*14 for _ in range(4)]
    V_224 = [[0]*14 for _ in range(8)]
    W_224 = [[0]*14 for _ in range(8)]
    
    for r in range(7):
        # First half (k in 0,1)
        for i in range(2):
            for j in range(2):
                U_224[i*2+j][r] = U_222[i*2+j][r]
        for j in range(2):
            for k in range(2):
                V_224[j*2+k][r] = V_222[j*2+k][r]
        for k in range(2):
            for i in range(2):
                W_224[k*2+i][r] = W_222[k*2+i][r]
                
        # Second half (k in 2,3)
        r2 = r + 7
        for i in range(2):
            for j in range(2):
                U_224[i*2+j][r2] = U_222[i*2+j][r]
        for j in range(2):
            for k in range(2):
                V_224[j*2+(k+2)][r2] = V_222[j*2+k][r]
        for k in range(2):
            for i in range(2):
                W_224[(k+2)*2+i][r2] = W_222[k*2+i][r]

    # Full 4x4x8 tensor (Rank 98)
    # U shape (16, 98)
    # V shape (32, 98)
    # W shape (32, 98)
    n, m, p = 4, 4, 8
    rank = 98
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    for rA in range(7):
        for rB in range(14):
            r = rA * 14 + rB
            
            for iA in range(2):
                for jA in range(2):
                    uA = U_222[iA*2+jA][rA]
                    if uA != 0:
                        for iB in range(2):
                            for jB in range(2):
                                uB = U_224[iB*2+jB][rB]
                                if uB != 0:
                                    I = iA*2 + iB
                                    J = jA*2 + jB
                                    U[I*m+J][r] = uA * uB
                                    
            for jA in range(2):
                for kA in range(2):
                    vA = V_222[jA*2+kA][rA]
                    if vA != 0:
                        for jB in range(2):
                            for kB in range(4):
                                vB = V_224[jB*2+kB][rB]
                                if vB != 0:
                                    J = jA*2 + jB
                                    K = kA*4 + kB
                                    V[J*p+K][r] = vA * vB
                                    
            for kA in range(2):
                for iA in range(2):
                    wA = W_222[kA*2+iA][rA]
                    if wA != 0:
                        for kB in range(4):
                            for iB in range(2):
                                wB = W_224[kB*2+iB][rB]
                                if wB != 0:
                                    K = kA*4 + kB
                                    I = iA*2 + iB
                                    W[K*n+I][r] = wA * wB

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
