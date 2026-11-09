"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    # Rank 7 decomposition for 2x2x2 matrix multiplication (Strassen)
    # U, V, W are 4x7
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # Strassen coefficients
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Indexing: U[i*2+j, k], V[j*2+k, k], W[k*2+i, k]
    # r=0: M1
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[3][0] = 1 # C00=1, C11=1? No, C00=M1, C11=M1.
    # W index is k*2+i.
    # C00: i=0, k=0 -> idx 0. C11: i=1, k=1 -> idx 3.
    # M1 contributes to C00 and C11.
    W2[0][0] = 1
    W2[3][0] = 1
    
    # r=1: M2
    U2[2][1] = 1; U2[3][1] = 1 # A10, A11
    V2[0][1] = 1 # B00
    W2[2][1] = 1 # C10 (i=1, k=0 -> idx 2)
    
    # r=2: M3
    U2[0][2] = 1 # A00
    V2[1][2] = 1; V2[3][2] = -1 # B01, B11
    W2[1][2] = 1 # C01 (i=0, k=1 -> idx 1)
    
    # r=3: M4
    U2[3][3] = 1 # A11
    V2[2][3] = 1; V2[0][3] = -1 # B10, B00
    W2[0][3] = 1 # C00 (i=0, k=0 -> idx 0)
    W2[2][3] = 1 # C10 (i=1, k=0 -> idx 2)
    
    # r=4: M5
    U2[0][4] = 1; U2[1][4] = 1 # A00, A01
    V2[3][4] = 1 # B11
    W2[0][4] = -1 # C00
    W2[1][4] = 1 # C01
    
    # r=5: M6
    U2[2][5] = 1; U2[0][5] = -1 # A10, A00
    V2[0][5] = 1; V2[1][5] = 1 # B00, B01
    W2[3][5] = 1 # C11 (i=1, k=1 -> idx 3)
    
    # r=6: M7
    U2[1][6] = 1; U2[3][6] = -1 # A01, A11
    V2[2][6] = 1; V2[3][6] = 1 # B10, B11
    W2[0][6] = 1 # C00
    W2[3][6] = 1 # C11

    # Construct 4x4x4 decomposition of rank 49
    # U4, V4, W4 are 16x49
    rank44 = 49
    U4 = [[0]*rank44 for _ in range(16)]
    V4 = [[0]*rank44 for _ in range(16)]
    W4 = [[0]*rank44 for _ in range(16)]
    
    for r in range(7):
        for rp in range(7):
            col = r * 7 + rp
            # U4 index: i*4+j. i = I*2+ip, j = J*2+jp
            # U2 index: I*2+J, rp
            # U2[ip*2+jp, rp]
            for I in range(2):
                for J in range(2):
                    u_block = U2[I*2+J][r]
                    if u_block != 0:
                        for ip in range(2):
                            for jp in range(2):
                                u_local = U2[ip*2+jp][rp]
                                if u_local != 0:
                                    i = I*2 + ip
                                    j = J*2 + jp
                                    U4[i*4+j][col] = u_block * u_local
                                    
            # V4 index: j*4+k. j = J*2+jp, k = K*2+kp
            # V2 index: J*2+K, r
            # V2[jp*2+kp, rp]
            for J in range(2):
                for K in range(2):
                    v_block = V2[J*2+K][r]
                    if v_block != 0:
                        for jp in range(2):
                            for kp in range(2):
                                v_local = V2[jp*2+kp][rp]
                                if v_local != 0:
                                    j = J*2 + jp
                                    k = K*2 + kp
                                    V4[j*4+k][col] = v_block * v_local
                                    
            # W4 index: k*4+i. k = K*2+kp, i = I*2+ip
            # W2 index: K*2+I, r
            # W2[kp*2+ip, rp]
            for K in range(2):
                for I in range(2):
                    w_block = W2[K*2+I][r]
                    if w_block != 0:
                        for kp in range(2):
                            for ip in range(2):
                                w_local = W2[kp*2+ip][rp]
                                if w_local != 0:
                                    k = K*2 + kp
                                    i = I*2 + ip
                                    W4[k*4+i][col] = w_block * w_local

    # Construct 4x4x8 decomposition of rank 98
    # Split B into B1 and B2
    # U8 is 16x98, V8 is 32x98, W8 is 32x98
    n, m, p = 4, 4, 8
    rank = 98
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # First half: B1 (cols 0-3)
    for i in range(n*m):
        for c in range(49):
            U[i][c] = U4[i][c]
    for j in range(m): # j in 0..3
        for k in range(4): # k in 0..3 (part of B1)
            for c in range(49):
                V[j*p + k][c] = V4[j*4+k][c]
    for k in range(4): # k in 0..3
        for i in range(n):
            for c in range(49):
                W[k*n + i][c] = W4[k*4+i][c]
                
    # Second half: B2 (cols 4-7)
    for i in range(n*m):
        for c in range(49):
            U[i][c + 49] = U4[i][c]
    for j in range(m): # j in 0..3
        for k in range(4): # k in 0..3 (part of B2, global k = 4+k)
            for c in range(49):
                V[j*p + (4+k)][c + 49] = V4[j*4+k][c]
    for k in range(4): # k in 0..3 (global k = 4+k)
        for i in range(n):
            for c in range(49):
                W[(4+k)*n + i][c + 49] = W4[k*4+i][c]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
