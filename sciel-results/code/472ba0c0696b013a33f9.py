"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    # Kronecker product of two Strassen 2x2 decompositions: rank 7*7 = 49
    n, m, p = DIMENSIONS
    rank = 49
    
    # Strassen 2x2 decomposition tables
    # U2[i*2+j, r], V2[j*2+k, r], W2[k*2+i, r] for r in 0..6
    # r=0: M1=(a00+a11)(b00+b11), contributes +1 to C[0,0](idx 0), +1 to C[1,1](idx 3)
    # r=1: M2=(a10+a11)b00, contributes +1 to C[1,0](idx 1), -1 to C[1,1](idx 3)
    # r=2: M3=a00(b01-b11), contributes +1 to C[0,1](idx 2), +1 to C[1,1](idx 3)
    # r=3: M4=a11(b10-b00), contributes +1 to C[0,0](idx 0), +1 to C[1,0](idx 1)
    # r=4: M5=(a00+a01)b11, contributes -1 to C[0,0](idx 0), +1 to C[0,1](idx 2)
    # r=5: M6=(a10-a00)(b00+b01), contributes +1 to C[1,1](idx 3)
    # r=6: M7=(a01-a11)(b10+b11), contributes +1 to C[0,0](idx 0)
    
    # U2: rows are i*2+j, cols are r
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # r=0: U2: a00(idx0)=1, a11(idx3)=1; V2: b00(idx0)=1, b11(idx3)=1; W2: C[0,0](idx0)=1, C[1,1](idx3)=1
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[3][0] = 1
    
    # r=1: U2: a10(idx2)=1, a11(idx3)=1; V2: b00(idx0)=1; W2: C[1,0](idx1)=1, C[1,1](idx3)=-1
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[1][1] = 1; W2[3][1] = -1
    
    # r=2: U2: a00(idx0)=1; V2: b01(idx1)=1, b11(idx3)=-1; W2: C[0,1](idx2)=1, C[1,1](idx3)=1
    U2[0][2] = 1
    V2[1][2] = 1; V2[3][2] = -1
    W2[2][2] = 1; W2[3][2] = 1
    
    # r=3: U2: a11(idx3)=1; V2: b10(idx2)=1, b00(idx0)=-1; W2: C[0,0](idx0)=1, C[1,0](idx1)=1
    U2[3][3] = 1
    V2[2][3] = 1; V2[0][3] = -1
    W2[0][3] = 1; W2[1][3] = 1
    
    # r=4: U2: a00(idx0)=1, a01(idx1)=1; V2: b11(idx3)=1; W2: C[0,0](idx0)=-1, C[0,1](idx2)=1
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    W2[0][4] = -1; W2[2][4] = 1
    
    # r=5: U2: a10(idx2)=1, a00(idx0)=-1; V2: b00(idx0)=1, b01(idx1)=1; W2: C[1,1](idx3)=1
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[1][5] = 1
    W2[3][5] = 1
    
    # r=6: U2: a01(idx1)=1, a11(idx3)=-1; V2: b10(idx2)=1, b11(idx3)=1; W2: C[0,0](idx0)=1
    U2[1][6] = 1; U2[3][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    W2[0][6] = 1
    
    # Build 4x4 via Kronecker product
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(16)]
    W = [[0]*rank for _ in range(16)]
    
    for r1 in range(7):
        for r2 in range(7):
            c = r1 * 7 + r2
            # U[i*4+j, c] = U2[i1*2+j1, r1] * U2[i0*2+j0, r2]
            for i1 in range(2):
                for j1 in range(2):
                    u1 = U2[i1*2+j1][r1]
                    if u1 == 0:
                        continue
                    for i0 in range(2):
                        for j0 in range(2):
                            u2 = U2[i0*2+j0][r2]
                            if u2 == 0:
                                continue
                            i = i1*2 + i0
                            j = j1*2 + j0
                            U[i*4+j][c] = u1 * u2
            
            # V[j*4+k, c] = V2[j1*2+k1, r1] * V2[j0*2+k0, r2]
            for j1 in range(2):
                for k1 in range(2):
                    v1 = V2[j1*2+k1][r1]
                    if v1 == 0:
                        continue
                    for j0 in range(2):
                        for k0 in range(2):
                            v2 = V2[j0*2+k0][r2]
                            if v2 == 0:
                                continue
                            j = j1*2 + j0
                            k = k1*2 + k0
                            V[j*4+k][c] = v1 * v2
            
            # W[k*4+i, c] = W2[k1*2+i1, r1] * W2[k0*2+i0, r2]
            for k1 in range(2):
                for i1 in range(2):
                    w1 = W2[k1*2+i1][r1]
                    if w1 == 0:
                        continue
                    for k0 in range(2):
                        for i0 in range(2):
                            w2 = W2[k0*2+i0][r2]
                            if w2 == 0:
                                continue
                            k = k1*2 + k0
                            i = i1*2 + i0
                            W[k*4+i][c] = w1 * w2
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
