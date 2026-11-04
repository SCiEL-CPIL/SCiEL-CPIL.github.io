import numpy as np

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # Strassen 2x2x2
    # U: (2*2, 7) = (4, 7)
    # V: (2*2, 7) = (4, 7)
    # W: (2*2, 7) = (4, 7)
    
    # Indices: i, j in {0, 1}
    # U[i*2+j, r]
    # V[j*2+k, r]
    # W[k*2+i, r]
    
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # M1 = (A11+A22)(B11+B22)
    U2[0*2+0][0] = 1; U2[1*2+1][0] = 1
    V2[0*2+0][0] = 1; V2[1*2+1][0] = 1
    W2[0*2+0][0] = 1; W2[1*2+1][0] = 1
    
    # M2 = (A21+A22)B11
    U2[1*2+0][1] = 1; U2[1*2+1][1] = 1
    V2[0*2+0][1] = 1
    W2[0*2+1][1] = 1; W2[1*2+1][1] = 1
    
    # M3 = A11(B12-B22)
    U2[0*2+0][2] = 1
    V2[0*2+1][2] = 1; V2[1*2+1][2] = -1
    W2[0*2+0][2] = 1; W2[1*2+1][2] = 1
    
    # M4 = A22(B21-B11)
    U2[1*2+1][3] = 1
    V2[1*2+0][3] = 1; V2[0*2+0][3] = -1
    W2[0*2+1][3] = 1; W2[1*2+1][3] = 1
    
    # M5 = (A11+A12)B22
    U2[0*2+0][4] = 1; U2[0*2+1][4] = 1
    V2[1*2+1][4] = 1
    W2[0*2+1][4] = 1; W2[1*2+1][4] = 1
    
    # M6 = (A21-A11)(B11+B12)
    U2[1*2+0][5] = 1; U2[0*2+0][5] = -1
    V2[0*2+0][5] = 1; V2[0*2+1][5] = 1
    W2[0*2+1][5] = 1; W2[1*2+1][5] = 1
    
    # M7 = (A12-A22)(B21+B22)
    U2[0*2+1][6] = 1; U2[1*2+1][6] = -1
    V2[1*2+0][6] = 1; V2[1*2+1][6] = 1
    W2[0*2+1][6] = 1; W2[1*2+1][6] = 1
    
    # 4x4x4 using Kronecker
    # Rank 49
    U44 = [[0]*49 for _ in range(16)]
    V44 = [[0]*49 for _ in range(16)]
    W44 = [[0]*49 for _ in range(16)]
    
    for i1 in range(2):
        for i2 in range(2):
            for j1 in range(2):
                for j2 in range(2):
                    for r1 in range(7):
                        for r2 in range(7):
                            r = r1 * 7 + r2
                            # U
                            val = U2[i1*2+j1][r1] * U2[i2*2+j2][r2]
                            if val != 0:
                                U44[(i1*2+i2)*2 + (j1*2+j2)][r] += val
                            # V
                            val = V2[j1*2+k1][r1] * V2[j2*2+k2][r2] # Wait, k is not defined here
                            # V index is j*k.
                            # We need to iterate k as well?
                            # No, V is (m*p, R). For 4x4, m=4, p=4.
                            # V index is j*4+k.
                            # j = j1*2+j2, k = k1*2+k2.
                            # V2 index is j*2+k.
                            # So V44[j*4+k, r] = sum V2[j1*2+k1, r1] V2[j2*2+k2, r2]
                            
                            # Let's do this properly with loops
                            pass
                            
    # This nested loop is getting complicated. Let's use numpy or a cleaner construction.
    
    # Actually, I can just construct the 49 rank terms directly.
    # But the code above is a start.
    
    # Let's refine the Kronecker construction.
    # U44[i, j, r] where i = i1*2+i2, j = j1*2+j2.
    # r = r1*7 + r2.
    # U44[i1*2+i2, j1*2+j2, r1*7+r2] = U2[i1*2+j1, r1] * U2[i2*2+j2, r2]
    
    U44 = [[0]*49 for _ in range(16)]
    V44 = [[0]*49 for _ in range(16)]
    W44 = [[0]*49 for _ in range(16)]
    
    for i1 in range(2):
        for j1 in range(2):
            for k1 in range(2):
                for r1 in range(7):
                    u1 = U2[i1*2+j1][r1]
                    v1 = V2[j1*2+k1][r1]
                    w1 = W2[k1*2+i1][r1]
                    if u1 == 0 and v1 == 0 and w1 == 0: continue
                    for i2 in range(2):
                        for j2 in range(2):
                            for k2 in range(2):
                                for r2 in range(7):
                                    u2 = U2[i2*2+j2][r2]
                                    v2 = V2[j2*2+k2][r2]
                                    w2 = W2[k2*2+i2][r2]
                                    if u2 == 0 and v2 == 0 and w2 == 0: continue
                                    
                                    r = r1*7 + r2
                                    i = i1*2 + i2
                                    j = j1*2 + j2
                                    k = k1*2 + k2
                                    
                                    U44[i*4+j][r] += u1*u2
                                    V44[j*4+k][r] += v1*v2
                                    W44[k*4+i][r] += w1*w2

    # 4x1x4 (Rank 1)
    # U: (4*1, 1) = (4, 1)
    # V: (1*4, 1) = (4, 1)
    # W: (4*4, 1) = (16, 1)
    U41 = [[0] for _ in range(4)]
    V14 = [[0] for _ in range(4)]
    W44_1 = [[0] for _ in range(16)]
    for i in range(4):
        U41[i][0] = 1
        for k in range(4):
            W44_1[k*4+i][0] = 1
    for k in range(4):
        V14[k][0] = 1

    # 4x5x1 (Rank 5)
    # U: (4*5, 5) = (20, 5)
    # V: (5*1, 5) = (5, 5)
    # W: (1*4, 5) = (4, 5)
    U45 = [[0]*5 for _ in range(20)]
    V51 = [[0]*5 for _ in range(5)]
    W14 = [[0]*5 for _ in range(4)]
    for j in range(5):
        for i in range(4):
            U45[i*5+j][j] = 1
        V51[j][j] = 1
        for i in range(4):
            W14[i][j] = 1

    # Combine 4x4x4 and 4x1x4 to get 4x5x4
    # n=4, m=5, p=4
    # U: (20, 50)
    # V: (20, 50)
    # W: (16, 50)
    
    U_454 = [[0]*50 for _ in range(20)]
    V_454 = [[0]*50 for _ in range(20)]
    W_454 = [[0]*50 for _ in range(16)]
    
    # Part 1: 4x4x4 (cols 0-48)
    for i in range(4):
        for j in range(4):
            for r in range(49):
                U_454[i*5+j][r] = U44[i*4+j][r]
    for j in range(4):
        for k in range(4):
            for r in range(49):
                V_454[j*4+k][r] = V44[j*4+k][r]
    for k in range(4):
        for i in range(4):
            for r in range(49):
                W_454[k*4+i][r] = W44[k*4+i][r]
                
    # Part 2: 4x1x4 (cols 49)
    # U: i*5+4 (j=4)
    for i in range(4):
        U_454[i*5+4][49] = U41[i][0]
    # V: 4*4+k (j=4)
    for k in range(4):
        V_454[4*4+k][49] = V14[k][0]
    # W: k*4+i
    for k in range(4):
        for i in range(4):
            W_454[k*4+i][49] = W44_1[k*4+i][0]

    # Combine 4x5x4 and 4x5x1 to get 4x5x6
    # n=4, m=5, p=6
    # U: (20, 55)
    # V: (30, 55)
    # W: (24, 55)
    
    U_final = [[0]*55 for _ in range(20)]
    V_final = [[0]*55 for _ in range(30)]
    W_final = [[0]*55 for _ in range(24)]
    
    # Part 1: 4x5x4 (cols 0-54)
    for i in range(4):
        for j in range(5):
            for r in range(50):
                U_final[i*5+j][r] = U_454[i*5+j][r]
    for j in range(5):
        for k in range(4):
            for r in range(50):
                V_final[j*4+k][r] = V_454[j*4+k][r]
    for k in range(4):
        for i in range(4):
            for r in range(50):
                W_final[k*4+i][r] = W_454[k*4+i][r]
                
    # Part 2: 4x5x1 (cols 50-54)
    # U: i*5+j
    for i in range(4):
        for j in range(5):
            for r in range(5):
                U_final[i*5+j][50+r] = U45[i*5+j][r]
    # V: j*1+0 (k=0) -> j*6+5? No, p=6.
    # V index is j*p+k.
    # In 4x5x1, p=1. V index is j*1+0 = j.
    # In final, p=6. V index is j*6+5 (k=5).
    for j in range(5):
        for r in range(5):
            V_final[j*6+5][50+r] = V51[j][r]
    # W: k*4+i. In 4x5x1, k=0.
    # In final, k=5.
    for i in range(4):
        for r in range(5):
            W_final[5*4+i][50+r] = W14[i][r]

    return {"U": U_final, "V": V_final, "W": W_final}
