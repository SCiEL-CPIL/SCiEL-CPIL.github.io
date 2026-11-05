import numpy as np

DIMENSIONS = [4, 4, 8]

def construct():
    # Step 1: Construct Strassen factors for 2x2x2
    n1, m1, p1 = 2, 2, 2
    rank1 = 7
    U1 = [[0]*rank1 for _ in range(n1*m1)]
    V1 = [[0]*rank1 for _ in range(m1*p1)]
    W1 = [[0]*rank1 for _ in range(p1*n1)]
    
    # M1: A00(B00+B11)
    U1[0][0] = 1
    V1[0][0] = 1; V1[3][0] = 1
    W1[0][0] = 1; W1[1][0] = -1; W1[3][0] = 1
    
    # M2: (A00+A10)B11
    U1[0][1] = 1; U1[2][1] = 1
    V1[3][1] = 1
    W1[1][1] = 1; W1[3][1] = -1
    
    # M3: (A01+A11)B00
    U1[1][2] = 1; U1[3][2] = 1
    V1[0][2] = 1
    W1[2][2] = 1; W1[3][2] = 1
    
    # M4: A11(B10+B00)
    U1[3][3] = 1
    V1[2][3] = 1; V1[0][3] = 1
    W1[0][3] = 1; W1[1][3] = -1; W1[3][3] = 1
    
    # M5: (A00+A11)(B11+B10)
    U1[0][4] = 1; U1[3][4] = 1
    V1[3][4] = 1; V1[2][4] = 1
    W1[0][4] = -1; W1[2][4] = 1
    
    # M6: (A01+A11)(B00+B10)
    U1[1][5] = 1; U1[3][5] = 1
    V1[0][5] = 1; V1[2][5] = 1
    W1[2][5] = -1; W1[1][5] = 1
    
    # M7: (A00+A10)(B11+B01)
    U1[0][6] = 1; U1[2][6] = 1
    V1[3][6] = 1; V1[1][6] = 1
    W1[0][6] = 1; W1[2][6] = -1; W1[1][6] = -1
    
    # Step 2: Construct 2x2x4 factors (Two independent 2x2x2 Strassen blocks)
    n2, m2, p2 = 2, 2, 4
    rank2 = 14
    U2 = [[0]*rank2 for _ in range(n2*m2)]
    V2 = [[0]*rank2 for _ in range(m2*p2)]
    W2 = [[0]*rank2 for _ in range(p2*n2)]
    
    for r in range(7):
        # First block (k in 0,1)
        for i in range(2):
            for j in range(2):
                U2[i*2+j][r] = U1[i*2+j][r]
        for j in range(2):
            for k in range(2):
                V2[j*4+k][r] = V1[j*2+k][r]
        for k in range(2):
            for i in range(2):
                W2[k*2+i][r] = W1[k*2+i][r]
                
        # Second block (k in 2,3)
        for i in range(2):
            for j in range(2):
                U2[i*2+j][r+7] = U1[i*2+j][r]
        for j in range(2):
            for k in range(2, 4):
                V2[j*4+k][r+7] = V1[j*2+(k-2)][r]
        for k in range(2, 4):
            for i in range(2):
                W2[k*2+i][r+7] = W1[(k-2)*2+i][r]
                
    # Step 3: Kronecker product to get 4x4x8
    n, m, p = 4, 4, 8
    rank = rank1 * rank2 # 7 * 14 = 98
    
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # Iterate over all combinations of factors
    for r1 in range(rank1):
        for r2 in range(rank2):
            r = r1 * rank2 + r2
            
            # Fill U
            for i1 in range(2):
                for j1 in range(2):
                    val1 = U1[i1*2+j1][r1]
                    if val1 == 0: continue
                    for i2 in range(2):
                        for j2 in range(2):
                            val2 = U2[i2*2+j2][r2]
                            if val2 == 0: continue
                            i = i1*2 + i2
                            j = j1*2 + j2
                            U[i*4+j][r] += val1 * val2
                            
            # Fill V
            for j1 in range(2):
                for k1 in range(2):
                    val1 = V1[j1*2+k1][r1]
                    if val1 == 0: continue
                    for j2 in range(2):
                        for k2 in range(4):
                            val2 = V2[j2*4+k2][r2]
                            if val2 == 0: continue
                            j = j1*2 + j2
                            k = k1*4 + k2
                            V[j*8+k][r] += val1 * val2
                            
            # Fill W
            for k1 in range(2):
                for i1 in range(2):
                    val1 = W1[k1*2+i1][r1]
                    if val1 == 0: continue
                    for k2 in range(4):
                        for i2 in range(2):
                            val2 = W2[k2*2+i2][r2]
                            if val2 == 0: continue
                            k = k1*4 + k2
                            i = i1*2 + i2
                            W[k*4+i][r] += val1 * val2
                            
    return {"U": U, "V": V, "W": W}
