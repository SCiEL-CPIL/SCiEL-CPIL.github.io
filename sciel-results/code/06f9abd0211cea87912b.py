import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # 1. Construct Strassen decomposition for 2x2 (Rank 7)
    # U2, V2, W2 are 4x7 matrices
    # Indices: U2[i*2+j, r], V2[j*2+k, r], W2[k*2+i, r]
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # M1
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[3][0] = 1
    
    # M2
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[2][1] = 1; W2[3][1] = -1 # C10 (k=1,i=0) -> idx 2, C11 (k=1,i=1) -> idx 3
    
    # M3
    U2[0][2] = 1
    V2[1][2] = 1; V2[3][2] = -1
    W2[1][2] = 1; W2[3][2] = 1 # C01 (k=0,i=1) -> idx 1, C11 (k=1,i=1) -> idx 3
    
    # M4
    U2[3][3] = 1
    V2[2][3] = 1; V2[0][3] = -1
    W2[0][3] = 1; W2[2][3] = 1 # C00 (k=0,i=0) -> idx 0, C10 (k=1,i=0) -> idx 2
    
    # M5
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    W2[0][4] = -1; W2[1][4] = 1 # C00 (k=0,i=0) -> idx 0, C01 (k=0,i=1) -> idx 1
    
    # M6
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[1][5] = 1
    W2[3][5] = 1 # C11 (k=1,i=1) -> idx 3
    
    # M7
    U2[1][6] = 1; U2[3][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    W2[0][6] = 1 # C00 (k=0,i=0) -> idx 0

    # 2. Construct Schoolbook decomposition for 3x3 (Rank 27)
    # U3, V3, W3 are 9x27 matrices
    U3 = [[0]*27 for _ in range(9)]
    V3 = [[0]*27 for _ in range(9)]
    W3 = [[0]*27 for _ in range(9)]
    
    r3 = 0
    for i1 in range(3):
        for j1 in range(3):
            for k1 in range(3):
                U3[i1*3 + j1][r3] = 1
                V3[j1*3 + k1][r3] = 1
                W3[k1*3 + i1][r3] = 1
                r3 += 1

    # 3. Combine via Kronecker Product
    rank = 27 * 7
    U = [[0]*rank for _ in range(36)]
    V = [[0]*rank for _ in range(36)]
    W = [[0]*rank for _ in range(36)]
    
    for r3 in range(27):
        for r2 in range(7):
            r = r3 * 7 + r2
            
            # Fill U
            for i in range(6):
                i1, i2 = divmod(i, 2)
                for j in range(6):
                    j1, j2 = divmod(j, 2)
                    U[i*6 + j][r] = U3[i1*3 + j1][r3] * U2[i2*2 + j2][r2]
            
            # Fill V
            for j in range(6):
                j1, j2 = divmod(j, 2)
                for k in range(6):
                    k1, k2 = divmod(k, 2)
                    V[j*6 + k][r] = V3[j1*3 + k1][r3] * V2[j2*2 + k2][r2]
            
            # Fill W
            for k in range(6):
                k1, k2 = divmod(k, 2)
                for i in range(6):
                    i1, i2 = divmod(i, 2)
                    W[k*6 + i][r] = W3[k1*3 + i1][r3] * W2[k2*2 + i2][r2]
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
