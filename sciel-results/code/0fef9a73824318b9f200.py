import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS
    
    # 1. Construct 2x2 Strassen U, V, W
    # U: 4x7, V: 4x7, W: 4x7
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # c=0: M1 = (A00+A11)(B00+B11)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1 # C00, C11
    
    # c=1: M2 = (A10+A11)B00
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[1, 1] = 1; W2[3, 1] = -1 # C10, C11
    
    # c=2: M3 = A00(B10-B11)
    U2[0, 2] = 1
    V2[2, 2] = 1; V2[3, 2] = -1
    W2[2, 2] = 1; W2[3, 2] = 1 # C01, C11
    
    # c=3: M4 = A11(B01-B00)
    U2[3, 3] = 1
    V2[1, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[1, 3] = 1 # C00, C10
    
    # c=4: M5 = (A00+A01)B11
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[2, 4] = 1 # C00, C01
    
    # c=5: M6 = (A10-A00)(B00+B01)
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[1, 5] = 1; W2[3, 5] = 1 # C10, C11
    
    # c=6: M7 = (A01-A11)(B10+B11)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1; W2[2, 6] = 1 # C00, C01

    # 2. Construct 4x4 by 4x4 U, V, W (Rank 49)
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    for cb in range(7):
        for cs in range(7):
            c = cb * 7 + cs
            for i in range(4):
                for j in range(4):
                    i1, i0 = divmod(i, 2)
                    j1, j0 = divmod(j, 2)
                    U4[i*4+j, c] = U2[i1*2+j1, cb] * U2[i0*2+j0, cs]
            
            for j in range(4):
                for k in range(4):
                    j1, j0 = divmod(j, 2)
                    k1, k0 = divmod(k, 2)
                    V4[j*4+k, c] = V2[j1*2+k1, cb] * V2[j0*2+k0, cs]
            
            for k in range(4):
                for i in range(4):
                    k1, k0 = divmod(k, 2)
                    i1, i0 = divmod(i, 2)
                    W4[k*4+i, c] = W2[k1*2+i1, cb] * W2[k0*2+i0, cs]

    # 3. Construct 4x4 by 4x1 U, V, W (Rank 16)
    U_tail = np.zeros((16, 16))
    V_tail = np.zeros((4, 16))
    W_tail = np.zeros((4, 16))
    
    for i in range(4):
        for j in range(4):
            c = i*4 + j
            U_tail[i*4+j, c] = 1
            V_tail[j, c] = 1
            W_tail[i, c] = 1

    # 4. Combine
    rank = 49 + 49 + 16
    U_total = np.zeros((16, rank))
    V_total = np.zeros((36, rank))
    W_total = np.zeros((36, rank))
    
    # Part 1: B1 (cols 0-3)
    U_total[:, 0:49] = U4
    for j in range(4):
        for k in range(4):
            V_total[j*9+k, 0:49] = V4[j*4+k, :]
    for k in range(4):
        for i in range(4):
            W_total[k*4+i, 0:49] = W4[k*4+i, :]
            
    # Part 2: B2 (cols 4-7)
    U_total[:, 49:98] = U4
    for j in range(4):
        for k in range(4):
            V_total[j*9+(4+k), 49:98] = V4[j*4+k, :]
    for k in range(4):
        for i in range(4):
            W_total[(4+k)*4+i, 49:98] = W4[k*4+i, :]
            
    # Part 3: B3 (col 8)
    U_total[:, 98:114] = U_tail
    for j in range(4):
        V_total[j*9+8, 98:114] = V_tail[j, :]
    for i in range(4):
        W_total[8*4+i, 98:114] = W_tail[i, :]
        
    # Convert to lists
    return {
        "U": U_total.tolist(),
        "V": V_total.tolist(),
        "W": W_total.tolist()
    }
