import numpy as np

def construct():
    n, m, p = 4, 4, 4
    rank = 49
    
    # Initialize U, V, W
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2 decomposition
    # U2, V2, W2 are 4x7 matrices
    # Indices: i, j, k in {0, 1}
    # U2 row: i*2 + j
    # V2 row: j*2 + k
    # W2 row: k*2 + i
    
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # c=0: M1
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[3][0] = 1
    
    # c=1: M2
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[2][1] = 1; W2[3][1] = -1
    
    # c=2: M3
    U2[0][2] = 1
    V2[1][2] = 1; V2[3][2] = -1
    W2[1][2] = 1; W2[3][2] = 1
    
    # c=3: M4
    U2[3][3] = 1
    V2[2][3] = 1; V2[0][3] = -1
    W2[0][3] = 1; W2[2][3] = 1
    
    # c=4: M5
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    W2[0][4] = -1; W2[1][4] = 1
    
    # c=5: M6
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[1][5] = 1
    W2[3][5] = 1
    
    # c=6: M7
    U2[1][6] = 1; U2[3][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    W2[0][6] = 1
    
    # Construct 4x4 from 2x2
    for i in range(4):
        for j in range(4):
            i1, i2 = i // 2, i % 2
            j1, j2 = j // 2, j % 2
            idx_u = i * 4 + j
            for c1 in range(7):
                for c2 in range(7):
                    c = c1 * 7 + c2
                    val_u = U2[i1 * 2 + j1][c1] * U2[i2 * 2 + j2][c2]
                    U[idx_u][c] = val_u
                    
    for j in range(4):
        for k in range(4):
            j1, j2 = j // 2, j % 2
            k1, k2 = k // 2, k % 2
            idx_v = j * 4 + k
            for c1 in range(7):
                for c2 in range(7):
                    c = c1 * 7 + c2
                    val_v = V2[j1 * 2 + k1][c1] * V2[j2 * 2 + k2][c2]
                    V[idx_v][c] = val_v
                    
    for k in range(4):
        for i in range(4):
            k1, k2 = k // 2, k % 2
            i1, i2 = i // 2, i % 2
            idx_w = k * 4 + i
            for c1 in range(7):
                for c2 in range(7):
                    c = c1 * 7 + c2
                    val_w = W2[k1 * 2 + i1][c1] * W2[k2 * 2 + i2][c2]
                    W[idx_w][c] = val_w

    return {"U": U, "V": V, "W": W}
