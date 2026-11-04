import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Define 2x2 Strassen Decomposition (Rank 7)
    # U2: 4x7, V2: 4x7, W2: 4x7
    # Indices: U2[i,j], V2[j,k], W2[k,i]
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # t=0: (a+d)(e+h)
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[3][0] = 1
    
    # t=1: (c+d)e
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[1][1] = 1; W2[3][1] = -1
    
    # t=2: a(f-h)
    U2[0][2] = 1
    V2[1][2] = 1; V2[3][2] = -1
    W2[2][2] = 1; W2[3][2] = 1
    
    # t=3: d(g-e)
    U2[3][3] = 1
    V2[2][3] = 1; V2[0][3] = -1
    W2[0][3] = 1; W2[1][3] = 1
    
    # t=4: (a+b)h
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    W2[0][4] = -1; W2[2][4] = 1
    
    # t=5: (c-a)(e+f)
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[1][5] = 1
    W2[3][5] = 1
    
    # t=6: (b-d)(g+h)
    U2[1][6] = 1; U2[3][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    W2[0][6] = 1

    # Define 3x3 Schoolbook Decomposition (Rank 27)
    # U3: 9x27, V3: 9x27, W3: 9x27
    U3 = [[0]*27 for _ in range(9)]
    V3 = [[0]*27 for _ in range(9)]
    W3 = [[0]*27 for _ in range(9)]
    
    for i3 in range(3):
        for j3 in range(3):
            for k3 in range(3):
                t3 = i3 * 9 + j3 * 3 + k3
                r_u3 = i3 * 3 + j3
                r_v3 = j3 * 3 + k3
                r_w3 = k3 * 3 + i3
                U3[r_u3][t3] = 1
                V3[r_v3][t3] = 1
                W3[r_w3][t3] = 1

    # Construct 6x6x6 Decomposition via Kronecker Product
    # Rank = 7 * 27 = 189
    rank = 189
    U = [[0]*rank for _ in range(36)]
    V = [[0]*rank for _ in range(36)]
    W = [[0]*rank for _ in range(36)]
    
    for t2 in range(7):
        for t3 in range(27):
            c = t2 * 27 + t3
            
            # Fill U
            for i in range(6):
                i3 = i // 2
                i2 = i % 2
                for j in range(6):
                    j3 = j // 2
                    j2 = j % 2
                    
                    r_u3 = i3 * 3 + j3
                    r_u2 = i2 * 2 + j2
                    
                    val_u3 = U3[r_u3][t3]
                    val_u2 = U2[r_u2][t2]
                    U[i * 6 + j][c] = val_u3 * val_u2
            
            # Fill V
            for j in range(6):
                j3 = j // 2
                j2 = j % 2
                for k in range(6):
                    k3 = k // 2
                    k2 = k % 2
                    
                    r_v3 = j3 * 3 + k3
                    r_v2 = j2 * 2 + k2
                    
                    val_v3 = V3[r_v3][t3]
                    val_v2 = V2[r_v2][t2]
                    V[j * 6 + k][c] = val_v3 * val_v2
            
            # Fill W
            for k in range(6):
                k3 = k // 2
                k2 = k % 2
                for i in range(6):
                    i3 = i // 2
                    i2 = i % 2
                    
                    r_w3 = k3 * 3 + i3
                    r_w2 = k2 * 2 + i2
                    
                    val_w3 = W3[r_w3][t3]
                    val_w2 = W2[r_w2][t2]
                    W[k * 6 + i][c] = val_w3 * val_w2

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
