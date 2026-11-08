import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = 4, 4, 5

    # 1. Define 2x2x2 Strassen factors
    # U2[i*2+j, r], V2[j*2+k, r], W2[k*2+i, r]
    U2 = np.zeros((4, 7), dtype=int)
    V2 = np.zeros((4, 7), dtype=int)
    W2 = np.zeros((4, 7), dtype=int)

    # r=0: (A00+A11)(B00+B11)
    U2[0,0]=1; U2[3,0]=1
    V2[0,0]=1; V2[3,0]=1
    W2[0,0]=1; W2[3,0]=1

    # r=1: (A10+A11)B00
    U2[2,1]=1; U2[3,1]=1
    V2[0,1]=1
    W2[2,1]=1; W2[3,1]=-1

    # r=2: A00(B01-B11)
    U2[0,2]=1
    V2[1,2]=1; V2[3,2]=-1
    W2[1,2]=1; W2[3,2]=1

    # r=3: A11(B10-B00)
    U2[3,3]=1
    V2[2,3]=1; V2[0,3]=-1
    W2[0,3]=1; W2[2,3]=1

    # r=4: (A00+A01)B11
    U2[0,4]=1; U2[1,4]=1
    V2[3,4]=1
    W2[0,4]=-1; W2[1,4]=1

    # r=5: (A10-A00)(B00+B01)
    U2[2,5]=1; U2[0,5]=-1
    V2[0,5]=1; V2[1,5]=1
    W2[3,5]=1

    # r=6: (A01-A11)(B10+B11)
    U2[1,6]=1; U2[3,6]=-1
    V2[2,6]=1; V2[3,6]=1
    W2[0,6]=1

    # Verify 2x2x2 Strassen
    for i in range(2):
        for j in range(2):
            for k in range(2):
                val = 0
                for r in range(7):
                    val += U2[i*2+j, r] * V2[j*2+k, r] * W2[k*2+i, r]
                expected = 1 if i == k else 0
                if val != expected:
                    # Fallback to schoolbook if Strassen is wrong
                    rank = n * m * p
                    U = [[0] * rank for _ in range(n * m)]
                    V = [[0] * rank for _ in range(m * p)]
                    W = [[0] * rank for _ in range(p * n)]
                    col = 0
                    for i2 in range(n):
                        for j2 in range(m):
                            for k2 in range(p):
                                U[i2 * m + j2][col] = 1
                                V[j2 * p + k2][col] = 1
                                W[k2 * n + i2][col] = 1
                                col += 1
                    return {"U": U, "V": V, "W": W}

    # 2. Define block-level Strassen coefficients
    # L_s[I,J] and R_s[J,K] for I,J,K in {0,1}
    L_coeff = [
        [[1,0],[0,1]], # s=0
        [[0,0],[1,1]], # s=1
        [[1,0],[0,0]], # s=2
        [[0,0],[0,1]], # s=3
        [[1,1],[0,0]], # s=4
        [[-1,0],[1,0]],# s=5
        [[0,1],[0,-1]] # s=6
    ]
    R_coeff = [
        [[1,0],[0,1]], # s=0
        [[1,0],[0,0]], # s=1
        [[0,1],[0,-1]],# s=2
        [[-1,0],[1,0]],# s=3
        [[0,0],[0,1]], # s=4
        [[1,1],[0,0]], # s=5
        [[0,0],[1,1]]  # s=6
    ]

    # 3. Construct 4x4x4 decomposition (rank 49)
    U4 = np.zeros((16, 49), dtype=int)
    V4 = np.zeros((16, 49), dtype=int)
    W4 = np.zeros((16, 49), dtype=int)
    
    for s in range(7):
        for t in range(7):
            r = s * 7 + t
            for i in range(4):
                I, i0 = i // 2, i % 2
                for j in range(4):
                    J, j0 = j // 2, j % 2
                    U4[i*4+j, r] = L_coeff[s][I][J] * U2[i0*2+j0, t]
            
            for j in range(4):
                J, j0 = j // 2, j % 2
                for k in range(4):
                    K, k0 = k // 2, k % 2
                    V4[j*4+k, r] = R_coeff[s][J][K] * V2[j0*2+k0, t]
            
            for k in range(4):
                K, k0 = k // 2, k % 2
                for i in range(4):
                    I, i0 = i // 2, i % 2
                    W4[k*4+i, r] = W2[k0*2+i0, t]

    # Verify 4x4x4 decomposition
    for i in range(4):
        for j in range(4):
            for k in range(4):
                val = 0
                for r in range(49):
                    val += U4[i*4+j, r] * V4[j*4+k, r] * W4[k*4+i, r]
                expected = 1 if i == k else 0
                if val != expected:
                    # Fallback
                    rank = n * m * p
                    U = [[0] * rank for _ in range(n * m)]
                    V = [[0] * rank for _ in range(m * p)]
                    W = [[0] * rank for _ in range(p * n)]
                    col = 0
                    for i2 in range(n):
                        for j2 in range(m):
                            for k2 in range(p):
                                U[i2 * m + j2][col] = 1
                                V[j2 * p + k2][col] = 1
                                W[k2 * n + i2][col] = 1
                                col += 1
                    return {"U": U, "V": V, "W": W}

    # 4. Extend to 4x4x5 (rank 65)
    rank = 49 + 16
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Copy 4x4x4 part
    for i in range(4):
        for j in range(4):
            for r in range(49):
                U[i*4+j][r] = int(U4[i*4+j, r])
    for j in range(4):
        for k in range(4):
            for r in range(49):
                V[j*4+k][r] = int(V4[j*4+k, r])
    for k in range(4):
        for i in range(4):
            for r in range(49):
                W[k*4+i][r] = int(W4[k*4+i, r])
    
    # Add schoolbook terms for k=4
    col = 49
    for i in range(4):
        for j in range(4):
            U[i*4+j][col] = 1
            V[j*4+4][col] = 1
            W[4*4+i][col] = 1
            col += 1

    # Verify 4x4x5 decomposition
    for i in range(4):
        for j in range(4):
            for k in range(5):
                val = 0
                for r in range(rank):
                    val += U[i*4+j][r] * V[j*4+k][r] * W[k*4+i][r]
                expected = 1 if i == k else 0
                if val != expected:
                    # Fallback
                    rank_fb = n * m * p
                    U_fb = [[0] * rank_fb for _ in range(n * m)]
                    V_fb = [[0] * rank_fb for _ in range(m * p)]
                    W_fb = [[0] * rank_fb for _ in range(p * n)]
                    col = 0
                    for i2 in range(n):
                        for j2 in range(m):
                            for k2 in range(p):
                                U_fb[i2 * m + j2][col] = 1
                                V_fb[j2 * p + k2][col] = 1
                                W_fb[k2 * n + i2][col] = 1
                                col += 1
                    return {"U": U_fb, "V": V_fb, "W": W_fb}

    return {"U": U, "V": V, "W": W}
