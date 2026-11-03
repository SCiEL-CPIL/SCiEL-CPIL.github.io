import numpy as np

def construct():
    n, m, p = 4, 5, 5
    rank = 85
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2 coefficients
    # U2: 4x7, V2: 4x7, W2: 4x7
    # Indices 0..3 for 00, 01, 10, 11
    U2 = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    V2 = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, -1, 1, 0, 1]
    ]
    W2 = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    
    # Part 1: 4x4x4 Strassen (Rank 49)
    col = 0
    for s in range(7):
        for t in range(7):
            # Compute gamma_U, gamma_V, gamma_W
            # gamma_U[I', J'] = sum_{I,J} U2[I*2+J, t] * U2[I'*2+J', s]
            gamma_U = [[0]*4 for _ in range(4)]
            gamma_V = [[0]*4 for _ in range(4)]
            gamma_W = [[0]*4 for _ in range(4)]
            
            for I in range(2):
                for J in range(2):
                    # U2 index for block I,J is I*2+J
                    # t-th step coefficient for this block in X_s
                    c_t = U2[I*2+J][t]
                    if c_t == 0: continue
                    for Ip in range(2):
                        for Jp in range(2):
                            # s-th step coefficient for block Ip,Jp
                            c_s = U2[Ip*2+Jp][s]
                            if c_s == 0: continue
                            gamma_U[Ip][Jp] += c_t * c_s
                            
            for J in range(2):
                for K in range(2):
                    c_t = V2[J*2+K][t]
                    if c_t == 0: continue
                    for Jp in range(2):
                        for Kp in range(2):
                            c_s = V2[Jp*2+Kp][s]
                            if c_s == 0: continue
                            gamma_V[Jp][Kp] += c_t * c_s
                            
            for I in range(2):
                for J in range(2):
                    c_s = W2[I*2+J][s]
                    if c_s == 0: continue
                    for u in range(2):
                        for v in range(2):
                            c_t = W2[u*2+v][t]
                            if c_t == 0: continue
                            # C block I,J, entry u,v
                            gamma_W[I][J] += c_s * c_t # This is wrong, need to store per u,v
                            
            # Re-do gamma_W properly
            gamma_W_full = [[0]*4 for _ in range(4)] # indexed by i, j (0..3)
            for I in range(2):
                for J in range(2):
                    c_s = W2[I*2+J][s]
                    if c_s == 0: continue
                    for u in range(2):
                        for v in range(2):
                            c_t = W2[u*2+v][t]
                            if c_t == 0: continue
                            i = 2*I + u
                            j = 2*J + v
                            gamma_W_full[i][j] += c_s * c_t

            # Fill U, V, W
            for Ip in range(2):
                for Jp in range(2):
                    val = gamma_U[Ip][Jp]
                    if val == 0: continue
                    for u in range(2):
                        for v in range(2):
                            i = 2*Ip + u
                            j = 2*Jp + v
                            U[i*5 + j][col] = val
                            
            for Jp in range(2):
                for Kp in range(2):
                    val = gamma_V[Jp][Kp]
                    if val == 0: continue
                    for v in range(2):
                        for w in range(2):
                            j = 2*Jp + v
                            k = 2*Kp + w
                            V[j*5 + k][col] = val
                            
            for i in range(4):
                for j in range(4):
                    val = gamma_W_full[i][j]
                    if val == 0: continue
                    W[j*4 + i][col] = val
            col += 1
            
    # Part 2: A[0:4, 0:4] * B[0:4, 4] -> C[0:4, 4] (Rank 16)
    for i in range(4):
        for j in range(4):
            U[i*5 + j][col] = 1
            V[j*5 + 4][col] = 1
            W[4*4 + i][col] = 1
            col += 1
            
    # Part 3: A[0:4, 4] * B[4, 0:4] -> C[0:4, 0:4] (Rank 16)
    for i in range(4):
        for k in range(4):
            U[i*5 + 4][col] = 1
            V[4*5 + k][col] = 1
            W[k*4 + i][col] = 1
            col += 1
            
    # Part 4: A[0:4, 4] * B[4, 4] -> C[0:4, 4] (Rank 4)
    for i in range(4):
        U[i*5 + 4][col] = 1
        V[4*5 + 4][col] = 1
        W[4*4 + i][col] = 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
