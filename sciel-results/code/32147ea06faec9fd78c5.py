import numpy as np

def construct():
    n, m, p = 4, 5, 5
    
    # Helper to create a zero matrix
    def zeros(rows, cols):
        return [[0] * cols for _ in range(rows)]

    # 1. Strassen 2x2x2 (Rank 7)
    # U, V, W are 4x7
    U_s = zeros(4, 7)
    V_s = zeros(4, 7)
    W_s = zeros(4, 7)
    
    # M1
    U_s[0][0] = 1; U_s[3][0] = 1
    V_s[0][0] = 1; V_s[3][0] = 1
    W_s[0][0] = 1; W_s[3][0] = 1
    
    # M2
    U_s[2][1] = 1; U_s[3][1] = 1
    V_s[0][1] = 1
    W_s[2][1] = 1; W_s[3][1] = -1
    
    # M3
    U_s[0][2] = 1
    V_s[1][2] = 1; V_s[3][2] = -1
    W_s[1][2] = 1; W_s[3][2] = 1
    
    # M4
    U_s[3][3] = 1
    V_s[2][3] = 1; V_s[0][3] = -1
    W_s[0][3] = 1; W_s[2][3] = 1
    
    # M5
    U_s[0][4] = 1; U_s[1][4] = 1
    V_s[3][4] = 1
    W_s[1][4] = 1; W_s[0][4] = -1
    
    # M6
    U_s[2][5] = 1; U_s[0][5] = -1
    V_s[0][5] = 1; V_s[1][5] = 1
    W_s[3][5] = 1
    
    # M7
    U_s[1][6] = 1; U_s[3][6] = -1
    V_s[2][6] = 1; V_s[3][6] = 1
    W_s[0][6] = 1

    # 2. Kronecker product for 4x4x4 (Rank 49)
    # U_444 is 16x49, V_444 is 16x49, W_444 is 16x49
    U_444 = zeros(16, 49)
    V_444 = zeros(16, 49)
    W_444 = zeros(16, 49)
    
    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2
            for i1 in range(4):
                for j1 in range(4):
                    u_val = U_s[i1][r1] * U_s[j1][r2]
                    if u_val != 0:
                        idx_u = i1 * 4 + j1
                        U_444[idx_u][r] += u_val
                    v_val = V_s[i1][r1] * V_s[j1][r2]
                    if v_val != 0:
                        idx_v = i1 * 4 + j1
                        V_444[idx_v][r] += v_val
                    w_val = W_s[i1][r1] * W_s[j1][r2]
                    if w_val != 0:
                        idx_w = i1 * 4 + j1
                        W_444[idx_w][r] += w_val

    # 3. Schoolbook 4x4x1 (Rank 16)
    U_441 = zeros(16, 16)
    V_441 = zeros(4, 16)
    W_441 = zeros(4, 16)
    for i in range(4):
        for j in range(4):
            r = i * 4 + j
            U_441[i * 4 + j][r] = 1
            V_441[j][r] = 1
            W_441[i][r] = 1

    # 4. Schoolbook 4x1x4 (Rank 16)
    U_414 = zeros(4, 16)
    V_414 = zeros(4, 16)
    W_414 = zeros(16, 16)
    for i in range(4):
        for k in range(4):
            r = i * 4 + k
            U_414[i][r] = 1
            V_414[k][r] = 1
            W_414[k * 4 + i][r] = 1

    # 5. Schoolbook 4x1x1 (Rank 4)
    U_411 = zeros(4, 4)
    V_411 = zeros(1, 4)
    W_411 = zeros(4, 4)
    for i in range(4):
        U_411[i][i] = 1
        V_411[0][i] = 1
        W_411[i][i] = 1

    # Combine into 4x5x5
    total_rank = 49 + 16 + 16 + 4
    U = zeros(20, total_rank)
    V = zeros(25, total_rank)
    W = zeros(20, total_rank)
    
    # Comp 1: A1 B11 (4x4x4) -> C[0:4, 0:4]
    # U_idx = I*5+J, V_idx = J*5+K, W_idx = K*4+I
    for r in range(49):
        for idx_u in range(16):
            val = U_444[idx_u][r]
            if val != 0:
                i, j = divmod(idx_u, 4)
                U[i * 5 + j][r] += val
        for idx_v in range(16):
            val = V_444[idx_v][r]
            if val != 0:
                j, k = divmod(idx_v, 4)
                V[j * 5 + k][r] += val
        for idx_w in range(16):
            val = W_444[idx_w][r]
            if val != 0:
                k, i = divmod(idx_w, 4)
                W[k * 4 + i][r] += val

    # Comp 2: A1 B12 (4x4x1) -> C[0:4, 4]
    # K=4
    offset = 49
    for r in range(16):
        rr = offset + r
        for idx_u in range(16):
            val = U_441[idx_u][r]
            if val != 0:
                i, j = divmod(idx_u, 4)
                U[i * 5 + j][rr] += val
        for idx_v in range(4):
            val = V_441[idx_v][r]
            if val != 0:
                j = idx_v
                V[j * 5 + 4][rr] += val
        for idx_w in range(4):
            val = W_441[idx_w][r]
            if val != 0:
                i = idx_w
                W[4 * 4 + i][rr] += val

    # Comp 3: A2 B21 (4x1x4) -> C[0:4, 0:4]
    # J=4
    offset = 49 + 16
    for r in range(16):
        rr = offset + r
        for idx_u in range(4):
            val = U_414[idx_u][r]
            if val != 0:
                i = idx_u
                U[i * 5 + 4][rr] += val
        for idx_v in range(4):
            val = V_414[idx_v][r]
            if val != 0:
                k = idx_v
                V[4 * 5 + k][rr] += val
        for idx_w in range(16):
            val = W_414[idx_w][r]
            if val != 0:
                k, i = divmod(idx_w, 4)
                W[k * 4 + i][rr] += val

    # Comp 4: A2 B22 (4x1x1) -> C[0:4, 4]
    # J=4, K=4
    offset = 49 + 16 + 16
    for r in range(4):
        rr = offset + r
        for idx_u in range(4):
            val = U_411[idx_u][r]
            if val != 0:
                i = idx_u
                U[i * 5 + 4][rr] += val
        for idx_v in range(1):
            val = V_411[idx_v][r]
            if val != 0:
                V[4 * 5 + 4][rr] += val
        for idx_w in range(4):
            val = W_411[idx_w][r]
            if val != 0:
                i = idx_w
                W[4 * 4 + i][rr] += val

    return {"U": U, "V": V, "W": W}
