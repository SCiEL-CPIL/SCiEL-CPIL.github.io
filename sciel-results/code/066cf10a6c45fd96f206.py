import numpy as np

def construct():
    n, m, p = 4, 4, 7
    rank = 97
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # 1. 4x4x4 Strassen (Rank 49)
    # Indices i, j, k in 0..3
    # U[i, j, r], V[j, k, r], W[k, i, r]
    
    # Define 2x2x2 Strassen coefficients
    # t: 1..7
    # U_local[t][i_l, j_l], V_local[t][j_l, k_l], W_local[t][k_l, i_l]
    U_local = [[ [0]*4 for _ in range(4) ] for _ in range(8)] # 1-indexed t
    V_local = [[ [0]*4 for _ in range(4) ] for _ in range(8)]
    W_local = [[ [0]*4 for _ in range(4) ] for _ in range(8)]
    
    # t=1
    U_local[1][0,0] = 1; U_local[1][1,1] = 1
    V_local[1][0,0] = 1; V_local[1][1,1] = 1
    W_local[1][0,0] = 1; W_local[1][1,1] = 1
    # t=2
    U_local[2][1,0] = 1; U_local[2][1,1] = 1
    V_local[2][0,0] = 1
    W_local[2][0,1] = 1; W_local[2][1,1] = -1
    # t=3
    U_local[3][0,0] = 1
    V_local[3][0,1] = 1; V_local[3][1,1] = -1
    W_local[3][1,0] = 1; W_local[3][1,1] = 1
    # t=4
    U_local[4][1,1] = 1
    V_local[4][1,0] = 1; V_local[4][0,0] = -1
    W_local[4][0,0] = 1; W_local[4][0,1] = 1
    # t=5
    U_local[5][0,0] = 1; U_local[5][0,1] = 1
    V_local[5][1,1] = 1
    W_local[5][0,0] = -1; W_local[5][1,0] = 1
    # t=6
    U_local[6][1,0] = 1; U_local[6][0,0] = -1
    V_local[6][0,0] = 1; V_local[6][0,1] = 1
    W_local[6][1,1] = 1
    # t=7
    U_local[7][0,1] = 1; U_local[7][1,1] = -1
    V_local[7][1,0] = 1; V_local[7][1,1] = 1
    W_local[7][0,0] = 1
    
    # Block Strassen coefficients (same as local, but for blocks)
    # U_block[t][i_b, j_b], etc.
    U_block = U_local
    V_block = V_local
    W_block = W_local
    
    col = 0
    for t in range(1, 8):
        for l in range(1, 8):
            r = col
            # Global i, j, k
            for i in range(4):
                i_b = i // 2
                i_l = i % 2
                for j in range(4):
                    j_b = j // 2
                    j_l = j % 2
                    u_val = U_block[t][i_b, j_b] * U_local[l][i_l, j_l]
                    if u_val != 0:
                        U[i * 4 + j][r] = u_val
            for j in range(4):
                j_b = j // 2
                j_l = j % 2
                for k in range(4):
                    k_b = k // 2
                    k_l = k % 2
                    v_val = V_block[t][j_b, k_b] * V_local[l][j_l, k_l]
                    if v_val != 0:
                        V[j * 4 + k][r] = v_val
            for k in range(4):
                k_b = k // 2
                k_l = k % 2
                for i in range(4):
                    i_b = i // 2
                    i_l = i % 2
                    w_val = W_block[t][k_b, i_b] * W_local[l][k_l, i_l]
                    if w_val != 0:
                        W[k * 4 + i][r] = w_val
            col += 1
            
    # 2. 4x4x2 Schoolbook (Rank 32)
    # k in 4, 5
    for k_offset in range(2):
        k_global = 4 + k_offset
        for i in range(4):
            for j in range(4):
                r = col
                U[i * 4 + j][r] = 1
                V[j * 4 + k_global][r] = 1
                W[k_global * 4 + i][r] = 1
                col += 1
                
    # 3. 4x4x1 Schoolbook (Rank 16)
    # k in 6
    k_global = 6
    for i in range(4):
        for j in range(4):
            r = col
            U[i * 4 + j][r] = 1
            V[j * 4 + k_global][r] = 1
            W[k_global * 4 + i][r] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
