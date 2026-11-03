import time

DIMENSIONS = [4, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS  # 4, 4, 7
    
    # Build the Strassen 2x2x2 rank-7 decomposition
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[3][0] = 1
    
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[1][1] = 1; W2[3][1] = -1
    
    U2[0][2] = 1
    V2[1][2] = 1; V2[3][2] = -1
    W2[2][2] = 1; W2[3][2] = 1
    
    U2[3][3] = 1
    V2[2][3] = 1; V2[0][3] = -1
    W2[0][3] = 1; W2[1][3] = 1
    
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    W2[0][4] = -1; W2[2][4] = 1
    
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[1][5] = 1
    W2[3][5] = 1
    
    U2[1][6] = 1; U2[3][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    W2[0][6] = 1
    
    rank_strassen = 49
    rank_schoolbook = 48
    total_rank = rank_strassen + rank_schoolbook
    
    U = [[0]*total_rank for _ in range(16)]
    V = [[0]*total_rank for _ in range(28)]
    W = [[0]*total_rank for _ in range(28)]
    
    block_products = [
        {'A': [(0,0,1), (1,1,1)], 'B': [(0,0,1), (1,1,1)], 'C': [(0,0,1), (1,1,1)]},
        {'A': [(1,0,1), (1,1,1)], 'B': [(0,0,1)], 'C': [(1,0,1), (1,1,-1)]},
        {'A': [(0,0,1)], 'B': [(0,1,1), (1,1,-1)], 'C': [(0,1,1), (1,1,1)]},
        {'A': [(1,1,1)], 'B': [(1,0,1), (0,0,-1)], 'C': [(0,0,1), (1,0,1)]},
        {'A': [(0,0,1), (0,1,1)], 'B': [(1,1,1)], 'C': [(0,0,-1), (0,1,1)]},
        {'A': [(1,0,1), (0,0,-1)], 'B': [(0,0,1), (0,1,1)], 'C': [(1,1,1)]},
        {'A': [(0,1,1), (1,1,-1)], 'B': [(1,0,1), (1,1,1)], 'C': [(0,0,1), (1,1,-1)]},
    ]
    
    for bp_idx, bp in enumerate(block_products):
        if time.time() > deadline:
            break
        for inner_c in range(7):
            col = bp_idx * 7 + inner_c
            
            for (bi, bj, coeff) in bp['A']:
                for ii in range(2):
                    for jj in range(2):
                        global_i = bi * 2 + ii
                        global_j = bj * 2 + jj
                        U[global_i * 4 + global_j][col] += coeff * U2[ii * 2 + jj][inner_c]
            
            for (bj, bk, coeff) in bp['B']:
                for jj in range(2):
                    for kk in range(2):
                        global_j = bj * 2 + jj
                        global_k = bk * 2 + kk
                        V[global_j * 7 + global_k][col] += coeff * V2[jj * 2 + kk][inner_c]
            
            for (bi, bk, coeff) in bp['C']:
                for kk in range(2):
                    for ii in range(2):
                        global_k = bk * 2 + kk
                        global_i = bi * 2 + ii
                        W[global_k * 4 + global_i][col] += coeff * W2[kk * 2 + ii][inner_c]
    
    col_offset = 49
    for i in range(4):
        for j in range(4):
            for k in range(3):
                global_k = k + 4
                col = col_offset + i * 4 * 3 + j * 3 + k
                U[i * 4 + j][col] = 1
                V[j * 7 + global_k][col] = 1
                W[global_k * 4 + i][col] = 1
    
    return {"U": U, "V": V, "W": W}
