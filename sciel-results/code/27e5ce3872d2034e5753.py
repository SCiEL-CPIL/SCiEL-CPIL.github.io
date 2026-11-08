def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    import time
    import numpy as np
    start_time = time.time()
    deadline = start_time + 140
    
    # 2x2x2 Strassen rank-7 decomposition
    S_U = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1],
    ]
    S_V = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1],
    ]
    S_W = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0],
    ]
    
    # Build 4x4x2 rank-28 decomposition using 4 Strassen blocks
    # U_442: (16, 28), V_442: (8, 28), W_442: (8, 28)
    U_442 = [[0]*28 for _ in range(16)]
    V_442 = [[0]*28 for _ in range(8)]
    W_442 = [[0]*28 for _ in range(8)]
    
    col_offset = 0
    for bi in range(2):
        for bj in range(2):
            for c in range(7):
                col = col_offset + c
                for di in range(2):
                    for dj in range(2):
                        i = 2*bi + di
                        j = 2*bj + dj
                        U_442[i*4 + j][col] = S_U[di*2 + dj][c]
                for dj in range(2):
                    for dk in range(2):
                        j = 2*bj + dj
                        k = dk  # k in [0,2) for the 4x4x2 sub-problem
                        V_442[j*2 + k][col] = S_V[dj*2 + dk][c]
                for dk in range(2):
                    for di in range(2):
                        k = dk
                        i = 2*bi + di
                        W_442[k*4 + i][col] = S_W[dk*2 + di][c]
            col_offset += 7
    
    # Try to find a rank-21 decomposition for 4x4x4
    # If not found, use rank-56 (Strassen)
    U_444 = None
    V_444 = None
    W_444 = None
    
    # Try to find rank-21 decomposition
    if time.time() < deadline - 30:
        # Try a heuristic search
        # For now, fall back to rank-56 Strassen
        U_444 = [[0]*56 for _ in range(16)]
        V_444 = [[0]*56 for _ in range(16)]
        W_444 = [[0]*56 for _ in range(16)]
        
        col_offset = 0
        for bi in range(2):
            for bj in range(2):
                for bk in range(2):
                    for c in range(7):
                        col = col_offset + c
                        for di in range(2):
                            for dj in range(2):
                                i = 2*bi + di
                                j = 2*bj + dj
                                U_444[i*4 + j][col] = S_U[di*2 + dj][c]
                        for dj in range(2):
                            for dk in range(2):
                                j = 2*bj + dj
                                k = 2*bk + dk
                                V_444[j*4 + k][col] = S_V[dj*2 + dk][c]
                        for dk in range(2):
                            for di in range(2):
                                k = 2*bk + dk
                                i = 2*bi + di
                                W_444[k*4 + i][col] = S_W[dk*2 + di][c]
                    col_offset += 7
        
        rank_444 = 56
    else:
        rank_444 = 56
        U_444 = [[0]*56 for _ in range(16)]
        V_444 = [[0]*56 for _ in range(16)]
        W_444 = [[0]*56 for _ in range(16)]
        # ... same as above
    
    # Combine
    total_rank = rank_444 + 28
    U = [[0]*total_rank for _ in range(16)]
    V = [[0]*total_rank for _ in range(24)]
    W = [[0]*total_rank for _ in range(24)]
    
    # Copy 4x4x4 part
    for i in range(16):
        for s in range(rank_444):
            U[i][s] = U_444[i][s]
    for j in range(4):
        for k in range(4):
            for s in range(rank_444):
                V[j*6 + k][s] = V_444[j*4 + k][s]
    for k in range(4):
        for i in range(4):
            for s in range(rank_444):
                W[k*4 + i][s] = W_444[k*4 + i][s]
    
    # Copy 4x4x2 part
    for i in range(16):
        for s in range(28):
            U[i][rank_444 + s] = U_442[i][s]
    for j in range(4):
        for k in range(2):
            for s in range(28):
                V[j*6 + 4 + k][rank_444 + s] = V_442[j*2 + k][s]
    for k in range(2):
        for i in range(4):
            for s in range(28):
                W[(4 + k)*4 + i][rank_444 + s] = W_442[k*4 + i][s]
    
    return {"U": U, "V": V, "W": W}
