import time

DIMENSIONS = [2, 2, 6]

def construct():
    start_time = time.time()
    deadline = 120.0 - 5.0 # 5s safety margin
    
    n, m, p = DIMENSIONS
    # Rank is 3 * 7 = 21
    rank = 21
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Local U for Strassen 2x2
    # Indices: 0:a, 1:b, 2:c, 3:d
    U_loc = [
        [1, 0, 1, 0, 1, -1, 0], # a
        [0, 0, 0, 0, 1, 0, 1],  # b
        [0, 1, 0, 0, 0, 1, 0],  # c
        [1, 1, 0, 1, 0, 0, -1]  # d
    ]
    
    # For each of the 3 blocks
    for t in range(3):
        col_offset = t * 7
        
        # Fill U
        for i in range(4):
            for k in range(7):
                U[i][col_offset + k] = U_loc[i][k]
        
        # Fill V
        # Rows: 2t, 2t+1, 6+2t, 7+2t
        r_e = 2 * t
        r_f = 2 * t + 1
        r_g = 6 + 2 * t
        r_h = 7 + 2 * t
        
        # Col 1 (M1): e=1, h=1
        V[r_e][col_offset + 0] = 1
        V[r_h][col_offset + 0] = 1
        # Col 2 (M2): e=1
        V[r_e][col_offset + 1] = 1
        # Col 3 (M3): f=1, h=-1
        V[r_f][col_offset + 2] = 1
        V[r_h][col_offset + 2] = -1
        # Col 4 (M4): g=1, f=-1
        V[r_g][col_offset + 3] = 1
        V[r_f][col_offset + 3] = -1
        # Col 5 (M5): h=1
        V[r_h][col_offset + 4] = 1
        # Col 6 (M6): e=1, f=1
        V[r_e][col_offset + 5] = 1
        V[r_f][col_offset + 5] = 1
        # Col 7 (M7): g=1, h=1
        V[r_g][col_offset + 6] = 1
        V[r_h][col_offset + 6] = 1
        
        # Fill W
        # Rows: 4t, 4t+1, 4t+2, 4t+3
        r_p = 4 * t
        r_q = 4 * t + 2
        r_r = 4 * t + 1
        r_s = 4 * t + 3
        
        # Col 1 (M1): p=1, s=1
        W[r_p][col_offset + 0] = 1
        W[r_s][col_offset + 0] = 1
        # Col 2 (M2): r=1, s=-1
        W[r_r][col_offset + 1] = 1
        W[r_s][col_offset + 1] = -1
        # Col 3 (M3): q=1, s=1
        W[r_q][col_offset + 2] = 1
        W[r_s][col_offset + 2] = 1
        # Col 4 (M4): p=1, r=1
        W[r_p][col_offset + 3] = 1
        W[r_r][col_offset + 3] = 1
        # Col 5 (M5): p=-1, q=1
        W[r_p][col_offset + 4] = -1
        W[r_q][col_offset + 4] = 1
        # Col 6 (M6): s=1
        W[r_s][col_offset + 5] = 1
        # Col 7 (M7): p=1
        W[r_p][col_offset + 6] = 1
        
        if time.time() > deadline:
            break
            
    return {"U": U, "V": V, "W": W}
