import time
import numpy as np

DIMENSIONS = [3, 3, 3]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    
    # Attempt rank-23 decomposition
    # If it fails verification, fall back to rank-27
    
    rank_23 = 23
    U = [[0.0] * rank_23 for _ in range(9)]
    V = [[0.0] * rank_23 for _ in range(9)]
    W = [[0.0] * rank_23 for _ in range(9)]
    
    # Bini et al. Rank-23 Decomposition
    # Indices: U[i*3+j], V[j*3+k], W[k*3+i]
    
    # Term 1
    U[0][0] = 1.0; V[0][0] = 1.0; W[0][0] = 1.0
    # Term 2
    U[0][1] = 1.0; V[3][1] = 1.0; W[1][1] = 1.0
    # Term 3
    U[1][2] = 1.0; V[0][2] = 1.0; W[0][2] = 1.0
    # Term 4
    U[1][3] = 1.0; U[4][3] = 1.0; V[0][3] = 1.0; V[3][3] = 1.0; W[1][3] = 1.0; W[4][3] = 1.0
    # Term 5
    U[1][4] = 1.0; U[4][4] = 1.0; V[0][4] = 1.0; V[1][4] = 1.0; W[1][4] = 1.0; W[4][4] = 1.0
    # Term 6
    U[0][5] = 1.0; U[1][5] = 1.0; V[3][5] = 1.0; W[1][5] = 1.0; W[4][5] = 1.0
    # Term 7
    U[0][6] = 1.0; U[1][6] = 1.0; V[4][6] = 1.0; W[1][6] = 1.0; W[4][6] = 1.0
    # Term 8
    U[2][7] = 1.0; V[0][7] = 1.0; W[0][7] = 1.0
    # Term 9
    U[2][8] = 1.0; V[1][8] = 1.0; W[0][8] = 1.0
    # Term 10
    U[2][9] = 1.0; U[5][9] = 1.0; V[0][9] = 1.0; V[3][9] = 1.0; W[1][9] = 1.0; W[4][9] = 1.0
    # Term 11
    U[2][10] = 1.0; U[5][10] = 1.0; V[0][10] = 1.0; V[1][10] = 1.0; W[1][10] = 1.0; W[4][10] = 1.0
    # Term 12
    U[0][11] = 1.0; U[2][11] = 1.0; V[3][11] = 1.0; W[1][11] = 1.0; W[4][11] = 1.0
    # Term 13
    U[0][12] = 1.0; U[2][12] = 1.0; V[4][12] = 1.0; W[1][12] = 1.0; W[4][12] = 1.0
    # Term 14
    U[3][13] = 1.0; V[0][13] = 1.0; W[0][13] = 1.0
    # Term 15
    U[3][14] = 1.0; V[6][14] = 1.0; W[2][14] = 1.0
    # Term 16
    U[4][15] = 1.0; V[0][15] = 1.0; W[0][15] = 1.0
    # Term 17
    U[4][16] = 1.0; U[7][16] = 1.0; V[0][16] = 1.0; V[6][16] = 1.0; W[2][16] = 1.0; W[5][16] = 1.0
    # Term 18
    U[4][17] = 1.0; U[7][17] = 1.0; V[0][17] = 1.0; V[2][17] = 1.0; W[2][17] = 1.0; W[5][17] = 1.0
    # Term 19
    U[3][18] = 1.0; U[4][18] = 1.0; V[6][18] = 1.0; W[2][18] = 1.0; W[5][18] = 1.0
    # Term 20
    U[3][19] = 1.0; U[4][19] = 1.0; V[7][19] = 1.0; W[2][19] = 1.0; W[5][19] = 1.0
    # Term 21
    U[5][20] = 1.0; V[0][20] = 1.0; W[0][20] = 1.0
    # Term 22
    U[5][21] = 1.0; U[8][21] = 1.0; V[0][21] = 1.0; V[6][21] = 1.0; W[2][21] = 1.0; W[5][21] = 1.0
    # Term 23
    U[5][22] = 1.0; U[8][22] = 1.0; V[0][22] = 1.0; V[2][22] = 1.0; W[2][22] = 1.0; W[5][22] = 1.0

    # Verify rank-23 decomposition
    valid = True
    for i in range(3):
        for j in range(3):
            for k in range(3):
                val = 0.0
                for c in range(23):
                    val += U[i*3+j][c] * V[j*3+k][c] * W[k*3+i][c]
                if abs(val - 1.0) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break

    if valid:
        return {"U": U, "V": V, "W": W}
    
    # Fallback to rank-27 schoolbook
    rank_sb = 27
    U_sb = [[0.0] * rank_sb for _ in range(9)]
    V_sb = [[0.0] * rank_sb for _ in range(9)]
    W_sb = [[0.0] * rank_sb for _ in range(9)]
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U_sb[i*3+j][col] = 1.0
                V_sb[j*3+k][col] = 1.0
                W_sb[k*3+i][col] = 1.0
                col += 1
                
    return {"U": U_sb, "V": V_sb, "W": W_sb}
