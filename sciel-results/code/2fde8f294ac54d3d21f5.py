import numpy as np
import random
import time

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    deadline = time.time() + 140  # safety margin

    def schoolbook(n, m, p):
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W

    # Split 7 into 3 and 4
    # Block 1: 2x3x3
    U1, V1, W1 = schoolbook(2, 3, 3)
    R1 = len(U1[0]) # 18
    
    # Block 2: 2x3x4
    U2, V2, W2 = schoolbook(2, 3, 4)
    R2 = len(U2[0]) # 12
    
    total_rank = R1 + R2
    
    # Initialize full matrices
    U_total = [[0] * total_rank for _ in range(6)]
    V_total = [[0] * total_rank for _ in range(21)]
    W_total = [[0] * total_rank for _ in range(14)]
    
    # Copy Block 1
    # U: same rows 0..5
    for i in range(6):
        for c in range(R1):
            U_total[i][c] = U1[i][c]
            
    # V: rows j*7+k for k in 0..2
    # V1 rows are j*3+k
    for j in range(3):
        for k in range(3):
            row_v1 = j * 3 + k
            row_v_total = j * 7 + k
            for c in range(R1):
                V_total[row_v_total][c] = V1[row_v1][c]
                
    # W: rows k*2+i for k in 0..2
    # W1 rows are k*2+i
    for k in range(3):
        for i in range(2):
            row_w1 = k * 2 + i
            row_w_total = k * 2 + i
            for c in range(R1):
                W_total[row_w_total][c] = W1[row_w1][c]

    # Copy Block 2
    offset = R1
    # U: same rows 0..5
    for i in range(6):
        for c in range(R2):
            U_total[i][offset + c] = U2[i][c]
            
    # V: rows j*7+k for k in 3..6
    # V2 rows are j*4+k_local (k_local 0..3)
    for j in range(3):
        for k_local in range(4):
            row_v2 = j * 4 + k_local
            row_v_total = j * 7 + (3 + k_local)
            for c in range(R2):
                V_total[row_v_total][offset + c] = V2[row_v2][c]
                
    # W: rows k*2+i for k in 3..6
    # W2 rows are k_local*2+i (k_local 0..3)
    for k_local in range(4):
        for i in range(2):
            row_w2 = k_local * 2 + i
            row_w_total = (3 + k_local) * 2 + i
            for c in range(R2):
                W_total[row_w_total][offset + c] = W2[row_w2][c]

    return {"U": U_total, "V": V_total, "W": W_total}
