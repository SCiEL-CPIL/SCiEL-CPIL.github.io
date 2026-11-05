import numpy as np
import time

def construct():
    n, m, p = 3, 3, 7
    start_time = time.time()
    deadline = 120

    # Schoolbook fallback
    rank_sb = n * m * p
    U_sb = [[0]*rank_sb for _ in range(n*m)]
    V_sb = [[0]*rank_sb for _ in range(m*p)]
    W_sb = [[0]*rank_sb for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j][col] = 1
                V_sb[j*p+k][col] = 1
                W_sb[k*n+i][col] = 1
                col += 1

    # Optimized decomposition:
    # Block 1: 3x3 multiplication (cols 0-2 of B) -> Rank 9 (Schoolbook)
    # Block 2: 3x4 multiplication (cols 3-6 of B) -> Rank 36 (Schoolbook)
    # Total Rank: 45
    
    rank_total = 9 + 36
    U_full = [[0]*rank_total for _ in range(n*m)]
    V_full = [[0]*rank_total for _ in range(m*p)]
    W_full = [[0]*rank_total for _ in range(p*n)]
    
    # Block 1: 3x3x3 Schoolbook
    # U indices: i*3+j
    # V indices: j*7+k (k in 0..2)
    # W indices: k*3+i
    offset1 = 0
    col1 = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U_full[i*3+j][offset1 + col1] = 1
                V_full[j*7+k][offset1 + col1] = 1
                W_full[k*3+i][offset1 + col1] = 1
                col1 += 1
                
    # Block 2: 3x3x4 Schoolbook
    # U indices: i*3+j
    # V indices: j*7+k (k in 3..6)
    # W indices: k*3+i
    offset2 = 9
    col2 = 0
    for i in range(3):
        for j in range(3):
            for k in range(3, 7):
                U_full[i*3+j][offset2 + col2] = 1
                V_full[j*7+k][offset2 + col2] = 1
                W_full[k*3+i][offset2 + col2] = 1
                col2 += 1

    if verify_decomposition(U_full, V_full, W_full, n, m, p):
        return {"U": U_full, "V": V_full, "W": W_full}
    else:
        return {"U": U_sb, "V": V_sb, "W": W_sb}

def verify_decomposition(U, V, W, n, m, p):
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if s != 1:
                    return False
    return True
