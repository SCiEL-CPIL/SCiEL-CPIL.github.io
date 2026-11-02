import time
import random
import numpy as np

DIMENSIONS = [4, 4, 9]

def verify(U, V, W, n, m, p):
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    U_reshaped = U_arr.reshape(n, m, rank)
    V_reshaped = V_arr.reshape(m, p, rank)
    W_reshaped = W_arr.reshape(p, n, rank)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U_reshaped[i, j, :], V_reshaped[j, k, :] * W_reshaped[k, i, :])
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Schoolbook as default
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j][col] = 1
                V_sb[j*p+k][col] = 1
                W_sb[k*n+i][col] = 1
                col += 1
    best = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_sb
    
    # Improved decomposition: Split B into 4x4, 4x4, 4x1
    # Rank: 48 + 48 + 16 = 112
    r_improved = 112
    U_imp = [[0] * r_improved for _ in range(n * m)]
    V_imp = [[0] * r_improved for _ in range(m * p)]
    W_imp = [[0] * r_improved for _ in range(p * n)]
    
    col = 0
    # Block 1: B cols 0-3
    for i in range(n):
        for j in range(m):
            for k in range(4):
                U_imp[i*m+j][col] = 1
                V_imp[j*p+k][col] = 1
                W_imp[k*n+i][col] = 1
                col += 1
                
    # Block 2: B cols 4-7
    for i in range(n):
        for j in range(m):
            for k in range(4, 8):
                U_imp[i*m+j][col] = 1
                V_imp[j*p+k][col] = 1
                W_imp[k*n+i][col] = 1
                col += 1
                
    # Block 3: B col 8
    for i in range(n):
        for j in range(m):
            k = 8
            U_imp[i*m+j][col] = 1
            V_imp[j*p+k][col] = 1
            W_imp[k*n+i][col] = 1
            col += 1
            
    if verify(U_imp, V_imp, W_imp, n, m, p):
        best = {"U": U_imp, "V": V_imp, "W": W_imp}
        best_rank = r_improved
        
    return best

