import time
import numpy as np

DIMENSIONS = [5, 5, 5]

def get_strassen_2x2():
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    U[0][0] += 1; U[3][0] += 1
    V[0][0] += 1; V[3][0] += 1
    W[0][0] += 1; W[3][0] += 1
    
    U[2][1] += 1; U[3][1] += 1
    V[0][1] += 1
    W[1][1] += 1
    
    U[0][2] += 1
    V[1][2] += 1; V[3][2] += -1
    W[2][2] += 1
    
    U[3][3] += 1
    V[2][3] += 1; V[0][3] += -1
    W[0][3] += 1; W[2][3] += 1
    
    U[0][4] += 1; U[1][4] += 1
    V[3][4] += 1
    W[3][4] += 1
    
    U[2][5] += 1; U[0][5] += -1
    V[0][5] += 1; V[1][5] += 1
    W[0][5] += -1; W[1][5] += 1
    
    U[1][6] += 1; U[3][6] += -1
    V[2][6] += 1; V[3][6] += 1
    W[0][6] += 1; W[3][6] += 1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    U2, V2, W2 = get_strassen_2x2()
    
    rank_core = 49
    rank_border = 61
    total_rank = rank_core + rank_border
    
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    # 1. Fill core 4x4 (indices 0-3)
    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2
            for i1 in range(2):
                for j1 in range(2):
                    for k1 in range(2):
                        for i2 in range(2):
                            for j2 in range(2):
                                for k2 in range(2):
                                    i = i1 * 2 + i2
                                    j = j1 * 2 + j2
                                    k = k1 * 2 + k2
                                    
                                    u_val = U2[i1*2+j1][r1] * U2[i2*2+j2][r2]
                                    if u_val != 0:
                                        U[i*m + j][r] += u_val
                                        
                                    v_val = V2[j1*2+k1][r1] * V2[j2*2+k2][r2]
                                    if v_val != 0:
                                        V[j*p + k][r] += v_val
                                        
                                    w_val = W2[k1*2+i1][r1] * W2[k2*2+i2][r2]
                                    if w_val != 0:
                                        W[k*n + i][r] += w_val
                                        
    # 2. Fill border terms (indices involving 4)
    col = rank_core
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i < 4 and j < 4 and k < 4:
                    continue
                U[i*m + j][col] = 1.0
                V[j*p + k][col] = 1.0
                W[k*n + i][col] = 1.0
                col += 1
                
    if time.time() - start_time > 140:
        pass # Should be fast enough
        
    return {"U": U, "V": V, "W": W}
