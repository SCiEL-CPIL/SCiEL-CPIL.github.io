import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = 4, 4, 6
    rank = 84
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2x2 (Rank 7)
    # U_s: (4, 7), V_s: (4, 7), W_s: (4, 7)
    U_s = [[0]*7 for _ in range(4)]
    V_s = [[0]*7 for _ in range(4)]
    W_s = [[0]*7 for _ in range(4)]
    
    # M1: (a+d)(e+h)
    U_s[0][0] = 1; U_s[3][0] = 1
    V_s[0][0] = 1; V_s[3][0] = 1
    W_s[0][0] = 1; W_s[3][0] = 1
    
    # M2: (c+d)(e+h)
    U_s[2][1] = 1; U_s[3][1] = 1
    V_s[0][1] = 1; V_s[3][1] = 1
    W_s[0][1] = -1
    
    # M3: a(f-h)
    U_s[0][2] = 1
    V_s[1][2] = 1; V_s[3][2] = -1
    W_s[1][2] = 1; W_s[3][2] = -1
    
    # M4: d(g-e)
    U_s[3][3] = 1
    V_s[2][3] = 1; V_s[0][3] = -1
    W_s[0][3] = 1; W_s[2][3] = -1
    
    # M5: (a+b)h
    U_s[0][4] = 1; U_s[1][4] = 1
    V_s[3][4] = 1
    W_s[1][4] = 1; W_s[3][4] = 1
    
    # M6: (c+d)g
    U_s[2][5] = 1; U_s[3][5] = 1
    V_s[2][5] = 1
    W_s[2][5] = 1; W_s[3][5] = -1
    
    # M7: (a-c)(e+f)
    U_s[0][6] = 1; U_s[2][6] = -1
    V_s[0][6] = 1; V_s[1][6] = 1
    W_s[0][6] = 1; W_s[1][6] = -1
    
    # Schoolbook 2x2x3 (Rank 12)
    # U_b: (4, 12), V_b: (6, 12), W_b: (6, 12)
    U_b = [[0]*12 for _ in range(4)]
    V_b = [[0]*12 for _ in range(6)]
    W_b = [[0]*12 for _ in range(6)]
    
    col_b = 0
    for i in range(2):
        for j in range(2):
            for k in range(3):
                U_b[i*2+j][col_b] = 1
                V_b[j*3+k][col_b] = 1
                W_b[k*2+i][col_b] = 1
                col_b += 1
                
    # Combine
    for c in range(7):
        for d in range(12):
            col = c * 12 + d
            # U: (4x4) -> (2x2) (2x2)
            for i1 in range(2):
                for j1 in range(2):
                    u_val_s = U_s[i1*2+j1][c]
                    if u_val_s == 0: continue
                    for i2 in range(2):
                        for j2 in range(2):
                            u_val_b = U_b[i2*2+j2][d]
                            if u_val_b == 0: continue
                            I = i1*2 + i2
                            J = j1*2 + j2
                            U[I][col] += u_val_s * u_val_b
                            
            # V: (4x6) -> (2x2) (2x3)
            # V index: j_global * 6 + k_global
            # j_global = j1*2 + j2
            # k_global = k1*3 + k2
            for j1 in range(2):
                for k1 in range(2):
                    v_val_s = V_s[j1*2+k1][c]
                    if v_val_s == 0: continue
                    for j2 in range(2):
                        for k2 in range(3):
                            v_val_b = V_b[j2*3+k2][d]
                            if v_val_b == 0: continue
                            J = j1*2 + j2
                            K = k1*3 + k2
                            V[J*6+K][col] += v_val_s * v_val_b
                            
            # W: (6x4) -> (2x3) (2x2)
            # W index: k_global * 4 + i_global
            # k_global = k1*3 + k2
            # i_global = i1*2 + i2
            for k1 in range(2):
                for i1 in range(2):
                    w_val_s = W_s[k1*2+i1][c]
                    if w_val_s == 0: continue
                    for k2 in range(3):
                        for i2 in range(2):
                            w_val_b = W_b[k2*2+i2][d]
                            if w_val_b == 0: continue
                            K = k1*3 + k2
                            I = i1*2 + i2
                            W[K*4+I][col] += w_val_s * w_val_b
                            
    return {"U": U, "V": V, "W": W}
