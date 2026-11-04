import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    n, m, p = 6, 6, 6
    
    # 1. Strassen 2x2
    U_str = np.zeros((4, 7), dtype=int)
    V_str = np.zeros((4, 7), dtype=int)
    W_str = np.zeros((4, 7), dtype=int)
    
    # M1
    U_str[0, 0] = 1; U_str[3, 0] = 1
    V_str[0, 0] = 1; V_str[3, 0] = 1
    W_str[0, 0] = 1; W_str[3, 0] = 1
    
    # M2
    U_str[2, 1] = 1; U_str[3, 1] = 1
    V_str[0, 1] = 1
    W_str[1, 1] = 1; W_str[3, 1] = -1
    
    # M3
    U_str[0, 2] = 1
    V_str[1, 2] = 1; V_str[3, 2] = -1
    W_str[2, 2] = 1; W_str[3, 2] = 1
    
    # M4
    U_str[3, 3] = 1
    V_str[0, 3] = -1; V_str[2, 3] = 1
    W_str[0, 3] = 1; W_str[1, 3] = 1
    
    # M5
    U_str[0, 4] = 1; U_str[1, 4] = 1
    V_str[3, 4] = 1
    W_str[0, 4] = -1; W_str[2, 4] = 1
    
    # M6
    U_str[0, 5] = -1; U_str[2, 5] = 1
    V_str[0, 5] = 1; V_str[1, 5] = 1
    W_str[3, 5] = 1
    
    # M7
    U_str[1, 6] = 1; U_str[3, 6] = -1
    V_str[2, 6] = 1; V_str[3, 6] = 1
    W_str[0, 6] = 1

    # 2. Schoolbook 3x3
    U_sch = np.zeros((9, 27), dtype=int)
    V_sch = np.zeros((9, 27), dtype=int)
    W_sch = np.zeros((9, 27), dtype=int)
    
    for i in range(3):
        for j in range(3):
            for k in range(3):
                r = i * 9 + j * 3 + k
                U_sch[i * 3 + j, r] = 1
                V_sch[j * 3 + k, r] = 1
                W_sch[k * 3 + i, r] = 1

    # 3. Kronecker
    rank = 189
    U = np.zeros((36, rank), dtype=int)
    V = np.zeros((36, rank), dtype=int)
    W = np.zeros((36, rank), dtype=int)
    
    for r_s in range(7):
        for r_c in range(27):
            c = r_s * 27 + r_c
            if time.time() > deadline:
                break
                
            # U
            for I in range(2):
                for J in range(2):
                    u_val = U_str[I*2+J, r_s]
                    if u_val != 0:
                        for i in range(3):
                            for j in range(3):
                                u_sch_val = U_sch[i*3+j, r_c]
                                if u_sch_val != 0:
                                    R = I*3 + i
                                    C = J*3 + j
                                    U[R*6+C, c] = u_val * u_sch_val
                                    
            # V
            for J in range(2):
                for K in range(2):
                    v_val = V_str[J*2+K, r_s]
                    if v_val != 0:
                        for j in range(3):
                            for k in range(3):
                                v_sch_val = V_sch[j*3+k, r_c]
                                if v_sch_val != 0:
                                    C = J*3 + j
                                    K_glob = K*3 + k
                                    V[C*6+K_glob, c] = v_val * v_sch_val
                                    
            # W
            for K in range(2):
                for I in range(2):
                    w_val = W_str[K*2+I, r_s]
                    if w_val != 0:
                        for k in range(3):
                            for i in range(3):
                                w_sch_val = W_sch[k*3+i, r_c]
                                if w_sch_val != 0:
                                    K_glob = K*3 + k
                                    R = I*3 + i
                                    W[K_glob*6+R, c] = w_val * w_sch_val

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
