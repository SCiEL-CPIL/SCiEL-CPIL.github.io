import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 140.0
    
    n, m, p = 4, 4, 8
    rank = 98
    
    U_2 = np.zeros((4, 7))
    V_2 = np.zeros((4, 7))
    W_2 = np.zeros((4, 7))
    
    U_2[0, 0] = 1; U_2[3, 0] = 1
    V_2[0, 0] = 1; V_2[3, 0] = 1
    W_2[0, 0] = 1; W_2[3, 0] = 1
    
    U_2[2, 1] = 1; U_2[3, 1] = 1
    V_2[0, 1] = 1
    W_2[2, 1] = 1; W_2[3, 1] = -1
    
    U_2[0, 2] = 1
    V_2[1, 2] = 1; V_2[3, 2] = -1
    W_2[1, 2] = 1; W_2[3, 2] = 1
    
    U_2[3, 3] = 1
    V_2[2, 3] = 1; V_2[0, 3] = -1
    W_2[0, 3] = 1; W_2[2, 3] = 1
    
    U_2[0, 4] = 1; U_2[1, 4] = 1
    V_2[3, 4] = 1
    W_2[0, 4] = -1; W_2[1, 4] = 1
    
    U_2[2, 5] = 1; U_2[0, 5] = -1
    V_2[0, 5] = 1; V_2[1, 5] = 1
    W_2[3, 5] = 1
    
    U_2[1, 6] = 1; U_2[3, 6] = -1
    V_2[2, 6] = 1; V_2[3, 6] = 1
    W_2[0, 6] = 1
    
    A_specs = [
        [(1, 0), (1, 3)],
        [(1, 2), (1, 3)],
        [(1, 0)],
        [(1, 3)],
        [(1, 0), (1, 1)],
        [(1, 2), (-1, 0)],
        [(1, 1), (-1, 3)]
    ]
    
    B_specs = [
        [(1, 0), (1, 3)],
        [(1, 0)],
        [(1, 1), (-1, 3)],
        [(1, 2), (-1, 0)],
        [(1, 3)],
        [(1, 0), (1, 1)],
        [(1, 2), (1, 3)]
    ]
    
    C_specs = [
        [(1, 0), (1, 3)],
        [(1, 2), (-1, 3)],
        [(1, 1), (1, 3)],
        [(1, 0), (1, 2)],
        [(-1, 0), (1, 1)],
        [(1, 3)],
        [(1, 0)]
    ]
    
    U_44 = np.zeros((16, 49))
    V_44 = np.zeros((16, 49))
    W_44 = np.zeros((16, 49))
    
    for k_p in range(7):
        for m_s in range(7):
            col = k_p * 7 + m_s
            
            for r in range(2):
                for c in range(2):
                    a_eff_idx = r * 2 + c
                    coeff_u2 = U_2[a_eff_idx, m_s]
                    if coeff_u2 == 0: continue
                    for coeff_a, block_idx in A_specs[k_p]:
                        bi, bj = divmod(block_idx, 2)
                        global_r = bi * 2 + r
                        global_c = bj * 2 + c
                        global_a_idx = global_r * 4 + global_c
                        U_44[global_a_idx, col] += coeff_a * coeff_u2
            
            for r in range(2):
                for c in range(2):
                    b_eff_idx = r * 2 + c
                    coeff_v2 = V_2[b_eff_idx, m_s]
                    if coeff_v2 == 0: continue
                    for coeff_b, block_idx in B_specs[k_p]:
                        bi, bj = divmod(block_idx, 2)
                        global_r = bi * 2 + r
                        global_c = bj * 2 + c
                        global_b_idx = global_r * 4 + global_c
                        V_44[global_b_idx, col] += coeff_b * coeff_v2
            
            for r in range(2):
                for c in range(2):
                    p_eff_idx = r * 2 + c
                    coeff_w2 = W_2[p_eff_idx, m_s]
                    if coeff_w2 == 0: continue
                    for coeff_c, c_block_idx in C_specs[k_p]:
                        bi, bj = divmod(c_block_idx, 2)
                        global_i = bi * 2 + r
                        global_k = bj * 2 + c
                        global_w_idx = global_k * 4 + global_i
                        W_44[global_w_idx, col] += coeff_c * coeff_w2
                        
    U = np.zeros((16, 98))
    V = np.zeros((32, 98))
    W = np.zeros((32, 98))
    
    U[:, 0:49] = U_44
    U[:, 49:98] = U_44
    
    for j in range(4):
        for k in range(4):
            v_idx_44 = j * 4 + k
            v_idx_8_left = j * 8 + k
            v_idx_8_right = j * 8 + (k + 4)
            V[v_idx_8_left, 0:49] = V_44[v_idx_44, 0:49]
            V[v_idx_8_right, 49:98] = V_44[v_idx_44, 0:49]
            
    for k in range(4):
        for i in range(4):
            w_idx_44 = k * 4 + i
            w_idx_8_left = k * 4 + i
            w_idx_8_right = (k + 4) * 4 + i
            W[w_idx_8_left, 0:49] = W_44[w_idx_44, 0:49]
            W[w_idx_8_right, 49:98] = W_44[w_idx_44, 0:49]

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
