import time
import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    DEADLINE = start_time + 140  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # Strassen decomposition for 2x2 matrix multiplication
    U_local = np.zeros((4, 7), dtype=int)
    V_local = np.zeros((4, 7), dtype=int)
    W_local = np.zeros((4, 7), dtype=int)
    
    # M1: (A11+A22)*(B11+B22) -> C11, C22
    U_local[0, 0] = 1; U_local[3, 0] = 1
    V_local[0, 0] = 1; V_local[3, 0] = 1
    W_local[0, 0] = 1
    W_local[3, 0] = 1
    
    # M2: (A21+A22)*B11 -> C21, C22
    U_local[2, 1] = 1; U_local[3, 1] = 1
    V_local[0, 1] = 1
    W_local[2, 1] = 1
    W_local[3, 1] = -1
    
    # M3: A11*(B12-B22) -> C12, C22
    U_local[0, 2] = 1
    V_local[1, 2] = 1; V_local[3, 2] = -1
    W_local[1, 2] = 1
    W_local[3, 2] = 1
    
    # M4: A22*(B21-B11) -> C11, C21
    U_local[3, 3] = 1
    V_local[2, 3] = 1; V_local[0, 3] = -1
    W_local[0, 3] = 1
    W_local[2, 3] = 1
    
    # M5: (A11+A12)*B22 -> C11, C12
    U_local[0, 4] = 1; U_local[1, 4] = 1
    V_local[3, 4] = 1
    W_local[0, 4] = -1
    W_local[1, 4] = 1
    
    # M6: (A21-A11)*(B11+B12) -> C22
    U_local[2, 5] = 1; U_local[0, 5] = -1
    V_local[0, 5] = 1; V_local[1, 5] = 1
    W_local[3, 5] = 1
    
    # M7: (A12-A22)*(B21+B22) -> C11
    U_local[1, 6] = 1; U_local[3, 6] = -1
    V_local[2, 6] = 1; V_local[3, 6] = 1
    W_local[0, 6] = 1
    
    total_rank = 56
    
    U = np.zeros((n * m, total_rank), dtype=int)
    V = np.zeros((m * p, total_rank), dtype=int)
    W = np.zeros((p * n, total_rank), dtype=int)
    
    for j_group in range(2):
        a_map = [
            0 * 4 + j_group * 2,
            0 * 4 + j_group * 2 + 1,
            1 * 4 + j_group * 2,
            1 * 4 + j_group * 2 + 1
        ]
        
        for k_group in range(4):
            b_map = [
                (j_group * 2) * 8 + k_group * 2,
                (j_group * 2) * 8 + k_group * 2 + 1,
                (j_group * 2 + 1) * 8 + k_group * 2,
                (j_group * 2 + 1) * 8 + k_group * 2 + 1
            ]
            
            c_map = [
                (k_group * 2) * 2 + 0,
                (k_group * 2 + 1) * 2 + 0,
                (k_group * 2) * 2 + 1,
                (k_group * 2 + 1) * 2 + 1
            ]
            
            base_r = (j_group * 4 + k_group) * 7
            
            for l in range(7):
                r = base_r + l
                for la in range(4):
                    val = U_local[la, l]
                    if val != 0:
                        U[a_map[la], r] += val
                for lb in range(4):
                    val = V_local[lb, l]
                    if val != 0:
                        V[b_map[lb], r] += val
                for lc in range(4):
                    val = W_local[lc, l]
                    if val != 0:
                        W[c_map[lc], r] += val
    
    if time.time() > DEADLINE:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
