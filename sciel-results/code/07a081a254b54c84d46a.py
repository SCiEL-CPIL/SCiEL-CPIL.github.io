import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s safety margin

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n

    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W, rank

    # Strassen decomposition for 2x2x2
    # U: 4x7, V: 4x7, W: 4x7
    # Indices for A: 0:a11, 1:a12, 2:a21, 3:a22
    # Indices for B: 0:b11, 1:b12, 2:b21, 3:b22
    # Indices for C: 0:c11, 1:c12, 2:c21, 3:c22
    
    U_loc = np.zeros((4, 7))
    V_loc = np.zeros((4, 7))
    W_loc = np.zeros((4, 7))
    
    # m1
    U_loc[0, 0] = 1; U_loc[3, 0] = 1
    V_loc[0, 0] = 1; V_loc[3, 0] = 1
    W_loc[0, 0] = 1; W_loc[3, 0] = 1
    
    # m2
    U_loc[2, 1] = 1; U_loc[3, 1] = 1
    V_loc[0, 1] = 1
    W_loc[2, 1] = 1; W_loc[3, 1] = -1
    
    # m3
    U_loc[0, 2] = 1
    V_loc[1, 2] = 1; V_loc[3, 2] = -1
    W_loc[1, 2] = 1; W_loc[3, 2] = 1
    
    # m4
    U_loc[3, 3] = 1
    V_loc[0, 3] = -1; V_loc[2, 3] = 1
    W_loc[0, 3] = 1; W_loc[2, 3] = 1
    
    # m5
    U_loc[0, 4] = 1; U_loc[1, 4] = 1
    V_loc[3, 4] = 1
    W_loc[0, 4] = -1; W_loc[1, 4] = 1
    
    # m6
    U_loc[0, 5] = -1; U_loc[2, 5] = 1
    V_loc[0, 5] = 1; V_loc[1, 5] = 1
    W_loc[3, 5] = 1
    
    # m7
    U_loc[1, 6] = 1; U_loc[3, 6] = -1
    V_loc[2, 6] = 1; V_loc[3, 6] = 1
    W_loc[0, 6] = 1

    # Construct global U, V, W for 2x4x8
    # Rank = 8 blocks * 7 = 56
    rank = 56
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    # Define blocks
    # Block 0: A_L (A[0,1,2,3]), B_T0 (B[0,1,8,9]), C_00 (C[0,1,8,9])
    # Block 1: A_L, B_T1 (B[2,3,10,11]), C_01 (C[2,3,10,11])
    # Block 2: A_L, B_T2 (B[4,5,12,13]), C_02 (C[4,5,12,13])
    # Block 3: A_L, B_T3 (B[6,7,14,15]), C_03 (C[6,7,14,15])
    # Block 4: A_R (A[2,3,6,7]), B_B0 (B[16,17,24,25]), C_10 (C[0,1,8,9])
    # Block 5: A_R, B_B1 (B[18,19,26,27]), C_11 (C[2,3,10,11])
    # Block 6: A_R, B_B2 (B[20,21,28,29]), C_12 (C[4,5,12,13])
    # Block 7: A_R, B_B3 (B[22,23,30,31]), C_13 (C[6,7,14,15])
    
    blocks = [
        # (A_indices, B_indices, C_indices)
        ([0, 1, 2, 3], [0, 1, 8, 9], [0, 1, 8, 9]),
        ([0, 1, 2, 3], [2, 3, 10, 11], [2, 3, 10, 11]),
        ([0, 1, 2, 3], [4, 5, 12, 13], [4, 5, 12, 13]),
        ([0, 1, 2, 3], [6, 7, 14, 15], [6, 7, 14, 15]),
        ([2, 3, 6, 7], [16, 17, 24, 25], [0, 1, 8, 9]),
        ([2, 3, 6, 7], [18, 19, 26, 27], [2, 3, 10, 11]),
        ([2, 3, 6, 7], [20, 21, 28, 29], [4, 5, 12, 13]),
        ([2, 3, 6, 7], [22, 23, 30, 31], [6, 7, 14, 15]),
    ]
    
    for b_idx, (a_idx, b_idx_list, c_idx) in enumerate(blocks):
        base_col = b_idx * 7
        for r in range(7):
            col = base_col + r
            for loc_idx, g_idx in enumerate(a_idx):
                U[g_idx, col] = U_loc[loc_idx, r]
            for loc_idx, g_idx in enumerate(b_idx_list):
                V[g_idx, col] = V_loc[loc_idx, r]
            for loc_idx, g_idx in enumerate(c_idx):
                W[g_idx, col] = W_loc[loc_idx, r]
                
    # Remap W indices from row-major (i*p+k) to required (k*n+i)
    W_final = np.zeros((pn, rank))
    for i in range(n):
        for k in range(p):
            my_idx = i * p + k
            req_idx = k * n + i
            W_final[req_idx, :] = W[my_idx, :]
            
    W = W_final
    
    # Check validity
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if not np.isclose(val, 1.0, atol=1e-9):
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
            
    if not valid:
        # Fallback to schoolbook if something went wrong
        U, V, W, rank = schoolbook()
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
