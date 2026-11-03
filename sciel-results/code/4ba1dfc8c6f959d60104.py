import numpy as np
import time

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Strassen decomposition for 2x2x2 (rank 7)
    U_S = np.zeros((4, 7))
    V_S = np.zeros((4, 7))
    W_S = np.zeros((4, 7))
    
    # M1
    U_S[0, 0] = 1; U_S[3, 0] = 1
    V_S[0, 0] = 1; V_S[3, 0] = 1
    W_S[0, 0] = 1; W_S[1, 0] = 1; W_S[2, 0] = 1; W_S[3, 0] = 1
    
    # M2
    U_S[2, 1] = 1; U_S[3, 1] = 1
    V_S[0, 1] = 1
    W_S[2, 1] = 1; W_S[3, 1] = 1
    
    # M3
    U_S[0, 2] = 1
    V_S[2, 2] = 1; V_S[3, 2] = -1
    W_S[0, 2] = 1; W_S[2, 2] = 1
    
    # M4
    U_S[3, 3] = 1
    V_S[1, 3] = 1; V_S[0, 3] = -1
    W_S[1, 3] = 1; W_S[3, 3] = 1
    
    # M5
    U_S[0, 4] = 1; U_S[1, 4] = 1
    V_S[3, 4] = 1
    W_S[1, 4] = 1; W_S[3, 4] = 1
    
    # M6
    U_S[2, 5] = 1; U_S[0, 5] = -1
    V_S[0, 5] = 1; V_S[1, 5] = 1
    W_S[2, 5] = 1; W_S[0, 5] = 1
    
    # M7
    U_S[1, 6] = 1; U_S[3, 6] = -1
    V_S[2, 6] = 1; V_S[3, 6] = 1
    W_S[3, 6] = 1; W_S[1, 6] = 1
    
    # Part 1: 4x4x4 multiplication
    U_1 = np.kron(U_S, U_S)
    V_1 = np.kron(V_S, V_S)
    W_1 = np.kron(W_S, W_S)
    
    # Part 2: 4x4x2 multiplication
    # S_4 for 2x2x1
    U_4 = np.zeros((4, 4))
    V_4 = np.zeros((2, 4))
    W_4 = np.zeros((2, 4))
    
    U_4[0, 0] = 1
    V_4[0, 0] = 1
    W_4[0, 0] = 1
    
    U_4[1, 1] = 1
    V_4[1, 1] = 1
    W_4[0, 1] = 1
    
    U_4[2, 2] = 1
    V_4[0, 2] = 1
    W_4[1, 2] = 1
    
    U_4[3, 3] = 1
    V_4[1, 3] = 1
    W_4[1, 3] = 1
    
    U_2 = np.kron(U_S, U_4)
    V_2 = np.kron(V_S, V_4)
    W_2 = np.kron(W_S, W_4)
    
    # Combine
    R1 = U_1.shape[1]
    R2 = U_2.shape[1]
    R_total = R1 + R2
    
    U_total = np.hstack([U_1, U_2])
    
    # V_total: 24 rows
    V_total = np.zeros((24, R_total))
    
    # Map V_1 rows (16) to global V rows (24)
    # V_1 rows are row-major 4x4 (j, k) with k in 0..3
    # Global index: j * 6 + k
    for r in range(16):
        j = r // 4
        k = r % 4
        global_idx = j * 6 + k
        V_total[global_idx, :R1] = V_1[r, :]
        
    # Map V_2 rows (8) to global V rows (24)
    # V_2 rows are ordered: (0,0), (1,0), (0,1), (1,1), (2,0), (3,0), (2,1), (3,1)
    # Corresponding to (j, k_local) where k_local is 0 or 1
    # Global k = 4 + k_local
    v2_order = [
        (0, 0), # j=0, k_l=0 -> g=4
        (1, 0), # j=1, k_l=0 -> g=10
        (0, 1), # j=0, k_l=1 -> g=5
        (1, 1), # j=1, k_l=1 -> g=11
        (2, 0), # j=2, k_l=0 -> g=16
        (3, 0), # j=3, k_l=0 -> g=22
        (2, 1), # j=2, k_l=1 -> g=17
        (3, 1)  # j=3, k_l=1 -> g=23
    ]
    for r, (j, k_l) in enumerate(v2_order):
        global_idx = j * 6 + (4 + k_l)
        V_total[global_idx, R1:] = V_2[r, :]
        
    # W_total: 24 rows
    W_total = np.zeros((24, R_total))
    
    # Map W_1 rows (16) to global W rows (24)
    # W_1 rows are row-major 4x4 (k, i) with k in 0..3
    # Global index: k * 4 + i
    for r in range(16):
        k = r // 4
        i = r % 4
        global_idx = k * 4 + i
        W_total[global_idx, :R1] = W_1[r, :]
        
    # Map W_2 rows (8) to global W rows (24)
    # W_2 rows are ordered: (0,0), (0,1), (1,0), (1,1), (2,0), (2,1), (3,0), (3,1)
    # Corresponding to (i, k_local) where k_local is 0 or 1
    # Global k = 4 + k_local
    w2_order = [
        (0, 0), # i=0, k_l=0 -> g=16
        (0, 1), # i=0, k_l=1 -> g=20
        (1, 0), # i=1, k_l=0 -> g=17
        (1, 1), # i=1, k_l=1 -> g=21
        (2, 0), # i=2, k_l=0 -> g=18
        (2, 1), # i=2, k_l=1 -> g=22
        (3, 0), # i=3, k_l=0 -> g=19
        (3, 1)  # i=3, k_l=1 -> g=23
    ]
    for r, (i, k_l) in enumerate(w2_order):
        global_idx = (4 + k_l) * 4 + i
        W_total[global_idx, R1:] = W_2[r, :]
        
    return {
        "U": U_total.tolist(),
        "V": V_total.tolist(),
        "W": W_total.tolist()
    }

if __name__ == "__main__":
    res = construct()
    print(f"Rank: {len(res['U'][0])}")
