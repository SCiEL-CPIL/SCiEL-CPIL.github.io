import numpy as np
import time

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 145  # Safety margin before 150s limit
    
    # Schoolbook solution (guaranteed correct, rank 48)
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school), dtype=np.int64)
    V_school = np.zeros((m * p, rank_school), dtype=np.int64)
    W_school = np.zeros((p * n, rank_school), dtype=np.int64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1

    # Target rank 44 decomposition using block structure
    # 4 blocks of 3x2x2, each rank 11
    target_rank = 44
    U = np.zeros((n * m, target_rank), dtype=np.int64)
    V = np.zeros((m * p, target_rank), dtype=np.int64)
    W = np.zeros((p * n, target_rank), dtype=np.int64)
    
    # Define local patterns for 3x2x2 block
    # Local A is 3x2 (6 entries), Local B is 2x2 (4 entries), Local C is 3x2 (6 entries)
    
    # Strassen terms for top 2x2 part (indices 0-3 in A, 0-3 in B, 0-3 in C)
    strassen_U = [
        [1, 0, 0, 1, 0, 0], # m1
        [0, 0, 1, 1, 0, 0], # m2
        [1, 0, 0, 0, 0, 0], # m3
        [0, 0, 0, 1, 0, 0], # m4
        [1, 1, 0, 0, 0, 0], # m5
        [-1, 0, 1, 0, 0, 0], # m6
        [0, 1, 0, -1, 0, 0]  # m7
    ]
    strassen_V = [
        [1, 0, 0, 1], # m1
        [1, 0, 0, 0], # m2
        [0, 1, 0, -1], # m3
        [-1, 0, 1, 0], # m4
        [0, 0, 0, 1], # m5
        [1, 1, 0, 0], # m6
        [0, 0, 1, 1]  # m7
    ]
    strassen_W = [
        [1, 0, 0, 1, 0, 0], # m1 -> c00, c11
        [0, 0, 1, 0, 0, 0], # m2 -> c10
        [0, 1, 0, 1, 0, 0], # m3 -> c01, c11
        [1, 0, 1, 0, 0, 0], # m4 -> c00, c10
        [-1, 1, 0, 0, 0, 0], # m5 -> c00(-), c01(+)
        [0, 0, 0, 1, 0, 0], # m6 -> c11
        [1, 0, 0, 0, 0, 0]  # m7 -> c00
    ]
    
    # Trivial terms for bottom 1x2 part (indices 4-5 in A, 0-3 in B, 4-5 in C)
    trivial_U = [
        [0, 0, 0, 0, 1, 0], # xe
        [0, 0, 0, 0, 0, 1], # yg
        [0, 0, 0, 0, 1, 0], # xf
        [0, 0, 0, 0, 0, 1]  # yh
    ]
    trivial_V = [
        [1, 0, 0, 0], # e
        [0, 0, 1, 0], # g
        [0, 1, 0, 0], # f
        [0, 0, 0, 1]  # h
    ]
    trivial_W = [
        [0, 0, 0, 0, 1, 0], # c20
        [0, 0, 0, 0, 1, 0], # c20
        [0, 0, 0, 0, 0, 1], # c21
        [0, 0, 0, 0, 0, 1]  # c21
    ]
    
    col = 0
    # Iterate over 4 blocks
    for b in range(4):
        j_start = (b % 2) * 2
        k_start = (b // 2) * 2
        
        # Process 7 Strassen terms
        for t in range(7):
            u_loc = strassen_U[t]
            v_loc = strassen_V[t]
            w_loc = strassen_W[t]
            
            # Map U
            for l in range(6):
                if u_loc[l] != 0:
                    i_loc = l // 2
                    j_loc = l % 2
                    j_glob = j_start + j_loc
                    u_glob_idx = i_loc * m + j_glob
                    U[u_glob_idx, col] += u_loc[l]
            
            # Map V
            for l in range(4):
                if v_loc[l] != 0:
                    j_loc = l // 2
                    k_loc = l % 2
                    j_glob = j_start + j_loc
                    k_glob = k_start + k_loc
                    v_glob_idx = j_glob * p + k_glob
                    V[v_glob_idx, col] += v_loc[l]
            
            # Map W
            for l in range(6):
                if w_loc[l] != 0:
                    i_loc = l // 2
                    k_loc = l % 2
                    k_glob = k_start + k_loc
                    i_glob = i_loc
                    w_glob_idx = k_glob * n + i_glob
                    W[w_glob_idx, col] += w_loc[l]
            
            col += 1
        
        # Process 4 Trivial terms
        for t in range(4):
            u_loc = trivial_U[t]
            v_loc = trivial_V[t]
            w_loc = trivial_W[t]
            
            # Map U
            for l in range(6):
                if u_loc[l] != 0:
                    i_loc = l // 2
                    j_loc = l % 2
                    j_glob = j_start + j_loc
                    u_glob_idx = i_loc * m + j_glob
                    U[u_glob_idx, col] += u_loc[l]
            
            # Map V
            for l in range(4):
                if v_loc[l] != 0:
                    j_loc = l // 2
                    k_loc = l % 2
                    j_glob = j_start + j_loc
                    k_glob = k_start + k_loc
                    v_glob_idx = j_glob * p + k_glob
                    V[v_glob_idx, col] += v_loc[l]
            
            # Map W
            for l in range(6):
                if w_loc[l] != 0:
                    i_loc = l // 2
                    k_loc = l % 2
                    k_glob = k_start + k_loc
                    i_glob = i_loc
                    w_glob_idx = k_glob * n + i_glob
                    W[w_glob_idx, col] += w_loc[l]
            
            col += 1
    
    # Verify the result
    def verify(U, V, W, n, m, p):
        rank = U.shape[1]
        tensor = np.zeros((n, m, p))
        for r in range(rank):
            u = U[:, r]
            v = V[:, r]
            w = W[:, r]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        tensor[i, j, k] += u[i * m + j] * v[j * p + k] * w[k * n + i]
        target = np.ones((n, m, p))
        return np.allclose(tensor, target, atol=1e-9)
    
    if verify(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook if verification fails
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}

