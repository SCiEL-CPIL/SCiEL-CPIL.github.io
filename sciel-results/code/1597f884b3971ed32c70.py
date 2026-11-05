import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Partition 5 into 2, 2, 1
    parts = [2, 2, 1]
    offsets = [0, 2, 4]
    
    # Total rank calculation
    total_rank = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                if parts[i] == 2 and parts[j] == 2 and parts[k] == 2:
                    total_rank += 7
                else:
                    total_rank += parts[i] * parts[j] * parts[k]
    
    # Allocate matrices
    U = np.zeros((n * m, total_rank), dtype=np.float64)
    V = np.zeros((m * p, total_rank), dtype=np.float64)
    W = np.zeros((p * n, total_rank), dtype=np.float64)
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                si = parts[i]
                sj = parts[j]
                sk = parts[k]
                
                off_i = offsets[i]
                off_j = offsets[j]
                off_k = offsets[k]
                
                if si == 2 and sj == 2 and sk == 2:
                    # Strassen 2x2x2
                    # Factors defined as (u_vec, v_vec, w_vec) of length 4
                    # Indices: 0:00, 1:01, 2:10, 3:11
                    strassen_factors = [
                        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]), # M1
                        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]), # M2
                        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),  # M3
                        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),  # M4
                        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),  # M5
                        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),  # M6
                        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])   # M7
                    ]
                    
                    for uf, vf, wf in strassen_factors:
                        for idx in range(4):
                            if uf[idx] != 0:
                                lr = idx // 2
                                lc = idx % 2
                                U[(off_i + lr) * m + (off_j + lc), col] += uf[idx]
                            if vf[idx] != 0:
                                lr = idx // 2
                                lc = idx % 2
                                V[(off_j + lr) * p + (off_k + lc), col] += vf[idx]
                            if wf[idx] != 0:
                                lr = idx // 2
                                lc = idx % 2
                                W[(off_k + lc) * n + (off_i + lr), col] += wf[idx]
                        col += 1
                else:
                    # Schoolbook
                    for li in range(si):
                        for lj in range(sj):
                            for lk in range(sk):
                                U_idx = (off_i + li) * m + (off_j + lj)
                                V_idx = (off_j + lj) * p + (off_k + lk)
                                W_idx = (off_k + lk) * n + (off_i + li)
                                U[U_idx, col] = 1.0
                                V[V_idx, col] = 1.0
                                W[W_idx, col] = 1.0
                                col += 1
    
    assert col == total_rank
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
