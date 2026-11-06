import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = 5, 5, 5
    
    # Use 3+2 partition for 5
    block_sizes = [3, 2]
    offsets = [0, 3, 5]
    
    # Strassen factors for 2x2x2
    def get_strassen_factors():
        factors = []
        factors.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]))
        factors.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]))
        factors.append(([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]))
        factors.append(([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]))
        factors.append(([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]))
        factors.append(([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]))
        factors.append(([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]))
        return factors
    
    strassen_factors = get_strassen_factors()
    
    # For 3x3x3 blocks, we can try to find a better decomposition
    # For now, use schoolbook for 3x3x3 (rank 27)
    # But we can potentially do better. Let's first compute the total rank with schoolbook for 3x3x3
    
    total_rank = 0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                ni = block_sizes[i]
                mj = block_sizes[j]
                pk = block_sizes[k]
                if ni == 2 and mj == 2 and pk == 2:
                    total_rank += 7
                else:
                    # For 3x3x3, use schoolbook for now
                    total_rank += ni * mj * pk
    
    # U, V, W matrices
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    col = 0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                if time.time() > deadline:
                    break
                ni = block_sizes[i]
                mj = block_sizes[j]
                pk = block_sizes[k]
                
                if ni == 2 and mj == 2 and pk == 2:
                    for (uf, vf, wf) in strassen_factors:
                        local_to_global_U = [
                            (offsets[i] + 0) * m + (offsets[j] + 0),
                            (offsets[i] + 0) * m + (offsets[j] + 1),
                            (offsets[i] + 1) * m + (offsets[j] + 0),
                            (offsets[i] + 1) * m + (offsets[j] + 1),
                        ]
                        local_to_global_V = [
                            (offsets[j] + 0) * p + (offsets[k] + 0),
                            (offsets[j] + 0) * p + (offsets[k] + 1),
                            (offsets[j] + 1) * p + (offsets[k] + 0),
                            (offsets[j] + 1) * p + (offsets[k] + 1),
                        ]
                        local_to_global_W = [
                            (offsets[k] + 0) * n + (offsets[i] + 0),
                            (offsets[k] + 1) * n + (offsets[i] + 0),
                            (offsets[k] + 0) * n + (offsets[i] + 1),
                            (offsets[k] + 1) * n + (offsets[i] + 1),
                        ]
                        
                        for idx in range(4):
                            if uf[idx] != 0:
                                U[local_to_global_U[idx]][col] += uf[idx]
                            if vf[idx] != 0:
                                V[local_to_global_V[idx]][col] += vf[idx]
                            if wf[idx] != 0:
                                W[local_to_global_W[idx]][col] += wf[idx]
                        col += 1
                else:
                    # Schoolbook for 3x3x3 or other sizes
                    for li in range(ni):
                        for lj in range(mj):
                            for lk in range(pk):
                                U_idx = (offsets[i] + li) * m + (offsets[j] + lj)
                                V_idx = (offsets[j] + lj) * p + (offsets[k] + lk)
                                W_idx = (offsets[k] + lk) * n + (offsets[i] + li)
                                U[U_idx][col] = 1.0
                                V[V_idx][col] = 1.0
                                W[W_idx][col] = 1.0
                                col += 1
    
    assert col == total_rank, f"Column mismatch: {col} != {total_rank}"
    
    return {"U": U, "V": V, "W": W}
