import time
import numpy as np

DIMENSIONS = [5, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    n, m, p = 5, 5, 5
    
    # Block sizes for partitioning 5 = 4 + 1
    block_sizes = [4, 1]
    
    # Block offsets
    offsets = [0]
    for s in block_sizes:
        offsets.append(offsets[-1] + s)
    
    # Strassen decomposition for 2x2x2
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
    
    # Generate 4x4x4 factors via Kronecker product of 2x2x2 factors
    def get_4x4x4_factors():
        factors_4 = []
        for uf, vf, wf in strassen_factors:
            for uf2, vf2, wf2 in strassen_factors:
                uf4 = [uf[i//2] * uf2[i%2] for i in range(16)]
                vf4 = [vf[i//2] * vf2[i%2] for i in range(16)]
                wf4 = [wf[i//2] * wf2[i%2] for i in range(16)]
                factors_4.append((uf4, vf4, wf4))
        return factors_4
    
    factors_4 = get_4x4x4_factors()
    
    # Calculate total rank
    total_rank = 0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                ni = block_sizes[i]
                mj = block_sizes[j]
                pk = block_sizes[k]
                if ni == 4 and mj == 4 and pk == 4:
                    total_rank += 49
                else:
                    total_rank += ni * mj * pk
    
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
                
                if ni == 4 and mj == 4 and pk == 4:
                    # Use 4x4x4 factors (Kronecker of Strassen)
                    for (uf4, vf4, wf4) in factors_4:
                        local_to_global_U = [
                            (offsets[i] + (idx // 2)) * m + (offsets[j] + (idx % 2))
                            for idx in range(16)
                        ]
                        local_to_global_V = [
                            (offsets[j] + (idx // 2)) * p + (offsets[k] + (idx % 2))
                            for idx in range(16)
                        ]
                        local_to_global_W = [
                            (offsets[k] + (idx // 2)) * n + (offsets[i] + (idx % 2))
                            for idx in range(16)
                        ]
                        
                        for idx in range(16):
                            if uf4[idx] != 0:
                                U[local_to_global_U[idx]][col] += uf4[idx]
                            if vf4[idx] != 0:
                                V[local_to_global_V[idx]][col] += vf4[idx]
                            if wf4[idx] != 0:
                                W[local_to_global_W[idx]][col] += wf4[idx]
                        col += 1
                else:
                    # Schoolbook for other blocks
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
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    assert col == total_rank, f"Column mismatch: {col} != {total_rank}"
    
    return {"U": U, "V": V, "W": W}
