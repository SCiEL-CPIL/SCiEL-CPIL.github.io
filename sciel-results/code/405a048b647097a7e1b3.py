import time
import numpy as np

DIMENSIONS = [5, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = 5, 5, 5
    
    # Block sizes for partitioning 5 = 3 + 1 + 1
    block_sizes = [3, 1, 1]
    
    # Block offsets
    offsets = [0]
    for s in block_sizes:
        offsets.append(offsets[-1] + s)
    
    # Calculate total rank (all schoolbook for now)
    total_rank = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                ni = block_sizes[i]
                mj = block_sizes[j]
                pk = block_sizes[k]
                total_rank += ni * mj * pk
    
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                if time.time() > deadline:
                    pass
                
                ni = block_sizes[i]
                mj = block_sizes[j]
                pk = block_sizes[k]
                
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
