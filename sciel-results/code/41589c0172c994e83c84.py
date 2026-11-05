import time
import numpy as np

DIMENSIONS = [5, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    n, m, p = 5, 5, 5
    
    # Block sizes for partitioning 5 = 2 + 2 + 1
    block_sizes = [2, 2, 1]
    
    # Block offsets
    offsets = [0]
    for s in block_sizes:
        offsets.append(offsets[-1] + s)
    
    # Strassen decomposition for 2x2x2
    def get_strassen_factors():
        """Return list of (U_vec, V_vec, W_vec) for Strassen 2x2x2.
        U_vec has length 4: indices [0,1,2,3] = [a11,a12,a21,a22]
        V_vec has length 4: indices [0,1,2,3] = [b11,b12,b21,b22]
        W_vec has length 4: indices [0,1,2,3] = [c11,c12,c21,c22]
        """
        factors = []
        # M1 = (a11+a22)(b11+b22)
        factors.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]))
        # M2 = (a21+a22)b11
        factors.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]))
        # M3 = a11(b12-b22)
        factors.append(([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]))
        # M4 = a22(b21-b11)
        factors.append(([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]))
        # M5 = (a11+a12)b22
        factors.append(([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]))
        # M6 = (a21-a11)(b11+b12)
        factors.append(([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]))
        # M7 = (a12-a22)(b21+b22)
        factors.append(([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]))
        
        return factors
    
    strassen_factors = get_strassen_factors()
    
    # Total rank calculation
    total_rank = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                ni = block_sizes[i]
                mj = block_sizes[j]
                pk = block_sizes[k]
                if ni == 2 and mj == 2 and pk == 2:
                    total_rank += 7
                else:
                    total_rank += ni * mj * pk
    
    # Allocate matrices
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                if time.time() > deadline:
                    # If we exceed deadline, we return what we have so far, 
                    # but since the construction is deterministic and fast, 
                    # this should not happen. 
                    pass

                ni = block_sizes[i]
                mj = block_sizes[j]
                pk = block_sizes[k]
                
                if ni == 2 and mj == 2 and pk == 2:
                    # Use Strassen
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
                    # Schoolbook for this block
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
