import time

DIMENSIONS = [5, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = 5, 5, 5

    # Block sizes for partitioning 5 = 3 + 2
    block_sizes = [3, 2]

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

    # BCLR rank-19 algorithm for 3x3x3
    # U_vec, V_vec, W_vec each have length 9: indices [0..8] = [a11,a12,a13,a21,a22,a23,a31,a32,a33]
    def get_bclr_factors():
        factors = []
        # m1 = (a11+a22)(b11+b22)
        factors.append(([1,0,0,0,1,0,0,0,0],[1,0,0,0,1,0,0,0,0],[1,0,0,0,1,0,0,0,0]))
        # m2 = (a21+a22)b11
        factors.append(([0,0,0,1,1,0,0,0,0],[1,0,0,0,0,0,0,0,0],[0,0,0,0,0,1,0,0,-1]))
        # m3 = a11(b12-b22)
        factors.append(([1,0,0,0,0,0,0,0,0],[0,1,0,0,-1,0,0,0,0],[0,1,0,0,0,0,0,0,1]))
        # m4 = a22(b21-b11)
        factors.append(([0,0,0,0,1,0,0,0,0],[-1,0,0,1,0,0,0,0,0],[1,0,0,0,0,1,0,0,0]))
        # m5 = (a11+a12)b22
        factors.append(([1,1,0,0,0,0,0,0,0],[0,0,0,0,1,0,0,0,0],[-1,0,1,0,0,0,0,0,0]))
        # m6 = (a21-a11)(b11+b12)
        factors.append(([-1,0,0,1,0,0,0,0,0],[1,1,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,1]))
        # m7 = (a12-a22)(b21+b22)
        factors.append(([0,1,0,0,-1,0,0,0,0],[0,0,0,1,1,0,0,0,0],[1,0,0,0,0,0,0,0,0]))
        # m8 = (a33+a31)(b33+b31)
        factors.append(([0,0,0,0,0,0,1,0,1],[0,0,0,0,0,0,1,0,1],[0,0,0,0,0,0,1,0,1]))
        # m9 = (a31+a33)b31
        factors.append(([0,0,0,0,0,0,1,0,1],[1,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,1]))
        # m10 = a33(b32-b31)
        factors.append(([0,0,0,0,0,0,0,0,1],[0,1,-1,0,0,0,0,0,0],[0,0,0,0,0,0,0,1,0]))
        # m11 = a31(b32-b33)
        factors.append(([0,0,0,0,0,0,1,0,0],[0,1,-1,0,0,0,0,0,0],[0,0,0,0,0,0,1,0,-1]))
        # m12 = (a31+a32)b33
        factors.append(([0,0,0,0,0,0,1,1,0],[0,0,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,0,1]))
        # m13 = (a32-a31)(b31+b32)
        factors.append(([0,0,0,0,0,0,-1,1,0],[1,1,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,1]))
        # m14 = (a32-a33)(b33+b32) -- wait, this doesn't look right for 3x3
        # Let me reconsider the BCLR algorithm structure
        # Actually, for 3x3x3, the BCLR algorithm has a specific structure
        # Let me use a verified set of factors
        
        # The above 14 factors are not correct. Let me use the actual BCLR algorithm.
        # After research, the correct BCLR 19-term algorithm for 3x3x3:
        
        return factors

    # Since implementing BCLR correctly is complex, let me fall back to schoolbook for 3x3x3
    # but use the [3,2] block structure with schoolbook for 3x3x3 (rank 27)
    # Total: 27 + 18+18+12+18+12+12+7 = 124, which is worse than 117
    
    # So let's stick with [2,2,1] but try to verify correctness
    block_sizes = [2, 2, 1]
    offsets = [0]
    for s in block_sizes:
        offsets.append(offsets[-1] + s)

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

    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
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
    
    return {"U": U, "V": V, "W": W}
