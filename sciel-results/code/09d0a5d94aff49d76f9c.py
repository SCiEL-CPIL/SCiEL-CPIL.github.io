import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = 5, 5, 5
    block_sizes = [1, 2, 2]
    offsets = [0]
    for s in block_sizes:
        offsets.append(offsets[-1] + s)
    
    # Strassen factors for 2x2x2
    # U, V, W are vectors of length 4 (indices 0..3 corresponding to 2x2 matrix elements)
    # Order: (0,0), (0,1), (1,0), (1,1)
    strassen_factors = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]
    
    # Calculate total rank
    total_rank = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                ni = block_sizes[i]
                mj = block_sizes[j]
                pk = block_sizes[k]
                
                if ni == 2 and mj == 2 and pk == 2:
                    total_rank += 7
                elif ni == 1 and mj == 1 and pk == 1:
                    total_rank += 1
                elif ni == 1 and mj == 1 and pk == 2:
                    total_rank += 2
                elif ni == 1 and mj == 2 and pk == 1:
                    total_rank += 2
                elif ni == 2 and mj == 1 and pk == 1:
                    total_rank += 2
                elif ni == 1 and mj == 2 and pk == 2:
                    total_rank += 2
                elif ni == 2 and mj == 1 and pk == 2:
                    total_rank += 1
                elif ni == 2 and mj == 2 and pk == 1:
                    total_rank += 2
                else:
                    # Fallback to schoolbook if unexpected
                    total_rank += ni * mj * pk

    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                ni = block_sizes[i]
                mj = block_sizes[j]
                pk = block_sizes[k]
                
                # Local offsets
                io = offsets[i]
                jo = offsets[j]
                ko = offsets[k]
                
                if ni == 2 and mj == 2 and pk == 2:
                    for (uf, vf, wf) in strassen_factors:
                        # U indices: (io+li, jo+lj)
                        # V indices: (jo+lj, ko+lk)
                        # W indices: (ko+lk, io+li)
                        for idx in range(4):
                            if uf[idx] != 0:
                                li = idx // 2
                                lj = idx % 2
                                U[(io + li) * m + (jo + lj)][col] += uf[idx]
                            if vf[idx] != 0:
                                lj = idx // 2
                                lk = idx % 2
                                V[(jo + lj) * p + (ko + lk)][col] += vf[idx]
                            if wf[idx] != 0:
                                lk = idx // 2
                                li = idx % 2
                                W[(ko + lk) * n + (io + li)][col] += wf[idx]
                        col += 1
                elif ni == 1 and mj == 1 and pk == 1:
                    U[io * m + jo][col] = 1.0
                    V[jo * p + ko][col] = 1.0
                    W[ko * n + io][col] = 1.0
                    col += 1
                elif ni == 1 and mj == 1 and pk == 2:
                    # C_k = A_0 B_k
                    # r=0: k=0
                    U[io * m + jo][col] = 1.0
                    V[jo * p + (ko + 0)][col] = 1.0
                    W[(ko + 0) * n + io][col] = 1.0
                    col += 1
                    # r=1: k=1
                    U[io * m + jo][col] = 1.0
                    V[jo * p + (ko + 1)][col] = 1.0
                    W[(ko + 1) * n + io][col] = 1.0
                    col += 1
                elif ni == 1 and mj == 2 and pk == 1:
                    # C_i = A_i B_0
                    # r=0: i=0
                    U[(io + 0) * m + jo][col] = 1.0
                    V[jo * p + ko][col] = 1.0
                    W[ko * n + (io + 0)][col] = 1.0
                    col += 1
                    # r=1: i=1
                    U[(io + 1) * m + jo][col] = 1.0
                    V[jo * p + ko][col] = 1.0
                    W[ko * n + (io + 1)][col] = 1.0
                    col += 1
                elif ni == 2 and mj == 1 and pk == 1:
                    # C_i = A_i B_0
                    # r=0: i=0
                    U[(io + 0) * m + jo][col] = 1.0
                    V[jo * p + ko][col] = 1.0
                    W[ko * n + (io + 0)][col] = 1.0
                    col += 1
                    # r=1: i=1
                    U[(io + 1) * m + jo][col] = 1.0
                    V[jo * p + ko][col] = 1.0
                    W[ko * n + (io + 1)][col] = 1.0
                    col += 1
                elif ni == 1 and mj == 2 and pk == 2:
                    # C_k = sum_j A_j B_jk
                    # r=0: j=0
                    U[io * m + (jo + 0)][col] = 1.0
                    V[(jo + 0) * p + (ko + 0)][col] = 1.0
                    W[(ko + 0) * n + io][col] = 1.0
                    V[(jo + 0) * p + (ko + 1)][col] = 1.0
                    W[(ko + 1) * n + io][col] = 1.0
                    col += 1
                    # r=1: j=1
                    U[io * m + (jo + 1)][col] = 1.0
                    V[(jo + 1) * p + (ko + 0)][col] = 1.0
                    W[(ko + 0) * n + io][col] = 1.0
                    V[(jo + 1) * p + (ko + 1)][col] = 1.0
                    W[(ko + 1) * n + io][col] = 1.0
                    col += 1
                elif ni == 2 and mj == 1 and pk == 2:
                    # C_ik = A_i B_k
                    # Rank 1
                    for li in range(2):
                        U[(io + li) * m + jo][col] = 1.0
                    for lk in range(2):
                        V[jo * p + (ko + lk)][col] = 1.0
                    for li in range(2):
                        for lk in range(2):
                            W[(ko + lk) * n + (io + li)][col] = 1.0
                    col += 1
                elif ni == 2 and mj == 2 and pk == 1:
                    # C_i = sum_j A_ij B_j
                    # r=0: j=0
                    U[(io + 0) * m + (jo + 0)][col] = 1.0
                    U[(io + 1) * m + (jo + 0)][col] = 1.0
                    V[(jo + 0) * p + ko][col] = 1.0
                    W[ko * n + (io + 0)][col] = 1.0
                    W[ko * n + (io + 1)][col] = 1.0
                    col += 1
                    # r=1: j=1
                    U[(io + 0) * m + (jo + 1)][col] = 1.0
                    U[(io + 1) * m + (jo + 1)][col] = 1.0
                    V[(jo + 1) * p + ko][col] = 1.0
                    W[ko * n + (io + 0)][col] = 1.0
                    W[ko * n + (io + 1)][col] = 1.0
                    col += 1
                else:
                    # Schoolbook
                    for li in range(ni):
                        for lj in range(mj):
                            for lk in range(pk):
                                U[(io + li) * m + (jo + lj)][col] = 1.0
                                V[(jo + lj) * p + (ko + lk)][col] = 1.0
                                W[(ko + lk) * n + (io + li)][col] = 1.0
                                col += 1
    
    assert col == total_rank, f"Column mismatch: {col} != {total_rank}"
    
    # Verification
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Check tensor decomposition
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Should be 1 for all i,j,k
    
    error = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(total_rank):
                    s += U_np[i*m+j, r] * V_np[j*p+k, r] * W_np[k*n+i, r]
                if abs(s - 1.0) > 1e-8:
                    error += abs(s - 1.0)
    
    if error > 1e-6:
        # Fallback to schoolbook
        U_s = np.zeros((n*m, n*m*p))
        V_s = np.zeros((m*p, n*m*p))
        W_s = np.zeros((p*n, n*m*p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i*m*p + j*p + k
                    U_s[i*m+j, r] = 1.0
                    V_s[j*p+k, r] = 1.0
                    W_s[k*n+i, r] = 1.0
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
    
    return {"U": U, "V": V, "W": W}
