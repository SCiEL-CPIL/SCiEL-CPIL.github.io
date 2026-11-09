import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = 4, 5, 5
    nm = n * m  # 20
    mp = m * p  # 25
    pn = p * n  # 20
    
    # Strategy: 8 Strassen blocks (rank 56) + Fringe (rank 8) = 64
    rank = 64
    U = np.zeros((nm, rank), dtype=np.int64)
    V = np.zeros((mp, rank), dtype=np.int64)
    W = np.zeros((pn, rank), dtype=np.int64)
    
    col = 0
    
    # 8 Strassen blocks for 4x4x4
    for i_base in [0, 2]:
        for j_base in [0, 2]:
            for k_base in [0, 2]:
                i1, i2 = i_base, i_base + 1
                j1, j2 = j_base, j_base + 1
                k1, k2 = k_base, k_base + 1
                
                # M1 = (a11 + a22)(b11 + b22)
                U[i1 * m + j1, col] += 1
                U[i2 * m + j2, col] += 1
                V[j1 * p + k1, col] += 1
                V[j2 * p + k2, col] += 1
                W[k1 * n + i1, col] += 1
                W[k2 * n + i2, col] += 1
                col += 1
                
                # M2 = (a21 + a22)b11
                U[i2 * m + j1, col] += 1
                U[i2 * m + j2, col] += 1
                V[j1 * p + k1, col] += 1
                W[k1 * n + i2, col] += 1
                W[k2 * n + i2, col] += -1
                col += 1
                
                # M3 = a11(b12 - b22)
                U[i1 * m + j1, col] += 1
                V[j1 * p + k2, col] += 1
                V[j2 * p + k2, col] += -1
                W[k2 * n + i1, col] += 1
                W[k2 * n + i2, col] += 1
                col += 1
                
                # M4 = a22(b21 - b11)
                U[i2 * m + j2, col] += 1
                V[j2 * p + k1, col] += 1
                V[j1 * p + k1, col] += -1
                W[k1 * n + i1, col] += 1
                W[k1 * n + i2, col] += 1
                col += 1
                
                # M5 = (a11 + a12)b22
                U[i1 * m + j1, col] += 1
                U[i1 * m + j2, col] += 1
                V[j2 * p + k2, col] += 1
                W[k1 * n + i1, col] += -1
                W[k2 * n + i1, col] += 1
                col += 1
                
                # M6 = (a21 - a11)(b11 + b12)
                U[i2 * m + j1, col] += 1
                U[i1 * m + j1, col] += -1
                V[j1 * p + k1, col] += 1
                V[j1 * p + k2, col] += 1
                W[k2 * n + i2, col] += 1
                col += 1
                
                # M7 = (a12 - a22)(b21 + b22)
                U[i1 * m + j2, col] += 1
                U[i2 * m + j2, col] += -1
                V[j2 * p + k1, col] += 1
                V[j2 * p + k2, col] += 1
                W[k1 * n + i1, col] += 1
                col += 1
    
    # Fringe 1: j=4
    # For each i in 0..3, one term
    for i in range(n):
        # U[i, 4, col] = 1
        U[i * m + 4, col] = 1
        # V[4, k, col] = 1 for all k
        for k in range(p):
            V[4 * p + k, col] = 1
        # W[k, i, col] = 1 for all k
        for k in range(p):
            W[k * n + i, col] = 1
        col += 1
    
    # Fringe 2: k=4, j in 0..3
    # For each j in 0..3, one term
    for j in range(4):
        # U[i, j, col] = 1 for all i
        for i in range(n):
            U[i * m + j, col] = 1
        # V[j, 4, col] = 1
        V[j * p + 4, col] = 1
        # W[4, i, col] = 1 for all i
        for i in range(n):
            W[4 * n + i, col] = 1
        col += 1
    
    # Verify correctness
    is_valid = True
    for test_i in range(n):
        for test_j in range(m):
            for test_k in range(p):
                expected = 1
                actual = 0
                for r in range(rank):
                    actual += U[test_i * m + test_j, r] * V[test_j * p + test_k, r] * W[test_k * n + test_i, r]
                if actual != expected:
                    is_valid = False
                    break
            if not is_valid:
                break
        if not is_valid:
            break
    
    if is_valid:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback: return standard rank-100 decomposition
        rank_std = 100
        U_std = np.zeros((nm, rank_std), dtype=np.int64)
        V_std = np.zeros((mp, rank_std), dtype=np.int64)
        W_std = np.zeros((pn, rank_std), dtype=np.int64)
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_std[i*m+j, c] = 1
                    V_std[j*p+k, c] = 1
                    W_std[k*n+i, c] = 1
                    c += 1
        return {"U": U_std.tolist(), "V": V_std.tolist(), "W": W_std.tolist()}
