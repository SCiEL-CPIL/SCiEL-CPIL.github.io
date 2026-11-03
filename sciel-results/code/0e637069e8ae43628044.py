import time
import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    
    n, m, p = DIMENSIONS  # 2, 4, 8
    rank = 56
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Strassen 2x2 local tensors
    U_L = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ])
    V_L = np.array([
        [1, 1, -1, -1, 0, 1, 1],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, -1, 1, 0, 1]
    ])
    W_L = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ])
    
    for s in range(2):
        for j in range(4):
            base = (j * 2 + s) * 7
            ac = [0 * m + s * 2, 0 * m + s * 2 + 1, 1 * m + s * 2, 1 * m + s * 2 + 1]
            bc = [s * 2 * p + 2 * j, s * 2 * p + 2 * j + 1, (s * 2 + 1) * p + 2 * j, (s * 2 + 1) * p + 2 * j + 1]
            cc = [2 * j * n + 0, (2 * j + 1) * n + 0, 2 * j * n + 1, (2 * j + 1) * n + 1]
            for r in range(7):
                for l in range(4):
                    if U_L[l, r]:
                        U[ac[l], base + r] += U_L[l, r]
                    if V_L[l, r]:
                        V[bc[l], base + r] += V_L[l, r]
                    if W_L[l, r]:
                        W[cc[l], base + r] += W_L[l, r]
    
    # Quick verification
    np.random.seed(123)
    A_t = np.random.randint(-3, 4, (n, m))
    B_t = np.random.randint(-3, 4, (m, p))
    C_exp = A_t @ B_t
    C_rec = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            s = 0
            for r in range(rank):
                us = sum(U[i * m + j, r] * A_t[i, j] for j in range(m))
                vs = sum(V[j * p + k, r] * B_t[j, k] for j in range(m))
                s += us * vs * W[k * n + i, r]
            C_rec[i, k] = s
    ok = np.array_equal(C_rec, C_exp)
    
    if time.time() - start_time > 140:
        pass
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
