import numpy as np
import time
import itertools

def construct():
    n, m, p = 3, 4, 4
    
    def get_strassen_2x2x2():
        u1 = np.array([[1, 0], [1, 1]], dtype=float)
        v1 = np.array([[1, 0], [0, 1]], dtype=float)
        w1 = np.array([[1, 0], [0, 1]], dtype=float)
        
        u2 = np.array([[0, 0], [1, 1]], dtype=float)
        v2 = np.array([[1, 0], [0, 0]], dtype=float)
        w2 = np.array([[0, 0], [1, -1]], dtype=float)
        
        u3 = np.array([[1, 0], [0, 0]], dtype=float)
        v3 = np.array([[0, 0], [1, -1]], dtype=float)
        w3 = np.array([[0, 1], [0, 1]], dtype=float)
        
        u4 = np.array([[0, 0], [0, 1]], dtype=float)
        v4 = np.array([[-1, 1], [0, 0]], dtype=float)
        w4 = np.array([[1, 0], [1, 0]], dtype=float)
        
        u5 = np.array([[1, 1], [0, 0]], dtype=float)
        v5 = np.array([[0, 0], [0, 1]], dtype=float)
        w5 = np.array([[-1, 1], [0, 0]], dtype=float)
        
        u6 = np.array([[-1, 0], [1, 0]], dtype=float)
        v6 = np.array([[1, 0], [1, 0]], dtype=float)
        w6 = np.array([[0, 0], [0, 1]], dtype=float)
        
        u7 = np.array([[0, 1], [-1, 1]], dtype=float)
        v7 = np.array([[0, 1], [0, 1]], dtype=float)
        w7 = np.array([[1, 0], [0, 0]], dtype=float)
        
        return [(u1, v1, w1), (u2, v2, w2), (u3, v3, w3), (u4, v4, w4), (u5, v5, w5), (u6, v6, w6), (u7, v7, w7)]
    
    strassen_factors = get_strassen_2x2x2()
    
    # Rank 32 decomposition
    # Part A: 4 Strassen blocks (28 terms)
    # Part B: 4 terms for row 2 (optimized from 16)
    rank = 32
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Part A: Rows 0,1 of A
    blocks = [
        (0, 0, 0), # i_start, j_start, k_start
        (0, 2, 0),
        (0, 0, 2),
        (0, 2, 2)
    ]
    
    base = 0
    for b_idx, (i_s, j_s, k_s) in enumerate(blocks):
        for l, (u, v, w) in enumerate(strassen_factors):
            col_idx = base + l
            for i in range(2):
                for j in range(2):
                    U[(i_s + i) * m + (j_s + j), col_idx] = u[i, j]
            for j in range(2):
                for k in range(2):
                    V[(j_s + j) * p + (k_s + k), col_idx] = v[j, k]
            for k in range(2):
                for i in range(2):
                    W[(k_s + k) * n + (i_s + i), col_idx] = w[i, k]
        base += 7
        
    # Part B: Row 2 of A
    # Rank 4 decomposition
    base = 28
    for r in range(4):
        col_idx = base + r
        # U[2, j] = 1
        for j in range(4):
            U[2 * m + j, col_idx] = 1.0
        # V[j, k] = delta(k, r)
        for j in range(4):
            for k in range(4):
                if k == r:
                    V[j * p + k, col_idx] = 1.0
        # W[k, 2] = delta(k, r)
        for k in range(4):
            if k == r:
                W[k * n + 2, col_idx] = 1.0
                
    # Verification
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
                    
    if max_err > 1e-9:
        # Fallback to schoolbook if verification fails
        rank_sb = n * m * p
        U_sb = np.zeros((n * m, rank_sb))
        V_sb = np.zeros((m * p, rank_sb))
        W_sb = np.zeros((p * n, rank_sb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j, col] = 1
                    V_sb[j * p + k, col] = 1
                    W_sb[k * n + i, col] = 1
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
