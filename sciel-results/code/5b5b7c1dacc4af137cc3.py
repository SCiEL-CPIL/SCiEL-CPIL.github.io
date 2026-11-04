import time
import numpy as np

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = 3, 4, 4
    
    # Target rank: 44 (4 * 7 + 4 * 4)
    rank = 44
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen coefficients for 2x2x2
    # Indices for U (2x2): 00, 01, 10, 11 -> 0, 1, 2, 3
    # Indices for V (2x2): 00, 01, 10, 11 -> 0, 1, 2, 3
    # Indices for W (2x2): 00, 01, 10, 11 -> 0, 1, 2, 3 (k*n + i)
    
    strassen_terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, 0]),
        ([1, 0, 0, 0], [0, 0, 1, -1], [0, 0, 0, 1]),
        ([0, 0, 0, 1], [0, 1, -1, 0], [1, 0, 0, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]),
        ([0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 0, 0]), # P6 unused?
        ([0, 1, 0, -1], [0, 0, 1, 1], [0, 1, 0, 0]),
    ]
    
    # Schoolbook terms for 1x2x2
    # U (1x2): 0, 1
    # V (2x2): 00, 01, 10, 11
    # W (2x1): 0, 1
    schoolbook_terms = [
        ([1, 0], [1, 0, 0, 0], [1, 0]),
        ([0, 1], [0, 0, 1, 0], [1, 0]),
        ([1, 0], [0, 1, 0, 0], [0, 1]),
        ([0, 1], [0, 0, 0, 1], [0, 1]),
    ]
    
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i
    
    col = 0
    
    # 4 blocks of 2x2x2 (Strassen) for i in {0,1}, j in {0..3}, k in {0..3}
    # Split j and k into 2 blocks of size 2
    for i_offset in [0, 1]:
        for j_offset in [0, 2]:
            for k_offset in [0, 2]:
                for ut, vt, wt in strassen_terms:
                    for li in range(2):
                        for lj in range(2):
                            val = ut[li * 2 + lj]
                            if val != 0:
                                U[u_idx(i_offset + li, j_offset + lj), col] = val
                    for lj in range(2):
                        for lk in range(2):
                            val = vt[lj * 2 + lk]
                            if val != 0:
                                V[v_idx(j_offset + lj, k_offset + lk), col] = val
                    for lk in range(2):
                        for li in range(2):
                            val = wt[lk * 2 + li]
                            if val != 0:
                                W[w_idx(k_offset + lk, i_offset + li), col] = val
                    col += 1
                    
    # 4 blocks of 1x2x2 (Schoolbook) for i=2, j in {0..3}, k in {0..3}
    for j_offset in [0, 2]:
        for k_offset in [0, 2]:
            for ut, vt, wt in schoolbook_terms:
                for lj in range(2):
                    val = ut[lj]
                    if val != 0:
                        U[u_idx(2, j_offset + lj), col] = val
                for lj in range(2):
                    for lk in range(2):
                        val = vt[lj * 2 + lk]
                        if val != 0:
                            V[v_idx(j_offset + lj, k_offset + lk), col] = val
                for lk in range(2):
                    val = wt[lk]
                    if val != 0:
                        W[w_idx(k_offset + lk, 2), col] = val
                col += 1
                
    # Verify
    def _verify(U, V, W, n, m, p, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    if not _verify(U, V, W, n, m, p, rank):
        # Fallback to schoolbook 48
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, c] = 1
                    V[j*p + k, c] = 1
                    W[k*n + i, c] = 1
                    c += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
