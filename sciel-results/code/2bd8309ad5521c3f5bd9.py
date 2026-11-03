import numpy as np
import time

DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = DIMENSIONS
    # Global matrices
    U = np.zeros((n * m, 99))
    V = np.zeros((m * p, 99))
    W = np.zeros((p * n, 99))
    
    col_offset = 0
    
    # Define Strassen 2x2x2 generator
    def get_strassen_222():
        # Returns U, V, W for 2x2x2
        # U shape (4, 7), V shape (4, 7), W shape (4, 7)
        u = np.zeros((4, 7))
        v = np.zeros((4, 7))
        w = np.zeros((4, 7))
        
        # M1 = (A00+A11)(B00+B11)
        u[0, 0] = 1; u[3, 0] = 1
        v[0, 0] = 1; v[3, 0] = 1
        w[0, 0] = 1; w[3, 0] = 1 # C00, C11 parts? No, W maps to C indices.
        # C00 gets M1, C11 gets M1
        # W[k*n+i, c]. For 2x2x2, n=2, m=2, p=2.
        # C indices: 00, 01, 10, 11.
        # W row index = k*n + i.
        # M1 contributes to C00 (k=0, i=0 -> 0) and C11 (k=1, i=1 -> 3).
        w[0, 0] = 1
        w[3, 0] = 1
        
        # M2 = (A10+A11)B00
        u[2, 1] = 1; u[3, 1] = 1
        v[0, 1] = 1
        # C10 gets M2 (k=0, i=1 -> 2)
        w[2, 1] = 1
        
        # M3 = A00(B10-B11)
        u[0, 2] = 1
        v[2, 2] = 1; v[3, 2] = -1
        # C01 gets M3 (k=1, i=0 -> 1)
        w[1, 2] = 1
        
        # M4 = A11(B01-B00)
        u[3, 3] = 1
        v[1, 3] = 1; v[0, 3] = -1
        # C00 gets M4, C10 gets M4
        w[0, 3] = 1
        w[2, 3] = 1
        
        # M5 = (A00+A01)B11
        u[0, 4] = 1; u[1, 4] = 1
        v[3, 4] = 1
        # C00 gets -M5, C01 gets M5
        w[0, 4] = -1
        w[1, 4] = 1
        
        # M6 = (A10-A00)(B00+B01)
        u[2, 5] = 1; u[0, 5] = -1
        v[0, 5] = 1; v[1, 5] = 1
        # C11 gets M6
        w[3, 5] = 1
        
        # M7 = (A01-A11)(B10+B11)
        u[1, 6] = 1; u[3, 6] = -1
        v[2, 6] = 1; v[3, 6] = 1
        # C00 gets M7
        w[0, 6] = 1
        
        return u, v, w

    # Define Schoolbook generator for n x m x p
    def get_schoolbook(n, m, p):
        rank = n * m * p
        u = np.zeros((n * m, rank))
        v = np.zeros((m * p, rank))
        w = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u[i * m + j, col] = 1
                    v[j * p + k, col] = 1
                    w[k * n + i, col] = 1
                    col += 1
        return u, v, w

    # Helper to embed local matrices into global
    def embed(global_U, global_V, global_W, local_U, local_V, local_W, 
              i_start, i_end, j_start, j_end, k_start, k_end, col_offset):
        n_local = i_end - i_start
        m_local = j_end - j_start
        p_local = k_end - k_start
        rank_local = local_U.shape[1]
        
        for r in range(rank_local):
            # U
            for i in range(n_local):
                for j in range(m_local):
                    val = local_U[i * m_local + j, r]
                    if val != 0:
                        global_U[(i_start + i) * m + (j_start + j), col_offset + r] += val
            # V
            for j in range(m_local):
                for k in range(p_local):
                    val = local_V[j * p_local + k, r]
                    if val != 0:
                        global_V[(j_start + j) * p + (k_start + k), col_offset + r] += val
            # W
            for k in range(p_local):
                for i in range(n_local):
                    val = local_W[k * n_local + i, r]
                    if val != 0:
                        global_W[(k_start + k) * n + (i_start + i), col_offset + r] += val

    # Partition indices
    # I: 0,1 | 2
    # J: 0,1 | 2,3 | 4
    # K: 0,1 | 2,3 | 4,5 | 6
    
    i_blocks = [(0, 2), (2, 3)]
    j_blocks = [(0, 2), (2, 4), (4, 5)]
    k_blocks = [(0, 2), (2, 4), (4, 6), (6, 7)]
    
    for i_s, i_e in i_blocks:
        for j_s, j_e in j_blocks:
            for k_s, k_e in k_blocks:
                n_l = i_e - i_s
                m_l = j_e - j_s
                p_l = k_e - k_s
                
                if n_l == 2 and m_l == 2 and p_l == 2:
                    lu, lv, lw = get_strassen_222()
                else:
                    lu, lv, lw = get_schoolbook(n_l, m_l, p_l)
                    
                embed(U, V, W, lu, lv, lw, i_s, i_e, j_s, j_e, k_s, k_e, col_offset)
                col_offset += lu.shape[1]
                
    return {"U": U, "V": V, "W": W}
