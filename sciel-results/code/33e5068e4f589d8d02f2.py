import numpy as np

def construct():
    n, m, p = 4, 5, 5
    
    # Define the optimal partition
    i_parts = [(0, 2), (2, 4)]
    j_parts = [(0, 2), (2, 4), (4, 5)]
    k_parts = [(0, 2), (2, 4), (4, 5)]
    
    def build_strassen_2x2x2():
        U = np.zeros((4, 7), dtype=float)
        V = np.zeros((4, 7), dtype=float)
        W = np.zeros((4, 7), dtype=float)
        
        # Term 1: M1 = (A00+A11)(B00+B11)
        U[0, 0] = 1.0; U[3, 0] = 1.0
        V[0, 0] = 1.0; V[3, 0] = 1.0
        W[0, 0] = 1.0; W[3, 0] = 1.0
        
        # Term 2: M2 = (A10+A11)B00
        U[2, 1] = 1.0; U[3, 1] = 1.0
        V[0, 1] = 1.0
        W[2, 1] = 1.0
        
        # Term 3: M3 = A00(B10-B11)
        U[0, 2] = 1.0
        V[2, 2] = 1.0; V[3, 2] = -1.0
        W[1, 2] = 1.0; W[3, 2] = 1.0
        
        # Term 4: M4 = A11(B01-B10)
        U[3, 3] = 1.0
        V[1, 3] = 1.0; V[2, 3] = -1.0
        W[0, 3] = 1.0; W[2, 3] = 1.0
        
        # Term 5: M5 = (A00+A01)B11
        U[0, 4] = 1.0; U[1, 4] = 1.0
        V[3, 4] = 1.0
        W[0, 4] = -1.0; W[1, 4] = 1.0
        
        # Term 6: M6 = (A10-A00)(B00+B10)
        U[2, 5] = 1.0; U[0, 5] = -1.0
        V[0, 5] = 1.0; V[2, 5] = 1.0
        W[3, 5] = 1.0
        
        # Term 7: M7 = (A01-A11)(B01+B11)
        U[1, 6] = 1.0; U[3, 6] = -1.0
        V[1, 6] = 1.0; V[3, 6] = 1.0
        W[0, 6] = 1.0; W[1, 6] = 1.0
        
        return U, V, W

    def schoolbook_block(n_b, m_b, p_b):
        rank_b = n_b * m_b * p_b
        U_b = np.zeros((n_b * m_b, rank_b), dtype=float)
        V_b = np.zeros((m_b * p_b, rank_b), dtype=float)
        W_b = np.zeros((p_b * n_b, rank_b), dtype=float)
        col = 0
        for i in range(n_b):
            for j in range(m_b):
                for k in range(p_b):
                    U_b[i * m_b + j, col] = 1.0
                    V_b[j * p_b + k, col] = 1.0
                    W_b[k * n_b + i, col] = 1.0
                    col += 1
        return U_b, V_b, W_b

    U_s, V_s, W_s = build_strassen_2x2x2()
    
    total_rank = 0
    block_list = []
    
    for i_start, i_end in i_parts:
        for j_start, j_end in j_parts:
            for k_start, k_end in k_parts:
                n_b = i_end - i_start
                m_b = j_end - j_start
                p_b = k_end - k_start
                
                if n_b == 2 and m_b == 2 and p_b == 2:
                    U_b, V_b, W_b = U_s.copy(), V_s.copy(), W_s.copy()
                else:
                    U_b, V_b, W_b = schoolbook_block(n_b, m_b, p_b)
                
                total_rank += U_b.shape[1]
                block_list.append((i_start, j_start, k_start, i_end, j_end, k_end, U_b, V_b, W_b))
    
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)
    
    col = 0
    for (i_start, j_start, k_start, i_end, j_end, k_end, U_b, V_b, W_b) in block_list:
        rank_block = U_b.shape[1]
        nb_i = i_end - i_start
        nb_j = j_end - j_start
        nb_k = k_end - k_start
        for li in range(nb_i):
            for lj in range(nb_j):
                gi = i_start + li
                gj = j_start + lj
                U[gi * m + gj, col:col+rank_block] = U_b[li * nb_j + lj, :]
        for lj in range(nb_j):
            for lk in range(nb_k):
                gj = j_start + lj
                gk = k_start + lk
                V[gj * p + gk, col:col+rank_block] = V_b[lj * nb_k + lk, :]
        for lk in range(nb_k):
            for li in range(nb_i):
                gk = k_start + lk
                gi = i_start + li
                W[gk * n + gi, col:col+rank_block] = W_b[lk * nb_i + li, :]
        col += rank_block
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
