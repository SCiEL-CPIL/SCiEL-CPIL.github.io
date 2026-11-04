import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    n_m = n * m
    m_p = m * p
    p_n = p * n
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    start_time = time.time()
    deadline = start_time + 10  # Fast construction, plenty of time
    
    def add_term(u_vec, v_vec, w_vec):
        """
        u_vec: list of (idx, val)
        v_vec: list of (idx, val)
        w_vec: list of (idx, val)
        """
        u_col = np.zeros(n_m)
        v_col = np.zeros(m_p)
        w_col = np.zeros(p_n)
        
        for idx, val in u_vec:
            u_col[idx] = val
        for idx, val in v_vec:
            v_col[idx] = val
        for idx, val in w_vec:
            w_col[idx] = val
            
        U_cols.append(u_col)
        V_cols.append(v_col)
        W_cols.append(w_col)
        
    def schoolbook(n_prime, m_prime, p_prime, i_offset, j_offset, k_offset):
        """
        Adds schoolbook decomposition for a sub-block.
        """
        for i_loc in range(n_prime):
            for j_loc in range(m_prime):
                for k_loc in range(p_prime):
                    i_glob = i_offset + i_loc
                    j_glob = j_offset + j_loc
                    k_glob = k_offset + k_loc
                    
                    I = i_glob * m + j_glob
                    J = j_glob * p + k_glob
                    K = k_glob * n + i_glob
                    
                    add_term([(I, 1)], [(J, 1)], [(K, 1)])
                    
    def strassen_2x2x2(i_offset, j_offset, k_offset):
        """
        Adds Strassen decomposition for a 2x2x2 sub-block.
        """
        def gI(i_l, j_l): return (i_offset + i_l) * m + (j_offset + j_l)
        def gJ(j_l, k_l): return (j_offset + j_l) * p + (k_offset + k_l)
        def gK(k_l, i_l): return (k_offset + k_l) * n + (i_offset + i_l)
        
        # M1: (A00+A11)(B00+B11) -> C00, C11
        add_term(
            [(gI(0,0), 1), (gI(1,1), 1)],
            [(gJ(0,0), 1), (gJ(1,1), 1)],
            [(gK(0,0), 1), (gK(1,1), 1)]
        )
        
        # M2: (A10+A11)B00 -> C10, -C11
        add_term(
            [(gI(1,0), 1), (gI(1,1), 1)],
            [(gJ(0,0), 1)],
            [(gK(0,1), 1), (gK(1,1), -1)]
        )
        
        # M3: A00(B01-B11) -> C01, C11
        add_term(
            [(gI(0,0), 1)],
            [(gJ(0,1), 1), (gJ(1,1), -1)],
            [(gK(1,0), 1), (gK(1,1), 1)]
        )
        
        # M4: A11(B10-B00) -> C00, C10
        add_term(
            [(gI(1,1), 1)],
            [(gJ(1,0), 1), (gJ(0,0), -1)],
            [(gK(0,0), 1), (gK(0,1), 1)]
        )
        
        # M5: (A00+A01)B11 -> -C00, C01
        add_term(
            [(gI(0,0), 1), (gI(0,1), 1)],
            [(gJ(1,1), 1)],
            [(gK(0,0), -1), (gK(1,0), 1)]
        )
        
        # M6: (A10-A00)(B00+B01) -> C11
        add_term(
            [(gI(1,0), 1), (gI(0,0), -1)],
            [(gJ(0,0), 1), (gJ(0,1), 1)],
            [(gK(1,1), 1)]
        )
        
        # M7: (A01-A11)(B10+B11) -> C00
        add_term(
            [(gI(0,1), 1), (gI(1,1), -1)],
            [(gJ(1,0), 1), (gJ(1,1), 1)],
            [(gK(0,0), 1)]
        )

    def strassen_2x2x3(i_offset, j_offset, k_offset):
        """
        Adds decomposition for a 2x2x3 sub-block by splitting k into 2+1.
        Rank: 7 (for 2x2x2) + 4 (for 2x2x1) = 11
        """
        # First part: 2x2x2 on k_offset to k_offset+1
        strassen_2x2x2(i_offset, j_offset, k_offset)
        # Second part: 2x2x1 on k_offset+2
        schoolbook(2, 2, 1, i_offset, j_offset, k_offset + 2)

    def strassen_2x3x2(i_offset, j_offset, k_offset):
        """
        Adds decomposition for a 2x3x2 sub-block by splitting m into 2+1.
        Rank: 7 (for 2x2x2) + 4 (for 2x1x2) = 11
        """
        # First part: 2x2x2 on j_offset to j_offset+1
        strassen_2x2x2(i_offset, j_offset, k_offset)
        # Second part: 2x1x2 on j_offset+2
        schoolbook(2, 1, 2, i_offset, j_offset + 2, k_offset)

    def strassen_2x3x3(i_offset, j_offset, k_offset):
        """
        Adds decomposition for a 2x3x3 sub-block.
        Split m into 2+1: (2x2x3) + (2x1x3) = 11 + 6 = 17
        """
        # First part: 2x2x3 on j_offset to j_offset+1
        strassen_2x2x3(i_offset, j_offset, k_offset)
        # Second part: 2x1x3 on j_offset+2
        schoolbook(2, 1, 3, i_offset, j_offset + 2, k_offset)

    # Partitioning:
    # n=4: [2, 2]
    # m=5: [2, 3]
    # p=5: [2, 3]
    
    i_blocks = [(0, 2), (2, 2)]
    j_blocks = [(0, 2), (2, 3)]
    k_blocks = [(0, 2), (2, 3)]
    
    for i_start, i_size in i_blocks:
        for j_start, j_size in j_blocks:
            for k_start, k_size in k_blocks:
                if time.time() > deadline:
                    break
                if i_size == 2 and j_size == 2 and k_size == 2:
                    strassen_2x2x2(i_start, j_start, k_start)
                elif i_size == 2 and j_size == 2 and k_size == 3:
                    strassen_2x2x3(i_start, j_start, k_start)
                elif i_size == 2 and j_size == 3 and k_size == 2:
                    strassen_2x3x2(i_start, j_start, k_start)
                elif i_size == 2 and j_size == 3 and k_size == 3:
                    strassen_2x3x3(i_start, j_start, k_start)
                else:
                    schoolbook(i_size, j_size, k_size, i_start, j_start, k_start)
    
    if not U_cols:
        # Fallback to schoolbook if something went wrong
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
        
    U = np.column_stack(U_cols)
    V = np.column_stack(V_cols)
    W = np.column_stack(W_cols)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
