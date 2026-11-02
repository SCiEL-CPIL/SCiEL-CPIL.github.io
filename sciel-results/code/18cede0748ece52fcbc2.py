import time
import sys

DIMENSIONS = [3, 3, 8]

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    n, m, p = DIMENSIONS
    # n=3, m=3, p=8
    
    U_terms = []
    V_terms = []
    W_terms = []
    
    def add_term(u_vec, v_vec, w_vec):
        U_terms.append(u_vec)
        V_terms.append(v_vec)
        W_terms.append(w_vec)
        
    def make_vec(dim, indices_values):
        vec = [0] * dim
        for idx, val in indices_values:
            vec[idx] += val
        return vec
        
    n_dim = n * m # 9
    m_dim = m * p # 24
    p_dim = p * n # 24
    
    # Strassen decomposition for 2x2x2
    # A: 2x2, B: 2x2, C: 2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B00)
    # M4 = A11(B01-B11)
    # M5 = (A00+A01)(B10+B11)
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    def add_strassen_block(i_off, j_off, k_off):
        # i_off, j_off: offset in A (rows, cols)
        # k_off: offset in B (cols)
        # B rows are j_off, j_off+1
        
        # A indices
        a00 = (i_off * 3 + j_off)
        a01 = (i_off * 3 + j_off + 1)
        a10 = ((i_off + 1) * 3 + j_off)
        a11 = ((i_off + 1) * 3 + j_off + 1)
        
        # B indices
        b00 = (j_off * 8 + k_off)
        b01 = (j_off * 8 + k_off + 1)
        b10 = ((j_off + 1) * 8 + k_off)
        b11 = ((j_off + 1) * 8 + k_off + 1)
        
        # C indices
        c00 = (k_off * 3 + i_off)
        c01 = ((k_off + 1) * 3 + i_off)
        c10 = (k_off * 3 + i_off + 1)
        c11 = ((k_off + 1) * 3 + i_off + 1)
        
        # M1
        u1 = make_vec(n_dim, [(a00, 1), (a11, 1)])
        v1 = make_vec(m_dim, [(b00, 1), (b11, 1)])
        w1 = make_vec(p_dim, [(c00, 1), (c11, 1)])
        add_term(u1, v1, w1)
        
        # M2
        u2 = make_vec(n_dim, [(a10, 1), (a11, 1)])
        v2 = make_vec(m_dim, [(b00, 1)])
        w2 = make_vec(p_dim, [(c10, 1)])
        add_term(u2, v2, w2)
        
        # M3
        u3 = make_vec(n_dim, [(a00, 1)])
        v3 = make_vec(m_dim, [(b10, 1), (b00, -1)])
        w3 = make_vec(p_dim, [(c01, 1), (c10, 1)])
        add_term(u3, v3, w3)
        
        # M4
        u4 = make_vec(n_dim, [(a11, 1)])
        v4 = make_vec(m_dim, [(b01, 1), (b11, -1)])
        w4 = make_vec(p_dim, [(c00, 1), (c10, 1)])
        add_term(u4, v4, w4)
        
        # M5
        u5 = make_vec(n_dim, [(a00, 1), (a01, 1)])
        v5 = make_vec(m_dim, [(b10, 1), (b11, 1)])
        w5 = make_vec(p_dim, [(c00, -1), (c01, 1)])
        add_term(u5, v5, w5)
        
        # M6
        u6 = make_vec(n_dim, [(a10, 1), (a00, -1)])
        v6 = make_vec(m_dim, [(b00, 1), (b01, 1)])
        w6 = make_vec(p_dim, [(c11, 1)])
        add_term(u6, v6, w6)
        
        # M7
        u7 = make_vec(n_dim, [(a01, 1), (a11, -1)])
        v7 = make_vec(m_dim, [(b10, 1), (b11, 1)])
        w7 = make_vec(p_dim, [(c00, 1), (c11, 1)])
        add_term(u7, v7, w7)
        
        # Note: The W coefficients above are derived from the Strassen formulas:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        # 
        # Let's verify the W vectors added:
        # M1: w1 has c00=1, c11=1. (Matches M1 in C00, C11)
        # M2: w2 has c10=1. (Matches M2 in C10, -M2 in C11? No, w2 only has c10. 
        #      Wait, C11 has -M2. So we need a term for -M2 in C11?
        #      The decomposition is sum_c U_c V_c W_c.
        #      If W_c has multiple non-zero entries, it contributes to multiple C entries.
        #      My W vectors above only have 1 or 2 non-zeros.
        #      Let's check M2.
        #      C10 = M2 + M4.
        #      C11 = M1 - M2 + M3 + M6.
        #      So M2 contributes +1 to C10 and -1 to C11.
        #      My w2 only has c10=1. It is missing c11=-1.
        #      I must fix the W vectors to include all contributions.
        
        # Let's redo the W vectors carefully.
        # M1: C00=1, C11=1. w1 = [c00:1, c11:1]
        # M2: C10=1, C11=-1. w2 = [c10:1, c11:-1]
        # M3: C01=1, C10=1. w3 = [c01:1, c10:1]
        # M4: C00=1, C10=1. w4 = [c00:1, c10:1]
        # M5: C00=-1, C01=1. w5 = [c00:-1, c01:1]
        # M6: C11=1. w6 = [c11:1]
        # M7: C00=1, C11=1. w7 = [c00:1, c11:1]
        
        # I will clear the terms and re-add with correct W.
        # For efficiency in this thought process, I'll just note the correction.
        
    # Given the complexity, I will implement the full correct version in the final code.
    
    # Clear previous partial work
    U_terms.clear()
    V_terms.clear()
    W_terms.clear()
    
    def add_strassen_block_correct(i_off, j_off, k_off):
        a00 = (i_off * 3 + j_off)
        a01 = (i_off * 3 + j_off + 1)
        a10 = ((i_off + 1) * 3 + j_off)
        a11 = ((i_off + 1) * 3 + j_off + 1)
        
        b00 = (j_off * 8 + k_off)
        b01 = (j_off * 8 + k_off + 1)
        b10 = ((j_off + 1) * 8 + k_off)
        b11 = ((j_off + 1) * 8 + k_off + 1)
        
        c00 = (k_off * 3 + i_off)
        c01 = ((k_off + 1) * 3 + i_off)
        c10 = (k_off * 3 + i_off + 1)
        c11 = ((k_off + 1) * 3 + i_off + 1)
        
        # M1
        add_term(make_vec(n_dim, [(a00, 1), (a11, 1)]), 
                 make_vec(m_dim, [(b00, 1), (b11, 1)]), 
                 make_vec(p_dim, [(c00, 1), (c11, 1)]))
        
        # M2
        add_term(make_vec(n_dim, [(a10, 1), (a11, 1)]), 
                 make_vec(m_dim, [(b00, 1)]), 
                 make_vec(p_dim, [(c10, 1), (c11, -1)]))
        
        # M3
        add_term(make_vec(n_dim, [(a00, 1)]), 
                 make_vec(m_dim, [(b10, 1), (b00, -1)]), 
                 make_vec(p_dim, [(c01, 1), (c10, 1)]))
        
        # M4
        add_term(make_vec(n_dim, [(a11, 1)]), 
                 make_vec(m_dim, [(b01, 1), (b11, -1)]), 
                 make_vec(p_dim, [(c00, 1), (c10, 1)]))
        
        # M5
        add_term(make_vec(n_dim, [(a00, 1), (a01, 1)]), 
                 make_vec(m_dim, [(b10, 1), (b11, 1)]), 
                 make_vec(p_dim, [(c00, -1), (c01, 1)]))
        
        # M6
        add_term(make_vec(n_dim, [(a10, 1), (a00, -1)]), 
                 make_vec(m_dim, [(b00, 1), (b01, 1)]), 
                 make_vec(p_dim, [(c11, 1)]))
        
        # M7
        add_term(make_vec(n_dim, [(a01, 1), (a11, -1)]), 
                 make_vec(m_dim, [(b10, 1), (b11, 1)]), 
                 make_vec(p_dim, [(c00, 1), (c11, 1)]))
        
    # 4 Strassen blocks for A[0:2, 0:2] * B[0:2, k]
    for k_off in [0, 2, 4, 6]:
        add_strassen_block_correct(0, 0, k_off)
        
    # Cross terms
    # 1. A[0:2, 2] * B[2, :]
    for i in [0, 1]:
        for k in range(8):
            u_idx = i * 3 + 2
            v_idx = 2 * 8 + k
            w_idx = k * 3 + i
            add_term(make_vec(n_dim, [(u_idx, 1)]), 
                     make_vec(m_dim, [(v_idx, 1)]), 
                     make_vec(p_dim, [(w_idx, 1)]))
                     
    # 2. A[2, 0:2] * B[0:2, :]
    for j in [0, 1]:
        for k in range(8):
            u_idx = 2 * 3 + j
            v_idx = j * 8 + k
            w_idx = k * 3 + 2
            add_term(make_vec(n_dim, [(u_idx, 1)]), 
                     make_vec(m_dim, [(v_idx, 1)]), 
                     make_vec(p_dim, [(w_idx, 1)]))
                     
    # 3. A[2, 2] * B[2, :]
    for k in range(8):
        u_idx = 2 * 3 + 2
        v_idx = 2 * 8 + k
        w_idx = k * 3 + 2
        add_term(make_vec(n_dim, [(u_idx, 1)]), 
                 make_vec(m_dim, [(v_idx, 1)]), 
                 make_vec(p_dim, [(w_idx, 1)]))
                 
    # Convert to matrices
    rank = len(U_terms)
    U_mat = [[0] * rank for _ in range(n_dim)]
    V_mat = [[0] * rank for _ in range(m_dim)]
    W_mat = [[0] * rank for _ in range(p_dim)]
    
    for c in range(rank):
        for i in range(n_dim):
            if U_terms[c][i] != 0:
                U_mat[i][c] = U_terms[c][i]
        for j in range(m_dim):
            if V_terms[c][j] != 0:
                V_mat[j][c] = V_terms[c][j]
        for k in range(p_dim):
            if W_terms[c][k] != 0:
                W_mat[k][c] = W_terms[c][k]
                
    return {"U": U_mat, "V": V_mat, "W": W_mat}
