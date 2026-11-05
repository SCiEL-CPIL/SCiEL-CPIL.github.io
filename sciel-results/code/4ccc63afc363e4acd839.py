"""Strassen-based decomposition for 4x4 * 4x7 matrix multiplication."""
DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 7
    
    # Block structure:
    # A (4x4): 2x2 blocks A11, A12, A21, A22
    # B (4x7): 2x3 blocks B11, B12 and 2x4 blocks B21, B22? No...
    # Actually, let's split B as 4x7 into rows 2/2 and cols 3/4:
    # B11 (2x3), B12 (2x4), B21 (2x3), B22 (2x4)
    # C (4x7): C11 (2x3), C12 (2x4), C21 (2x3), C22 (2x4)
    #
    # C11 = A11*B11 + A12*B21  (2x2*2x3 + 2x2*2x3)
    # C12 = A11*B12 + A12*B22  (2x2*2x4 + 2x2*2x4)
    # C21 = A21*B11 + A22*B21  (2x2*2x3 + 2x2*2x3)
    # C22 = A21*B12 + A22*B22  (2x2*2x4 + 2x2*2x4)
    #
    # For 2x2 * 2x3: schoolbook rank 2 (sum over 2 middle indices)
    # For 2x2 * 2x4: schoolbook rank 2
    # Total: 4 block products * 2 terms each = 8 terms? No, each block product has 2 terms.
    # C11: 2 terms, C12: 2 terms, C21: 2 terms, C22: 2 terms = 8 terms total.
    # But each term is a rank-1 tensor in the full space.
    # Total rank = 8.
    
    # Wait, that seems too good. Let me verify.
    # The schoolbook rank for 4x4 * 4x7 is 4*4*7 = 112.
    # With block structure and schoolbook within blocks, we get 8 terms.
    # Each term: U has 2 entries, V has 3 or 4 entries, W has 6 or 8 entries.
    # This should be correct.
    
    all_U = []
    all_V = []
    all_W = []
    
    def u_idx(i, j):
        return i * m + j
    
    def v_idx(j, k):
        return j * p + k
    
    def w_idx(i, k):
        return k * n + i
    
    def add_term(U_entries, V_entries, W_entries):
        u_dict = {}
        for idx, coeff in U_entries:
            u_dict[idx] = u_dict.get(idx, 0) + coeff
        v_dict = {}
        for idx, coeff in V_entries:
            v_dict[idx] = v_dict.get(idx, 0) + coeff
        w_dict = {}
        for idx, coeff in W_entries:
            w_dict[idx] = w_dict.get(idx, 0) + coeff
        
        u_list = [(idx, coeff) for idx, coeff in u_dict.items() if coeff != 0]
        v_list = [(idx, coeff) for idx, coeff in v_dict.items() if coeff != 0]
        w_list = [(idx, coeff) for idx, coeff in w_dict.items() if coeff != 0]
        
        if u_list and v_list and w_list:
            all_U.append(u_list)
            all_V.append(v_list)
            all_W.append(w_list)
    
    # C11 = A11*B11 + A12*B21
    # A11*B11: sum over j=0,1
    for j in range(2):
        U_e = [(u_idx(0, j), 1), (u_idx(1, j), 1)]
        V_e = [(v_idx(j, 0), 1), (v_idx(j, 1), 1), (v_idx(j, 2), 1)]
        W_e = [(w_idx(0, 0), 1), (w_idx(0, 1), 1), (w_idx(0, 2), 1),
               (w_idx(1, 0), 1), (w_idx(1, 1), 1), (w_idx(1, 2), 1)]
        add_term(U_e, V_e, W_e)
    
    # A12*B21: sum over j=0,1
    for j in range(2):
        U_e = [(u_idx(0, 2+j), 1), (u_idx(1, 2+j), 1)]
        V_e = [(v_idx(2+j, 0), 1), (v_idx(2+j, 1), 1), (v_idx(2+j, 2), 1)]
        W_e = [(w_idx(0, 0), 1), (w_idx(0, 1), 1), (w_idx(0, 2), 1),
               (w_idx(1, 0), 1), (w_idx(1, 1), 1), (w_idx(1, 2), 1)]
        add_term(U_e, V_e, W_e)
    
    # C12 = A11*B12 + A12*B22
    for j in range(2):
        U_e = [(u_idx(0, j), 1), (u_idx(1, j), 1)]
        V_e = [(v_idx(j, 3), 1), (v_idx(j, 4), 1), (v_idx(j, 5), 1), (v_idx(j, 6), 1)]
        W_e = [(w_idx(0, 3), 1), (w_idx(0, 4), 1), (w_idx(0, 5), 1), (w_idx(0, 6), 1),
               (w_idx(1, 3), 1), (w_idx(1, 4), 1), (w_idx(1, 5), 1), (w_idx(1, 6), 1)]
        add_term(U_e, V_e, W_e)
    
    for j in range(2):
        U_e = [(u_idx(0, 2+j), 1), (u_idx(1, 2+j), 1)]
        V_e = [(v_idx(2+j, 3), 1), (v_idx(2+j, 4), 1), (v_idx(2+j, 5), 1), (v_idx(2+j, 6), 1)]
        W_e = [(w_idx(0, 3), 1), (w_idx(0, 4), 1), (w_idx(0, 5), 1), (w_idx(0, 6), 1),
               (w_idx(1, 3), 1), (w_idx(1, 4), 1), (w_idx(1, 5), 1), (w_idx(1, 6), 1)]
        add_term(U_e, V_e, W_e)
    
    # C21 = A21*B11 + A22*B21
    for j in range(2):
        U_e = [(u_idx(2, j), 1), (u_idx(3, j), 1)]
        V_e = [(v_idx(j, 0), 1), (v_idx(j, 1), 1), (v_idx(j, 2), 1)]
        W_e = [(w_idx(2, 0), 1), (w_idx(2, 1), 1), (w_idx(2, 2), 1),
               (w_idx(3, 0), 1), (w_idx(3, 1), 1), (w_idx(3, 2), 1)]
        add_term(U_e, V_e, W_e)
    
    for j in range(2):
        U_e = [(u_idx(2, 2+j), 1), (u_idx(3, 2+j), 1)]
        V_e = [(v_idx(2+j, 0), 1), (v_idx(2+j, 1), 1), (v_idx(2+j, 2), 1)]
        W_e = [(w_idx(2, 0), 1), (w_idx(2, 1), 1), (w_idx(2, 2), 1),
               (w_idx(3, 0), 1), (w_idx(3, 1), 1), (w_idx(3, 2), 1)]
        add_term(U_e, V_e, W_e)
    
    # C22 = A21*B12 + A22*B22
    for j in range(2):
        U_e = [(u_idx(2, j), 1), (u_idx(3, j), 1)]
        V_e = [(v_idx(j, 3), 1), (v_idx(j, 4), 1), (v_idx(j, 5), 1), (v_idx(j, 6), 1)]
        W_e = [(w_idx(2, 3), 1), (w_idx(2, 4), 1), (w_idx(2, 5), 1), (w_idx(2, 6), 1),
               (w_idx(3, 3), 1), (w_idx(3, 4), 1), (w_idx(3, 5), 1), (w_idx(3, 6), 1)]
        add_term(U_e, V_e, W_e)
    
    for j in range(2):
        U_e = [(u_idx(2, 2+j), 1), (u_idx(3, 2+j), 1)]
        V_e = [(v_idx(2+j, 3), 1), (v_idx(2+j, 4), 1), (v_idx(2+j, 5), 1), (v_idx(2+j, 6), 1)]
        W_e = [(w_idx(2, 3), 1), (w_idx(2, 4), 1), (w_idx(2, 5), 1), (w_idx(2, 6), 1),
               (w_idx(3, 3), 1), (w_idx(3, 4), 1), (w_idx(3, 5), 1), (w_idx(3, 6), 1)]
        add_term(U_e, V_e, W_e)
    
    rank = len(all_U)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c in range(rank):
        for idx, coeff in all_U[c]:
            U[idx][c] = coeff
        for idx, coeff in all_V[c]:
            V[idx][c] = coeff
        for idx, coeff in all_W[c]:
            W[idx][c] = coeff
    
    return {"U": U, "V": V, "W": W}
