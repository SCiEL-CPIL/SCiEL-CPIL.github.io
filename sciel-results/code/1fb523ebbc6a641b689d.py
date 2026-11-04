import time

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    # We will use a decomposition of rank 65.
    # 49 terms for 4x4 * 4x4 (Strassen)
    # 16 terms for 4x4 * 4x1 (Schoolbook)
    
    rank = 65
    U = [[0] * rank for _ in range(n * m)]  # 16 x 65
    V = [[0] * rank for _ in range(m * p)]  # 20 x 65
    W = [[0] * rank for _ in range(p * n)]  # 20 x 65
    
    # Helper to add a rank-1 term
    def add_term(col, u_list, v_list, w_list):
        for i, j, c in u_list:
            if c != 0:
                U[i * m + j][col] += c
        for j, k, c in v_list:
            if c != 0:
                V[j * p + k][col] += c
        for k, i, c in w_list:
            if c != 0:
                W[k * n + i][col] += c

    col = 0
    
    # --- Part 1: 4x4 * 4x4 using Strassen ---
    # Split A (4x4) into 2x2 blocks: A00, A01, A10, A11
    # Split B (4x4) into 2x2 blocks: B00, B01, B10, B11
    
    # Strassen 7 multiplications for 2x2 blocks
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Each M is a 2x2 matrix product.
    # We decompose each 2x2 product using Strassen (7 scalar mults).
    
    # Define the 7 Strassen combinations for the inner 2x2
    # P = [[p00, p01], [p10, p11]], Q = [[q00, q01], [q10, q11]]
    # s1 = (p00+p11)(q00+q11)
    # s2 = (p10+p11)q00
    # s3 = p00(q01-q11)
    # s4 = p11(q10-q00)
    # s5 = (p00+p01)q11
    # s6 = (p10-p00)(q00+q01)
    # s7 = (p01-p11)(q10+q11)
    
    # Result C = [[s1+s4-s5+s7, s3+s5], [s2+s4, s1-s2+s3+s6]]
    
    def get_super_term_indices(super_idx, inner_idx):
        u_list = []
        v_list = []
        w_list = []
        
        # Determine P and Q structure based on super_idx
        if super_idx == 0: # M1
            p_blocks = [1, 0, 0, 1] # A00 + A11
            q_blocks = [1, 0, 0, 1] # B00 + B11
        elif super_idx == 1: # M2
            p_blocks = [0, 0, 1, 1] # A10 + A11
            q_blocks = [1, 0, 0, 0] # B00
        elif super_idx == 2: # M3
            p_blocks = [1, 0, 0, 0] # A00
            q_blocks = [0, 1, 0, -1] # B01 - B11
        elif super_idx == 3: # M4
            p_blocks = [0, 0, 0, 1] # A11
            q_blocks = [-1, 0, 1, 0] # B10 - B00
        elif super_idx == 4: # M5
            p_blocks = [1, 1, 0, 0] # A00 + A01
            q_blocks = [0, 0, 0, 1] # B11
        elif super_idx == 5: # M6
            p_blocks = [-1, 0, 1, 0] # A10 - A00
            q_blocks = [1, 1, 0, 0] # B00 + B01
        elif super_idx == 6: # M7
            p_blocks = [0, 1, 0, -1] # A01 - A11
            q_blocks = [0, 0, 1, 1] # B10 + B11
        else:
            return u_list, v_list, w_list
            
        def get_p_element(p_elem_idx):
            res = []
            r = p_elem_idx // 2
            c = p_elem_idx % 2
            
            for b_idx, coeff in enumerate(p_blocks):
                if coeff == 0: continue
                if b_idx == 0: # A00
                    gi = r; gj = c
                elif b_idx == 1: # A01
                    gi = r; gj = 2 + c
                elif b_idx == 2: # A10
                    gi = 2 + r; gj = c
                elif b_idx == 3: # A11
                    gi = 2 + r; gj = 2 + c
                
                res.append((gi, gj, coeff))
            return res
            
        def get_q_element(q_elem_idx):
            res = []
            r = q_elem_idx // 2
            c = q_elem_idx % 2
            
            for b_idx, coeff in enumerate(q_blocks):
                if coeff == 0: continue
                if b_idx == 0: # B00
                    gj = r; gk = c
                elif b_idx == 1: # B01
                    gj = r; gk = 2 + c
                elif b_idx == 2: # B10
                    gj = 2 + r; gk = c
                elif b_idx == 3: # B11
                    gj = 2 + r; gk = 2 + c
                
                res.append((gj, gk, coeff))
            return res
            
        # Determine P and Q elements for this inner term
        if inner_idx == 0:
            p_elems = [(0, 1), (3, 1)] # P00, P11
            q_elems = [(0, 1), (3, 1)] # Q00, Q11
            c_contrib = [(0, 0, 1), (1, 1, 1)]
        elif inner_idx == 1:
            p_elems = [(2, 1), (3, 1)] # P10, P11
            q_elems = [(0, 1)] # Q00
            c_contrib = [(1, 0, 1), (1, 1, -1)]
        elif inner_idx == 2:
            p_elems = [(0, 1)] # P00
            q_elems = [(1, 1), (3, -1)] # Q01, -Q11
            c_contrib = [(0, 1, 1), (1, 1, 1)]
        elif inner_idx == 3:
            p_elems = [(3, 1)] # P11
            q_elems = [(2, 1), (0, -1)] # Q10, -Q00
            c_contrib = [(0, 0, 1), (1, 0, 1)]
        elif inner_idx == 4:
            p_elems = [(0, 1), (1, 1)] # P00, P01
            q_elems = [(3, 1)] # Q11
            c_contrib = [(0, 0, -1), (0, 1, 1)]
        elif inner_idx == 5:
            p_elems = [(2, 1), (0, -1)] # P10, -P00
            q_elems = [(0, 1), (1, 1)] # Q00, Q01
            c_contrib = [(1, 1, 1)]
        elif inner_idx == 6:
            p_elems = [(1, 1), (3, -1)] # P01, -P11
            q_elems = [(2, 1), (3, 1)] # Q10, Q11
            c_contrib = [(0, 0, 1)]
        
        u_list = []
        v_list = []
        
        for p_elem, coeff in p_elems:
            for i, j, c in get_p_element(p_elem):
                u_list.append((i, j, coeff * c))
        
        for q_elem, coeff in q_elems:
            for j, k, c in get_q_element(q_elem):
                v_list.append((j, k, coeff * c))
        
        w_list = []
        
        # Determine the blocks and signs for this super term
        if super_idx == 0:
            blocks = [(0, 0, 1), (1, 1, 1)] # (block_row, block_col, sign)
        elif super_idx == 1:
            blocks = [(1, 0, 1), (1, 1, -1)]
        elif super_idx == 2:
            blocks = [(0, 1, 1), (1, 1, 1)]
        elif super_idx == 3:
            blocks = [(0, 0, 1), (1, 0, 1)]
        elif super_idx == 4:
            blocks = [(0, 0, -1), (0, 1, 1)]
        elif super_idx == 5:
            blocks = [(1, 1, 1)]
        elif super_idx == 6:
            blocks = [(0, 0, 1)]
        
        for br, bc, sign in blocks:
            for lr, lc, local_coeff in c_contrib:
                gi = br * 2 + lr
                gk = bc * 2 + lc
                w_list.append((gk, gi, sign * local_coeff))
        
        return u_list, v_list, w_list
    
    # Now iterate over all super terms and inner terms
    for super_idx in range(7):
        for inner_idx in range(7):
            u_list, v_list, w_list = get_super_term_indices(super_idx, inner_idx)
            if u_list and v_list and w_list:
                add_term(col, u_list, v_list, w_list)
                col += 1
    
    # Part 2: 4x4x1 (rank 16)
    # A (4x4) * B_2 (4x1) -> C_2 (4x1)
    # B_2 is column 4 of B.
    # C_2 is column 4 of C.
    
    for i in range(4):
        for j in range(4):
            U[i*4 + j][col] = 1
            V[j*5 + 4][col] = 1 # B index: row j, col 4
            W[4*4 + i][col] = 1 # C index: k=4, i. W[k*n+i] = W[4*4+i]
            col += 1

    return {"U": U, "V": V, "W": W}
