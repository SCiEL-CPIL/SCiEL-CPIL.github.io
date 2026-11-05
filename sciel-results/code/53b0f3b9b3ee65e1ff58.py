import time
import sys

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    # We will use a decomposition of rank 65.
    # 49 terms for 4x4 * 4x4 (Double Strassen)
    # 16 terms for 4x4 * 4x1 (Schoolbook)
    
    rank = 65
    U = [[0] * rank for _ in range(n * m)]  # 16 x 65
    V = [[0] * rank for _ in range(m * p)]  # 20 x 65
    W = [[0] * rank for _ in range(p * n)]  # 20 x 65
    
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
    
    # --- Part 1: 4x4 * 4x4 using Double Strassen ---
    # This part decomposes the multiplication of the first 4 columns of B (4x4)
    # into the first 4 columns of C (4x4).
    
    # Inner Strassen definitions for 2x2 * 2x2
    # P = [[P00, P01], [P10, P11]], Q = [[Q00, Q01], [Q10, Q11]]
    # R = P*Q
    # 7 terms:
    # 0: (P00+P11)(Q00+Q11) -> R00 += 1, R11 += 1
    # 1: (P10+P11)Q00       -> R10 += 1, R11 -= 1
    # 2: P00(Q01-Q11)       -> R01 += 1, R11 += 1
    # 3: P11(Q10-Q00)       -> R00 += 1, R10 += 1
    # 4: (P00+P01)Q11       -> R00 -= 1, R01 += 1
    # 5: (P10-P00)(Q00+Q01) -> R11 += 1
    # 6: (P01-P11)(Q10+Q11) -> R00 += 1
    
    # p_elems, q_elems, r_contribs
    # elem_idx: 0:P00, 1:P01, 2:P10, 3:P11
    inner_defs = [
        ([(0,1),(3,1)], [(0,1),(3,1)], [(0,0,1),(1,1,1)]),
        ([(2,1),(3,1)], [(0,1)], [(1,0,1),(1,1,-1)]),
        ([(0,1)], [(1,1),(3,-1)], [(0,1,1),(1,1,1)]),
        ([(3,1)], [(2,1),(0,-1)], [(0,0,1),(1,0,1)]),
        ([(0,1),(1,1)], [(3,1)], [(0,0,-1),(0,1,1)]),
        ([(2,1),(0,-1)], [(0,1),(1,1)], [(1,1,1)]),
        ([(1,1),(3,-1)], [(2,1),(3,1)], [(0,0,1)])
    ]
    
    # Super Strassen definitions for 4x4 * 4x4
    # A blocks: A00, A01, A10, A11
    # B blocks: B00, B01, B10, B11
    # C blocks: C00, C01, C10, C11
    #
    # M1=(A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    # M2=(A10+A11)B00       -> C10 += M2, C11 -= M2
    # M3=A00(B01-B11)       -> C01 += M3, C11 += M3
    # M4=A11(B10-B00)       -> C00 += M4, C10 += M4
    # M5=(A00+A01)B11       -> C00 -= M5, C01 += M5
    # M6=(A10-A00)(B00+B01) -> C11 += M6
    # M7=(A01-A11)(B10+B11) -> C00 += M7
    
    # p_block_coeffs: [A00, A01, A10, A11]
    # q_block_coeffs: [B00, B01, B10, B11]
    # c_block_contribs: [(c_row, c_col, coeff)]
    super_defs = [
        ([1,0,0,1], [1,0,0,1], [(0,0,1),(1,1,1)]),
        ([0,0,1,1], [1,0,0,0], [(1,0,1),(1,1,-1)]),
        ([1,0,0,0], [0,1,0,-1], [(0,1,1),(1,1,1)]),
        ([0,0,0,1], [-1,0,1,0], [(0,0,1),(1,0,1)]),
        ([1,1,0,0], [0,0,0,1], [(0,0,-1),(0,1,1)]),
        ([-1,0,1,0], [1,1,0,0], [(1,1,1)]),
        ([0,1,0,-1], [0,0,1,1], [(0,0,1)])
    ]
    
    def get_p_element(p_elem_idx, p_block_coeffs):
        """
        Map local 2x2 P element index to global A indices.
        p_elem_idx: 0:00, 1:01, 2:10, 3:11
        Returns list of (i, j, coeff)
        """
        r = p_elem_idx // 2
        c = p_elem_idx % 2
        res = []
        for b_idx, coeff in enumerate(p_block_coeffs):
            if coeff == 0:
                continue
            if b_idx == 0:    # A00
                gi, gj = r, c
            elif b_idx == 1:  # A01
                gi, gj = r, 2 + c
            elif b_idx == 2:  # A10
                gi, gj = 2 + r, c
            elif b_idx == 3:  # A11
                gi, gj = 2 + r, 2 + c
            res.append((gi, gj, coeff))
        return res
    
    def get_q_element(q_elem_idx, q_block_coeffs):
        """
        Map local 2x2 Q element index to global B indices.
        q_elem_idx: 0:00, 1:01, 2:10, 3:11
        Returns list of (j, k, coeff)
        """
        r = q_elem_idx // 2
        c = q_elem_idx % 2
        res = []
        for b_idx, coeff in enumerate(q_block_coeffs):
            if coeff == 0:
                continue
            if b_idx == 0:    # B00
                gj, gk = r, c
            elif b_idx == 1:  # B01
                gj, gk = r, 2 + c
            elif b_idx == 2:  # B10
                gj, gk = 2 + r, c
            elif b_idx == 3:  # B11
                gj, gk = 2 + r, 2 + c
            res.append((gj, gk, coeff))
        return res
    
    # Build the 49 terms
    for s_idx, (p_bc, q_bc, c_contribs) in enumerate(super_defs):
        for i_idx, (p_elems, q_elems, r_contribs) in enumerate(inner_defs):
            u_list = []
            v_list = []
            w_list = []
            
            # Build U list (A part)
            for p_elem, coeff in p_elems:
                for gi, gj, c in get_p_element(p_elem, p_bc):
                    u_list.append((gi, gj, coeff * c))
            
            # Build V list (B part)
            for q_elem, coeff in q_elems:
                for gj, gk, c in get_q_element(q_elem, q_bc):
                    v_list.append((gj, gk, coeff * c))
            
            # Build W list (C part)
            for br, bc, sign in c_contribs:
                for lr, lc, local_coeff in r_contribs:
                    gi = br * 2 + lr
                    gk = bc * 2 + lc
                    w_list.append((gk, gi, sign * local_coeff))
            
            if u_list and v_list and w_list:
                add_term(col, u_list, v_list, w_list)
                col += 1
    
    # --- Part 2: 4x4 * 4x1 (Schoolbook) ---
    # This part handles the 5th column of B (index 4) contributing to the 5th column of C (index 4).
    # C[i, 4] = sum_j A[i, j] * B[j, 4]
    for i in range(4):
        for j in range(4):
            U[i * m + j][col] = 1
            V[j * p + 4][col] = 1
            W[4 * n + i][col] = 1  # k=4, i=i. W index is k*n + i
            col += 1
    
    return {"U": U, "V": V, "W": W}
