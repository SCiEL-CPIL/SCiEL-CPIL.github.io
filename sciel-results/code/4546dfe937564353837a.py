"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    # Rank 65 decomposition:
    # 49 terms for 4x4 * 4x4 (Strassen squared)
    # 16 terms for 4x4 * 4x1 (Schoolbook)
    
    rank = 65
    U = np.zeros((n * m, rank))  # 16 x 65
    V = np.zeros((m * p, rank))  # 20 x 65
    W = np.zeros((p * n, rank))  # 20 x 65
    
    col = 0
    
    # --- Part 1: 4x4 * 4x4 using Strassen (49 terms) ---
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
    
    # Each M is a 2x2 matrix product, decomposed with Strassen (7 scalar mults)
    
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
    
    # We need to map the "super" Strassen terms to the inner Strassen terms.
    
    # Super Term 1: P = A00+A11, Q = B00+B11
    # Super Term 2: P = A10+A11, Q = B00
    # Super Term 3: P = A00, Q = B01-B11
    # Super Term 4: P = A11, Q = B10-B00
    # Super Term 5: P = A00+A01, Q = B11
    # Super Term 6: P = A10-A00, Q = B00+B01
    # Super Term 7: P = A01-A11, Q = B10+B11
    
    # Block definitions:
    # A00: i in 0,1; j in 0,1
    # A01: i in 0,1; j in 2,3
    # A10: i in 2,3; j in 0,1
    # A11: i in 2,3; j in 2,3
    
    # B00: j in 0,1; k in 0,1
    # B01: j in 0,1; k in 2,3
    # B10: j in 2,3; k in 0,1
    # B11: j in 2,3; k in 2,3
    
    # For each super term, we have linear combinations of blocks for P and Q
    # For each inner Strassen term, we have linear combinations of P and Q elements
    
    # Let's build the U, V, W entries directly
    
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
    
    # Define the super terms
    # Each super term has:
    # - p_blocks: coefficients for A00, A01, A10, A11
    # - q_blocks: coefficients for B00, B01, B10, B11
    
    super_terms = [
        # M1: (A00+A11)(B00+B11)
        ([1, 0, 0, 1], [1, 0, 0, 1]),
        # M2: (A10+A11)B00
        ([0, 0, 1, 1], [1, 0, 0, 0]),
        # M3: A00(B01-B11)
        ([1, 0, 0, 0], [0, 1, 0, -1]),
        # M4: A11(B10-B00)
        ([0, 0, 0, 1], [-1, 0, 1, 0]),
        # M5: (A00+A01)B11
        ([1, 1, 0, 0], [0, 0, 0, 1]),
        # M6: (A10-A00)(B00+B01)
        ([-1, 0, 1, 0], [1, 1, 0, 0]),
        # M7: (A01-A11)(B10+B11)
        ([0, 1, 0, -1], [0, 0, 1, 1]),
    ]
    
    # Inner Strassen terms
    # Each inner term has:
    # - p_elems: list of (p_elem_idx, coeff) where p_elem_idx is 0:00, 1:01, 2:10, 3:11
    # - q_elems: list of (q_elem_idx, coeff)
    # - s_contrib: coefficients for S[0,0], S[0,1], S[1,0], S[1,1]
    
    inner_terms = [
        # s1: (P00+P11)(Q00+Q11)
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [1, 0, 0, 1]),
        # s2: (P10+P11)Q00
        ([(2, 1), (3, 1)], [(0, 1)], [0, 0, 1, -1]),
        # s3: P00(Q01-Q11)
        ([(0, 1)], [(1, 1), (3, -1)], [0, 1, 0, 1]),
        # s4: P11(Q10-Q00)
        ([(3, 1)], [(2, 1), (0, -1)], [1, 0, 1, 0]),
        # s5: (P00+P01)Q11
        ([(0, 1), (1, 1)], [(3, 1)], [-1, 1, 0, 0]),
        # s6: (P10-P00)(Q00+Q01)
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [0, 0, 0, 1]),
        # s7: (P01-P11)(Q10+Q11)
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [1, 0, 0, 0]),
    ]
    
    # C block contributions for each super term
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    c_block_contrib = [
        # M1: C00 (+1), C11 (+1)
        [1, 0, 0, 1],
        # M2: C10 (+1), C11 (-1)
        [0, 0, 1, -1],
        # M3: C01 (+1), C11 (+1)
        [0, 1, 0, 1],
        # M4: C00 (+1), C10 (+1)
        [1, 0, 1, 0],
        # M5: C00 (-1), C01 (+1)
        [-1, 1, 0, 0],
        # M6: C11 (+1)
        [0, 0, 0, 1],
        # M7: C00 (+1)
        [1, 0, 0, 0],
    ]
    
    # Function to get P element linear combo
    def get_p_element(p_elem_idx, p_blocks):
        # p_elem_idx: 0:00, 1:01, 2:10, 3:11
        # Returns list of (i, j, coeff)
        res = []
        r = p_elem_idx // 2
        c = p_elem_idx % 2
        
        for b_idx, coeff in enumerate(p_blocks):
            if coeff == 0: continue
            # Global i, j for block b_idx at local (r,c)
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
    
    # Function to get Q element linear combo
    def get_q_element(q_elem_idx, q_blocks):
        # q_elem_idx: 0:00, 1:01, 2:10, 3:11
        # Returns list of (j, k, coeff)
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
    
    # Add all 49 terms
    for s_idx, (p_blocks, q_blocks) in enumerate(super_terms):
        for t_idx, (p_elems, q_elems, s_contrib) in enumerate(inner_terms):
            u_list = []
            v_list = []
            w_list = []
            
            # Build u_list
            for p_idx, p_coeff in p_elems:
                elems = get_p_element(p_idx, p_blocks)
                for i, j, c in elems:
                    u_list.append((i, j, c * p_coeff))
                    
            # Build v_list
            for q_idx, q_coeff in q_elems:
                elems = get_q_element(q_idx, q_blocks)
                for j, k, c in elems:
                    v_list.append((j, k, c * q_coeff))
                    
            # Build w_list
            c_block = c_block_contrib[s_idx]
            
            for s_pos in range(4):
                s_coeff = s_contrib[s_pos]
                if s_coeff == 0: continue
                # s_pos 0: S[0,0]. 1: S[0,1]. 2: S[1,0]. 3: S[1,1]
                s_r = s_pos // 2
                s_c = s_pos % 2
                
                for c_b in range(4):
                    c_coeff = c_block[c_b]
                    if c_coeff == 0: continue
                    
                    total_coeff = s_coeff * c_coeff
                    if total_coeff == 0: continue
                    
                    # Determine global i, k
                    if c_b == 0:
                        gi = s_r; gk = s_c
                    elif c_b == 1:
                        gi = s_r; gk = 2 + s_c
                    elif c_b == 2:
                        gi = 2 + s_r; gk = s_c
                    elif c_b == 3:
                        gi = 2 + s_r; gk = 2 + s_c
                        
                    w_list.append((gk, gi, total_coeff))
                    
            add_term(col, u_list, v_list, w_list)
            col += 1
    
    # --- Part 2: 4x4 * 4x1 using Schoolbook (16 terms) ---
    # A (4x4) * B_col (4x1)
    # B_col is column 4 of B.
    # C_col is column 4 of C.
    # C[i,4] = sum_j A[i,j] B[j,4]
    
    for i in range(4):
        for j in range(4):
            u_list = [(i, j, 1)]
            v_list = [(j, 4, 1)]
            w_list = [(4, i, 1)]
            add_term(col, u_list, v_list, w_list)
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
