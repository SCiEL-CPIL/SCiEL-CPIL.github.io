"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    rank = 65
    U = np.zeros((n * m, rank), dtype=int)  # 16 x 65
    V = np.zeros((m * p, rank), dtype=int)  # 20 x 65
    W = np.zeros((p * n, rank), dtype=int)  # 20 x 65
    
    # Super-level Strassen block coefficients
    # P blocks: (coeff_A00, coeff_A01, coeff_A10, coeff_A11)
    # Q blocks: (coeff_B00, coeff_B01, coeff_B10, coeff_B11)
    super_p_blocks = [
        [1, 0, 0, 1],   # M1: A00+A11
        [0, 0, 1, 1],   # M2: A10+A11
        [1, 0, 0, 0],   # M3: A00
        [0, 0, 0, 1],   # M4: A11
        [1, 1, 0, 0],   # M5: A00+A01
        [-1, 0, 1, 0],  # M6: A10-A00
        [0, 1, 0, -1],  # M7: A01-A11
    ]
    super_q_blocks = [
        [1, 0, 0, 1],   # M1: B00+B11
        [1, 0, 0, 0],   # M2: B00
        [0, 1, 0, -1],  # M3: B01-B11
        [-1, 0, 1, 0],  # M4: B10-B00
        [0, 0, 0, 1],   # M5: B11
        [1, 1, 0, 0],   # M6: B00+B01
        [0, 0, 1, 1],   # M7: B10+B11
    ]
    
    # Inner Strassen: which P and Q elements are multiplied
    # Each entry: (list of (p_elem_idx, coeff), list of (q_elem_idx, coeff))
    # p_elem_idx: 0=P00, 1=P01, 2=P10, 3=P11
    # q_elem_idx: 0=Q00, 1=Q01, 2=Q10, 3=Q11
    inner_pq = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)]),   # s1: (P00+P11)(Q00+Q11)
        ([(2, 1), (3, 1)], [(0, 1)]),             # s2: (P10+P11)Q00
        ([(0, 1)], [(1, 1), (3, -1)]),            # s3: P00(Q01-Q11)
        ([(3, 1)], [(2, 1), (0, -1)]),            # s4: P11(Q10-Q00)
        ([(0, 1), (1, 1)], [(3, 1)]),             # s5: (P00+P01)Q11
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)]),   # s6: (P10-P00)(Q00+Q01)
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)]),   # s7: (P01-P11)(Q10+Q11)
    ]
    
    # Inner Strassen contribution to S = [[S00,S01],[S10,S11]]
    # indexed as [S00, S01, S10, S11]
    inner_s_contrib = [
        [1, 0, 0, 1],   # s1
        [0, 0, 1, -1],  # s2
        [0, 1, 0, 1],   # s3
        [1, 0, 1, 0],   # s4
        [-1, 1, 0, 0],  # s5
        [0, 0, 0, 1],   # s6
        [1, 0, 0, 0],   # s7
    ]
    
    # Super Strassen contribution to C blocks [C00, C01, C10, C11]
    super_c_contrib = [
        [1, 0, 0, 1],   # M1
        [0, 0, 1, -1],  # M2
        [0, 1, 0, 1],   # M3
        [1, 0, 1, 0],   # M4
        [-1, 1, 0, 0],  # M5
        [0, 0, 0, 1],   # M6
        [1, 0, 0, 0],   # M7
    ]
    
    def get_global_ij(block_idx, r, c):
        """Get global (i,j) for A block at local position (r,c)"""
        if block_idx == 0:    # A00
            return r, c
        elif block_idx == 1:  # A01
            return r, 2 + c
        elif block_idx == 2:  # A10
            return 2 + r, c
        elif block_idx == 3:  # A11
            return 2 + r, 2 + c
    
    def get_global_jk(block_idx, r, c):
        """Get global (j,k) for B block at local position (r,c)"""
        if block_idx == 0:    # B00
            return r, c
        elif block_idx == 1:  # B01
            return r, 2 + c
        elif block_idx == 2:  # B10
            return 2 + r, c
        elif block_idx == 3:  # B11
            return 2 + r, 2 + c
    
    col = 0
    for s in range(7):
        p_blocks = super_p_blocks[s]
        q_blocks = super_q_blocks[s]
        p_elems, q_elems = inner_pq[s]  # Wait, this is wrong - inner_pq is indexed by inner term, not super term
        # Actually, inner_pq is the same for all super terms; it defines which P,Q elements are multiplied
        # The inner term index t determines the pattern
        for t in range(7):
            p_elems_t, q_elems_t = inner_pq[t]
            s_contrib = inner_s_contrib[t]
            c_contrib = super_c_contrib[s]
            
            # Build U contributions
            for p_idx, p_coeff in p_elems_t:
                r = p_idx // 2
                c = p_idx % 2
                for b_idx, block_coeff in enumerate(p_blocks):
                    if block_coeff == 0:
                        continue
                    gi, gj = get_global_ij(b_idx, r, c)
                    coeff = block_coeff * p_coeff
                    if coeff != 0:
                        U[gi * m + gj, col] += coeff
            
            # Build V contributions
            for q_idx, q_coeff in q_elems_t:
                r = q_idx // 2
                c = q_idx % 2
                for b_idx, block_coeff in enumerate(q_blocks):
                    if block_coeff == 0:
                        continue
                    gj, gk = get_global_jk(b_idx, r, c)
                    coeff = block_coeff * q_coeff
                    if coeff != 0:
                        V[gj * p + gk, col] += coeff
            
            # Build W contributions
            for s_pos in range(4):
                s_coeff = s_contrib[s_pos]
                if s_coeff == 0:
                    continue
                s_r = s_pos // 2
                s_c = s_pos % 2
                for c_b in range(4):
                    c_coeff = c_contrib[c_b]
                    if c_coeff == 0:
                        continue
                    total_coeff = s_coeff * c_coeff
                    if total_coeff == 0:
                        continue
                    if c_b == 0:
                        gi = s_r; gk = s_c
                    elif c_b == 1:
                        gi = s_r; gk = 2 + s_c
                    elif c_b == 2:
                        gi = 2 + s_r; gk = s_c
                    elif c_b == 3:
                        gi = 2 + s_r; gk = 2 + s_c
                    W[gk * n + gi, col] += total_coeff
            
            col += 1
    
    # Part 2: 4x4 * 4x1 using Schoolbook for column 4
    for i in range(4):
        for j in range(4):
            U[i * m + j, col] = 1
            V[j * p + 4, col] = 1
            W[4 * n + i, col] = 1
            col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
