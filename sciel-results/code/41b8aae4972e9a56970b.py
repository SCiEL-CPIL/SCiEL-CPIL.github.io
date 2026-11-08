"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    # Deadline guard: return a valid decomposition before timeout
    start_time = time.time()
    DEADLINE = 145  # seconds, with 5s safety margin
    
    # Fallback: Schoolbook decomposition (Rank 80)
    def get_schoolbook():
        rank_fb = n * m * p
        U_fb = [[0] * rank_fb for _ in range(n * m)]
        V_fb = [[0] * rank_fb for _ in range(m * p)]
        W_fb = [[0] * rank_fb for _ in range(p * n)]
        col_fb = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j][col_fb] = 1
                    V_fb[j * p + k][col_fb] = 1
                    W_fb[k * n + i][col_fb] = 1
                    col_fb += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}

    if time.time() - start_time > DEADLINE:
        return get_schoolbook()

    # Attempt Rank 65: Strassen for 4x4x4 (49 terms) + Schoolbook for 4x4x1 (16 terms)
    rank = 65
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
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
    
    # Strassen for 4x4 * 4x4
    # Super terms M1..M7
    # Inner terms S1..S7 for each 2x2 product
    
    # Coefficients for Strassen 2x2:
    # S1 = (P00+P11)(Q00+Q11)
    # S2 = (P10+P11)Q00
    # S3 = P00(Q01-Q11)
    # S4 = P11(Q10-Q00)
    # S5 = (P00+P01)Q11
    # S6 = (P10-P00)(Q00+Q01)
    # S7 = (P01-P11)(Q10+Q11)
    
    # Result C = [[S1+S4-S5+S7, S3+S5], [S2+S4, S1-S2+S3+S6]]
    
    # Mapping of Inner Term t to contributions to C00, C01, C10, C11 (local 2x2 result)
    # t=0 (S1): C00:+1, C11:+1
    # t=1 (S2): C10:+1, C11:-1
    # t=2 (S3): C01:+1, C11:+1
    # t=3 (S4): C00:+1, C10:+1
    # t=4 (S5): C00:-1, C01:+1
    # t=5 (S6): C11:+1
    # t=6 (S7): C00:+1
    
    inner_contribs = [
        [1, 0, 0, 1], # t=0
        [0, 0, 1, -1], # t=1
        [0, 1, 0, 1], # t=2
        [1, 0, 1, 0], # t=3
        [-1, 1, 0, 0], # t=4
        [0, 0, 0, 1], # t=5
        [1, 0, 0, 0]  # t=6
    ]
    
    # Super terms M1..M7 define P and Q blocks
    # P = sum c_ab A_ab, Q = sum d_bc B_bc
    # c_ab, d_bc are in {0, 1, -1}
    
    super_defs = [
        # [P_blocks (A00, A01, A10, A11), Q_blocks (B00, B01, B10, B11)]
        ([1, 0, 0, 1], [1, 0, 0, 1]), # M1
        ([0, 0, 1, 1], [1, 0, 0, 0]), # M2
        ([1, 0, 0, 0], [0, 1, 0, -1]), # M3
        ([0, 0, 0, 1], [-1, 0, 1, 0]), # M4
        ([1, 1, 0, 0], [0, 0, 0, 1]), # M5
        ([-1, 0, 1, 0], [1, 1, 0, 0]), # M6
        ([0, 1, 0, -1], [0, 0, 1, 1])  # M7
    ]
    
    # Contribution of Super Term M_s to Global C blocks (C00, C01, C10, C11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    super_contribs = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, -1], # M2
        [0, 1, 0, 1], # M3
        [1, 0, 1, 0], # M4
        [-1, 1, 0, 0], # M5
        [0, 0, 0, 1], # M6
        [1, 0, 0, 0]  # M7
    ]
    
    # Helper to expand a linear combination of a 2x2 block matrix into scalar terms
    # block_coeffs: [c00, c01, c10, c11] for the 4 blocks of A or B
    # is_A: True for A, False for B
    def expand_block_combination(block_coeffs, is_A):
        terms = []
        # Block 0: (0,0) to (1,1)
        # Block 1: (0,2) to (1,3)
        # Block 2: (2,0) to (3,1)
        # Block 3: (2,2) to (3,3)
        
        for b_idx, coeff in enumerate(block_coeffs):
            if coeff == 0: continue
            base_r = (b_idx // 2) * 2
            base_c = (b_idx % 2) * 2
            for lr in range(2):
                for lc in range(2):
                    if is_A:
                        gi = base_r + lr
                        gj = base_c + lc
                        terms.append((gi, gj, coeff))
                    else:
                        gj = base_r + lr
                        gk = base_c + lc
                        terms.append((gj, gk, coeff))
        return terms

    # Inner Strassen term definitions
    # Each term is a product of a linear combo of P elements and Q elements
    # P elements are P00, P01, P10, P11 (indices 0,1,2,3)
    # Q elements are Q00, Q01, Q10, Q11 (indices 0,1,2,3)
    
    inner_defs = [
        # [P_elems (idx, coeff), Q_elems (idx, coeff)]
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)]), # S1
        ([(2, 1), (3, 1)], [(0, 1)]),          # S2
        ([(0, 1)], [(1, 1), (3, -1)]),         # S3
        ([(3, 1)], [(2, 1), (0, -1)]),         # S4
        ([(0, 1), (1, 1)], [(3, 1)]),          # S5
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)]), # S6
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)])  # S7
    ]
    
    for s_idx in range(7):
        p_blocks, q_blocks = super_defs[s_idx]
        s_contribs = super_contribs[s_idx]
        
        for t_idx in range(7):
            if time.time() - start_time > DEADLINE:
                return get_schoolbook()
                
            p_elems, q_elems = inner_defs[t_idx]
            t_contribs = inner_contribs[t_idx]
            
            u_list = []
            v_list = []
            w_list = []
            
            # Expand U
            for p_idx, p_coeff in p_elems:
                # p_idx is 0..3 corresponding to P00, P01, P10, P11
                # We need to expand the linear combo of P blocks for this specific element
                # P is a 2x2 matrix formed by summing A blocks.
                # P[r,c] = sum_{b} p_blocks[b] * A_b[r,c]
                # So for a specific P element (r,c), the U terms are:
                # sum_{b} p_blocks[b] * A_b[r,c]
                
                pr = p_idx // 2
                pc = p_idx % 2
                
                for b_idx, b_coeff in enumerate(p_blocks):
                    if b_coeff == 0: continue
                    # Local (pr, pc) in block b_idx
                    base_r = (b_idx // 2) * 2
                    base_c = (b_idx % 2) * 2
                    gi = base_r + pr
                    gj = base_c + pc
                    u_list.append((gi, gj, b_coeff * p_coeff))
                    
            # Expand V
            for q_idx, q_coeff in q_elems:
                qr = q_idx // 2
                qc = q_idx % 2
                
                for b_idx, b_coeff in enumerate(q_blocks):
                    if b_coeff == 0: continue
                    base_r = (b_idx // 2) * 2
                    base_c = (b_idx % 2) * 2
                    gj = base_r + qr
                    gk = base_c + qc
                    v_list.append((gj, gk, b_coeff * q_coeff))
                    
            # Expand W
            # The scalar product contributes to local C elements based on t_contribs
            # t_contribs = [C00, C01, C10, C11] for the local 2x2 result S
            # The local S element (sr, sc) corresponds to global C elements based on s_contribs
            # s_contribs = [C00, C01, C10, C11] for the global 4x4 result
            
            # Wait, the logic is:
            # The inner Strassen term t produces a scalar value X.
            # X contributes to local S elements with weights t_contribs.
            # Each local S element (sr, sc) is part of the 2x2 result of P*Q.
            # This 2x2 result is then combined into the global 4x4 C using super_contribs.
            # Specifically, the super term s produces a 2x2 matrix S_s.
            # S_s is added to C blocks according to s_contribs.
            # So, X * t_contrib[s_pos] * s_contrib[c_block] contributes to C block c_block at position (sr, sc).
            
            for s_pos in range(4):
                t_c = t_contribs[s_pos]
                if t_c == 0: continue
                sr = s_pos // 2
                sc = s_pos % 2
                
                for c_block in range(4):
                    s_c = s_contribs[c_block]
                    if s_c == 0: continue
                    
                    total_coeff = t_c * s_c
                    if total_coeff == 0: continue
                    
                    # Map (c_block, sr, sc) to global (i, k)
                    base_i = (c_block // 2) * 2
                    base_k = (c_block % 2) * 2
                    gi = base_i + sr
                    gk = base_k + sc
                    
                    w_list.append((gk, gi, total_coeff))
                    
            add_term(col, u_list, v_list, w_list)
            col += 1
            
    # Part 2: Schoolbook for column 4
    for i in range(n):
        for j in range(m):
            if time.time() - start_time > DEADLINE:
                return get_schoolbook()
            u_list = [(i, j, 1)]
            v_list = [(j, 4, 1)]
            w_list = [(4, i, 1)]
            add_term(col, u_list, v_list, w_list)
            col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
