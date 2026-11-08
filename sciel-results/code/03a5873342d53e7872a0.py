"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    # Rank 65: 49 for 4x4*4x4 (Strassen) + 16 for 4x4*4x1 (Schoolbook)
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
    
    # --- Part 1: 4x4 * 4x4 using Strassen (Data-Driven Table) ---
    # Pre-computed table of 49 terms for 4x4x4 Strassen decomposition
    # Each term is (u_terms, v_terms, w_terms)
    # u_terms: list of (i, j, coeff)
    # v_terms: list of (j, k, coeff)
    # w_terms: list of (k, i, coeff)
    
    # Strassen 2x2x2 core:
    # s1 = (a00+a11)(b00+b11)
    # s2 = (a10+a11)b00
    # s3 = a00(b01-b11)
    # s4 = a11(b10-b00)
    # s5 = (a00+a01)b11
    # s6 = (a10-a00)(b00+b01)
    # s7 = (a01-a11)(b10+b11)
    
    # 4x4x4 decomposition:
    # We iterate over 7 super-terms and 7 inner-terms.
    
    # Super term definitions (A-side, B-side)
    # A-side: [A00, A01, A10, A11]
    # B-side: [B00, B01, B10, B11]
    super_A = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0],# M6
        [0, 1, 0, -1] # M7
    ]
    super_B = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1],# M3
        [-1, 0, 1, 0],# M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    
    # Inner term definitions (A-side, B-side) for 2x2x2
    # A-side: [A00, A01, A10, A11] (local 2x2)
    # B-side: [B00, B01, B10, B11] (local 2x2)
    inner_A = [
        [1, 0, 0, 1], # s1
        [0, 0, 1, 1], # s2
        [1, 0, 0, 0], # s3
        [0, 0, 0, 1], # s4
        [1, 1, 0, 0], # s5
        [-1, 0, 1, 0],# s6
        [0, 1, 0, -1] # s7
    ]
    inner_B = [
        [1, 0, 0, 1], # s1
        [1, 0, 0, 0], # s2
        [0, 1, 0, -1],# s3
        [-1, 0, 1, 0],# s4
        [0, 0, 0, 1], # s5
        [1, 1, 0, 0], # s6
        [0, 0, 1, 1]  # s7
    ]
    
    # Result contributions
    # Inner result S (2x2) contributions from s1..s7
    # S00 = s1 + s4 - s5 + s7
    # S01 = s3 + s5
    # S10 = s2 + s4
    # S11 = s1 - s2 + s3 + s6
    inner_S_contrib = [
        [1, 0, 0, 1], # s1
        [0, 0, 1, -1],# s2
        [0, 1, 0, 1], # s3
        [1, 0, 1, 0], # s4
        [-1, 1, 0, 0],# s5
        [0, 0, 0, 1], # s6
        [1, 0, 0, 0]  # s7
    ]
    
    # Super result C (4x4) contributions from M1..M7
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    super_C_contrib = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, -1],# M2
        [0, 1, 0, 1], # M3
        [1, 0, 1, 0], # M4
        [-1, 1, 0, 0],# M5
        [0, 0, 0, 1], # M6
        [1, 0, 0, 0]  # M7
    ]
    
    # Helper to map local 2x2 index to global 4x4 index
    # block: 0=A00, 1=A01, 2=A10, 3=A11
    # local_r, local_c: 0 or 1
    def map_A(block, lr, lc):
        gr = (block // 2) * 2 + lr
        gc = (block % 2) * 2 + lc
        return gr, gc
        
    def map_B(block, lr, lc):
        gr = (block // 2) * 2 + lr
        gc = (block % 2) * 2 + lc
        return gr, gc

    for s in range(7):
        for t in range(7):
            u_list = []
            v_list = []
            w_list = []
            
            # Process A-side (U)
            # Combine super_A[s] and inner_A[t]
            # super_A[s] gives weights for A00, A01, A10, A11
            # inner_A[t] gives weights for local A00, A01, A10, A11 within those blocks
            for b_idx in range(4):
                w_super = super_A[s][b_idx]
                if w_super == 0: continue
                for l_idx in range(4):
                    w_inner = inner_A[t][l_idx]
                    if w_inner == 0: continue
                    coeff = w_super * w_inner
                    if coeff == 0: continue
                    lr = l_idx // 2
                    lc = l_idx % 2
                    gi, gj = map_A(b_idx, lr, lc)
                    u_list.append((gi, gj, coeff))
                    
            # Process B-side (V)
            for b_idx in range(4):
                w_super = super_B[s][b_idx]
                if w_super == 0: continue
                for l_idx in range(4):
                    w_inner = inner_B[t][l_idx]
                    if w_inner == 0: continue
                    coeff = w_super * w_inner
                    if coeff == 0: continue
                    lr = l_idx // 2
                    lc = l_idx % 2
                    gj, gk = map_B(b_idx, lr, lc)
                    v_list.append((gj, gk, coeff))
                    
            # Process C-side (W)
            # Combine super_C_contrib[s] and inner_S_contrib[t]
            for c_block in range(4):
                w_super = super_C_contrib[s][c_block]
                if w_super == 0: continue
                for s_pos in range(4):
                    w_inner = inner_S_contrib[t][s_pos]
                    if w_inner == 0: continue
                    coeff = w_super * w_inner
                    if coeff == 0: continue
                    # s_pos maps to local C element
                    lr = s_pos // 2
                    lc = s_pos % 2
                    # c_block maps to global C block
                    gi = (c_block // 2) * 2 + lr
                    gk = (c_block % 2) * 2 + lc
                    # W index is k*n + i
                    w_list.append((gk, gi, coeff))
            
            add_term(col, u_list, v_list, w_list)
            col += 1
            
    # --- Part 2: 4x4 * 4x1 using Schoolbook ---
    for i in range(4):
        for j in range(4):
            u_list = [(i, j, 1)]
            v_list = [(j, 4, 1)]
            w_list = [(4, i, 1)]
            add_term(col, u_list, v_list, w_list)
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
