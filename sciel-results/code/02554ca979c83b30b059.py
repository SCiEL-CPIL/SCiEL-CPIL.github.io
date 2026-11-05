"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    deadline = time.time() + 140  # safety margin before 150s kill
    
    # Strategy: Use a two-level Strassen decomposition for the 4x4x4 part (rank 49)
    # and schoolbook for the 4x4x1 part (rank 16), totaling rank 65.
    # This is the same as the current approach but with cleaner implementation.
    
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
    
    # --- Part 1: 4x4 * 4x4 using two-level Strassen (rank 49) ---
    
    # Block structure for 4x4 matrices:
    # A = [[A00, A01], [A10, A11]] where each is 2x2
    # A00: rows 0-1, cols 0-1
    # A01: rows 0-1, cols 2-3
    # A10: rows 2-3, cols 0-1
    # A11: rows 2-3, cols 2-3
    
    def a_block_elem(block_idx, r, c):
        if block_idx == 0:  # A00
            return (r, c)
        elif block_idx == 1:  # A01
            return (r, 2 + c)
        elif block_idx == 2:  # A10
            return (2 + r, c)
        elif block_idx == 3:  # A11
            return (2 + r, 2 + c)
    
    def b_block_elem(block_idx, r, c):
        if block_idx == 0:  # B00
            return (r, c)
        elif block_idx == 1:  # B01
            return (r, 2 + c)
        elif block_idx == 2:  # B10
            return (2 + r, c)
        elif block_idx == 3:  # B11
            return (2 + r, 2 + c)
    
    def c_block_elem(block_idx, r, c):
        if block_idx == 0:  # C00
            return (r, c)
        elif block_idx == 1:  # C01
            return (r, 2 + c)
        elif block_idx == 2:  # C10
            return (2 + r, c)
        elif block_idx == 3:  # C11
            return (2 + r, 2 + c)
    
    # Level 1 Strassen: 7 products for 4x4 * 4x4
    level1_products = [
        # S0 = (A00 + A11) * (B00 + B11)
        {
            'p_blocks': [(0, 1), (3, 1)],
            'q_blocks': [(0, 1), (3, 1)],
            'c_contrib': {0: 1, 3: 1}
        },
        # S1 = (A10 + A11) * B00
        {
            'p_blocks': [(2, 1), (3, 1)],
            'q_blocks': [(0, 1)],
            'c_contrib': {2: 1, 3: -1}
        },
        # S2 = A00 * (B01 - B11)
        {
            'p_blocks': [(0, 1)],
            'q_blocks': [(1, 1), (3, -1)],
            'c_contrib': {1: 1, 3: 1}
        },
        # S3 = A11 * (B10 - B00)
        {
            'p_blocks': [(3, 1)],
            'q_blocks': [(2, 1), (0, -1)],
            'c_contrib': {0: 1, 2: 1}
        },
        # S4 = (A00 + A01) * B11
        {
            'p_blocks': [(0, 1), (1, 1)],
            'q_blocks': [(3, 1)],
            'c_contrib': {0: -1, 1: 1}
        },
        # S5 = (A10 - A00) * (B00 + B01)
        {
            'p_blocks': [(2, 1), (0, -1)],
            'q_blocks': [(0, 1), (1, 1)],
            'c_contrib': {3: 1}
        },
        # S6 = (A01 - A11) * (B10 + B11)
        {
            'p_blocks': [(1, 1), (3, -1)],
            'q_blocks': [(2, 1), (3, 1)],
            'c_contrib': {0: 1}
        },
    ]
    
    # Level 2 Strassen: 7 products for 2x2 * 2x2
    level2_strassen = [
        # M0 = (p00 + p11) * (q00 + q11)
        {
            'p_elems': [(0, 1), (3, 1)],
            'q_elems': [(0, 1), (3, 1)],
            'c_contrib': {0: 1, 3: 1}
        },
        # M1 = (p10 + p11) * q00
        {
            'p_elems': [(2, 1), (3, 1)],
            'q_elems': [(0, 1)],
            'c_contrib': {2: 1, 3: -1}
        },
        # M2 = p00 * (q01 - q11)
        {
            'p_elems': [(0, 1)],
            'q_elems': [(1, 1), (3, -1)],
            'c_contrib': {1: 1, 3: 1}
        },
        # M3 = p11 * (q10 - q00)
        {
            'p_elems': [(3, 1)],
            'q_elems': [(2, 1), (0, -1)],
            'c_contrib': {0: 1, 2: 1}
        },
        # M4 = (p00 + p01) * q11
        {
            'p_elems': [(0, 1), (1, 1)],
            'q_elems': [(3, 1)],
            'c_contrib': {0: -1, 1: 1}
        },
        # M5 = (p10 - p00) * (q00 + q01)
        {
            'p_elems': [(2, 1), (0, -1)],
            'q_elems': [(0, 1), (1, 1)],
            'c_contrib': {3: 1}
        },
        # M6 = (p01 - p11) * (q10 + q11)
        {
            'p_elems': [(1, 1), (3, -1)],
            'q_elems': [(2, 1), (3, 1)],
            'c_contrib': {0: 1}
        },
    ]
    
    # Generate all 49 terms from two-level Strassen
    for l1_idx, l1_prod in enumerate(level1_products):
        for l2_idx, l2_prod in enumerate(level2_strassen):
            if time.time() > deadline:
                break
            
            u_list = []
            v_list = []
            w_list = []
            
            # Build u_list: linear combination of A elements
            for p_block_idx, p_block_coeff in l1_prod['p_blocks']:
                for p_elem_idx, p_elem_coeff in l2_prod['p_elems']:
                    r = p_elem_idx // 2
                    c = p_elem_idx % 2
                    gi, gj = a_block_elem(p_block_idx, r, c)
                    coeff = p_block_coeff * p_elem_coeff
                    u_list.append((gi, gj, coeff))
            
            # Build v_list: linear combination of B elements (for cols 0-3)
            for q_block_idx, q_block_coeff in l1_prod['q_blocks']:
                for q_elem_idx, q_elem_coeff in l2_prod['q_elems']:
                    r = q_elem_idx // 2
                    c = q_elem_idx % 2
                    gj, gk = b_block_elem(q_block_idx, r, c)
                    coeff = q_block_coeff * q_elem_coeff
                    v_list.append((gj, gk, coeff))
            
            # Build w_list: contributions to C (for cols 0-3)
            for si_elem_idx, si_coeff in l2_prod['c_contrib'].items():
                si_r = si_elem_idx // 2
                si_c = si_elem_idx % 2
                for l1_c_block, l1_c_coeff in l1_prod['c_contrib'].items():
                    gi, gj = c_block_elem(l1_c_block, si_r, si_c)
                    coeff = l1_c_coeff * si_coeff
                    w_list.append((gj, gi, coeff))  # W[k*n+i]
            
            add_term(col, u_list, v_list, w_list)
            col += 1
    
    # --- Part 2: 4x4 * 4x1 using schoolbook (rank 16) ---
    for i in range(n):
        for j in range(m):
            if time.time() > deadline:
                break
            u_list = [(i, j, 1)]
            v_list = [(j, 4, 1)]
            w_list = [(4, i, 1)]
            add_term(col, u_list, v_list, w_list)
            col += 1
    
    assert col == rank, f"Expected {rank} terms, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
