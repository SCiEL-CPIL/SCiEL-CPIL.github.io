import time
import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 20s margin before 150s kill
    
    n, m, p = 4, 4, 6
    rank = 77
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    def add_term(u_coeffs, v_coeffs, w_coeffs):
        nonlocal col
        if col >= rank:
            return
        for idx, c in u_coeffs:
            if c != 0:
                U[idx, col] += c
        for idx, c in v_coeffs:
            if c != 0:
                V[idx, col] += c
        for idx, c in w_coeffs:
            if c != 0:
                W[idx, col] += c
        col += 1
    
    # Strassen 2x2x2 terms: (U_coeffs, V_coeffs)
    # Each term is a pair of sparse vectors for the two factors
    # S1 = (A00+A11)(B00+B11)
    # S2 = (A10+A11)(B00)
    # S3 = (A00)(B01-B11)
    # S4 = (A11)(B10-B00)
    # S5 = (A00+A01)(B11)
    # S6 = (A10-A00)(B00+B01)
    # S7 = (A01-A11)(B10+B11)
    
    strassen_2x2 = [
        # S1
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        # S2
        ([(1,0,1), (1,1,1)], [(0,0,1)]),
        # S3
        ([(0,0,1)], [(0,1,1), (1,1,-1)]),
        # S4
        ([(1,1,1)], [(1,0,1), (0,0,-1)]),
        # S5
        ([(0,0,1), (0,1,1)], [(1,1,1)]),
        # S6
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)]),
        # S7
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)]),
    ]
    
    # Result mapping for 2x2 Strassen
    # C00 = S1 + S4 - S5 - S7
    # C01 = S3 + S5
    # C10 = S2 + S4
    # C11 = S1 - S2 + S3 + S6
    
    # result_map[c_block][s_idx] = coefficient
    result_map = {
        (0,0): {0: 1, 3: 1, 4: -1, 6: -1},
        (0,1): {2: 1, 4: 1},
        (1,0): {1: 1, 3: 1},
        (1,1): {0: 1, 1: -1, 2: 1, 5: 1},
    }
    
    # --- Part 1: Strassen for 4x4x4 (B cols 0-3, C cols 0-3) ---
    # Split into 2x2 blocks of 2x2 submatrices
    # A block (bx, by) covers A[2*bx:2*bx+2, 2*by:2*by+2]
    # B block (bx, by) covers B[2*bx:2*bx+2, 2*by:2*by+2]
    # C block (bx, by) covers C[2*bx:2*bx+2, 2*by:2*by+2]
    
    def a_idx(bx, by, r, c):
        return (2*bx + r) * m + (2*by + c)
    
    def b1_idx(bx, by, r, c):
        return (2*bx + r) * p + (2*by + c)
    
    def c1_idx(bx, by, r, c):
        return (2*by + c) * n + (2*bx + r)
    
    # Big Strassen combinations for 4x4x4
    # Each big multiplication M_i = (A_part) * (B_part)
    # A_part is a sum of A-blocks with signs
    # B_part is a sum of B-blocks with signs
    # The result contributes to C-blocks
    
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)(B00)
    # M3 = (A00)(B01 - B11)
    # M4 = (A11)(B10 - B00)
    # M5 = (A00 + A01)(B11)
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    big_combos = [
        # (a_blocks, b_blocks) where blocks are (bx, by, sign)
        ([((0,0,1), (1,1,1))], [((0,0,1), (1,1,1))]),   # M1
        ([((1,0,1), (1,1,1))], [((0,0,1))]),             # M2
        ([((0,0,1))], [((0,1,1), (1,1,-1))]),            # M3
        ([((1,1,1))], [((1,0,1), (0,0,-1))]),            # M4
        ([((0,0,1), (0,1,1))], [((1,1,1))]),             # M5
        ([((1,0,1), (0,0,-1))], [((0,0,1), (0,1,1))]),   # M6
        ([((0,1,1), (1,1,-1))], [((1,0,1), (1,1,1))]),   # M7
    ]
    
    # Big result mapping: which C-blocks each M_i contributes to
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    big_result_map = {
        (0,0): {0: 1, 3: 1, 4: -1, 6: -1},
        (0,1): {2: 1, 4: 1},
        (1,0): {1: 1, 3: 1},
        (1,1): {0: 1, 1: -1, 2: 1, 5: 1},
    }
    
    # Iterate through the 7 big multiplications
    for big_idx in range(7):
        if time.time() > deadline:
            break
        
        a_blocks, b_blocks = big_combos[big_idx]
        
        for s_idx in range(7):
            if time.time() > deadline:
                break
            
            u_coeffs_inner, v_coeffs_inner = strassen_2x2[s_idx]
            
            u_coeffs = []
            for (r, c, c_val) in u_coeffs_inner:
                for (bx, by, sign) in a_blocks:
                    idx = a_idx(bx, by, r, c)
                    u_coeffs.append((idx, sign * c_val))
            
            v_coeffs = []
            for (r, c, c_val) in v_coeffs_inner:
                for (bx, by, sign) in b_blocks:
                    idx = b1_idx(bx, by, r, c)
                    v_coeffs.append((idx, sign * c_val))
            
            w_coeffs = []
            for c_block, big_contribs in big_result_map.items():
                big_coeff = big_contribs.get(big_idx, 0)
                if big_coeff == 0:
                    continue
                
                for (cr, cc), s_contribs in result_map.items():
                    small_coeff = s_contribs.get(s_idx, 0)
                    if small_coeff == 0:
                        continue
                    
                    idx = c1_idx(c_block[0], c_block[1], cr, cc)
                    w_coeffs.append((idx, big_coeff * small_coeff))
            
            add_term(u_coeffs, v_coeffs, w_coeffs)
    
    # --- Part 2: Strassen-based 4x4x2 (B cols 4-5, C cols 4-5) ---
    # B has columns 4-5 (2 columns), C has columns 4-5 (2 columns)
    # Split into 2x2 blocks where B-block is 2x2 (rows 2*bx:2*bx+2, cols 4+cy:4+cy+1 for cy in 0,1)
    # Actually, B is 4x2, so each "block" in the Strassen sense is 2x2
    # B-block (bx, by) where by=0 covers cols 4, by=1 covers col 5? No, that's 2 cols total.
    # For 4x4x2: A is 4x4, B is 4x2, C is 4x2
    # We can think of B as two 2x2 blocks: B_left (cols 4) and B_right (col 5)
    # But Strassen needs square blocks. So we use 2x2x2 Strassen on 2x2 subproblems.
    
    # Split A into 2x2 blocks: A00, A01, A10, A11
    # Split B into 2x2 blocks: B0 (rows 0-1, cols 4-5), B1 (rows 2-3, cols 4-5)
    # Split C into 2x2 blocks: C0 (rows 0-1, cols 4-5), C1 (rows 2-3, cols 4-5)
    
    # C0 = A00 * B0 + A01 * B1
    # C1 = A10 * B0 + A11 * B1
    
    # We can use Strassen on each 2x2x2 multiplication, but that would be 4 * 7 = 28 terms.
    # Alternatively, we can combine:
    # C = A * B where A is 4x4, B is 4x2
    # This is equivalent to computing 4x2 outputs from 4x4 * 4x2
    
    # Let's use the direct approach with Strassen 2x2x2 for each pair
    # Products: (A_block, B_block, C_block)
    products = [
        ('00', 'T', 'top'),   # A00 * BT -> Ctop
        ('01', 'T', 'top'),   # A01 * BT -> Ctop
        ('10', 'B', 'bot'),   # A10 * BB -> Cbot
        ('11', 'B', 'bot'),   # A11 * BB -> Cbot
    ]
    
    for a_block, b_block, c_part in products:
        if time.time() > deadline:
            break
        
        for s_idx in range(7):
            if time.time() > deadline:
                break
            
            u_coeffs_inner, v_coeffs_inner = strassen_2x2[s_idx]
            
            u_coeffs = []
            for (r, c, c_val) in u_coeffs_inner:
                x, y = int(a_block[0]), int(a_block[1])
                idx = (2*x + r) * m + (2*y + c)
                u_coeffs.append((idx, c_val))
            
            v_coeffs = []
            for (r, c, c_val) in v_coeffs_inner:
                if b_block == 'T':
                    idx = r * p + (4 + c)
                else:
                    idx = (2 + r) * p + (4 + c)
                v_coeffs.append((idx, c_val))
            
            w_coeffs = []
            for (cr, cc), s_contribs in result_map.items():
                small_coeff = s_contribs.get(s_idx, 0)
                if small_coeff == 0:
                    continue
                
                if c_part == 'top':
                    idx = (4 + cc) * n + cr
                else:
                    idx = (4 + cc) * n + (2 + cr)
                
                w_coeffs.append((idx, small_coeff))
            
            add_term(u_coeffs, v_coeffs, w_coeffs)
    
    return {"U": U, "V": V, "W": W}
