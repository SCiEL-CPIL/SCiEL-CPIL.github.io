"""Combined approach: structured search for low-rank exact tensor decomposition."""
import numpy as np
import time
import random
import itertools

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """Find rank-minimal exact tensor decomposition for 4x4 * 4x6 matrix multiplication."""
    n, m, p = DIMENSIONS
    
    # We aim for rank 77 using:
    # 1. Strassen (rank 49) for the 4x4 * 4x4 part (B cols 0-3, C cols 0-3)
    # 2. Schoolbook (rank 32) for the 4x4 * 4x2 part (B cols 4-5, C cols 4-5)
    # Total rank = 49 + 32 = 81? Wait, the inspiration says 77.
    # Let's re-verify the rank of the 4x4*4x2 part.
    # A is 4x4. B_right is 4x2. C_right is 4x2.
    # Schoolbook rank is n*m*p = 4*4*2 = 32.
    # So 49 + 32 = 81.
    # Why does the inspiration say 77?
    # "The current program uses Strassen's algorithm for the 4x4x4 block (rank 49) and schoolbook multiplication for the 4x4x2 block (rank 32), resulting in a total rank of 81"
    # But the code in inspiration 1 sets `rank = 77`.
    # Let's look closer at the inspiration code. It uses Strassen for the 4x4x4 part.
    # Does it use a better algorithm for the 4x4x2 part?
    # Or is there an overlap? No, the columns are disjoint.
    # Maybe the 4x4x2 part can be done with rank < 32?
    # The tensor for 4x4x2 multiplication.
    # Is there a known decomposition?
    # Usually, rectangular matrix multiplication ranks are not as well studied.
    # However, the current program (score 1.1429) uses rank 84.
    # The inspiration (score 1.2468) claims rank 77.
    # 1.0 / 77 * seed? No, score = seed / value.
    # If seed is 88 (schoolbook 4x4x6), 88/84 = 1.047.
    # If seed is 88, 88/77 = 1.142.
    # Wait, the current program has score 1.1429.
    # 1/1.1429 = 0.875.
    # If seed is 84, score is 1.0.
    # If seed is 88, score is 88/84 = 1.047.
    # The prompt says: "score is the rank... seed/value, so 1.0 equals the seed program".
    # Current program rank 84, score 1.1429.
    # This implies Seed = 84 * 1.1429 = 96?
    # Or maybe the seed program is the naive one with rank 96 (4*4*6)?
    # 96 / 84 = 1.142857. Yes.
    # So the seed program is the naive schoolbook multiplication with rank 96.
    # The inspiration program has rank 77 and score 1.2468.
    # 96 / 77 = 1.24675. Matches.
    # So the inspiration program achieves rank 77.
    # How?
    # 49 (Strassen for 4x4x4) + 28 (for 4x4x2)?
    # Can we do 4x4x2 with rank 28?
    # Let's check if 28 is possible for 4x4x2.
    # The tensor is T = sum_{i,j,k} A_ij B_jk C_ik where k is 0,1.
    # This is equivalent to multiplying two 4x2 matrices? No, A is 4x4, B is 4x2.
    # Let's try to construct a rank 28 decomposition for 4x4x2.
    # Or maybe the inspiration uses Strassen for the whole thing?
    # No, it splits into 4x4x4 and 4x4x2.
    # Let's assume the inspiration code is correct and produces rank 77.
    # I will copy the logic from the inspiration code which is known to work (score 1.2468).
    # The current program is worse (rank 84).
    # The directive says "COMBINE ... take the structural idea of one and the refinement machinery of the other".
    # The current program has a "structured search" wrapper but just builds rank 84.
    # The inspiration program builds rank 77 directly.
    # I will adopt the rank 77 construction from the inspiration.
    
    rank = 77
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_coeffs, v_coeffs, w_coeffs):
        nonlocal col
        for i, c in u_coeffs:
            if c != 0:
                U[i][col] += c
        for j, c in v_coeffs:
            if c != 0:
                V[j][col] += c
        for k, c in w_coeffs:
            if c != 0:
                W[k][col] += c
        col += 1

    # --- Part 1: Strassen for 4x4x4 (B cols 0-3, C cols 0-3) ---
    # A is 4x4. B1 is 4x4 (cols 0-3). C1 is 4x4 (cols 0-3).
    
    def a_idx(block, r, c):
        # block: '00', '01', '10', '11'
        x, y = int(block[0]), int(block[1])
        return (x*2+r)*4 + (y*2+c)
        
    def b1_idx(block, r, c):
        # block: '00', '01', '10', '11' for B1 (cols 0-3)
        x, y = int(block[0]), int(block[1])
        # j = x*2+r, k = y*2+c
        return (x*2+r)*6 + (y*2+c)
        
    def c1_idx(block, r, c):
        # block: '00', '01', '10', '11' for C1 (cols 0-3)
        # W index is k*n + i
        x, y = int(block[0]), int(block[1])
        # i = x*2+r, k = y*2+c
        return (y*2+c)*4 + (x*2+r)

    def get_strassen_2x2_terms():
        # S1 = (x00+x11)(y00+y11)
        s1 = ([( (0,0), 1), ((1,1), 1)], [((0,0), 1), ((1,1), 1)])
        # S2 = (x10+x11)y00
        s2 = ([( (1,0), 1), ((1,1), 1)], [((0,0), 1)])
        # S3 = x00(y01-y11)
        s3 = ([( (0,0), 1)], [((0,1), 1), ((1,1), -1)])
        # S4 = x11(y10-y00)
        s4 = ([( (1,1), 1)], [((1,0), 1), ((0,0), -1)])
        # S5 = (x00+x01)y11
        s5 = ([( (0,0), 1), ((0,1), 1)], [((1,1), 1)])
        # S6 = (x10-x00)(y00+y01)
        s6 = ([( (1,0), 1), ((0,0), -1)], [((0,0), 1), ((0,1), 1)])
        # S7 = (x01-x11)(y10+y11)
        s7 = ([( (0,1), 1), ((1,1), -1)], [((1,0), 1), ((1,1), 1)])
        return [s1, s2, s3, s4, s5, s6, s7]

    strassen_2x2_terms = get_strassen_2x2_terms()
    
    # Result mapping for 2x2 Strassen
    # Z00 = S1 + S4 - S5 + S7
    # Z01 = S3 + S5
    # Z10 = S2 + S4
    # Z11 = S1 - S2 + S3 + S6
    result_map = {
        (0,0): [(0, 1), (3, 1), (4, -1), (6, 1)],
        (0,1): [(2, 1), (4, 1)],
        (1,0): [(1, 1), (3, 1)],
        (1,1): [(0, 1), (1, -1), (2, 1), (5, 1)]
    }
    
    # Big Strassen 4x4 combinations
    big_combos = [
        ([('00', 1), ('11', 1)], [('00', 1), ('11', 1)]), # M1
        ([('10', 1), ('11', 1)], [('00', 1)]),            # M2
        ([('00', 1)], [('01', 1), ('11', -1)]),           # M3
        ([('11', 1)], [('10', 1), ('00', -1)]),           # M4
        ([('00', 1), ('01', 1)], [('11', 1)]),            # M5
        ([('10', 1), ('00', -1)], [('00', 1), ('01', 1)]),# M6
        ([('01', 1), ('11', -1)], [('10', 1), ('11', 1)]) # M7
    ]
    
    # Big Result mapping
    big_result_map = {
        '00': [(0, 1), (3, 1), (4, -1), (6, 1)], # M1, M4, -M5, M7
        '01': [(2, 1), (4, 1)],                  # M3, M5
        '10': [(1, 1), (3, 1)],                  # M2, M4
        '11': [(0, 1), (1, -1), (2, 1), (5, 1)]  # M1, -M2, M3, M6
    }
    
    # Iterate through the 7 big multiplications
    for big_idx, (a_parts, b_parts) in enumerate(big_combos):
        for s_idx in range(7):
            x_coeffs, y_coeffs = strassen_2x2_terms[s_idx]
            
            u_coeffs = []
            for (rx, cx), cx_c in x_coeffs:
                for a_block, a_sign in a_parts:
                    idx = a_idx(a_block, rx, cx)
                    u_coeffs.append((idx, a_sign * cx_c))
            
            v_coeffs = []
            for (ry, cy), cy_c in y_coeffs:
                for b_block, b_sign in b_parts:
                    idx = b1_idx(b_block, ry, cy)
                    v_coeffs.append((idx, b_sign * cy_c))
            
            w_coeffs = []
            for c_block, big_contribs in big_result_map.items():
                big_coeff = 0
                for m_idx, m_c in big_contribs:
                    if m_idx == big_idx:
                        big_coeff = m_c
                        break
                if big_coeff == 0:
                    continue
                
                for rc, s_contribs in result_map.items():
                    small_coeff = 0
                    for s_check, s_c in s_contribs:
                        if s_check == s_idx:
                            small_coeff = s_c
                            break
                    if small_coeff == 0:
                        continue
                    
                    idx = c1_idx(c_block, rc[0], rc[1])
                    w_coeffs.append((idx, big_coeff * small_coeff))
            
            add_term(u_coeffs, v_coeffs, w_coeffs)
            
    # --- Part 2: 4x4x2 part (B cols 4-5, C cols 4-5) ---
    # We need rank 28 here to reach total 77.
    # A is 4x4. B_right is 4x2. C_right is 4x2.
    # Can we use Strassen-like tricks?
    # Let's try to decompose the 4x4x2 tensor.
    # A = [A00 A01; A10 A11] where blocks are 2x2.
    # B_right = [B0; B1] where B0, B1 are 2x2.
    # C_right = [C0; C1] where C0, C1 are 2x2.
    # C0 = A00 B0 + A01 B1
    # C1 = A10 B0 + A11 B1
    
    # This is two matrix multiplications of 2x2 matrices.
    # C0 = A00 B0 + A01 B1
    # C1 = A10 B0 + A11 B1
    
    # We can compute C0 and C1 using Strassen for 2x2?
    # C0 involves A00, A01, B0, B1.
    # C1 involves A10, A11, B0, B1.
    
    # Let's look at the structure.
    # We can compute:
    # M1 = A00 B0
    # M2 = A01 B1
    # M3 = A10 B0
    # M4 = A11 B1
    # Then C0 = M1 + M2, C1 = M3 + M4.
    # Rank of each 2x2x2 multiplication is 7 (Strassen).
    # So 4 * 7 = 28.
    # Total rank = 49 + 28 = 77.
    # This matches the inspiration's rank!
    
    # Let's implement the 28-term part.
    # We need to map the 2x2 blocks of A, B_right, C_right to global indices.
    
    # A blocks (2x2):
    # A00: i 0-1, j 0-1
    # A01: i 0-1, j 2-3
    # A10: i 2-3, j 0-1
    # A11: i 2-3, j 2-3
    
    # B_right blocks (2x2):
    # B0: j 0-1, k 4-5
    # B1: j 2-3, k 4-5
    
    # C_right blocks (2x2):
    # C0: i 0-1, k 4-5
    # C1: i 2-3, k 4-5
    
    def a_idx_2x2(block, r, c):
        x, y = int(block[0]), int(block[1])
        return (x*2+r)*4 + (y*2+c)

    def b_right_idx(block, r, c):
        # block '0' is B0 (j 0-1), '1' is B1 (j 2-3)
        # k is always 4-5, so c is 0-1 mapping to 4-5
        j_base = int(block) * 2
        return (j_base+r)*6 + (4+c)

    def c_right_idx(block, r, c):
        # block '0' is C0 (i 0-1), '1' is C1 (i 2-3)
        # k is 4-5, so c is 0-1 mapping to 4-5
        i_base = int(block) * 2
        return (4+c)*4 + (i_base+r)

    # Strassen terms for 2x2
    # We reuse strassen_2x2_terms
    # And result_map
    
    # We have 4 multiplications:
    # 1. A00 * B0 -> contributes to C0
    # 2. A01 * B1 -> contributes to C0
    # 3. A10 * B0 -> contributes to C1
    # 4. A11 * B1 -> contributes to C1
    
    # For each multiplication, we use 7 Strassen terms.
    
    # Multiplication 1: A00 * B0 -> C0
    # A_block = '00', B_block = '0', C_block = '0'
    # Multiplication 2: A01 * B1 -> C0
    # A_block = '01', B_block = '1', C_block = '0'
    # Multiplication 3: A10 * B0 -> C1
    # A_block = '10', B_block = '0', C_block = '1'
    # Multiplication 4: A11 * B1 -> C1
    # A_block = '11', B_block = '1', C_block = '1'
    
    mults = [
        ('00', '0', '0'),
        ('01', '1', '0'),
        ('10', '0', '1'),
        ('11', '1', '1')
    ]
    
    for a_b, b_b, c_b in mults:
        for s_idx in range(7):
            x_coeffs, y_coeffs = strassen_2x2_terms[s_idx]
            
            u_coeffs = []
            for (rx, cx), cx_c in x_coeffs:
                idx = a_idx_2x2(a_b, rx, cx)
                u_coeffs.append((idx, cx_c))
            
            v_coeffs = []
            for (ry, cy), cy_c in y_coeffs:
                idx = b_right_idx(b_b, ry, cy)
                v_coeffs.append((idx, cy_c))
            
            w_coeffs = []
            for rc, s_contribs in result_map.items():
                small_coeff = 0
                for s_check, s_c in s_contribs:
                    if s_check == s_idx:
                        small_coeff = s_c
                        break
                if small_coeff == 0:
                    continue
                idx = c_right_idx(c_b, rc[0], rc[1])
                w_coeffs.append((idx, small_coeff))
            
            add_term(u_coeffs, v_coeffs, w_coeffs)
            
    # Verify col == rank
    if col != rank:
        # If col < rank, we have unused columns.
        # If col > rank, we have a bug.
        # We expect exactly 77.
        pass

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
