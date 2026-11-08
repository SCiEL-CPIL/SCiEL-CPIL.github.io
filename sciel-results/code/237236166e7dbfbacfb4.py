"""
Refactored solution for 4x4x5 matrix multiplication tensor decomposition.
Uses Rank 65: 49 (Strassen for 4x4x4) + 16 (Schoolbook for 4x4x1).
Structure changed to use pre-computed term definitions for distinctness and speed.
"""

import numpy as np

DIMENSIONS = [4, 4, 5]

# Pre-compute the rank-1 terms.
# Each term is a tuple (u_sparsity, v_sparsity, w_sparsity)
# where sparsity is a list of (index, value).
# U index: i*4 + j
# V index: j*5 + k
# W index: k*4 + i

def _build_terms():
    terms = []
    
    # Part 1: Strassen 4x4x4 (cols 0-3 of B, cols 0-3 of C)
    # Strassen recursion: 4x4 -> 7 mults of 2x2. Each 2x2 -> 7 mults of 1x1.
    # Total 49 terms.
    
    # Helper to add a scalar term
    def add_scalar_term(i, j, k, coeff):
        # This adds to the current term being built
        pass # Placeholder
    
    # We will build the 49 terms explicitly based on Strassen formulas.
    # A is 4x4. Split into A00, A01, A10, A11 (2x2 blocks).
    # B00, B01, B10, B11 (2x2 blocks of B[:, 0:4]).
    
    # Strassen M's:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Each M is a 2x2 product P Q.
    # P, Q are 2x2.
    # We decompose P Q into 7 scalar products.
    
    # Strassen 2x2 formulas for S = P Q:
    # s1 = (p00+p11)(q00+q11)
    # s2 = (p10+p11)q00
    # s3 = p00(q01-q11)
    # s4 = p11(q10-q00)
    # s5 = (p00+p01)q11
    # s6 = (p10-p00)(q00+q01)
    # s7 = (p01-p11)(q10+q11)
    
    # S00 = s1+s4-s5+s7
    # S01 = s3+s5
    # S10 = s2+s4
    # S11 = s1-s2+s3+s6
    
    # M1 contributes to C00, C11.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We iterate over the 7 Super terms and 7 Inner terms.
    
    # Mapping of block indices to global (i, j) or (j, k)
    # Block (br, bc) in 2x2 grid of blocks.
    # A block br, bc corresponds to rows 2*br to 2*br+1, cols 2*bc to 2*bc+1.
    # Local (r, c) in 2x2 block -> Global (2*br+r, 2*bc+c).
    
    def get_A_indices(br, bc, lr, lc):
        return (2*br+lr, 2*bc+lc)
    
    def get_B_indices(br, bc, lr, lc):
        # B is 4x4 for this part (cols 0-3)
        return (2*br+lr, 2*bc+lc)
        
    def get_C_indices(cr, cc, lr, lc):
        # C is 4x4 for this part (cols 0-3)
        # Note: W index is k*n + i.
        # k is col (0-3), i is row (0-3).
        return (2*cc+lc, 2*cr+lr) # k, i

    # Super term definitions:
    # (p_blocks, q_blocks)
    # p_blocks: coeffs for A00, A01, A10, A11
    # q_blocks: coeffs for B00, B01, B10, B11
    
    super_terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1]), # M1
        ([0, 0, 1, 1], [1, 0, 0, 0]), # M2
        ([1, 0, 0, 0], [0, 1, 0, -1]), # M3
        ([0, 0, 0, 1], [-1, 0, 1, 0]), # M4
        ([1, 1, 0, 0], [0, 0, 0, 1]), # M5
        ([-1, 0, 1, 0], [1, 1, 0, 0]), # M6
        ([0, 1, 0, -1], [0, 0, 1, 1]), # M7
    ]
    
    # C block contributions for each Super term:
    # [C00, C01, C10, C11]
    c_contribs = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, -1], # M2
        [0, 1, 0, 1], # M3
        [1, 0, 1, 0], # M4
        [-1, 1, 0, 0], # M5
        [0, 0, 0, 1], # M6
        [1, 0, 0, 0], # M7
    ]
    
    # Inner Strassen term definitions:
    # p_elems: list of (block_idx_in_p, local_r, local_c, coeff)
    # q_elems: list of (block_idx_in_q, local_r, local_c, coeff)
    # s_contrib: [S00, S01, S10, S11]
    
    # p/q block indices correspond to the 4 blocks of the 2x2 P/Q matrix.
    # P00 is block 0, P01 is block 1, P10 is block 2, P11 is block 3.
    # Wait, P is formed by linear combo of A blocks.
    # P is a 2x2 matrix of "super-scalars".
    # Each "super-scalar" is a 2x2 matrix (a block).
    # So P00 is a 2x2 matrix.
    # In the Inner Strassen, we multiply elements of P and Q.
    # P00, P01, P10, P11 are 2x2 matrices.
    
    # Inner term 0: (P00+P11)(Q00+Q11)
    # Inner term 1: (P10+P11)Q00
    # Inner term 2: P00(Q01-Q11)
    # Inner term 3: P11(Q10-Q00)
    # Inner term 4: (P00+P01)Q11
    # Inner term 5: (P10-P00)(Q00+Q01)
    # Inner term 6: (P01-P11)(Q10+Q11)
    
    inner_defs = [
        ( # 0
            [(0, 0, 1), (3, 1, 1)], # P00, P11
            [(0, 0, 1), (3, 1, 1)], # Q00, Q11
            [1, 0, 0, 1] # S contrib
        ),
        ( # 1
            [(2, 1, 1), (3, 1, 1)], # P10, P11
            [(0, 0, 1)], # Q00
            [0, 0, 1, -1]
        ),
        ( # 2
            [(0, 0, 1)], # P00
            [(1, 1, 1), (3, 3, -1)], # Q01, Q11? 
            # Q01 is block 1 of Q. Q11 is block 3 of Q.
            # Q01 corresponds to Q[0,1]. Q[0,1] is top-right block.
            # My block indices: 0:TL, 1:TR, 2:BL, 3:BR.
            # So Q01 is index 1. Q11 is index 3.
            [0, 1, 0, 1]
        ),
        ( # 3
            [(3, 3, 1)], # P11 (BR block)
            [(2, 2, 1), (0, 0, -1)], # Q10 (BL), Q00 (TL)
            [1, 0, 1, 0]
        ),
        ( # 4
            [(0, 0, 1), (1, 1, 1)], # P00, P01
            [(3, 3, 1)], # Q11
            [-1, 1, 0, 0]
        ),
        ( # 5
            [(2, 2, 1), (0, 0, -1)], # P10, P00
            [(0, 0, 1), (1, 1, 1)], # Q00, Q01
            [0, 0, 0, 1]
        ),
        ( # 6
            [(1, 1, 1), (3, 3, -1)], # P01, P11
            [(2, 2, 1), (3, 3, 1)], # Q10, Q11
            [1, 0, 0, 0]
        ),
    ]
    
    # Expansion logic
    # For a given Super Term S and Inner Term I:
    # 1. Determine P and Q linear combos (u_list, v_list base)
    # 2. Expand Inner Term to specific (r, c) in P and Q (2x2 blocks)
    # 3. Multiply to get S (2x2)
    # 4. Map S to C blocks
    
    for s_idx, (p_blocks, q_blocks) in enumerate(super_terms):
        for i_idx, (p_elems, q_elems, s_contrib) in enumerate(inner_defs):
            u_list = []
            v_list = []
            w_list = []
            
            # Expand P and Q elements
            # p_elems: (block_idx, local_row, local_col, coeff) -- 
            # Wait, my previous logic in the seed was different.
            # Let's use the explicit 2x2 block expansion.
            
            # For each element in P involved in this inner term:
            # p_block_idx: 0,1,2,3 (TL, TR, BL, BR of the 2x2 super-matrix P)
            # local_r, local_c: position within the 2x2 block.
            
            for p_b, lr, lc, p_c in p_elems:
                # Get global A indices for this part of P
                # p_b 0: A00, 1: A01, 2: A10, 3: A11
                # Global i, j for A:
                # A00: i 0-1, j 0-1
                # A01: i 0-1, j 2-3
                # A10: i 2-3, j 0-1
                # A11: i 2-3, j 2-3
                
                if p_b == 0: gi, gj = lr, lc
                elif p_b == 1: gi, gj = lr, 2+lc
                elif p_b == 2: gi, gj = 2+lr, lc
                elif p_b == 3: gi, gj = 2+lr, 2+lc
                
                # Apply super term coefficient for this block
                # p_blocks[p_b] is the coeff
                # Note: p_elems might specify multiple blocks summing to P_element.
                # The 'p_c' in p_elems is the coeff of the P_element itself in the Inner term.
                # The p_blocks[p_b] is the coeff of the A_block in the P_element.
                
                total_c = p_c * p_blocks[p_b]
                if total_c != 0:
                    u_list.append((gi*4 + gj, total_c))
            
            for q_b, lr, lc, q_c in q_elems:
                if q_b == 0: gj, gk = lr, lc
                elif q_b == 1: gj, gk = lr, 2+lc
                elif q_b == 2: gj, gk = 2+lr, lc
                elif q_b == 3: gj, gk = 2+lr, 2+lc
                
                total_c = q_c * q_blocks[q_b]
                if total_c != 0:
                    v_list.append((gj*5 + gk, total_c)) # B col 0-3, so k is 0-3?
                    # V shape is (m*p, rank) = 4*5.
                    # j 0-3, k 0-4.
                    # For Strassen part, k is 0-3.
            
            # W part
            # s_contrib: [S00, S01, S10, S11]
            # c_contribs[s_idx]: [C00, C01, C10, C11]
            
            for s_pos in range(4):
                s_c = s_contrib[s_pos]
                if s_c == 0: continue
                s_r = s_pos // 2
                s_c_idx = s_pos % 2
                
                for c_b in range(4):
                    c_b_c = c_contribs[s_idx][c_b]
                    if c_b_c == 0: continue
                    
                    total_w_c = s_c * c_b_c
                    if total_w_c == 0: continue
                    
                    # Map (c_b, s_r, s_c_idx) to global (i, k)
                    # c_b 0: C00 (i 0-1, k 0-1)
                    # c_b 1: C01 (i 0-1, k 2-3)
                    # c_b 2: C10 (i 2-3, k 0-1)
                    # c_b 3: C11 (i 2-3, k 2-3)
                    
                    if c_b == 0: gi, gk = s_r, s_c_idx
                    elif c_b == 1: gi, gk = s_r, 2+s_c_idx
                    elif c_b == 2: gi, gk = 2+s_r, s_c_idx
                    elif c_b == 3: gi, gk = 2+s_r, 2+s_c_idx
                    
                    # W index: k*4 + i
                    w_list.append((gk*4 + gi, total_w_c))
                    
            terms.append((u_list, v_list, w_list))
            
    # Part 2: Schoolbook 4x4x1 (col 4 of B, col 4 of C)
    # 16 terms.
    for i in range(4):
        for j in range(4):
            u = [(i*4 + j, 1)]
            v = [(j*5 + 4, 1)] # B[j, 4]
            w = [(4*4 + i, 1)] # C[4, i]? No, k=4, i=i.
            # W index k*n + i. k=4, n=4. 16+i.
            terms.append((u, v, w))
            
    return terms

TERMS = _build_terms()
RANK = len(TERMS)

def construct():
    n, m, p = DIMENSIONS
    U = [[0] * RANK for _ in range(n * m)]
    V = [[0] * RANK for _ in range(m * p)]
    W = [[0] * RANK for _ in range(p * n)]
    
    for r, (u_list, v_list, w_list) in enumerate(TERMS):
        for idx, val in u_list:
            U[idx][r] += val
        for idx, val in v_list:
            V[idx][r] += val
        for idx, val in w_list:
            W[idx][r] += val
            
    return {"U": U, "V": V, "W": W}

