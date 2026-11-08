import time
import numpy as np

TIME_LIMIT_SECONDS = 110
START_TIME = time.time()

def _elapsed() -> float:
    return time.time() - START_TIME

def construct():
    n, m, p = 4, 4, 5
    
    # We'll build the decomposition directly
    # U: (n*m, rank) = (16, rank)
    # V: (m*p, rank) = (20, rank)  
    # W: (p*n, rank) = (20, rank)
    
    terms = []  # Each term: (U_vector, V_vector, W_vector) as sparse dicts
    
    def add_term(u_dict, v_dict, w_dict):
        terms.append((u_dict, v_dict, w_dict))
    
    # Strassen 2x2: multiply [[a,b],[c,d]] * [[e,f],[g,h]]
    # 7 products:
    # p1 = (a+b)(e+f)
    # p2 = (c+d)e
    # p3 = a(f-h)
    # p4 = d(g-e)
    # p5 = (a+d)(e+h)
    # p6 = (b-c)(e+g)
    # p7 = (b+d)(f-h)
    # Result:
    # [[p1+p4-p5+p7, p3+p5],
    #  [p2+p4,       p1-p2+p3+p5]]
    
    # For 4x4 x 4x4, split into 2x2 blocks:
    # A = [[A00, A01], [A10, A11]] where each Aij is 2x2
    # B = [[B00, B01], [B10, B11]] where each Bij is 2x2
    # C = [[C00, C01], [C10, C11]] where each Cij is 2x2
    
    # Strassen for 4x4 (7 super-terms, each a 2x2 mult = 7 sub-terms):
    # S1 = (A00+A11)(B00+B11)
    # S2 = (A10+A11)B00
    # S3 = A00(B01-B11)
    # S4 = A11(B10-B00)
    # S5 = (A00+A10)(B00+B10)
    # S6 = (A01-A10)(B00+B10)
    # S7 = (A01+A11)(B01-B10)
    # C00 = S1+S4-S5+S7
    # C01 = S3+S5
    # C10 = S2+S4
    # C11 = S1-S2+S3+S5
    
    # Each S_i is a 2x2 matrix multiplication, which itself uses 7 scalar products
    
    # Let me enumerate all 49 scalar products for 4x4x4 Strassen
    
    # First, define the 2x2 Strassen terms
    # For 2x2: inputs are 4 elements each for left and right, output is 4 elements
    # Left: [[l00, l01], [l10, l11]] indexed as (row, col) -> 0,1,2,3
    # Right: [[r00, r01], [r10, r11]] indexed as (row, col) -> 0,1,2,3
    
    strassen_2x2 = [
        # (left_elements, right_elements, output_elements)
        # Each element: (index, coefficient)
        # p1 = (l00+l11)(r00+r11)
        ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(1,1),(2,1),(3,-1)]),
        # p2 = (l10+l11)r00
        ([(2,1),(3,1)], [(0,1)], [(2,1),(3,-1)]),
        # p3 = l00(r01-r11)
        ([(0,1)], [(1,1),(3,-1)], [(1,1),(3,1)]),
        # p4 = l11(r10-r00)
        ([(3,1)], [(2,1),(0,-1)], [(2,1),(3,-1)]),
        # p5 = (l00+l10)(r00+r10)
        ([(0,1),(2,1)], [(0,1),(2,1)], [(0,1),(1,1),(2,1),(3,-1)]),
        # p6 = (l01-l10)(r00+r10)
        ([(1,1),(2,-1)], [(0,1),(2,1)], [(0,-1),(1,1),(2,1),(3,1)]),
        # p7 = (l01+l11)(r01-r10)
        ([(1,1),(3,1)], [(1,1),(2,-1)], [(1,1),(3,1)]),
    ]
    
    # Wait, I need to be more careful with the output contributions.
    # Let me redo this. For 2x2 Strassen:
    # p1 = (a+b)(e+f)
    # p2 = (c+d)e  
    # p3 = a(f-h)
    # p4 = d(g-e)
    # p5 = (a+d)(e+h)
    # p6 = (b-c)(e+g)
    # p7 = (b+d)(f-h)
    # C00 = p1+p4-p5+p7
    # C01 = p3+p5
    # C10 = p2+p4
    # C11 = p1-p2+p3+p5
    
    # Let me define the 2x2 Strassen more carefully
    # Left matrix elements: a=(0,0), b=(0,1), c=(1,0), d=(1,1)
    # Right matrix elements: e=(0,0), f=(0,1), g=(1,0), h=(1,1)
    
    s2x2 = [
        # (left: [(idx, coeff)], right: [(idx, coeff)], output: [(idx, coeff)])
        # p1 = (a+b)(e+f)
        ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(1,1),(2,1),(3,-1)]),
        # p2 = (c+d)e
        ([(2,1),(3,1)], [(0,1)], [(2,1),(3,-1)]),
        # p3 = a(f-h)
        ([(0,1)], [(1,1),(3,-1)], [(1,1),(3,1)]),
        # p4 = d(g-e)
        ([(3,1)], [(2,1),(0,-1)], [(2,1),(3,-1)]),
        # p5 = (a+d)(e+h)
        ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(1,1),(2,1),(3,-1)]),
        # p6 = (b-c)(e+g)
        ([(1,1),(2,-1)], [(0,1),(2,1)], [(0,-1),(1,1),(2,1),(3,1)]),
        # p7 = (b+d)(f-h)
        ([(1,1),(3,1)], [(1,1),(3,-1)], [(1,1),(3,1)]),
    ]
    
    # Hmm, I'm getting confused. Let me just directly construct the 4x4x4 Strassen decomposition.
    
    # Actually, let me take a step back and think about this differently.
    
    # The current program already achieves rank 65. Let me verify it's correct and see if I can improve.
    
    # Let me try a different approach: directly construct rank 64 or 63.
    
    # One idea: Can we merge some of the schoolbook terms for the 5th column with Strassen terms?
    
    # Actually, let me just write a clean rank-65 construction and verify it works.
    
    # For now, let me replicate the current approach but more cleanly, and add a time guard.
    
    rank = 65
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_list, v_list, w_list):
        nonlocal col
        for i, j, c in u_list:
            if c != 0:
                U[i * m + j][col] += c
        for j, k, c in v_list:
            if c != 0:
                V[j * p + k][col] += c
        for k, i, c in w_list:
            if c != 0:
                W[k * n + i][col] += c
        col += 1
    
    # Strassen 4x4x4 decomposition (49 terms)
    # Using the recursive structure
    
    # 2x2 Strassen terms
    s2 = [
        ([(0,1),(3,1)], [(0,1),(3,1)]),  # p1 = (a+b)(e+f)
        ([(2,1),(3,1)], [(0,1)]),         # p2 = (c+d)e
        ([(0,1)], [(1,1),(3,-1)]),        # p3 = a(f-h)
        ([(3,1)], [(2,1),(0,-1)]),        # p4 = d(g-e)
        ([(0,1),(3,1)], [(0,1),(3,1)]),   # p5 = (a+d)(e+h)
        ([(1,1),(2,-1)], [(0,1),(2,1)]),  # p6 = (b-c)(e+g)
        ([(1,1),(3,1)], [(1,1),(3,-1)]),  # p7 = (b+d)(f-h)
    ]
    
    # Output contributions for 2x2 Strassen
    s2_out = [
        [(0,1),(1,1),(2,1),(3,-1)],  # p1
        [(2,1),(3,-1)],               # p2
        [(1,1),(3,1)],                # p3
        [(2,1),(3,-1)],               # p4
        [(0,1),(1,1),(2,1),(3,-1)],   # p5
        [(0,-1),(1,1),(2,1),(3,1)],   # p6
        [(1,1),(3,1)],                # p7
    ]
    
    # 4x4 Strassen super-terms
    # S1 = (A00+A11)(B00+B11)
    # S2 = (A10+A11)B00
    # S3 = A00(B01-B11)
    # S4 = A11(B10-B00)
    # S5 = (A00+A10)(B00+B10)
    # S6 = (A01-A10)(B00+B10)
    # S7 = (A01+A11)(B01-B10)
    
    # For each super-term, we need to map the 2x2 indices to 4x4 indices
    # 4x4 matrix indices: (i,j) where i,j in {0,1,2,3}
    # Block A00: i in {0,1}, j in {0,1} -> 2x2 index (i%2, j%2)
    # Block A01: i in {0,1}, j in {2,3} -> 2x2 index (i%2, (j-2)%2)
    # Block A10: i in {2,3}, j in {0,1} -> 2x2 index ((i-2)%2, j%2)
    # Block A11: i in {2,3}, j in {2,3} -> 2x2 index ((i-2)%2, (j-2)%2)
    
    def block_to_global(block, r, c):
        """Convert 2x2 block index (r,c) to global 4x4 index"""
        if block == 0:  # top-left
            return (r, c)
        elif block == 1:  # top-right
            return (r, 2+c)
        elif block == 2:  # bottom-left
            return (2+r, c)
        elif block == 3:  # bottom-right
            return (2+r, 2+c)
    
    # Super-term definitions: (left_blocks, right_blocks)
    # Each block is a 2x2 matrix, and we apply Strassen 2x2 to them
    super_terms = [
        # (left_block_list, right_block_list)
        # S1 = (A00+A11)(B00+B11)
        ([0, 3], [0, 3]),
        # S2 = (A10+A11)B00
        ([2, 3], [0]),
        # S3 = A00(B01-B11)
        ([0], [1, 3]),
        # S4 = A11(B10-B00)
        ([3], [2, 0]),
        # S5 = (A00+A10)(B00+B10)
        ([0, 2], [0, 2]),
        # S6 = (A01-A10)(B00+B10)
        ([1, 2], [0, 2]),
        # S7 = (A01+A11)(B01-B10)
        ([1, 3], [1, 2]),
    ]
    
    # Signs for each block in each super-term
    super_signs = [
        [1, 1],   # S1: A00 + A11, B00 + B11
        [1, 1],   # S2: A10 + A11, B00
        [1, 1, -1],  # S3: A00, B01 - B11
        [1, 1, -1],  # S4: A11, B10 - B00
        [1, 1],   # S5: A00 + A10, B00 + B10
        [1, -1],  # S6: A01 - A10, B00 + B10
        [1, 1],   # S7: A01 + A11, B01 - B10
    ]
    
    # Hmm, this is getting complicated. Let me just use the existing working code.
    
    # The current program works and achieves rank 65. Let me just clean it up and add a time guard.
    
    # Actually, let me try to see if rank 64 is possible by merging one schoolbook term.
    
    # For now, let me return the rank-65 solution with a time guard.
    
    # Re-initialize
    rank = 65
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    
    def add_term(u_list, v_list, w_list):
        nonlocal col
        for i, j, c in u_list:
            if c != 0:
                U[i * m + j][col] += c
        for j, k, c in v_list:
            if c != 0:
                V[j * p + k][col] += c
        for k, i, c in w_list:
            if c != 0:
                W[k * n + i][col] += c
        col += 1
    
    # Use the same structure as the current program
    super_terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0]),
        ([1, 0, 0, 0], [0, 1, 0, -1]),
        ([0, 0, 0, 1], [-1, 0, 1, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1]),
        ([-1, 0, 1, 0], [1, 1, 0, 0]),
        ([0, 1, 0, -1], [0, 0, 1, 1])
    ]
    
    c_contrib = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]
    
    inner_terms = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        ([(2, 1), (3, 1)], [(0, 1)]),
        ([(0, 1)], [(1, 1), (3, -1)]),
        ([(3, 1)], [(2, 1), (0, -1)]),
        ([(0, 1), (1, 1)], [(3, 1)]),
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)]),
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)])
    ]
    
    s_contrib = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]
    
    def get_p_global(p_elem_idx, p_blocks):
        r = p_elem_idx // 2
        c = p_elem_idx % 2
        res = []
        for b_idx, coeff in enumerate(p_blocks):
            if coeff == 0: continue
            if b_idx == 0: gi, gj = r, c
            elif b_idx == 1: gi, gj = r, 2+c
            elif b_idx == 2: gi, gj = 2+r, c
            elif b_idx == 3: gi, gj = 2+r, 2+c
            res.append((gi, gj, coeff))
        return res
    
    def get_q_global(q_elem_idx, q_blocks):
        r = q_elem_idx // 2
        c = q_elem_idx % 2
        res = []
        for b_idx, coeff in enumerate(q_blocks):
            if coeff == 0: continue
            if b_idx == 0: gj, gk = r, c
            elif b_idx == 1: gj, gk = r, 2+c
            elif b_idx == 2: gj, gk = 2+r, c
            elif b_idx == 3: gj, gk = 2+r, 2+c
            res.append((gj, gk, coeff))
        return res
    
    for s_idx in range(7):
        p_blocks, q_blocks = super_terms[s_idx]
        for t_idx in range(7):
            p_elems, q_elems = inner_terms[t_idx]
            
            u_list = []
            v_list = []
            w_list = []
            
            for p_idx, p_coeff in p_elems:
                elems = get_p_global(p_idx, p_blocks)
                for i, j, c in elems:
                    u_list.append((i, j, c * p_coeff))
                    
            for q_idx, q_coeff in q_elems:
                elems = get_q_global(q_idx, q_blocks)
                for j, k, c in elems:
                    v_list.append((j, k, c * q_coeff))
            
            for s_pos in range(4):
                sc = s_contrib[t_idx][s_pos]
                if sc == 0: continue
                s_r = s_pos // 2
                s_c = s_pos % 2
                
                for c_pos in range(4):
                    cc = c_contrib[s_idx][c_pos]
                    if cc == 0: continue
                    
                    total_coeff = sc * cc
                    if total_coeff == 0: continue
                    
                    if c_pos == 0: gi, gk = s_r, s_c
                    elif c_pos == 1: gi, gk = s_r, 2+s_c
                    elif c_pos == 2: gi, gk = 2+s_r, s_c
                    elif c_pos == 3: gi, gk = 2+s_r, 2+s_c
                    
                    w_list.append((gk, gi, total_coeff))
            
            add_term(u_list, v_list, w_list)
    
    # Schoolbook 4x4x1 for the 5th column
    for i in range(4):
        for j in range(4):
            u_list = [(i, j, 1)]
            v_list = [(j, 4, 1)]
            w_list = [(4, i, 1)]
            add_term(u_list, v_list, w_list)
    
    if _elapsed() > TIME_LIMIT_SECONDS:
        # Return whatever we have so far
        pass
    
    return {"U": U, "V": V, "W": W}
