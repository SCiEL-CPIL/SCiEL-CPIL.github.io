import time
import sys

DIMENSIONS = [4, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 130s safety margin within 150s limit
    
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    # We will use a decomposition of rank 65.
    # 49 terms for 4x4 * 4x4 (Strassen)
    # 16 terms for 4x4 * 4x1 (Schoolbook)
    
    rank = 65
    U = [[0] * rank for _ in range(n * m)]  # 16 x 65
    V = [[0] * rank for _ in range(m * p)]  # 20 x 65
    W = [[0] * rank for _ in range(p * n)]  # 20 x 65
    
    # Helper to add a rank-1 term
    # u_indices: list of (i, j, coeff)
    # v_indices: list of (j, k, coeff)
    # w_indices: list of (k, i, coeff)
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
    
    # --- Part 1: 4x4 * 4x4 using Strassen ---
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
    
    # Each M is a 2x2 matrix product.
    # We decompose each 2x2 product using Strassen (7 scalar mults).
    
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
    # P is 2x2. P[0,0] = A[0,0]+A[2,2], P[0,1] = A[0,1]+A[2,3], etc.
    # Q is 2x2. Q[0,0] = B[0,0]+B[2,2], etc.
    
    # Function to get U, V, W lists for a super term
    def get_super_term_indices(super_idx, inner_idx):
        # Returns u_list, v_list, w_list for the specific scalar multiplication
        # super_idx: 0..6
        # inner_idx: 0..6
        
        # Define the linear combinations for P and Q for each super term
        # P is sum of blocks of A. Q is sum of blocks of B.
        
        # Block definitions:
        # A00: i<2, j<2. A01: i<2, j>=2. A10: i>=2, j<2. A11: i>=2, j>=2.
        # B00: j<2, k<2. B01: j<2, k>=2. B10: j>=2, k<2. B11: j>=2, k>=2.
        
        # Coefficients for A blocks in P
        # Coefficients for B blocks in Q
        
        # We'll build u_list and v_list by expanding the linear combinations.
        
        u_list = []
        v_list = []
        w_list = []
        
        # Determine P and Q structure based on super_idx
        # P blocks: (coeff_A00, coeff_A01, coeff_A10, coeff_A11)
        # Q blocks: (coeff_B00, coeff_B01, coeff_B10, coeff_B11)
        
        if super_idx == 0: # M1
            p_blocks = [1, 0, 0, 1] # A00 + A11
            q_blocks = [1, 0, 0, 1] # B00 + B11
            # C contribution: C00 += M1, C11 += M1
            # C00 is top-left 2x2 of C. C11 is bottom-right 2x2.
            # So for each inner term, we add to C00 and C11 with the same coefficients
            # as the inner Strassen result.
            # Inner Strassen result for C:
            # C00 = s1+s4-s5+s7
            # C01 = s3+s5
            # C10 = s2+s4
            # C11 = s1-s2+s3+s6
            
            # We need to add the contribution to C00 (global rows 0,1 cols 0,1) 
            # and C11 (global rows 2,3 cols 2,3).
            # The inner Strassen terms s1..s7 contribute to C00, C01, C10, C11.
            # So for each inner term, we need to know which C elements it contributes to
            # and with what coefficient.
            
            # Let's define the contribution of each inner term to the local 2x2 result.
            # local_result[0,0] += coeff_s1 * s1 + coeff_s4 * s4 - coeff_s5 * s5 + coeff_s7 * s7
            # local_result[0,1] += coeff_s3 * s3 + coeff_s5 * s5
            # local_result[1,0] += coeff_s2 * s2 + coeff_s4 * s4
            # local_result[1,1] += coeff_s1 * s1 - coeff_s2 * s2 + coeff_s3 * s3 + coeff_s6 * s6
            
            # For super term 0, the local result is added to both C00 and C11.
            # So the global C elements are:
            # C[0,0] += local[0,0]
            # C[0,1] += local[0,1]
            # C[1,0] += local[1,0]
            # C[1,1] += local[1,1]
            # C[2,2] += local[0,0]
            # C[2,3] += local[0,1]
            # C[3,2] += local[1,0]
            # C[3,3] += local[1,1]
            
            # So for each inner term, we add to both C00 and C11 with the same coefficients.
            
        elif super_idx == 1: # M2
            p_blocks = [0, 0, 1, 1] # A10 + A11
            q_blocks = [1, 0, 0, 0] # B00
            # C contribution: C10 += M2, C11 -= M2
            # C10 is bottom-left 2x2. C11 is bottom-right 2x2.
            
        elif super_idx == 2: # M3
            p_blocks = [1, 0, 0, 0] # A00
            q_blocks = [0, 1, 0, -1] # B01 - B11
            # C contribution: C01 += M3, C11 += M3
            
        elif super_idx == 3: # M4
            p_blocks = [0, 0, 0, 1] # A11
            q_blocks = [-1, 0, 1, 0] # B10 - B00
            # C contribution: C00 += M4, C10 += M4
            
        elif super_idx == 4: # M5
            p_blocks = [1, 1, 0, 0] # A00 + A01
            q_blocks = [0, 0, 0, 1] # B11
            # C contribution: C00 -= M5, C01 += M5
            
        elif super_idx == 5: # M6
            p_blocks = [-1, 0, 1, 0] # A10 - A00
            q_blocks = [1, 1, 0, 0] # B00 + B01
            # C contribution: C11 += M6
            
        elif super_idx == 6: # M7
            p_blocks = [0, 1, 0, -1] # A01 - A11
            q_blocks = [0, 0, 1, 1] # B10 + B11
            # C contribution: C00 += M7
            
        else:
            return u_list, v_list, w_list
            
        # Now, for the inner Strassen term inner_idx, determine which elements of P and Q are multiplied
        # P = [[P00, P01], [P10, P11]]
        # Q = [[Q00, Q01], [Q10, Q11]]
        
        # Inner Strassen terms:
        # 0: (P00+P11)(Q00+Q11)
        # 1: (P10+P11)Q00
        # 2: P00(Q01-Q11)
        # 3: P11(Q10-Q00)
        # 4: (P00+P01)Q11
        # 5: (P10-P00)(Q00+Q01)
        # 6: (P01-P11)(Q10+Q11)
        
        # We need to identify the linear combination of P elements and Q elements.
        
        # Helper to get P element linear combo
        # P_ab is a 2x2 matrix.
        # P_ab[0,0] corresponds to A block element at local (0,0)
        # Local (0,0) of A00 is global (0,0).
        # Local (0,1) of A00 is global (0,1).
        # Local (1,0) of A00 is global (1,0).
        # Local (1,1) of A00 is global (1,1).
        # Local (0,0) of A01 is global (0,2).
        # ...
        
        def get_p_element(p_elem_idx):
            # p_elem_idx: 0:00, 1:01, 2:10, 3:11
            # Returns list of (i, j, coeff)
            res = []
            # Determine which block and local position
            # p_elem_idx maps to local row r, col c in the 2x2 P matrix.
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
            
        def get_q_element(q_elem_idx):
            # q_elem_idx: 0:00, 1:01, 2:10, 3:11
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
            
        # Determine P and Q elements for this inner term
        # inner_idx 0: (P00+P11)(Q00+Q11)
        if inner_idx == 0:
            p_elems = [(0, 1), (3, 1)] # P00, P11
            q_elems = [(0, 1), (3, 1)] # Q00, Q11
            # Contribution to local result:
            # local[0,0] += 1
            # local[1,1] += 1
            c_contrib = [(0, 0, 1), (1, 1, 1)]
        elif inner_idx == 1:
            p_elems = [(2, 1), (3, 1)] # P10, P11
            q_elems = [(0, 1)] # Q00
            # Contribution to local result:
            # local[1,0] += 1
            # local[1,1] -= 1
            c_contrib = [(1, 0, 1), (1, 1, -1)]
        elif inner_idx == 2:
            p_elems = [(0, 1)] # P00
            q_elems = [(1, 1), (3, -1)] # Q01, -Q11
            # Contribution to local result:
            # local[0,1] += 1
            # local[1,1] += 1
            c_contrib = [(0, 1, 1), (1, 1, 1)]
        elif inner_idx == 3:
            p_elems = [(3, 1)] # P11
            q_elems = [(2, 1), (0, -1)] # Q10, -Q00
            # Contribution to local result:
            # local[0,0] += 1
            # local[1,0] += 1
            c_contrib = [(0, 0, 1), (1, 0, 1)]
        elif inner_idx == 4:
            p_elems = [(0, 1), (1, 1)] # P00, P01
            q_elems = [(3, 1)] # Q11
            # Contribution to local result:
            # local[0,0] -= 1
            # local[0,1] += 1
            c_contrib = [(0, 0, -1), (0, 1, 1)]
        elif inner_idx == 5:
            p_elems = [(2, 1), (0, -1)] # P10, -P00
            q_elems = [(0, 1), (1, 1)] # Q00, Q01
            # Contribution to local result:
            # local[1,1] += 1
            c_contrib = [(1, 1, 1)]
        elif inner_idx == 6:
            p_elems = [(1, 1), (3, -1)] # P01, -P11
            q_elems = [(2, 1), (3, 1)] # Q10, Q11
            # Contribution to local result:
            # local[0,0] += 1
            c_contrib = [(0, 0, 1)]
        
        # Now build u_list, v_list, w_list
        u_list = []
        v_list = []
        
        for p_elem, coeff in p_elems:
            for i, j, c in get_p_element(p_elem):
                u_list.append((i, j, coeff * c))
        
        for q_elem, coeff in q_elems:
            for j, k, c in get_q_element(q_elem):
                v_list.append((j, k, coeff * c))
        
        # Now determine the global C elements to which this inner term contributes.
        # The local result is a 2x2 matrix. The super term determines which 2x2 block(s)
        # of the global C it is added to, and with what sign.
        
        # For super_idx 0 (M1): added to C00 and C11 with +1
        # For super_idx 1 (M2): added to C10 with +1, C11 with -1
        # For super_idx 2 (M3): added to C01 with +1, C11 with +1
        # For super_idx 3 (M4): added to C00 with +1, C10 with +1
        # For super_idx 4 (M5): added to C00 with -1, C01 with +1
        # For super_idx 5 (M6): added to C11 with +1
        # For super_idx 6 (M7): added to C00 with +1
        
        # c_contrib gives the local (r, c, coeff) for the local 2x2 result.
        # We need to map local (r, c) to global (i, k) for each block.
        
        w_list = []
        
        # Determine the blocks and signs for this super term
        if super_idx == 0:
            blocks = [(0, 0, 1), (1, 1, 1)] # (block_row, block_col, sign)
        elif super_idx == 1:
            blocks = [(1, 0, 1), (1, 1, -1)]
        elif super_idx == 2:
            blocks = [(0, 1, 1), (1, 1, 1)]
        elif super_idx == 3:
            blocks = [(0, 0, 1), (1, 0, 1)]
        elif super_idx == 4:
            blocks = [(0, 0, -1), (0, 1, 1)]
        elif super_idx == 5:
            blocks = [(1, 1, 1)]
        elif super_idx == 6:
            blocks = [(0, 0, 1)]
        
        for br, bc, sign in blocks:
            for lr, lc, local_coeff in c_contrib:
                # Global row i = br * 2 + lr
                # Global col k = bc * 2 + lc
                gi = br * 2 + lr
                gk = bc * 2 + lc
                w_list.append((gk, gi, sign * local_coeff))
        
        return u_list, v_list, w_list
    
    # Now iterate over all super terms and inner terms
    for super_idx in range(7):
        for inner_idx in range(7):
            if time.time() > deadline:
                break
            u_list, v_list, w_list = get_super_term_indices(super_idx, inner_idx)
            if u_list and v_list and w_list:
                add_term(col, u_list, v_list, w_list)
                col += 1
    
    # Part 2: 4x4x1 (rank 16)
    # A (4x4) * B_2 (4x1) -> C_2 (4x1)
    # B_2 is column 4 of B.
    # C_2 is column 4 of C.
    
    for i in range(4):
        for j in range(4):
            if time.time() > deadline:
                break
            U[i*4 + j][col] = 1
            V[j*5 + 4][col] = 1 # B index: row j, col 4
            W[4*4 + i][col] = 1 # C index: k=4, i. W[k*n+i] = W[4*4+i]
            col += 1

    # Trim to actual rank used
    if col < rank:
        U = [row[:col] for row in U]
        V = [row[:col] for row in V]
        W = [row[:col] for row in W]

    return {"U": U, "V": V, "W": W}
