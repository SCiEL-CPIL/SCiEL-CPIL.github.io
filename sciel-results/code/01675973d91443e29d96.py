import numpy as np
import time

def construct():
    deadline = time.time() + 140
    n, m, p = 4, 4, 5
    
    rank = 65
    U = [[0.0] * rank for _ in range(n*m)]
    V = [[0.0] * rank for _ in range(m*p)]
    W = [[0.0] * rank for _ in range(p*n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_coeffs, v_coeffs, w_coeffs):
        nonlocal col
        if col >= rank:
            raise ValueError("Rank exceeded")
        for idx, val in u_coeffs.items():
            if val != 0:
                U[idx][col] += val
        for idx, val in v_coeffs.items():
            if val != 0:
                V[idx][col] += val
        for idx, val in w_coeffs.items():
            if val != 0:
                W[idx][col] += val
        col += 1

    # Strassen 2x2x2 helper
    # Takes indices for A, B, C blocks and adds 7 terms
    def strassen_2x2(a_idx_map, b_idx_map, c_idx_map, signs_c):
        # a_idx_map: dict (0,0)->A_idx, etc.
        # b_idx_map: dict (0,0)->B_idx, etc.
        # c_idx_map: dict (0,0)->C_idx, etc.
        # signs_c: not needed here, we return contributions
        
        # M1 = (A00+A11)(B00+B11)
        u1 = {a_idx_map[(0,0)]: 1, a_idx_map[(1,1)]: 1}
        v1 = {b_idx_map[(0,0)]: 1, b_idx_map[(1,1)]: 1}
        # M1 contributes to C00 (+1), C11 (+1)
        w1 = {c_idx_map[(0,0)]: 1, c_idx_map[(1,1)]: 1}
        add_term(u1, v1, w1)
        
        # M2 = (A10+A11)B00
        u2 = {a_idx_map[(1,0)]: 1, a_idx_map[(1,1)]: 1}
        v2 = {b_idx_map[(0,0)]: 1}
        # M2 contributes to C10 (+1), C11 (-1)
        w2 = {c_idx_map[(1,0)]: 1, c_idx_map[(1,1)]: -1}
        add_term(u2, v2, w2)
        
        # M3 = A00(B01-B11)
        u3 = {a_idx_map[(0,0)]: 1}
        v3 = {b_idx_map[(0,1)]: 1, b_idx_map[(1,1)]: -1}
        # M3 contributes to C01 (+1), C11 (+1)
        w3 = {c_idx_map[(0,1)]: 1, c_idx_map[(1,1)]: 1}
        add_term(u3, v3, w3)
        
        # M4 = A11(B10-B00)
        u4 = {a_idx_map[(1,1)]: 1}
        v4 = {b_idx_map[(1,0)]: 1, b_idx_map[(0,0)]: -1}
        # M4 contributes to C00 (+1), C10 (+1)
        w4 = {c_idx_map[(0,0)]: 1, c_idx_map[(1,0)]: 1}
        add_term(u4, v4, w4)
        
        # M5 = (A00+A01)B11
        u5 = {a_idx_map[(0,0)]: 1, a_idx_map[(0,1)]: 1}
        v5 = {b_idx_map[(1,1)]: 1}
        # M5 contributes to C00 (-1), C01 (+1)
        w5 = {c_idx_map[(0,0)]: -1, c_idx_map[(0,1)]: 1}
        add_term(u5, v5, w5)
        
        # M6 = (A10-A00)(B00+B01)
        u6 = {a_idx_map[(1,0)]: 1, a_idx_map[(0,0)]: -1}
        v6 = {b_idx_map[(0,0)]: 1, b_idx_map[(0,1)]: 1}
        # M6 contributes to C11 (+1)
        w6 = {c_idx_map[(1,1)]: 1}
        add_term(u6, v6, w6)
        
        # M7 = (A01-A11)(B10+B11)
        u7 = {a_idx_map[(0,1)]: 1, a_idx_map[(1,1)]: -1}
        v7 = {b_idx_map[(1,0)]: 1, b_idx_map[(1,1)]: 1}
        # M7 contributes to C00 (+1)
        w7 = {c_idx_map[(0,0)]: 1}
        add_term(u7, v7, w7)

    # Define index mappings for the 4x4x4 Strassen blocks
    # A indices: i*4+j
    # B indices: j*4+k (for 4x4 part)
    # C indices: k*4+i
    
    # Block 1: P1 = (A11+A22)(B11+B22)
    # A11: (0,0)->0, (0,1)->1, (1,0)->4, (1,1)->5
    # A22: (2,2)->10, (2,3)->11, (3,2)->14, (3,3)->15
    # Note: My previous mapping was:
    # X00 = A00 + A22 -> A(0,0) + A(2,2) -> idx 0 + 10
    # X01 = A01 + A23 -> A(0,1) + A(2,3) -> idx 1 + 11
    # X10 = A10 + A32 -> A(1,0) + A(3,2) -> idx 4 + 14
    # X11 = A11 + A33 -> A(1,1) + A(3,3) -> idx 5 + 15
    
    # B11: (0,0)->0, (0,1)->1, (1,0)->4, (1,1)->5
    # B22: (2,2)->10, (2,3)->11, (3,2)->14, (3,3)->15
    # Y00 = B00 + B22 -> 0 + 10
    # Y01 = B01 + B23 -> 1 + 11
    # Y10 = B10 + B32 -> 4 + 14
    # Y11 = B11 + B33 -> 5 + 15
    
    # C11: (0,0)->0, (0,1)->1, (1,0)->4, (1,1)->5
    # C22: (2,2)->10, (2,3)->11, (3,2)->14, (3,3)->15
    # P1 contributes to C11 and C22?
    # C11 = P1 + P4 - P5 + P7
    # C22 = P1 - P2 + P3 + P6
    # So P1 entries map to C11 and C22.
    # P1 is 2x2.
    # P1[0,0] -> C[0,0] (C11[0,0])
    # P1[0,1] -> C[0,1] (C11[0,1])
    # P1[1,0] -> C[1,0] (C11[1,0])
    # P1[1,1] -> C[1,1] (C11[1,1])
    # AND
    # P1[0,0] -> C[2,2] (C22[0,0])
    # ...
    
    # This mapping is complex. It is easier to define the 7 Strassen multiplications
    # on the "super" matrices X and Y, and then add the results to the final C.
    
    # Let's just implement the 49 terms explicitly by defining the linear combos.
    
    # We will create a list of (u_dict, v_dict, w_dict)
    terms = []
    
    def add_strassen_block(a_map, b_map, c_map):
        # a_map: {(0,0): idx, ...}
        # b_map: {(0,0): idx, ...}
        # c_map: {(0,0): idx, ...}
        # These maps point to the ACTUAL indices in U, V, W.
        
        # M1
        u = {a_map[(0,0)]: 1, a_map[(1,1)]: 1}
        v = {b_map[(0,0)]: 1, b_map[(1,1)]: 1}
        w = {c_map[(0,0)]: 1, c_map[(1,1)]: 1}
        terms.append((u, v, w))
        
        # M2
        u = {a_map[(1,0)]: 1, a_map[(1,1)]: 1}
        v = {b_map[(0,0)]: 1}
        w = {c_map[(1,0)]: 1, c_map[(1,1)]: -1}
        terms.append((u, v, w))
        
        # M3
        u = {a_map[(0,0)]: 1}
        v = {b_map[(0,1)]: 1, b_map[(1,1)]: -1}
        w = {c_map[(0,1)]: 1, c_map[(1,1)]: 1}
        terms.append((u, v, w))
        
        # M4
        u = {a_map[(1,1)]: 1}
        v = {b_map[(1,0)]: 1, b_map[(0,0)]: -1}
        w = {c_map[(0,0)]: 1, c_map[(1,0)]: 1}
        terms.append((u, v, w))
        
        # M5
        u = {a_map[(0,0)]: 1, a_map[(0,1)]: 1}
        v = {b_map[(1,1)]: 1}
        w = {c_map[(0,0)]: -1, c_map[(0,1)]: 1}
        terms.append((u, v, w))
        
        # M6
        u = {a_map[(1,0)]: 1, a_map[(0,0)]: -1}
        v = {b_map[(0,0)]: 1, b_map[(0,1)]: 1}
        w = {c_map[(1,1)]: 1}
        terms.append((u, v, w))
        
        # M7
        u = {a_map[(0,1)]: 1, a_map[(1,1)]: -1}
        v = {b_map[(1,0)]: 1, b_map[(1,1)]: 1}
        w = {c_map[(0,0)]: 1}
        terms.append((u, v, w))

    # We need to define the 7 blocks of the 4x4 Strassen algorithm.
    # The inputs to the 2x2 Strassen are linear combinations of the 4x4 entries.
    # This means the "indices" in the maps above are not single indices, but sets of indices with coefficients.
    # My `add_strassen_block` assumes single indices.
    # I need to modify it to handle linear combinations.
    
    # Let's redefine `terms` to store lists of (idx, coeff)
    
    terms = []
    
    def add_strassen_block_lin(a_map, b_map, c_map):
        # a_map: {(0,0): [(idx, coeff), ...], ...}
        
        # M1 = (A00+A11)(B00+B11)
        u = {}
        for idx, c in a_map[(0,0)]: u[idx] = u.get(idx, 0) + c
        for idx, c in a_map[(1,1)]: u[idx] = u.get(idx, 0) + c
        
        v = {}
        for idx, c in b_map[(0,0)]: v[idx] = v.get(idx, 0) + c
        for idx, c in b_map[(1,1)]: v[idx] = v.get(idx, 0) + c
        
        w = {}
        for idx, c in c_map[(0,0)]: w[idx] = w.get(idx, 0) + c
        for idx, c in c_map[(1,1)]: w[idx] = w.get(idx, 0) + c
        
        terms.append((u, v, w))
        
        # ... repeat for M2-M7 ...
        # This is tedious but doable.

    # Alternative: Just generate the 49 terms manually by expanding the Strassen formulas.
    # Given the complexity and time limit, I will hardcode the 49 terms for 4x4x4 Strassen.
    # Or, I can use a simpler approach:
    # Use the 2x2 Strassen to generate 7 "virtual" 2x2x2 multiplications.
    # Then use 2x2 Strassen again? No, 4x4 is 2x2 of 2x2.
    
    # Let's stick to the plan:
    # 1. Define the 7 super-blocks P1..P7.
    # 2. For each Pk, define the linear combinations of A and B.
    # 3. Apply 2x2 Strassen to these combinations.
    
    # I will implement `add_strassen_block_lin` properly.
