import numpy as np
import time

DIMENSIONS = [5, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n, m, p = DIMENSIONS
    # We are targeting rank 110
    rank = 110
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Part 1: 4x4 Strassen Decomposition (49 terms)
    # We generate the terms for 4x4 multiplication
    # Local indices 0..3 map to global 0..3
    
    def add_term(u_dict, v_dict, w_dict, col_idx):
        # u_dict: { (i, j): coeff } for A
        # v_dict: { (j, k): coeff } for B
        # w_dict: { (k, i): coeff } for C
        for (i, j), c in u_dict.items():
            if c != 0:
                U[i * m + j][col_idx] += c
        for (j, k), c in v_dict.items():
            if c != 0:
                V[j * p + k][col_idx] += c
        for (k, i), c in w_dict.items():
            if c != 0:
                W[k * n + i][col_idx] += c

    # Define the 7 Strassen terms for 2x2
    # Each term is (u_dict, v_dict, w_dict)
    # Indices are (row, col)
    strassen_2x2 = [
        # P1
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # P2
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        # P3
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        # P4
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        # P5
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        # P6
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        # P7
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1, (1,1):-1})
    ]
    
    # Recursive function to generate terms for NxN using Strassen
    # Returns a list of (u_dict, v_dict, w_dict)
    def get_strassen_terms(N):
        if N == 1:
            return [({(0,0):1}, {(0,0):1}, {(0,0):1})]
        
        half = N // 2
        # Get terms for the block level (N/2 x N/2)
        # We need to combine the 7 Strassen products
        # Each product is a matrix product of size half x half
        # So we need the decomposition for half x half
        
        # First, get the decomposition for the half-size blocks
        base_terms = get_strassen_terms(half)
        
        # Now, apply the 7 Strassen formulas at the block level
        # The block indices are 00, 01, 10, 11
        # We need to map local indices to global indices
        # Block 00: rows 0..half-1, cols 0..half-1
        # Block 01: rows 0..half-1, cols half..N-1
        # Block 10: rows half..N-1, cols 0..half-1
        # Block 11: rows half..N-1, cols half..N-1
        
        all_terms = []
        
        # Define the block combinations for the 7 Strassen products
        # Each product is defined by linear combinations of blocks for A and B
        # And the resulting C blocks are linear combinations of the products
        
        # We will iterate through the 7 Strassen formulas
        # For each formula, we have a set of (A_blocks, B_blocks)
        # We expand each product using base_terms
        
        # Formula 1: (A00 + A11) (B00 + B11) -> C00, C11
        # Formula 2: (A10 + A11) B00 -> C10, C11
        # Formula 3: A00 (B01 - B11) -> C01, C11
        # Formula 4: A11 (B10 - B00) -> C00, C10
        # Formula 5: (A00 + A01) B11 -> C00, C01
        # Formula 6: (A10 - A00) (B00 + B01) -> C11
        # Formula 7: (A01 - A11) (B10 + B11) -> C00, C11
        
        # Helper to map local index (i, j) in a block to global (I, J)
        def map_block(block_id, i, j):
            if block_id == 0: # 00
                return (i, j)
            elif block_id == 1: # 01
                return (i, j + half)
            elif block_id == 2: # 10
                return (i + half, j)
            elif block_id == 3: # 11
                return (i + half, j + half)
        
        # Helper to add a term to a dictionary
        def add_to_dict(d, key, val):
            if val == 0: return
            d[key] = d.get(key, 0) + val
        
        # The 7 formulas as (list of (A_block, coeff), list of (B_block, coeff), list of (C_block, coeff))
        # Note: The C coefficients are how the product contributes to the C blocks
        formulas = [
            ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]), # P1
            ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),        # P2
            ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),        # P3
            ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),        # P4
            ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),        # P5
            ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),        # P6
            ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1), (3, -1)]) # P7
        ]
        
        for a_blocks, b_blocks, c_blocks in formulas:
            # For each base term (u, v, w) for the half-size product
            for u_base, v_base, w_base in base_terms:
                u_new = {}
                v_new = {}
                w_new = {}
                
                # Expand A
                for block_id, coeff_a in a_blocks:
                    for (i, j), coeff_u in u_base.items():
                        gi, gj = map_block(block_id, i, j)
                        add_to_dict(u_new, (gi, gj), coeff_a * coeff_u)
                        
                # Expand B
                for block_id, coeff_b in b_blocks:
                    for (i, j), coeff_v in v_base.items():
                        gi, gj = map_block(block_id, i, j)
                        add_to_dict(v_new, (gi, gj), coeff_b * coeff_v)
                        
                # Expand C
                for block_id, coeff_c in c_blocks:
                    for (i, j), coeff_w in w_base.items():
                        gi, gj = map_block(block_id, i, j)
                        add_to_dict(w_new, (gi, gj), coeff_c * coeff_w)
                        
                all_terms.append((u_new, v_new, w_new))
                
        return all_terms

    terms_4x4 = get_strassen_terms(4)
    
    col = 0
    for u, v, w in terms_4x4:
        add_term(u, v, w, col)
        col += 1
        
    # Part 2: Cross Terms (61 terms)
    # 1. a d^T: A[i,4] B[4,k] -> C[i,k] for i,k in 0..3
    for i in range(4):
        for k in range(4):
            add_term({(i, 4): 1}, {(4, k): 1}, {(k, i): 1}, col)
            col += 1
            
    # 2. A1 c: A[i,j] B[j,4] -> C[i,4] for i,j in 0..3
    for i in range(4):
        for j in range(4):
            add_term({(i, j): 1}, {(j, 4): 1}, {(4, i): 1}, col)
            col += 1
            
    # 3. a beta: A[i,4] B[4,4] -> C[i,4] for i in 0..3
    for i in range(4):
        add_term({(i, 4): 1}, {(4, 4): 1}, {(4, i): 1}, col)
        col += 1
        
    # 4. b^T B1: A[4,j] B[j,k] -> C[4,k] for j,k in 0..3
    for j in range(4):
        for k in range(4):
            add_term({(4, j): 1}, {(j, k): 1}, {(k, 4): 1}, col)
            col += 1
            
    # 5. alpha d^T: A[4,4] B[4,k] -> C[4,k] for k in 0..3
    for k in range(4):
        add_term({(4, 4): 1}, {(4, k): 1}, {(k, 4): 1}, col)
        col += 1
        
    # 6. b^T c: A[4,j] B[j,4] -> C[4,4] for j in 0..3
    for j in range(4):
        add_term({(4, j): 1}, {(j, 4): 1}, {(4, 4): 1}, col)
        col += 1
        
    # 7. alpha beta: A[4,4] B[4,4] -> C[4,4]
    add_term({(4, 4): 1}, {(4, 4): 1}, {(4, 4): 1}, col)
    col += 1
    
    return {"U": U, "V": V, "W": W}
