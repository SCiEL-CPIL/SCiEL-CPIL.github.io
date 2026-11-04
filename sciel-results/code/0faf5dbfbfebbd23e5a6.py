import numpy as np

# EVOLVE-BLOCK-START

def _get_strassen_base_case():
    """
    Returns the 7 terms of the Strassen decomposition for 2x2x2.
    Each term is a dict with keys 'U', 'V', 'W', each a list of (idx, val).
    """
    terms = []
    
    # Term 1: M1 = (a+d)(e+h)
    # U: a(0), d(3)
    # V: e(0), h(3)
    # W: C00(0), C11(3) -> coeffs 1, 1
    terms.append({
        'U': [(0, 1), (3, 1)],
        'V': [(0, 1), (3, 1)],
        'W': [(0, 1), (3, 1)]
    })
    
    # Term 2: M2 = (c+d)e
    # U: c(2), d(3)
    # V: e(0)
    # W: C10(1), C11(3) -> coeffs 1, -1
    terms.append({
        'U': [(2, 1), (3, 1)],
        'V': [(0, 1)],
        'W': [(1, 1), (3, -1)]
    })
    
    # Term 3: M3 = a(f-h)
    # U: a(0)
    # V: f(1), h(3) -> coeffs 1, -1
    # W: C01(2), C11(3) -> coeffs 1, 1
    terms.append({
        'U': [(0, 1)],
        'V': [(1, 1), (3, -1)],
        'W': [(2, 1), (3, 1)]
    })
    
    # Term 4: M4 = d(g-e)
    # U: d(3)
    # V: g(2), e(0) -> coeffs 1, -1
    # W: C00(0), C01(2), C10(1) -> coeffs -1, 1, 1
    terms.append({
        'U': [(3, 1)],
        'V': [(2, 1), (0, -1)],
        'W': [(0, -1), (2, 1), (1, 1)]
    })
    
    # Term 5: M5 = (a+b)h
    # U: a(0), b(1)
    # V: h(3)
    # W: C00(0) -> coeff 1
    terms.append({
        'U': [(0, 1), (1, 1)],
        'V': [(3, 1)],
        'W': [(0, 1)]
    })
    
    # Term 6: M6 = (c-a)(e+f)
    # U: c(2), a(0) -> coeffs 1, -1
    # V: e(0), f(1)
    # W: C11(3) -> coeff 1
    terms.append({
        'U': [(2, 1), (0, -1)],
        'V': [(0, 1), (1, 1)],
        'W': [(3, 1)]
    })
    
    # Term 7: M7 = (b-d)(g+h)
    # U: b(1), d(3) -> coeffs 1, -1
    # V: g(2), h(3)
    # W: C00(0) -> coeff -1
    terms.append({
        'U': [(1, 1), (3, -1)],
        'V': [(2, 1), (3, 1)],
        'W': [(0, -1)]
    })
    
    return terms

def _construct_4x4x4():
    """
    Construct rank-49 decomposition for 4x4x4.
    Returns U, V, W as lists of lists.
    """
    base_terms = _get_strassen_base_case()
    rank = 49
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(16)]
    
    # Strassen terms definitions for 4x4x4
    # Each term defines how to map local 2x2 indices to global 4x4 indices
    # term: (U_map, V_map, W_map)
    # U_map: list of (local_row_offset, local_col_offset, multiplier)
    # Actually, it's easier to define which blocks are summed.
    
    # M1 = (A00+A11)(B00+B11)
    # A00: i=0,1 j=0,1. A11: i=2,3 j=2,3.
    # B00: j=0,1 k=0,1. B11: j=2,3 k=2,3.
    # C00: i=0,1 k=0,1. C11: i=2,3 k=2,3.
    
    def add_term(col_offset, A_blocks, B_blocks, C_blocks):
        # A_blocks: list of (row_offset, col_offset)
        # B_blocks: list of (row_offset, col_offset)
        # C_blocks: list of (row_offset, col_offset)
        
        for s, term in enumerate(base_terms):
            col = col_offset + s
            
            # Process U
            for local_idx, val in term['U']:
                li = local_idx // 2
                lj = local_idx % 2
                for (ro, co) in A_blocks:
                    global_idx = (li + ro) * 4 + (lj + co)
                    U[global_idx][col] += val
            
            # Process V
            for local_idx, val in term['V']:
                lj = local_idx // 2
                lk = local_idx % 2
                for (ro, co) in B_blocks:
                    global_idx = (lj + ro) * 4 + (lk + co)
                    V[global_idx][col] += val
            
            # Process W
            for local_idx, val in term['W']:
                lk = local_idx // 2
                li = local_idx % 2
                for (ro, co) in C_blocks:
                    global_idx = (lk + ro) * 4 + (li + co)
                    W[global_idx][col] += val

    # Define the 7 Strassen terms for 4x4x4
    # A00: (0,0), A01: (0,2), A10: (2,0), A11: (2,2)
    
    # M1 = (A00+A11)(B00+B11)
    add_term(0, [(0,0), (2,2)], [(0,0), (2,2)], [(0,0), (2,2)])
    
    # M2 = (A10+A11)B00
    add_term(7, [(2,0), (2,2)], [(0,0)], [(2,0), (2,2)])
    
    # M3 = A00(B01-B11)
    add_term(14, [(0,0)], [(0,2), (2,2)], [(0,2), (2,2)])
    # Note: B01 is (0,2), B11 is (2,2).
    # The subtraction is handled by the base case V values?
    # No, the base case V values are for a single matrix multiplication.
    # Here we are passing a list of blocks to be summed.
    # The base case computes sum(A_blocks) * sum(B_blocks).
    # This is correct for M1, M2, M5, M6, M7.
    # For M3: A00 * (B01 - B11).
    # sum(B_blocks) with B11 having negative weight?
    # My add_term function just sums values. It doesn't support negative weights for blocks.
    # I need to modify add_term to accept weights for blocks.
    
    # Let's refine add_term to accept weights.
    
    return U, V, W

def _construct_4x4x4_v2():
    base_terms = _get_strassen_base_case()
    rank = 49
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(16)]
    
    def add_term(col_offset, A_blocks, B_blocks, C_blocks):
        # A_blocks: list of (row_offset, col_offset, weight)
        for s, term in enumerate(base_terms):
            col = col_offset + s
            
            for local_idx, val in term['U']:
                li = local_idx // 2
                lj = local_idx % 2
                for (ro, co, w) in A_blocks:
                    global_idx = (li + ro) * 4 + (lj + co)
                    U[global_idx][col] += val * w
            
            for local_idx, val in term['V']:
                lj = local_idx // 2
                lk = local_idx % 2
                for (ro, co, w) in B_blocks:
                    global_idx = (lj + ro) * 4 + (lk + co)
                    V[global_idx][col] += val * w
            
            for local_idx, val in term['W']:
                lk = local_idx // 2
                li = local_idx % 2
                for (ro, co, w) in C_blocks:
                    global_idx = (lk + ro) * 4 + (li + co)
                    W[global_idx][col] += val * w

    # M1 = (A00+A11)(B00+B11)
    add_term(0, [(0,0,1), (2,2,1)], [(0,0,1), (2,2,1)], [(0,0,1), (2,2,1)])
    
    # M2 = (A10+A11)B00
    add_term(7, [(2,0,1), (2,2,1)], [(0,0,1)], [(2,0,1), (2,2,-1)])
    # Wait, C10 = M2 + M4. C11 = M1 - M2 + M3 + M6.
    # M2 contributes +1 to C10, -1 to C11.
    # C10 is block (2,0). C11 is block (2,2).
    # So C_blocks for M2 should be [(2,0,1), (2,2,-1)].
    
    # M3 = A00(B01-B11)
    # C01 = M3 + M4. C11 = M1 - M2 + M3 + M6.
    # M3 contributes +1 to C01, +1 to C11.
    # C01 is block (0,2). C11 is block (2,2).
    add_term(14, [(0,0,1)], [(0,2,1), (2,2,-1)], [(0,2,1), (2,2,1)])
    
    # M4 = A11(B10-B00)
    # C00 = M1 - M4 + M5 - M7.
    # C01 = M3 + M4.
    # C10 = M2 + M4.
    # M4 contributes -1 to C00, +1 to C01, +1 to C10.
    # C00 is (0,0). C01 is (0,2). C10 is (2,0).
    add_term(21, [(2,2,1)], [(2,0,1), (0,0,-1)], [(0,0,-1), (0,2,1), (2,0,1)])
    
    # M5 = (A00+A01)B11
    # C00 = M1 - M4 + M5 - M7.
    # M5 contributes +1 to C00.
    add_term(28, [(0,0,1), (0,2,1)], [(2,2,1)], [(0,0,1)])
    
    # M6 = (A10-A00)(B00+B01)
    # C11 = M1 - M2 + M3 + M6.
    # M6 contributes +1 to C11.
    add_term(35, [(2,0,1), (0,0,-1)], [(0,0,1), (0,2,1)], [(2,2,1)])
    
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 - M4 + M5 - M7.
    # M7 contributes -1 to C00.
    add_term(42, [(0,2,1), (2,2,-1)], [(2,0,1), (2,2,1)], [(0,0,-1)])
    
    return U, V, W

def _construct_4x4x2():
    base_terms = _get_strassen_base_case()
    rank = 28
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(8)]
    W = [[0] * rank for _ in range(8)]
    
    def add_term_442(col_offset, A_blocks, B_blocks, C_blocks):
        # A_blocks: (row_off, col_off, weight) relative to 4x4 A
        # B_blocks: (row_off, col_off, weight) relative to 4x2 B
        # C_blocks: (row_off, col_off, weight) relative to 4x2 C
        
        for s, term in enumerate(base_terms):
            col = col_offset + s
            
            for local_idx, val in term['U']:
                li = local_idx // 2
                lj = local_idx % 2
                for (ro, co, w) in A_blocks:
                    global_idx = (li + ro) * 4 + (lj + co)
                    U[global_idx][col] += val * w
            
            for local_idx, val in term['V']:
                lj = local_idx // 2
                lk = local_idx % 2
                for (ro, co, w) in B_blocks:
                    global_idx = (lj + ro) * 2 + (lk + co)
                    V[global_idx][col] += val * w
            
            for local_idx, val in term['W']:
                lk = local_idx // 2
                li = local_idx % 2
                for (ro, co, w) in C_blocks:
                    global_idx = (lk + ro) * 4 + (li + co)
                    W[global_idx][col] += val * w

    # C = A_L B_T + A_R B_B
    # A_L: cols 0-1. A_R: cols 2-3.
    # B_T: rows 0-1. B_B: rows 2-3.
    
    # A_L B_T
    # A_LT: rows 0-1, cols 0-1. A_LB: rows 2-3, cols 0-1.
    # B_T: rows 0-1, cols 0-1.
    # C_1T: rows 0-1, cols 0-1. C_1B: rows 2-3, cols 0-1.
    
    # We use Strassen on A_LT B_T and A_LB B_T?
    # No, A_LT B_T is 2x2x2.
    # We can just use the base case directly for each 2x2x2.
    # But we have 4 such multiplications:
    # 1. A_LT B_T -> C_1T
    # 2. A_LB B_T -> C_1B
    # 3. A_RT B_B -> C_2T
    # 4. A_RB B_B -> C_2B
    
    # Each is rank 7. Total 28.
    
    # 1. A_LT B_T
    # A: (0,0,1). B: (0,0,1). C: (0,0,1).
    add_term_442(0, [(0,0,1)], [(0,0,1)], [(0,0,1)])
    
    # 2. A_LB B_T
    # A: (2,0,1). B: (0,0,1). C: (2,0,1).
    add_term_442(7, [(2,0,1)], [(0,0,1)], [(2,0,1)])
    
    # 3. A_RT B_B
    # A: (0,2,1). B: (2,0,1). C: (0,0,1).
    add_term_442(14, [(0,2,1)], [(2,0,1)], [(0,0,1)])
    
    # 4. A_RB B_B
    # A: (2,2,1). B: (2,0,1). C: (2,0,1).
    add_term_442(21, [(2,2,1)], [(2,0,1)], [(2,0,1)])
    
    return U, V, W

def construct():
    U1, V1, W1 = _construct_4x4x4_v2()
    U2, V2, W2 = _construct_4x4x2()
    
    # Pad V1 to 24 rows
    V1_padded = [row[:] for row in V1]
    for _ in range(8):
        V1_padded.append([0] * 49)
        
    # Pad V2 to 24 rows
    # V2 is 8x28. We need to map its rows to the correct positions in the 24-row V.
    # V2 rows correspond to B (4x2) indices j*2+k.
    # In final V (4x6), these correspond to j*6 + (4+k).
    V2_padded = [[0] * 28 for _ in range(24)]
    for r in range(8):
        j = r // 2
        k = r % 2
        target_row = j * 6 + (4 + k)
        V2_padded[target_row] = V2[r][:]
        
    # Pad W1 to 24 rows
    # W1 rows correspond to C (4x4) indices k*4+i.
    # In final W (6x4), these correspond to k*4+i.
    # So W1 rows 0-15 map to W rows 0-15.
    W1_padded = [row[:] for row in W1]
    for _ in range(8):
        W1_padded.append([0] * 49)
        
    # Pad W2 to 24 rows
    # W2 rows correspond to C (4x2) indices k*4+i.
    # In final W (6x4), these correspond to k*4+i.
    # So W2 rows 0-7 map to W rows 0-7.
    W2_padded = [[0] * 28 for _ in range(24)]
    for r in range(8):
        W2_padded[r] = W2[r][:]
        
    # Concatenate
    U = [U1[i] + U2[i] for i in range(16)]
    V = [V1_padded[i] + V2_padded[i] for i in range(24)]
    W = [W1_padded[i] + W2_padded[i] for i in range(24)]
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
