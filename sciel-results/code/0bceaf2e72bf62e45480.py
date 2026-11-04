def construct():
    n, m, p = 4, 4, 6
    
    # Total rank: 49 (for 4x4x4) + 28 (for 4x4x2) = 77
    rank = 77
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen 2x2x2 decomposition
    # a_map: dict {local_idx: (global_u_idx, coeff)}
    # b_map: dict {local_idx: (global_v_idx, coeff)}
    # c_map: dict {local_idx: (global_w_idx, coeff)}
    # local_idx: 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1)
    def add_strassen_2x2(a_map, b_map, c_map):
        nonlocal col
        # Strassen products M1..M7
        # M1 = (a00+a11)(b00+b11)
        # M2 = (a10+a11)b00
        # M3 = a00(b01-b11)
        # M4 = a11(b10-b00)
        # M5 = (a00+a01)b11
        # M6 = (a10-a00)(b00+b01)
        # M7 = (a01-a11)(b10+b11)
        
        # C00 = M1 - M4 + M5 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M3 + M2 - M6
        
        # Each M is a scalar product of two linear forms.
        # We add 7 rank-1 terms.
        
        # M1
        # A: 00 (+1), 11 (+1)
        # B: 00 (+1), 11 (+1)
        # C: 00 (+1), 11 (+1)
        def add_term(a_indices, b_indices, c_indices, c_coeffs):
            nonlocal col
            for idx, coeff in a_indices:
                U[idx][col] += coeff
            for idx, coeff in b_indices:
                V[idx][col] += coeff
            for idx, coeff in c_indices:
                W[idx][col] += coeff
            col += 1

        # M1
        a_idx = [(a_map[0][0], a_map[0][1]), (a_map[3][0], a_map[3][1])]
        b_idx = [(b_map[0][0], b_map[0][1]), (b_map[3][0], b_map[3][1])]
        c_idx = [(c_map[0][0], c_map[0][1]), (c_map[3][0], c_map[3][1])]
        add_term(a_idx, b_idx, c_idx, None)
        
        # M2
        # A: 10 (+1), 11 (+1)
        # B: 00 (+1)
        # C: 10 (+1), 11 (+1)
        a_idx = [(a_map[2][0], a_map[2][1]), (a_map[3][0], a_map[3][1])]
        b_idx = [(b_map[0][0], b_map[0][1])]
        c_idx = [(c_map[2][0], c_map[2][1]), (c_map[3][0], c_map[3][1])]
        add_term(a_idx, b_idx, c_idx, None)
        
        # M3
        # A: 00 (+1)
        # B: 01 (+1), 11 (-1)
        # C: 01 (+1), 11 (-1)
        a_idx = [(a_map[0][0], a_map[0][1])]
        b_idx = [(b_map[1][0], b_map[1][1]), (b_map[3][0], -b_map[3][1])]
        c_idx = [(c_map[1][0], c_map[1][1]), (c_map[3][0], -c_map[3][1])]
        add_term(a_idx, b_idx, c_idx, None)
        
        # M4
        # A: 11 (+1)
        # B: 10 (+1), 00 (-1)
        # C: 00 (-1), 10 (+1)
        a_idx = [(a_map[3][0], a_map[3][1])]
        b_idx = [(b_map[2][0], b_map[2][1]), (b_map[0][0], -b_map[0][1])]
        c_idx = [(c_map[0][0], -c_map[0][1]), (c_map[2][0], c_map[2][1])]
        add_term(a_idx, b_idx, c_idx, None)
        
        # M5
        # A: 00 (+1), 01 (+1)
        # B: 11 (+1)
        # C: 00 (+1), 01 (+1)
        a_idx = [(a_map[0][0], a_map[0][1]), (a_map[1][0], a_map[1][1])]
        b_idx = [(b_map[3][0], b_map[3][1])]
        c_idx = [(c_map[0][0], c_map[0][1]), (c_map[1][0], c_map[1][1])]
        add_term(a_idx, b_idx, c_idx, None)
        
        # M6
        # A: 10 (+1), 00 (-1)
        # B: 00 (+1), 01 (+1)
        # C: 11 (-1)
        a_idx = [(a_map[2][0], a_map[2][1]), (a_map[0][0], -a_map[0][1])]
        b_idx = [(b_map[0][0], b_map[0][1]), (b_map[1][0], b_map[1][1])]
        c_idx = [(c_map[3][0], -c_map[3][1])]
        add_term(a_idx, b_idx, c_idx, None)
        
        # M7
        # A: 01 (+1), 11 (-1)
        # B: 10 (+1), 11 (+1)
        # C: 00 (-1)
        a_idx = [(a_map[1][0], a_map[1][1]), (a_map[3][0], -a_map[3][1])]
        b_idx = [(b_map[2][0], b_map[2][1]), (b_map[3][0], b_map[3][1])]
        c_idx = [(c_map[0][0], -c_map[0][1])]
        add_term(a_idx, b_idx, c_idx, None)
        
        return col

    # Part 1: A (4x4) * B1 (4x4) -> C1 (4x4)
    # B1 is columns 0-3 of B.
    # A is 4x4.
    
    # Define maps for Strassen blocks
    # A blocks:
    # A00: rows 0-1, cols 0-1
    # A01: rows 0-1, cols 2-3
    # A10: rows 2-3, cols 0-1
    # A11: rows 2-3, cols 2-3
    
    # B1 blocks:
    # B00: rows 0-1, cols 0-1
    # B01: rows 0-1, cols 2-3
    # B10: rows 2-3, cols 0-1
    # B11: rows 2-3, cols 2-3
    
    # C1 blocks:
    # C00: rows 0-1, cols 0-1
    # C01: rows 0-1, cols 2-3
    # C10: rows 2-3, cols 0-1
    # C11: rows 2-3, cols 2-3
    
    # Helper to get U index for A[i,j]
    def u_idx(i, j): return i * m + j
    # Helper to get V index for B[j,k]
    def v_idx(j, k): return j * p + k
    # Helper to get W index for C[i,k] (W[k*n+i])
    def w_idx(i, k): return k * n + i
    
    # Strassen P1 = (A00+A11)(B00+B11)
    # A00: 0,0; 0,1; 1,0; 1,1
    # A11: 2,2; 2,3; 3,2; 3,3
    a_map = {
        0: (u_idx(0,0), 1),  # A00[0,0]
        1: (u_idx(0,1), 1),  # A00[0,1]
        2: (u_idx(1,0), 1),  # A00[1,0]
        3: (u_idx(1,1), 1)   # A00[1,1]
    }
    # Wait, the map is for the "virtual" 2x2 matrix.
    # For P1, virtual A[0,0] = A00[0,0] + A11[0,0] = A[0,0] + A[2,2]
    # virtual A[0,1] = A00[0,1] + A11[0,1] = A[0,1] + A[2,3]
    # virtual A[1,0] = A00[1,0] + A11[1,0] = A[1,0] + A[3,2]
    # virtual A[1,1] = A00[1,1] + A11[1,1] = A[1,1] + A[3,3]
    
    # My `add_strassen_2x2` expects `a_map` to map local indices 0..3 to (global_idx, coeff).
    # But here, one local index corresponds to TWO global indices.
    # So I need to modify `add_term` or `add_strassen_2x2` to handle lists of (idx, coeff).
    
    # Let's redefine `add_strassen_2x2` to take `a_map` as {local_idx: [(global_idx, coeff), ...]}
    
    def add_strassen_2x2_v2(a_map, b_map, c_map):
        nonlocal col
        
        def add_term(a_indices, b_indices, c_indices):
            nonlocal col
            for idx, coeff in a_indices:
                U[idx][col] += coeff
            for idx, coeff in b_indices:
                V[idx][col] += coeff
            for idx, coeff in c_indices:
                W[idx][col] += coeff
            col += 1

        # M1: (a0+a1)(b0+b1) where a0, a1 are local (0,0) and (1,1)
        # Local 0: a_map[0]
        # Local 3: a_map[3]
        a_idx = a_map[0] + a_map[3]
        b_idx = b_map[0] + b_map[3]
        c_idx = c_map[0] + c_map[3]
        add_term(a_idx, b_idx, c_idx)
        
        # M2: (a2+a3)b0
        a_idx = a_map[2] + a_map[3]
        b_idx = b_map[0]
        c_idx = c_map[2] + c_map[3]
        add_term(a_idx, b_idx, c_idx)
        
        # M3: a0(b1-b3)
        a_idx = a_map[0]
        b_idx = [(idx, coeff) for idx, coeff in b_map[1]] + [(idx, -coeff) for idx, coeff in b_map[3]]
        c_idx = [(idx, coeff) for idx, coeff in c_map[1]] + [(idx, -coeff) for idx, coeff in c_map[3]]
        add_term(a_idx, b_idx, c_idx)
        
        # M4: a3(b2-b0)
        a_idx = a_map[3]
        b_idx = [(idx, coeff) for idx, coeff in b_map[2]] + [(idx, -coeff) for idx, coeff in b_map[0]]
        c_idx = [(idx, -coeff) for idx, coeff in c_map[0]] + [(idx, coeff) for idx, coeff in c_map[2]]
        add_term(a_idx, b_idx, c_idx)
        
        # M5: (a0+a1)b3
        a_idx = a_map[0] + a_map[1]
        b_idx = b_map[3]
        c_idx = c_map[0] + c_map[1]
        add_term(a_idx, b_idx, c_idx)
        
        # M6: (a2-a0)(b0+b1)
        a_idx = [(idx, coeff) for idx, coeff in a_map[2]] + [(idx, -coeff) for idx, coeff in a_map[0]]
        b_idx = b_map[0] + b_map[1]
        c_idx = [(idx, -coeff) for idx, coeff in c_map[3]]
        add_term(a_idx, b_idx, c_idx)
        
        # M7: (a1-a3)(b2+b3)
        a_idx = [(idx, coeff) for idx, coeff in a_map[1]] + [(idx, -coeff) for idx, coeff in a_map[3]]
        b_idx = b_map[2] + b_map[3]
        c_idx = [(idx, -coeff) for idx, coeff in c_map[0]]
        add_term(a_idx, b_idx, c_idx)
        
        return col

    # Now construct maps for P1..P7 for the 4x4x4 part
    
    # P1 = (A00+A11)(B00+B11)
    a_map = {
        0: [(u_idx(0,0), 1), (u_idx(2,2), 1)],
        1: [(u_idx(0,1), 1), (u_idx(2,3), 1)],
        2: [(u_idx(1,0), 1), (u_idx(3,2), 1)],
        3: [(u_idx(1,1), 1), (u_idx(3,3), 1)]
    }
    b_map = {
        0: [(v_idx(0,0), 1), (v_idx(2,2), 1)],
        1: [(v_idx(0,1), 1), (v_idx(2,3), 1)],
        2: [(v_idx(1,0), 1), (v_idx(3,2), 1)],
        3: [(v_idx(1,1), 1), (v_idx(3,3), 1)]
    }
    c_map = {
        0: [(w_idx(0,0), 1), (w_idx(2,2), 1)],
        1: [(w_idx(0,1), 1), (w_idx(2,3), 1)],
        2: [(w_idx(1,0), 1), (w_idx(3,2), 1)],
        3: [(w_idx(1,1), 1), (w_idx(3,3), 1)]
    }
    add_strassen_2x2_v2(a_map, b_map, c_map)
    
    # P2 = (A10+A11)B00
    a_map = {
        0: [(u_idx(2,0), 1), (u_idx(2,2), 1)],
        1: [(u_idx(2,1), 1), (u_idx(2,3), 1)],
        2: [(u_idx(3,0), 1), (u_idx(3,2), 1)],
        3: [(u_idx(3,1), 1), (u_idx(3,3), 1)]
    }
    b_map = {
        0: [(v_idx(0,0), 1)],
        1: [(v_idx(0,1), 1)],
        2: [(v_idx(1,0), 1)],
        3: [(v_idx(1,1), 1)]
    }
    c_map = {
        0: [(w_idx(2,0), 1)],
        1: [(w_idx(2,1), 1)],
        2: [(w_idx(3,0), 1)],
        3: [(w_idx(3,1), 1)]
    }
    add_strassen_2x2_v2(a_map, b_map, c_map)
    
    # P3 = A00(B01-B11)
    a_map = {
        0: [(u_idx(0,0), 1)],
        1: [(u_idx(0,1), 1)],
        2: [(u_idx(1,0), 1)],
        3: [(u_idx(1,1), 1)]
    }
    b_map = {
        0: [(v_idx(0,2), 1), (v_idx(2,2), -1)],
        1: [(v_idx(0,3), 1), (v_idx(2,3), -1)],
        2: [(v_idx(1,2), 1), (v_idx(3,2), -1)],
        3: [(v_idx(1,3), 1), (v_idx(3,3), -1)]
    }
    c_map = {
        0: [(w_idx(0,2), 1)],
        1: [(w_idx(0,3), 1)],
        2: [(w_idx(1,2), 1)],
        3: [(w_idx(1,3), 1)]
    }
    add_strassen_2x2_v2(a_map, b_map, c_map)
    
    # P4 = A11(B10-B00)
    a_map = {
        0: [(u_idx(2,2), 1)],
        1: [(u_idx(2,3), 1)],
        2: [(u_idx(3,2), 1)],
        3: [(u_idx(3,3), 1)]
    }
    b_map = {
        0: [(v_idx(2,0), 1), (v_idx(0,0), -1)],
        1: [(v_idx(2,1), 1), (v_idx(0,1), -1)],
        2: [(v_idx(3,0), 1), (v_idx(1,0), -1)],
        3: [(v_idx(3,1), 1), (v_idx(1,1), -1)]
    }
    c_map = {
        0: [(w_idx(2,0), 1)],
        1: [(w_idx(2,1), 1)],
        2: [(w_idx(3,0), 1)],
        3: [(w_idx(3,1), 1)]
    }
    add_strassen_2x2_v2(a_map, b_map, c_map)
    
    # P5 = (A00+A01)B11
    a_map = {
        0: [(u_idx(0,0), 1), (u_idx(0,2), 1)],
        1: [(u_idx(0,1), 1), (u_idx(0,3), 1)],
        2: [(u_idx(1,0), 1), (u_idx(1,2), 1)],
        3: [(u_idx(1,1), 1), (u_idx(1,3), 1)]
    }
    b_map = {
        0: [(v_idx(2,2), 1)],
        1: [(v_idx(2,3), 1)],
        2: [(v_idx(3,2), 1)],
        3: [(v_idx(3,3), 1)]
    }
    c_map = {
        0: [(w_idx(0,2), 1)],
        1: [(w_idx(0,3), 1)],
        2: [(w_idx(1,2), 1)],
        3: [(w_idx(1,3), 1)]
    }
    add_strassen_2x2_v2(a_map, b_map, c_map)
    
    # P6 = (A10-A00)(B00+B01)
    a_map = {
        0: [(u_idx(2,0), 1), (u_idx(0,0), -1)],
        1: [(u_idx(2,1), 1), (u_idx(0,1), -1)],
        2: [(u_idx(3,0), 1), (u_idx(1,0), -1)],
        3: [(u_idx(3,1), 1), (u_idx(1,1), -1)]
    }
    b_map = {
        0: [(v_idx(0,0), 1), (v_idx(0,2), 1)],
        1: [(v_idx(0,1), 1), (v_idx(0,3), 1)],
        2: [(v_idx(1,0), 1), (v_idx(1,2), 1)],
        3: [(v_idx(1,1), 1), (v_idx(1,3), 1)]
    }
    c_map = {
        0: [(w_idx(2,0), 1)],
        1: [(w_idx(2,1), 1)],
        2: [(w_idx(3,0), 1)],
        3: [(w_idx(3,1), 1)]
    }
    add_strassen_2x2_v2(a_map, b_map, c_map)
    
    # P7 = (A01-A11)(B10+B11)
    a_map = {
        0: [(u_idx(0,2), 1), (u_idx(2,2), -1)],
        1: [(u_idx(0,3), 1), (u_idx(2,3), -1)],
        2: [(u_idx(1,2), 1), (u_idx(3,2), -1)],
        3: [(u_idx(1,3), 1), (u_idx(3,3), -1)]
    }
    b_map = {
        0: [(v_idx(2,0), 1), (v_idx(2,2), 1)],
        1: [(v_idx(2,1), 1), (v_idx(2,3), 1)],
        2: [(v_idx(3,0), 1), (v_idx(3,2), 1)],
        3: [(v_idx(3,1), 1), (v_idx(3,3), 1)]
    }
    c_map = {
        0: [(w_idx(2,0), 1)],
        1: [(w_idx(2,1), 1)],
        2: [(w_idx(3,0), 1)],
        3: [(w_idx(3,1), 1)]
    }
    add_strassen_2x2_v2(a_map, b_map, c_map)
    
    # Part 2: A (4x4) * B2 (4x2) -> C2 (4x2)
    # B2 is columns 4-5 of B.
    # C2 is columns 4-5 of C.
    
    # Split A into 2x2 blocks A00, A01, A10, A11.
    # B2 is 4x2. Split into B2_top (2x2) and B2_bot (2x2).
    # B2_top is rows 0-1, cols 4-5.
    # B2_bot is rows 2-3, cols 4-5.
    
    # C2 is 4x2. Split into C2_top (2x2) and C2_bot (2x2).
    # C2_top is rows 0-1, cols 4-5.
    # C2_bot is rows 2-3, cols 4-5.
    
    # C2_top = A00 * B2_top + A01 * B2_bot
    # C2_bot = A10 * B2_top + A11 * B2_bot
    
    # We need 4 products of 2x2x2.
    # Each uses Strassen (rank 7).
    # Total 28.
    
    # Product 1: A00 * B2_top -> contributes to C2_top
    a_map = {
        0: [(u_idx(0,0), 1)],
        1: [(u_idx(0,1), 1)],
        2: [(u_idx(1,0), 1)],
        3: [(u_idx(1,1), 1)]
    }
    b_map = {
        0: [(v_idx(0,4), 1)],
        1: [(v_idx(0,5), 1)],
        2: [(v_idx(1,4), 1)],
        3: [(v_idx(1,5), 1)]
    }
    c_map = {
        0: [(w_idx(0,4), 1)],
        1: [(w_idx(0,5), 1)],
        2: [(w_idx(1,4), 1)],
        3: [(w_idx(1,5), 1)]
    }
    add_strassen_2x2_v2(a_map, b_map, c_map)
    
    # Product 2: A01 * B2_bot -> contributes to C2_top
    a_map = {
        0: [(u_idx(0,2), 1)],
        1: [(u_idx(0,3), 1)],
        2: [(u_idx(1,2), 1)],
        3: [(u_idx(1,3), 1)]
    }
    b_map = {
        0: [(v_idx(2,4), 1)],
        1: [(v_idx(2,5), 1)],
        2: [(v_idx(3,4), 1)],
        3: [(v_idx(3,5), 1)]
    }
    c_map = {
        0: [(w_idx(0,4), 1)],
        1: [(w_idx(0,5), 1)],
        2: [(w_idx(1,4), 1)],
        3: [(w_idx(1,5), 1)]
    }
    add_strassen_2x2_v2(a_map, b_map, c_map)
    
    # Product 3: A10 * B2_top -> contributes to C2_bot
    a_map = {
        0: [(u_idx(2,0), 1)],
        1: [(u_idx(2,1), 1)],
        2: [(u_idx(3,0), 1)],
        3: [(u_idx(3,1), 1)]
    }
    b_map = {
        0: [(v_idx(0,4), 1)],
        1: [(v_idx(0,5), 1)],
        2: [(v_idx(1,4), 1)],
        3: [(v_idx(1,5), 1)]
    }
    c_map = {
        0: [(w_idx(2,4), 1)],
        1: [(w_idx(2,5), 1)],
        2: [(w_idx(3,4), 1)],
        3: [(w_idx(3,5), 1)]
    }
    add_strassen_2x2_v2(a_map, b_map, c_map)
    
    # Product 4: A11 * B2_bot -> contributes to C2_bot
    a_map = {
        0: [(u_idx(2,2), 1)],
        1: [(u_idx(2,3), 1)],
        2: [(u_idx(3,2), 1)],
        3: [(u_idx(3,3), 1)]
    }
    b_map = {
        0: [(v_idx(2,4), 1)],
        1: [(v_idx(2,5), 1)],
        2: [(v_idx(3,4), 1)],
        3: [(v_idx(3,5), 1)]
    }
    c_map = {
        0: [(w_idx(2,4), 1)],
        1: [(w_idx(2,5), 1)],
        2: [(w_idx(3,4), 1)],
        3: [(w_idx(3,5), 1)]
    }
    add_strassen_2x2_v2(a_map, b_map, c_map)
    
    return {"U": U, "V": V, "W": W}
