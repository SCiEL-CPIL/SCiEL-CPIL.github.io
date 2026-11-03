"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 56 using Strassen on 2x2 blocks
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 matmul
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # Output mapping:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Local indices for A (2x2):
    # a: (0,0), b: (0,1), c: (1,0), d: (1,1)
    # Local indices for B (2x2):
    # e: (0,0), f: (0,1), g: (1,0), h: (1,1)
    
    # Define the 7 Strassen terms
    # Each term is a dict of (local_A_index, coeff) and (local_B_index, coeff)
    # and a dict of (local_C_index, coeff)
    
    terms = []
    
    # M1
    terms.append({
        'A': {0: 1, 3: 1}, # a, d
        'B': {0: 1, 3: 1}, # e, h
        'C': {0: 1, 3: 1}  # C00, C11
    })
    # M2
    terms.append({
        'A': {2: 1, 3: 1}, # c, d
        'B': {0: 1},       # e
        'C': {2: 1, 3: -1} # C10, C11
    })
    # M3
    terms.append({
        'A': {0: 1},       # a
        'B': {1: 1, 3: -1},# f, h
        'C': {1: 1, 3: 1}  # C01, C11
    })
    # M4
    terms.append({
        'A': {3: 1},       # d
        'B': {2: 1, 0: -1},# g, e
        'C': {0: 1, 2: 1}  # C00, C10
    })
    # M5
    terms.append({
        'A': {0: 1, 1: 1}, # a, b
        'B': {3: 1},       # h
        'C': {0: -1, 1: 1} # C00, C01
    })
    # M6
    terms.append({
        'A': {2: 1, 0: -1},# c, a
        'B': {0: 1, 1: 1}, # e, f
        'C': {3: 1}        # C11
    })
    # M7
    terms.append({
        'A': {1: 1, 3: -1},# b, d
        'B': {2: 1, 3: 1}, # g, h
        'C': {0: 1}        # C00
    })
    
    # Iterate over the 8 sub-problems
    # side: 0 for L (cols 0-1 of A, rows 0-1 of B), 1 for R (cols 2-3 of A, rows 2-3 of B)
    # block: 0-3 (cols 0-1, 2-3, 4-5, 6-7 of the 2x8 B block)
    
    col_idx = 0
    for side in range(2):
        for block in range(4):
            for t in terms:
                # Determine global offsets
                # A global row: i * m + j
                # Local A index l: i_loc = l // 2, j_loc = l % 2
                # Global i = i_loc
                # Global j = side * 2 + j_loc
                
                # B global row: j * p + k
                # Local B index l: j_loc = l // 2, k_loc = l % 2
                # Global j = side * 2 + j_loc
                # Global k = block * 2 + k_loc
                
                # C global row: k * n + i
                # Local C index l: i_loc = l % 2, k_loc = l // 2
                # Global i = i_loc
                # Global k = block * 2 + k_loc
                
                for l_A, coeff_A in t['A'].items():
                    i_loc = l_A // 2
                    j_loc = l_A % 2
                    i_glob = i_loc
                    j_glob = side * 2 + j_loc
                    row_U = i_glob * m + j_glob
                    U[row_U][col_idx] += coeff_A
                
                for l_B, coeff_B in t['B'].items():
                    j_loc = l_B // 2
                    k_loc = l_B % 2
                    j_glob = side * 2 + j_loc
                    k_glob = block * 2 + k_loc
                    row_V = j_glob * p + k_glob
                    V[row_V][col_idx] += coeff_B
                
                for l_C, coeff_C in t['C'].items():
                    i_loc = l_C % 2
                    k_loc = l_C // 2
                    i_glob = i_loc
                    k_glob = block * 2 + k_loc
                    row_W = k_glob * n + i_glob
                    W[row_W][col_idx] += coeff_C
                
                col_idx += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
