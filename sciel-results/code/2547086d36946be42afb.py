import time
import sys

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    n, m, p = 4, 4, 4
    
    # Target rank: 49 (Strassen 4x4)
    # This is a structural improvement over the schoolbook rank 64.
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a Strassen 2x2 decomposition for a specific product.
    # a_coeffs: dict mapping (row, col) in 2x2 to coefficient for A
    # b_coeffs: dict mapping (row, col) in 2x2 to coefficient for B
    # c_coeffs: dict mapping (row, col) in 2x2 to coefficient for C
    # offset_a: (r, c) offset in global A
    # offset_b: (r, c) offset in global B
    # offset_c: (r, c) offset in global C
    # base_col: starting column index in U, V, W
    
    def add_strassen_2x2(a_coeffs, b_coeffs, c_coeffs, offset_a, offset_b, offset_c, base_col):
        # The 7 Strassen terms for 2x2:
        # M1 = (a00 + a11)(b00 + b11) -> c00 += 1, c11 += 1
        # M2 = (a10 + a11)b00 -> c10 += 1, c11 -= 1
        # M3 = a00(b01 - b11) -> c01 += 1, c11 += 1
        # M4 = a11(b10 - b00) -> c00 += 1, c10 += 1
        # M5 = (a00 + a01)b11 -> c00 -= 1, c01 += 1
        # M6 = (a10 - a00)(b00 + b01) -> c11 += 1
        # M7 = (a01 - a11)(b10 + b11) -> c00 += 1
        
        # Note: The C coefficients in the terms above are the contributions to the 2x2 result block.
        # We must add these contributions to the global C block defined by offset_c.
        
        terms = [
            # (A_dict, B_dict, C_dict)
            # A_dict keys are (r,c) in 0..1
            ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
            ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
            ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
            ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
            ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
            ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
            ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1})
        ]
        
        for t_idx, (a_t, b_t, c_t) in enumerate(terms):
            col = base_col + t_idx
            
            # Update U
            for (r, c), val in a_t.items():
                # The coefficient for this A entry in the current Strassen term
                # is val * (coefficient from a_coeffs)
                # Wait, a_coeffs is the linear combination of the 2x2 blocks.
                # So the actual coefficient for A[global_r, global_c] is val * a_coeffs[(r,c)]
                # But a_coeffs is a dict. If (r,c) is not in a_coeffs, it's 0.
                # Actually, the input to add_strassen_2x2 is the linear combination of the 2x2 entries.
                # So a_coeffs[(r,c)] is the multiplier for the 2x2 entry (r,c).
                # The Strassen term uses a specific linear combination of the 2x2 entries.
                # So the total coefficient for the global entry is a_coeffs.get((r,c), 0) * val.
                
                if (r, c) in a_coeffs:
                    u_idx = (offset_a[0] + r) * 4 + (offset_a[1] + c)
                    U[u_idx][col] += a_coeffs[(r, c)] * val
            
            # Update V
            for (r, c), val in b_t.items():
                if (r, c) in b_coeffs:
                    v_idx = (offset_b[0] + r) * 4 + (offset_b[1] + c)
                    V[v_idx][col] += b_coeffs[(r, c)] * val
            
            # Update W
            for (r, c), val in c_t.items():
                if (r, c) in c_coeffs:
                    w_idx = (offset_c[0] + r) * 4 + (offset_c[1] + c)
                    W[w_idx][col] += c_coeffs[(r, c)] * val

    # Define the 2x2 blocks
    # A11: (0,0), A12: (0,2), A21: (2,0), A22: (2,2)
    # B11: (0,0), B12: (0,2), B21: (2,0), B22: (2,2)
    # C11: (0,0), C12: (0,2), C21: (2,0), C22: (2,2)
    
    # M1 = (A11 + A22)(B11 + B22)
    # A part: A11 + A22
    a_m1 = {}
    for r in range(2):
        for c in range(2):
            a_m1[(r,c)] = 1 # From A11
            # Also from A22, but we handle the sum by adding contributions?
            # No, add_strassen_2x2 takes the coefficients for the 2x2 entries.
            # The 2x2 entries are the sums.
            # So a_m1[(r,c)] should be the coefficient for the (r,c) entry of the SUM matrix.
            # The SUM matrix is A11 + A22.
            # Entry (r,c) of A11 is A[0+r, 0+c].
            # Entry (r,c) of A22 is A[2+r, 2+c].
            # These are different global indices.
            # So I cannot just pass a single dict a_m1 to add_strassen_2x2.
            # I need to call add_strassen_2x2 for each block contribution separately?
            # No, that would double the rank.
            # The Strassen algorithm is linear.
            # M1 = (A11+A22)(B11+B22) = A11B11 + A11B22 + A22B11 + A22B22.
            # This expansion leads to 4 products, not 1.
            # So I must use the Strassen algorithm on the SUM matrices directly.
            # But the SUM matrices are not contiguous 2x2 blocks in the same way?
            # A11 + A22 is a matrix S_A where S_A[r,c] = A[r,c] + A[r+2,c+2].
            # This is a valid 2x2 matrix.
            # So I can define the coefficients for S_A.
            # S_A[0,0] = A[0,0] + A[2,2].
            # S_A[0,1] = A[0,1] + A[2,3].
            # S_A[1,0] = A[1,0] + A[3,2].
            # S_A[1,1] = A[1,1] + A[3,3].
            
            # So, for M1:
            # a_coeffs for the 2x2 S_A:
            # (0,0) -> 1 (contribution to A[0,0] and A[2,2])
            # This doesn't fit the `add_strassen_2x2` signature which assumes a single block offset.
            
            # I need a more general helper.
            # Let's redefine the helper to take explicit global indices.
            
            pass

    # Let's rewrite the helper to be more flexible.
    # It takes lists of (global_idx, coeff) for A, B, C.
    # And it adds the 7 Strassen terms.
    
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(16)]
    
    def add_strassen_explicit(a_entries, b_entries, c_entries, base_col):
        # a_entries: list of (global_idx, coeff)
        # b_entries: list of (global_idx, coeff)
        # c_entries: list of (global_idx, coeff)
        # These define the 2x2 matrices involved in the product.
        # The 2x2 matrices are defined by their 4 entries.
        # We need to map the 2x2 positions (0,0), (0,1), (1,0), (1,1) to these global indices.
        # Actually, the Strassen terms are defined on the 2x2 positions.
        # So I should pass the mapping from 2x2 pos to global idx and coeff.
        
        # Let's pass:
        # a_map: {(0,0): [(idx, coeff), ...], (0,1): ..., (1,0): ..., (1,1): ...}
        # This is getting complicated.
        
        # Simpler: 
        # The 2x2 matrix A' is defined by A'[r,c] = sum_k coeff_k * A[idx_k].
        # The Strassen term uses A'[r,c].
        # So the coefficient for A[idx_k] in the term is term_val * coeff_k.
        
        # Let's define the 2x2 matrices for M1..M7 explicitly.
        
        pass

    # I will hardcode the 49 terms by expanding the Strassen formulas.
    # This is tedious but safe.
    
    # M1 = (A11+A22)(B11+B22)
    # Let S_A = A11+A22, S_B = B11+B22.
    # S_A[0,0] = A[0,0]+A[2,2], S_A[0,1] = A[0,1]+A[2,3], S_A[1,0] = A[1,0]+A[3,2], S_A[1,1] = A[1,1]+A[3,3].
    # S_B[0,0] = B[0,0]+B[2,2], S_B[0,1] = B[0,1]+B[2,3], S_B[1,0] = B[1,0]+B[3,2], S_B[1,1] = B[1,1]+B[3,3].
    
    # Strassen 2x2 terms for S_A * S_B -> C11.
    # C11[0,0] = S_A[0,0]S_B[0,0] + S_A[1,1]S_B[1,1] - S_A[0,1]S_B[1,1] + S_A[0,1]S_B[0,1] - S_A[1,0]S_B[0,1] + S_A[0,0]S_B[1,1] - S_A[1,1]S_B[0,0] ... wait.
    
    # Let's use the standard Strassen 2x2 expansion:
    # m1 = (a00+a11)(b00+b11)
    # m2 = (a10+a11)b00
    # m3 = a00(b01-b11)
    # m4 = a11(b10-b00)
    # m5 = (a00+a01)b11
    # m6 = (a10-a00)(b00+b01)
    # m7 = (a01-a11)(b10+b11)
    
    # c00 = m1 + m4 - m5 + m7
    # c01 = m3 + m5
    # c10 = m2 + m4
    # c11 = m1 - m2 + m3 + m6
    
    # I will write a function that takes the linear combinations for a00, a01, a10, a11 and b00..b11 and adds the 7 terms.
    
    def add_2x2_product(a_combos, b_combos, c_combos, base_col):
        # a_combos: list of 4 items, each is a list of (global_idx, coeff) for a00, a01, a10, a11
        # b_combos: list of 4 items, each is a list of (global_idx, coeff) for b00, b01, b10, b11
        # c_combos: list of 4 items, each is a list of (global_idx, coeff) for c00, c01, c10, c11
        # The c_combos define how the 2x2 result maps to global C.
        
        # a00, a01, a10, a11 are the 2x2 entries.
        # Strassen terms:
        # m1: a=(0,0)+(1,1), b=(0,0)+(1,1)
        # m2: a=(1,0)+(1,1), b=(0,0)
        # m3: a=(0,0), b=(0,1)-(1,1)
        # m4: a=(1,1), b=(1,0)-(0,0)
        # m5: a=(0,0)+(0,1), b=(1,1)
        # m6: a=(1,0)-(0,0), b=(0,0)+(0,1)
        # m7: a=(0,1)-(1,1), b=(1,0)+(1,1)
        
        # C contributions:
        # c00: m1, m4, -m5, m7
        # c01: m3, m5
        # c10: m2, m4
        # c11: m1, -m2, m3, m6
        
        # I will iterate through the 7 terms.
        
        # Define the 2x2 indices for A and B in each term.
        # a_idx: list of (pos, coeff)
        # b_idx: list of (pos, coeff)
        
        terms = [
            # (a_idx, b_idx, c_idx_contribs)
            # c_idx_contribs: list of (pos, coeff)
            ([((0,0),1), ((1,1),1)], [((0,0),1), ((1,1),1)], [((0,0),1), ((1,1),1)]), # m1
            ([((1,0),1), ((1,1),1)], [((0,0),1)], [((1,0),1), ((1,1),-1)]), # m2
            ([((0,0),1)], [((0,1),1), ((1,1),-1)], [((0,1),1), ((1,1),1)]), # m3
            ([((1,1),1)], [((1,0),1), ((0,0),-1)], [((0,0),1), ((1,0),1)]), # m4
            ([((0,0),1), ((0,1),1)], [((1,1),1)], [((0,0),-1), ((0,1),1)]), # m5
            ([((1,0),1), ((0,0),-1)], [((0,0),1), ((0,1),1)], [((1,1),1)]), # m6
            ([((0,1),1), ((1,1),-1)], [((1,0),1), ((1,1),1)], [((0,0),1)]) # m7
        ]
        
        for t_idx, (a_idx, b_idx, c_idx) in enumerate(terms):
            col = base_col + t_idx
            
            # Update U
            for (pos, val) in a_idx:
                # pos is (r,c) in 0..1
                # a_combos[pos] is list of (global_idx, coeff)
                for (g_idx, coeff) in a_combos[pos]:
                    U[g_idx][col] += val * coeff
            
            # Update V
            for (pos, val) in b_idx:
                for (g_idx, coeff) in b_combos[pos]:
                    V[g_idx][col] += val * coeff
            
            # Update W
            for (pos, val) in c_idx:
                for (g_idx, coeff) in c_combos[pos]:
                    W[g_idx][col] += val * coeff

    # Now, define the combinations for M1..M7.
    
    # M1: (A11+A22)(B11+B22) -> C11
    # A11: (0,0)->0, (0,1)->1, (1,0)->4, (1,1)->5
    # A22: (2,0)->8, (2,1)->9, (3,0)->12, (3,1)->13
    # S_A[0,0] = A[0]+A[8]
    # S_A[0,1] = A[1]+A[9]
    # S_A[1,0] = A[4]+A[12]
    # S_A[1,1] = A[5]+A[13]
    
    a_m1 = [
        [(0,1), (8,1)], # 0,0
        [(1,1), (9,1)], # 0,1
        [(4,1), (12,1)], # 1,0
        [(5,1), (13,1)] # 1,1
    ]
    
    b_m1 = [
        [(0,1), (8,1)], # 0,0
        [(1,1), (9,1)], # 0,1
        [(4,1), (12,1)], # 1,0
        [(5,1), (13,1)] # 1,1
    ]
    
    c_m1 = [
        [(0,1)], # C11[0,0] -> C[0]
        [(2,1)], # C11[0,1] -> C[2]
        [(8,1)], # C11[1,0] -> C[8]
        [(10,1)] # C11[1,1] -> C[10]
    ]
    
    add_2x2_product(a_m1, b_m1, c_m1, 0)
    
    # M2: (A21+A22)B11 -> C21, C22
    # A21: (2,0)->8, (2,1)->9, (3,0)->12, (3,1)->13
    # A22: (2,0)->8... wait.
    # A21 is bottom-left. A[2,0], A[2,1], A[3,0], A[3,1]. Indices: 8, 9, 12, 13.
    # A22 is bottom-right. A[2,2], A[2,3], A[3,2], A[3,3]. Indices: 10, 11, 14, 15.
    # S_A = A21 + A22.
    # S_A[0,0] = A[8]+A[10]
    # S_A[0,1] = A[9]+A[11]
    # S_A[1,0] = A[12]+A[14]
    # S_A[1,1] = A[13]+A[15]
    
    a_m2 = [
        [(8,1), (10,1)],
        [(9,1), (11,1)],
        [(12,1), (14,1)],
        [(13,1), (15,1)]
    ]
    
    # B11: (0,0)->0, (0,1)->1, (1,0)->4, (1,1)->5
    b_m2 = [
        [(0,1)],
        [(1,1)],
        [(4,1)],
        [(5,1)]
    ]
    
    # Result is C21 (bottom-left) and C22 (bottom-right)?
    # M2 contributes to C21 and C22.
    # C21 = M2 + M4.
    # C22 = M1 - M2 + M3 + M6.
    # So M2 contributes to C21 with +1 and C22 with -1.
    # C21 is at (2,0)..(3,1). Indices 8,9,12,13.
    # C22 is at (2,2)..(3,3). Indices 10,11,14,15.
    
    c_m2 = [
        [(8,1), (10,-1)], # C[0,0] of result -> C21[0,0] (+1) and C22[0,0] (-1)
        [(9,1), (11,-1)], # C[0,1] -> C21[0,1] (+1) and C22[0,1] (-1)
        [(12,1), (14,-1)], # C[1,0] -> C21[1,0] (+1) and C22[1,0] (-1)
        [(13,1), (15,-1)] # C[1,1] -> C21[1,1] (+1) and C22[1,1] (-1)
    ]
    
    add_2x2_product(a_m2, b_m2, c_m2, 7)
    
    # M3: A11(B12-B22) -> C12, C22
    # A11: 0,1,4,5
    a_m3 = [
        [(0,1)],
        [(1,1)],
        [(4,1)],
        [(5,1)]
    ]
    
    # B12-B22
    # B12: (0,2)->2, (0,3)->3, (1,2)->6, (1,3)->7
    # B22: (2,2)->10, (2,3)->11, (3,2)->14, (3,3)->15
    # S_B[0,0] = B[2]-B[10]
    # S_B[0,1] = B[3]-B[11]
    # S_B[1,0] = B[6]-B[14]
    # S_B[1,1] = B[7]-B[15]
    
    b_m3 = [
        [(2,1), (10,-1)],
        [(3,1), (11,-1)],
        [(6,1), (14,-1)],
        [(7,1), (15,-1)]
    ]
    
    # M3 contributes to C12 (+1) and C22 (+1).
    # C12: (0,2)..(1,3). Indices 2,3,6,7.
    # C22: (2,2)..(3,3). Indices 10,11,14,15.
    
    c_m3 = [
        [(2,1), (10,1)],
        [(3,1), (11,1)],
        [(6,1), (14,1)],
        [(7,1), (15,1)]
    ]
    
    add_2x2_product(a_m3, b_m3, c_m3, 14)
    
    # M4: A22(B21-B11) -> C11, C21
    # A22: 10,11,14,15
    a_m4 = [
        [(10,1)],
        [(11,1)],
        [(14,1)],
        [(15,1)]
    ]
    
    # B21-B11
    # B21: (2,0)->8, (2,1)->9, (3,0)->12, (3,1)->13
    # B11: (0,0)->0, (0,1)->1, (1,0)->4, (1,1)->5
    # S_B[0,0] = B[8]-B[0]
    # S_B[0,1] = B[9]-B[1]
    # S_B[1,0] = B[12]-B[4]
    # S_B[1,1] = B[13]-B[5]
    
    b_m4 = [
        [(8,1), (0,-1)],
        [(9,1), (1,-1)],
        [(12,1), (4,-1)],
        [(13,1), (5,-1)]
    ]
    
    # M4 contributes to C11 (+1) and C21 (+1).
    # C11: 0,2,8,10
    # C21: 8,9,12,13
    
    c_m4 = [
        [(0,1), (8,1)],
        [(2,1), (9,1)],
        [(8,1), (12,1)],
        [(10,1), (13,1)]
    ]
    
    add_2x2_product(a_m4, b_m4, c_m4, 21)
    
    # M5: (A11+A12)B22 -> C11, C12
    # A11+A12
    # A11: 0,1,4,5
    # A12: (0,2)->2, (0,3)->3, (1,2)->6, (1,3)->7
    # S_A[0,0] = A[0]+A[2]
    # S_A[0,1] = A[1]+A[3]
    # S_A[1,0] = A[4]+A[6]
    # S_A[1,1] = A[5]+A[7]
    
    a_m5 = [
        [(0,1), (2,1)],
        [(1,1), (3,1)],
        [(4,1), (6,1)],
        [(5,1), (7,1)]
    ]
    
    # B22: 10,11,14,15
    b_m5 = [
        [(10,1)],
        [(11,1)],
        [(14,1)],
        [(15,1)]
    ]
    
    # M5 contributes to C11 (-1) and C12 (+1).
    # C11: 0,2,8,10
    # C12: 2,3,6,7
    
    c_m5 = [
        [(0,-1), (2,1)],
        [(2,-1), (3,1)],
        [(8,-1), (6,1)],
        [(10,-1), (7,1)]
    ]
    
    add_2x2_product(a_m5, b_m5, c_m5, 28)
    
    # M6: (A21-A11)(B11+B12) -> C22
    # A21-A11
    # A21: 8,9,12,13
    # A11: 0,1,4,5
    # S_A[0,0] = A[8]-A[0]
    # S_A[0,1] = A[9]-A[1]
    # S_A[1,0] = A[12]-A[4]
    # S_A[1,1] = A[13]-A[5]
    
    a_m6 = [
        [(8,1), (0,-1)],
        [(9,1), (1,-1)],
        [(12,1), (4,-1)],
        [(13,1), (5,-1)]
    ]
    
    # B11+B12
    # B11: 0,1,4,5
    # B12: 2,3,6,7
    # S_B[0,0] = B[0]+B[2]
    # S_B[0,1] = B[1]+B[3]
    # S_B[1,0] = B[4]+B[6]
    # S_B[1,1] = B[5]+B[7]
    
    b_m6 = [
        [(0,1), (2,1)],
        [(1,1), (3,1)],
        [(4,1), (6,1)],
        [(5,1), (7,1)]
    ]
    
    # M6 contributes to C22 (+1).
    # C22: 10,11,14,15
    
    c_m6 = [
        [(10,1)],
        [(11,1)],
        [(14,1)],
        [(15,1)]
    ]
    
    add_2x2_product(a_m6, b_m6, c_m6, 35)
    
    # M7: (A12-A22)(B21+B22) -> C11
    # A12-A22
    # A12: 2,3,6,7
    # A22: 10,11,14,15
    # S_A[0,0] = A[2]-A[10]
    # S_A[0,1] = A[3]-A[11]
    # S_A[1,0] = A[6]-A[14]
    # S_A[1,1] = A[7]-A[15]
    
    a_m7 = [
        [(2,1), (10,-1)],
        [(3,1), (11,-1)],
        [(6,1), (14,-1)],
        [(7,1), (15,-1)]
    ]
    
    # B21+B22
    # B21: 8,9,12,13
    # B22: 10,11,14,15
    # S_B[0,0] = B[8]+B[10]
    # S_B[0,1] = B[9]+B[11]
    # S_B[1,0] = B[12]+B[14]
    # S_B[1,1] = B[13]+B[15]
    
    b_m7 = [
        [(8,1), (10,1)],
        [(9,1), (11,1)],
        [(12,1), (14,1)],
        [(13,1), (15,1)]
    ]
    
    # M7 contributes to C11 (+1).
    # C11: 0,2,8,10
    
    c_m7 = [
        [(0,1)],
        [(2,1)],
        [(8,1)],
        [(10,1)]
    ]
    
    add_2x2_product(a_m7, b_m7, c_m7, 42)
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
