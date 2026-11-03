"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START

def get_strassen_2x2_decomp():
    """
    Returns U, V, W for 2x2x2 Strassen decomposition.
    U, V, W are lists of 7 vectors of length 4.
    """
    # u_r, v_r, w_r for r=1..7
    # A indices: 00, 01, 10, 11
    # B indices: 00, 01, 10, 11
    # C indices: 00, 01, 10, 11
    
    decomp = []
    
    # M1 = (A00 + A11)(B00 + B11)
    # u: [1, 0, 0, 1]
    # v: [1, 0, 0, 1]
    # w: [1, 0, 0, -1] (C00 += M1, C11 -= M1)
    decomp.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, -1]))
    
    # M2 = (A10 + A11)B00
    # u: [0, 0, 1, 1]
    # v: [1, 0, 0, 0]
    # w: [0, 0, 1, -1] (C10 += M2, C11 -= M2)
    decomp.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]))
    
    # M3 = A00(B01 - B11)
    # u: [1, 0, 0, 0]
    # v: [0, 1, 0, -1]
    # w: [0, 1, 0, 1] (C01 += M3, C11 += M3)
    decomp.append(([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]))
    
    # M4 = A11(B10 - B00)
    # u: [0, 0, 0, 1]
    # v: [0, 0, 1, -1]
    # w: [1, 0, 1, 0] (C00 += M4, C10 += M4)
    decomp.append(([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]))
    
    # M5 = (A00 + A10)B11
    # u: [1, 0, 1, 0]
    # v: [0, 0, 0, 1]
    # w: [-1, 1, 0, 0] (C00 -= M5, C01 += M5)
    decomp.append(([1, 0, 1, 0], [0, 0, 0, 1], [-1, 1, 0, 0]))
    
    # M6 = (A01 - A00)(B00 + B10)
    # u: [-1, 1, 0, 0]
    # v: [1, 0, 1, 0]
    # w: [0, 0, 0, 1] (C11 += M6)
    decomp.append(([ -1, 1, 0, 0], [1, 0, 1, 0], [0, 0, 0, 1]))
    
    # M7 = (A11 - A01)(B01 + B11)
    # u: [0, -1, 0, 1]
    # v: [0, 1, 0, 1]
    # w: [1, 0, 0, 0] (C00 += M7)
    decomp.append(([0, -1, 0, 1], [0, 1, 0, 1], [1, 0, 0, 0]))
    
    return decomp

def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=6
    
    # Total rank estimate: 49 (for 4x4x4) + 28 (for 4x4x2) = 77
    rank = 77
    
    U = [[0] * rank for _ in range(n * m)] # 16 x 77
    V = [[0] * rank for _ in range(m * p)] # 24 x 77
    W = [[0] * rank for _ in range(p * n)] # 24 x 77
    
    col = 0
    
    strassen_decomp = get_strassen_2x2_decomp()
    
    # --- Part 1: 4x4x4 Multiplication (A * B1) ---
    # B1 is the first 4 columns of B (cols 0-3)
    # A is 4x4, B1 is 4x4, C1 is 4x4
    
    # Strassen terms for 4x4x4
    # We iterate through the 7 Strassen multiplications of 2x2 blocks
    # Each is a 2x2x2 problem.
    
    # Define the block linear combinations for A and B
    # A blocks: 00, 01, 10, 11
    # B blocks: 00, 01, 10, 11
    
    # Term 1: (A00 + A11)(B00 + B11)
    # A_s = A00 + A11
    # B_s = B00 + B11
    # C_s contributes to C00 (+1) and C11 (-1)
    
    strassen_terms = [
        # (a_comb, b_comb, c_targets)
        # a_comb: list of (block_idx, coeff) for A
        # b_comb: list of (block_idx, coeff) for B
        # c_targets: list of (block_idx, coeff) for C
        ([ (0, 1), (3, 1) ], [ (0, 1), (3, 1) ], [ (0, 1), (3, -1) ]), # M1
        ([ (2, 1), (3, 1) ], [ (0, 1) ], [ (2, 1), (3, -1) ]),           # M2
        ([ (0, 1) ], [ (1, 1), (3, -1) ], [ (1, 1), (3, 1) ]),           # M3
        ([ (3, 1) ], [ (2, 1), (0, -1) ], [ (0, 1), (2, 1) ]),           # M4
        ([ (0, 1), (2, 1) ], [ (3, 1) ], [ (0, -1), (1, 1) ]),           # M5
        ([ (1, 1), (0, -1) ], [ (0, 1), (2, 1) ], [ (3, 1) ]),           # M6
        ([ (3, 1), (1, -1) ], [ (1, 1), (3, 1) ], [ (0, 1) ]),           # M7
    ]
    
    for a_comb, b_comb, c_targets in strassen_terms:
        # For each of the 7 local ranks in the 2x2x2 decomp
        for u_loc, v_loc, w_loc in strassen_decomp:
            # u_loc, v_loc, w_loc are length 4
            
            # Expand u_loc to global U vector (size 16)
            u_global = [0] * 16
            for l in range(4):
                lr = l // 2
                lc = l % 2
                coeff = u_loc[l]
                if coeff == 0: continue
                for blk, a_c in a_comb:
                    I = blk // 2
                    K = blk % 2
                    # Global A index: row = 2*I + lr, col = 2*K + lc
                    # Flat index: (2*I + lr) * 4 + (2*K + lc)
                    idx = (2*I + lr) * 4 + (2*K + lc)
                    u_global[idx] += coeff * a_c
            
            # Expand v_loc to global V vector (size 16 for B1 part, but V is size 24)
            # B1 occupies cols 0-3 of B.
            # Global B index: row = k, col = j.
            # B1 is B[0:4, 0:4].
            # So V indices 0-15 correspond to B1.
            v_global = [0] * 24
            for l in range(4):
                lr = l // 2
                lc = l % 2
                coeff = v_loc[l]
                if coeff == 0: continue
                for blk, b_c in b_comb:
                    K = blk // 2
                    J = blk % 2
                    # Global B index: row = 2*K + lr, col = 2*J + lc
                    # Flat index: (2*K + lr) * 4 + (2*J + lc)
                    # Note: m=4, p=6. B is 4x6.
                    # Index in V: k * p + j.
                    idx = (2*K + lr) * 6 + (2*J + lc)
                    v_global[idx] += coeff * b_c
            
            # Expand w_loc to global W vector (size 16 for C1 part, but W is size 24)
            # C1 is C[0:4, 0:4].
            # W index: k * n + i? No, W[k*n+i, r].
            # C is 4x6.
            # C1 is C[0:4, 0:4].
            # Indices in W: j * n + i.
            w_global = [0] * 24
            for l in range(4):
                lr = l // 2
                lc = l % 2
                coeff = w_loc[l]
                if coeff == 0: continue
                for blk, c_c in c_targets:
                    I = blk // 2
                    J = blk % 2
                    # Global C index: row = 2*I + lr, col = 2*J + lc
                    # Flat index in W: col * n + row = (2*J + lc) * 4 + (2*I + lr)
                    idx = (2*J + lc) * 4 + (2*I + lr)
                    w_global[idx] += coeff * c_c
            
            # Add to U, V, W at column col
            for r in range(16):
                U[r][col] = u_global[r]
            for r in range(24):
                V[r][col] = v_global[r]
            for r in range(24):
                W[r][col] = w_global[r]
            col += 1
    
    # --- Part 2: 4x4x2 Multiplication (A * B2) ---
    # B2 is the last 2 columns of B (cols 4-5)
    # A is 4x4, B2 is 4x2, C2 is 4x2
    
    # Split A into Atop (2x4) and Abot (2x4)
    # C2 = [ Ct ; Cb ]
    # Ct = At * B2 (2x2)
    # Cb = Ab * B2 (2x2)
    
    # Each is 2x4x2. Split middle dim 4 into 2+2.
    # At = [ At0 | At1 ] (2x2)
    # B2 = [ B20 ; B21 ] (2x2)
    # Ct = At0 B20 + At1 B21
    
    # We need to handle the two halves (Top and Bot) separately.
    # They are disjoint in the output rows.
    
    for half in range(2): # 0 for Top, 1 for Bot
        # A_half is rows 2*half to 2*half+1
        # B2 is rows 0 to 3, cols 4 to 5
        
        # Split B2 into B20 (rows 0-1) and B21 (rows 2-3)
        # B20 is 2x2, B21 is 2x2
        
        # At_half is 2x4. Split into At_half_0 (cols 0-1) and At_half_1 (cols 2-3).
        
        # Term 1: At_half_0 * B20
        # Term 2: At_half_1 * B21
        
        for sub_part in range(2): # 0 for left half of A, 1 for right half
            # A_sub is 2x2
            # B_sub is 2x2
            
            # Use 2x2x2 Strassen decomp
            for u_loc, v_loc, w_loc in strassen_decomp:
                u_global = [0] * 16
                v_global = [0] * 24
                w_global = [0] * 24
                
                for l in range(4):
                    lr = l // 2
                    lc = l % 2
                    coeff = u_loc[l]
                    if coeff == 0: continue
                    
                    # A_sub is A[2*half : 2*half+2, 2*sub_part : 2*sub_part+2]
                    # Global A row: 2*half + lr
                    # Global A col: 2*sub_part + lc
                    idx = (2*half + lr) * 4 + (2*sub_part + lc)
                    u_global[idx] += coeff
                    
                for l in range(4):
                    lr = l // 2
                    lc = l % 2
                    coeff = v_loc[l]
                    if coeff == 0: continue
                    
                    # B_sub is B2[2*sub_part : 2*sub_part+2, 4 : 6] ?
                    # B20 is B[0:2, 4:6]. B21 is B[2:4, 4:6].
                    # sub_part 0 -> rows 0-1. sub_part 1 -> rows 2-3.
                    # Global B row: 2*sub_part + lr
                    # Global B col: 4 + lc
                    idx = (2*sub_part + lr) * 6 + (4 + lc)
                    v_global[idx] += coeff
                    
                for l in range(4):
                    lr = l // 2
                    lc = l % 2
                    coeff = w_loc[l]
                    if coeff == 0: continue
                    
                    # C_sub is C[2*half : 2*half+2, 4 : 6]
                    # Global C row: 2*half + lr
                    # Global C col: 4 + lc
                    idx = (4 + lc) * 4 + (2*half + lr)
                    w_global[idx] += coeff
                
                for r in range(16):
                    U[r][col] = u_global[r]
                for r in range(24):
                    V[r][col] = v_global[r]
                for r in range(24):
                    W[r][col] = w_global[r]
                col += 1
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
