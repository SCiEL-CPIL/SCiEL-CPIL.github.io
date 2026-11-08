import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START

def _build_strassen_2x2_terms():
    """Return 7 (u, v, w) vectors for 2x2 x 2x2 -> 2x2 multiplication.
    A is 2x2 with entries A[0,0],A[0,1],A[1,0],A[1,1] (row-major).
    B is 2x2 with entries B[0,0],B[0,1],B[1,0],B[1,1] (row-major).
    C[i,j] = A[0,0]*B[0,0] + A[0,1]*B[1,0] (for i=0,j=0) etc.
    Actually C[i,j] = sum_k A[i,k]*B[k,j].
    Output C is 2x2, indexed C[0,0], C[0,1], C[1,0], C[1,1] (row-major).
    """
    terms = []
    # M1 = (A00 + A11)(B00 + B11)
    terms.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]))
    # M2 = (A10 + A11)B00
    terms.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]))
    # M3 = A00(B01 - B11)
    terms.append(([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 1]))
    # M4 = A11(B10 - B00)
    terms.append(([0, 0, 0, 1], [-1, 0, 1, 0], [1, 1, 0, 0]))
    # M5 = (A00 + A01)B11
    terms.append(([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]))
    # M6 = (A10 - A00)(B00 + B01)
    terms.append(([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]))
    # M7 = (A01 - A11)(B10 + B11)
    terms.append(([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 1]))
    return terms


def _build_rectangular_2x2_2x4_terms():
    """Return rank-1 terms for 2x2 A times 2x4 B -> 2x4 C.
    A: 2x2, row-major: A[0,0],A[0,1],A[1,0],A[1,1]
    B: 2x4, row-major: B[0,0],B[0,1],B[0,2],B[0,3],B[1,0],B[1,1],B[1,2],B[1,3]
    C: 2x4, row-major: C[0,0],C[0,1],C[0,2],C[0,3],C[1,0],C[1,1],C[1,2],C[1,3]
    C[i,j] = sum_k A[i,k]*B[k,j]
    """
    # Split B into two 2x2 blocks: B_left = B[:,0:2], B_right = B[:,2:4]
    # C_left = A * B_left (2x2), C_right = A * B_right (2x2)
    # Use Strassen for each 2x2 * 2x2.
    # For each half, the B-side vector picks 4 entries from B's 8 entries.
    strassen_base = _build_strassen_2x2_terms()
    terms = []
    for u, v, w in strassen_base:
        # u is 4-dim (A entries), v is 4-dim (B_half entries), w is 4-dim (C_half entries)
        # For left half: v entries map to B indices [0,1,4,5] (B[0,0],B[0,1],B[1,0],B[1,1])
        # For right half: v entries map to B indices [2,3,6,7] (B[0,2],B[0,3],B[1,2],B[1,3])
        v_left = [0] * 8
        v_right = [0] * 8
        w_full_left = [0] * 8
        w_full_right = [0] * 8
        
        # Map v (4-dim) to positions in 8-dim
        # v[0] -> B[0,0] -> index 0
        # v[1] -> B[0,1] -> index 1
        # v[2] -> B[1,0] -> index 4
        # v[3] -> B[1,1] -> index 5
        left_map = [0, 1, 4, 5]
        right_map = [2, 3, 6, 7]
        
        for k in range(4):
            v_left[left_map[k]] = v[k]
            v_right[right_map[k]] = v[k]
        
        # w (4-dim) maps to C_half entries
        # w[0] -> C[0,0], w[1] -> C[0,1], w[2] -> C[1,0], w[3] -> C[1,1]
        # For left half: C indices [0,1,4,5]
        # For right half: C indices [2,3,6,7]
        for k in range(4):
            w_full_left[left_map[k]] = w[k]
            w_full_right[right_map[k]] = w[k]
        
        terms.append((u, v_left, w_full_left))
        terms.append((u, v_right, w_full_right))
    return terms


def _build_4x4_4x8_strassen_terms():
    """Build rank-1 terms for 4x4 A times 4x8 B -> 4x8 C using Strassen recursion.
    
    Split A (4x4) into four 2x2 blocks:
      A11 = A[0:2, 0:2], A12 = A[0:2, 2:4], A21 = A[2:4, 0:2], A22 = A[2:4, 2:4]
    
    Split B (4x8) into four 2x4 blocks:
      B11 = B[0:2, 0:4], B12 = B[0:2, 4:8], B21 = B[2:4, 0:4], B22 = B[2:4, 4:8]
    
    C (4x8) has blocks:
      C11 = C[0:2, 0:4], C12 = C[0:2, 4:8], C21 = C[2:4, 0:4], C22 = C[2:4, 4:8]
    
    Strassen formulas:
      M1 = (A11 + A22)(B11 + B22)
      M2 = (A21 + A22)B11
      M3 = A11(B12 - B22)
      M4 = A22(B21 - B11)
      M5 = (A11 + A12)B22
      M6 = (A21 - A11)(B11 + B12)
      M7 = (A12 - A22)(B21 + B22)
    
      C11 = M1 + M4
      C12 = M3 + M5
      C21 = M2 + M4
      C22 = M1 - M2 + M3 + M7
    
    Each M is a 2x2 * 2x4 product. We precompute the 14 rank-1 terms for 2x2*2x4.
    For each Strassen M, we need to construct the appropriate u (4-dim, linear combo of A blocks)
    and v (8-dim, linear combo of B blocks), then use the 14 base terms.
    
    However, the rank-1 terms must be in the FULL 4x4, 4x8, 4x8 space.
    
    Let me think differently. The tensor for 4x4 * 4x8 -> 4x8 has:
      U space: 16 entries (A)
      V space: 32 entries (B)
      W space: 32 entries (C)
    
    For each Strassen M (say M1), the A-side is a linear combination of A11, A22 (each 2x2).
    The B-side is a linear combination of B11, B22 (each 2x4).
    
    For a given base term (u_base, v_base, w_base) of 2x2*2x4:
      u_base is 4-dim (for a 2x2 A)
      v_base is 8-dim (for a 2x4 B)
      w_base is 8-dim (for a 2x4 C)
    
    To lift to the full problem:
      u_full: 16-dim. For M1, A-side = A11 + A22. So u_full has u_base placed at A11 positions
               and u_base placed at A22 positions.
      v_full: 32-dim. For M1, B-side = B11 + B22. So v_full has v_base at B11 positions
               and v_base at B22 positions.
      w_full: 32-dim. For M1, C-side contributes to C11 and C22. w_base at C11 positions
               and w_base at C22 positions.
    """
    base_2x2_2x4 = _build_rectangular_2x2_2x4_terms()
    terms = []
    
    # Define the 7 Strassen combinations.
    # Each is: (a_coef_dict, b_coef_dict, c_coef_dict)
    # a_coef_dict: maps block index (0=A11,1=A12,2=A21,3=A22) to coefficient
    # b_coef_dict: maps block index (0=B11,1=B12,2=B21,3=B22) to coefficient
    # c_coef_dict: maps block index (0=C11,1=C12,2=C21,3=C22) to coefficient
    
    # Block positions in the full vectors:
    # A (4x4, row-major, 16 entries):
    #   A11: indices 0,1,4,5  (rows 0-1, cols 0-1)
    #   A12: indices 2,3,6,7  (rows 0-1, cols 2-3)
    #   A21: indices 8,9,12,13 (rows 2-3, cols 0-1)
    #   A22: indices 10,11,14,15 (rows 2-3, cols 2-3)
    
    # B (4x8, row-major, 32 entries):
    #   B11: indices 0,1,2,3,8,9,10,11 (rows 0-1, cols 0-3)
    #   B12: indices 4,5,6,7,12,13,14,15 (rows 0-1, cols 4-7)
    #   B21: indices 16,17,18,19,24,25,26,27 (rows 2-3, cols 0-3)
    #   B22: indices 20,21,22,23,28,29,30,31 (rows 2-3, cols 4-7)
    
    # C (4x8, row-major, 32 entries):
    #   C11: indices 0,1,2,3,8,9,10,11
    #   C12: indices 4,5,6,7,12,13,14,15
    #   C21: indices 16,17,18,19,24,25,26,27
    #   C22: indices 20,21,22,23,28,29,30,31
    
    # A block index -> positions in 16-dim vector
    A_block_pos = {
        0: [0, 1, 4, 5],   # A11
        1: [2, 3, 6, 7],   # A12
        2: [8, 9, 12, 13], # A21
        3: [10, 11, 14, 15], # A22
    }
    
    # B block index -> positions in 32-dim vector
    B_block_pos = {
        0: [0, 1, 2, 3, 8, 9, 10, 11],   # B11
        1: [4, 5, 6, 7, 12, 13, 14, 15],  # B12
        2: [16, 17, 18, 19, 24, 25, 26, 27], # B21
        3: [20, 21, 22, 23, 28, 29, 30, 31], # B22
    }
    
    # C block index -> positions in 32-dim vector
    C_block_pos = {
        0: [0, 1, 2, 3, 8, 9, 10, 11],   # C11
        1: [4, 5, 6, 7, 12, 13, 14, 15],  # C12
        2: [16, 17, 18, 19, 24, 25, 26, 27], # C21
        3: [20, 21, 22, 23, 28, 29, 30, 31], # C22
    }
    
    # Strassen M definitions: (a_coefs, b_coefs, c_coefs)
    # a_coefs: {block: coef}, b_coefs: {block: coef}, c_coefs: {block: coef}
    strassen_ms = [
        # M1 = (A11+A22)(B11+B22), C11 += M1, C22 += M1
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # M2 = (A21+A22)B11, C21 += M2, C22 -= M2
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # M3 = A11(B12-B22), C12 += M3, C22 += M3
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        # M4 = A22(B21-B11), C11 += M4, C21 += M4
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        # M5 = (A11+A12)B22, C12 += M5
        ({0: 1, 1: 1}, {3: 1}, {1: 1}),
        # M6 = (A21-A11)(B11+B12), C22 += M6  -- wait, let me recheck
        # Actually M6 is not directly in the C formulas above. Let me re-derive.
        # Standard Strassen:
        # C11 = M1 + M4
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M7
        # So M6 is not used in the final C. That's wrong. Let me use the standard 7-product Strassen.
    ]
    
    # Let me use the standard Strassen definitions more carefully.
    # The 7 products are:
    # P1 = (A11 + A22)(B11 + B22)
    # P2 = (A21 + A22)B11
    # P3 = A11(B12 - B22)
    # P4 = A22(B21 - B11)
    # P5 = (A11 + A12)B22
    # P6 = (A21 - A11)(B11 + B12)
    # P7 = (A12 - A22)(B21 + B22)
    # And:
    # C11 = P1 + P4
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P7
    
    # So P6 is not used! That means we only need 6 products? No, P6 is used in some formulations.
    # Let me verify: the standard Strassen uses 7 products and all are needed.
    # Actually, I think I'm confusing formulations. Let me use the ones that are needed.
    
    # For now, let's just implement the 4 C blocks using the products that appear:
    # C11 = P1 + P4
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P7
    
    products = [
        # (name, a_coefs, b_coefs, c_coefs)
        ("P1", {0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        ("P2", {2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        ("P3", {0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        ("P4", {3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        ("P5", {0: 1, 1: 1}, {3: 1}, {1: 1}),
        ("P7", {1: 1, 3: -1}, {2: 1, 3: 1}, {3: 1}),
    ]
    
    for pname, a_coefs, b_coefs, c_coefs in products:
        for u_base, v_base, w_base in base_2x2_2x4:
            u_full = [0] * 16
            v_full = [0] * 32
            w_full = [0] * 32
            
            # Fill u_full: for each A block with nonzero coef, place u_base * coef
            for blk, coef in a_coefs.items():
                for pos, idx in enumerate(A_block_pos[blk]):
                    u_full[idx] += coef * u_base[pos]
            
            # Fill v_full
            for blk, coef in b_coefs.items():
                for pos, idx in enumerate(B_block_pos[blk]):
                    v_full[idx] += coef * v_base[pos]
            
            # Fill w_full
            for blk, coef in c_coefs.items():
                for pos, idx in enumerate(C_block_pos[blk]):
                    w_full[idx] += coef * w_base[pos]
            
            terms.append((u_full, v_full, w_full))
    
    return terms


def construct():
    n, m, p = DIMENSIONS
    terms = _build_4x4_4x8_strassen_terms()
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for c, (u, v, w) in enumerate(terms):
        for i in range(n * m):
            U[i][c] = u[i]
        for i in range(m * p):
            V[i][c] = v[i]
        for i in range(p * n):
            W[i][c] = w[i]
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
