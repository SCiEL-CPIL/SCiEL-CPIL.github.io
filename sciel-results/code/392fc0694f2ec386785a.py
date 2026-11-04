import numpy as np

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = 4, 4, 7
    
    # Strassen 2x2x2 multiplication
    # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b12-b22)
    # M4 = a22(b21-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Each term is (u_indices, u_vals, v_indices, v_vals, w_indices, w_vals)
    # For 2x2x2: u is 4-dim (A), v is 4-dim (B), w is 4-dim (C)
    strassen_terms = [
        # M1 = (a11+a22)(b11+b22)
        ([(0,1), (2,1)], [1,1], [(0,1), (1,1)], [1,1], [(0,1), (3,1)], [1,1]),  # C11 += M1, C22 += M1
        # M2 = (a21+a22)b11
        ([(2,1), (3,1)], [1,1], [(0,1)], [1], [(2,1)], [1]),  # C21 += M2, C22 -= M2
        # M3 = a11(b12-b22)
        ([(0,1)], [1], [(1,1), (3,-1)], [1,-1], [(1,1), (3,1)], [1,1]),  # C12 += M3, C22 += M3
        # M4 = a22(b21-b11)
        ([(3,1)], [1], [(2,1), (0,-1)], [1,-1], [(0,1), (2,1)], [1,1]),  # C11 += M4, C21 += M4
        # M5 = (a11+a12)b22
        ([(0,1), (1,1)], [1,1], [(3,1)], [1], [(0,-1), (1,1)], [-1,1]),  # C11 -= M5, C12 += M5
        # M6 = (a21-a11)(b11+b12)
        ([(2,1), (0,-1)], [1,-1], [(0,1), (1,1)], [1,1], [(3,1)], [1]),  # C22 += M6
        # M7 = (a12-a22)(b21+b22)
        ([(1,1), (3,-1)], [1,-1], [(2,1), (3,1)], [1,1], [(0,1)], [1]),  # C11 += M7
    ]
    
    # Trivial 2x2x1 multiplication
    # A = [[a11, a12], [a21, a22]], B = [[b1], [b2]]
    # C1 = a11*b1 + a12*b2
    # C2 = a21*b1 + a22*b2
    # Rank 4 terms:
    trivial_2x2x1_terms = [
        # a11*b1
        ([(0,1)], [1], [(0,1)], [1], [(0,1)], [1]),
        # a12*b2
        ([(1,1)], [1], [(1,1)], [1], [(0,1)], [1]),
        # a21*b1
        ([(2,1)], [1], [(0,1)], [1], [(1,1)], [1]),
        # a22*b2
        ([(3,1)], [1], [(1,1)], [1], [(1,1)], [1]),
    ]
    
    # For 4x4x7, we split:
    # A is 4x4, B is 4x7
    # Split A into 4 blocks of 2x2 (rows 0-1, 2-3; cols 0-1, 2-3)
    # Split B into 7 columns: cols 0-1 use Strassen 2x2x2, cols 2-3 use Strassen 2x2x2, cols 4-6 use trivial 2x2x1
    # Actually, let's split B into: first 4 cols (2x2x2 blocks) and last 3 cols (2x2x1 blocks)
    # B = [B1 (4x4), B2 (4x3)]
    # C = [C1 (4x4), C2 (4x3)]
    # C1 = A * B1: use 4 Strassen 2x2x2 blocks (each 2x2 of A times 2x2 of B1)
    # C2 = A * B2: use 4 trivial 2x2x1 blocks (each 2x2 of A times 1 col of B2, 3 cols)
    
    # Let's be more careful with the block structure.
    # A is 4x4: A = [[A11, A12], [A21, A22]] where each Ai is 2x2
    # B is 4x7: B = [B1, B2] where B1 is 4x4, B2 is 4x3
    # B1 = [[B11, B12], [B21, B22]] where each Bij is 2x2
    # B2 = [[B21_1, B22_1], [B21_2, B22_2], [B21_3, B22_3]] where each is 2x1
    # C = [C1, C2] where C1 is 4x4, C2 is 4x3
    # C1 = A * B1 = [[A11*B11 + A12*B21, A11*B12 + A12*B22], [A21*B11 + A22*B21, A21*B12 + A22*B22]]
    # C2 = A * B2
    
    # For C1: 4 Strassen 2x2x2 multiplications
    # For C2: 3 trivial 2x2x1 multiplications (each with 4 terms)
    
    # Total rank: 4*7 + 3*4 = 28 + 12 = 40
    
    all_terms = []
    
    # Helper to add a term with offset
    def add_term(base_u, base_v, base_w, u_offset, v_offset, w_offset):
        u_indices = [idx + u_offset for idx in base_u]
        v_indices = [idx + v_offset for idx in base_v]
        w_indices = [idx + w_offset for idx in base_w]
        all_terms.append((u_indices, base_v_vals, v_indices, base_w_vals, w_indices, base_w_w))
    
    # This is getting complex. Let me use a different approach:
    # Directly construct the terms for 4x4x7 using block Strassen.
    
    # Let's define the terms more carefully.
    # A indices: 0-15 (row-major, 4x4)
    # B indices: 0-27 (row-major, 4x7)
    # C indices: 0-27 (row-major, 7x4)
    
    # Strassen 2x2x2: A is 2x2 (4 entries), B is 2x2 (4 entries), C is 2x2 (4 entries)
    # We need to map these to the global indices.
    
    # For C1 (4x4 part of C):
    # C1 = [[C11, C12], [C21, C22]] where each is 2x2
    # C11 = A11*B11 + A12*B21 (both 2x2x2)
    # C12 = A11*B12 + A12*B22 (both 2x2x2)
    # C21 = A21*B11 + A22*B21 (both 2x2x2)
    # C22 = A21*B12 + A22*B22 (both 2x2x2)
    
    # So we need 8 Strassen 2x2x2 multiplications for C1.
    # Each has rank 7, so total 56 for C1.
    
    # For C2 (4x3 part of C):
    # C2 = A * B2 where B2 is 4x3
    # Split A into 2x2 blocks: A11, A12, A21, A22
    # B2 has 3 columns: B2_col0, B2_col1, B2_col2 (each 4x1)
    # C2_col0 = A * B2_col0
    # C2_col1 = A * B2_col1
    # C2_col2 = A * B2_col2
    
    # For each column of C2, we use trivial 2x2x1 multiplication.
    # A * B_col where B_col is 4x1:
    # Split A into 2x2 blocks, B_col into 2x1 blocks
    # C_col = [[A11*B1 + A12*B2], [A21*B1 + A22*B2]] where B1, B2 are 2x1
    # This is 4 trivial 2x2x1 multiplications per column.
    # 3 columns * 4 = 12 terms.
    
    # Total rank: 56 + 12 = 68.
    
    # Let me implement this.
    
    def strassen_2x2x2_terms():
        # Returns list of (u_indices, u_vals, v_indices, v_vals, w_indices, w_vals)
        # for 2x2x2 Strassen, where indices are 0-3 for A, B, C
        return [
            # M1 = (a0+a3)(b0+b3) -> c0 += 1, c3 += 1
            ([(0,1), (3,1)], [1,1], [(0,1), (3,1)], [1,1], [(0,1), (3,1)], [1,1]),
            # M2 = (a2+a3)b0 -> c2 += 1, c3 -= 1
            ([(2,1), (3,1)], [1,1], [(0,1)], [1], [(2,1), (3,-1)], [1,-1]),
            # M3 = a0(b1-b3) -> c1 += 1, c3 += 1
            ([(0,1)], [1], [(1,1), (3,-1)], [1,-1], [(1,1), (3,1)], [1,1]),
            # M4 = a3(b2-b0) -> c0 += 1, c2 += 1
            ([(3,1)], [1], [(2,1), (0,-1)], [1,-1], [(0,1), (2,1)], [1,1]),
            # M5 = (a0+a1)b3 -> c0 -= 1, c1 += 1
            ([(0,1), (1,1)], [1,1], [(3,1)], [1], [(0,-1), (1,1)], [-1,1]),
            # M6 = (a2-a0)(b0+b1) -> c3 += 1
            ([(2,1), (0,-1)], [1,-1], [(0,1), (1,1)], [1,1], [(3,1)], [1]),
            # M7 = (a1-a3)(b2+b3) -> c0 += 1
            ([(1,1), (3,-1)], [1,-1], [(2,1), (3,1)], [1,1], [(0,1)], [1]),
        ]
    
    def trivial_2x2x1_terms():
        # Returns list of (u_indices, u_vals, v_indices, v_vals, w_indices, w_vals)
        # for 2x2x1 trivial, where indices are 0-3 for A, 0-1 for B, 0-1 for C
        return [
            # a0*b0 -> c0 += 1
            ([(0,1)], [1], [(0,1)], [1], [(0,1)], [1]),
            # a1*b1 -> c0 += 1
            ([(1,1)], [1], [(1,1)], [1], [(0,1)], [1]),
            # a2*b0 -> c1 += 1
            ([(2,1)], [1], [(0,1)], [1], [(1,1)], [1]),
            # a3*b1 -> c1 += 1
            ([(3,1)], [1], [(1,1)], [1], [(1,1)], [1]),
        ]
    
    terms = []
    
    # Helper to add terms with index offsets
    def add_strassen(a_offset, b_offset, c_offset):
        for u_idx, u_val, v_idx, v_val, w_idx, w_val in strassen_2x2x2_terms():
            u_indices = [idx + a_offset for idx, _ in u_idx]
            v_indices = [idx + b_offset for idx, _ in v_idx]
            w_indices = [idx + c_offset for idx, _ in w_idx]
            terms.append((u_indices, u_val, v_indices, v_val, w_indices, w_val))
    
    def add_trivial(a_offset, b_offset, c_offset):
        for u_idx, u_val, v_idx, v_val, w_idx, w_val in trivial_2x2x1_terms():
            u_indices = [idx + a_offset for idx, _ in u_idx]
            v_indices = [idx + b_offset for idx, _ in v_idx]
            w_indices = [idx + c_offset for idx, _ in w_idx]
            terms.append((u_indices, u_val, v_indices, v_val, w_indices, w_val))
    
    # C1 (4x4): 8 Strassen 2x2x2 multiplications
    # A11*B11: a_offset=0, b_offset=0, c_offset=0
    add_strassen(0, 0, 0)
    # A12*B21: a_offset=2, b_offset=2, c_offset=0
    add_strassen(2, 2, 0)
    # A11*B12: a_offset=0, b_offset=4, c_offset=1
    add_strassen(0, 4, 1)
    # A12*B22: a_offset=2, b_offset=6, c_offset=1
    add_strassen(2, 6, 1)
    # A21*B11: a_offset=8, b_offset=0, c_offset=4
    add_strassen(8, 0, 4)
    # A22*B21: a_offset=10, b_offset=2, c_offset=4
    add_strassen(10, 2, 4)
    # A21*B12: a_offset=8, b_offset=4, c_offset=5
    add_strassen(8, 4, 5)
    # A22*B22: a_offset=10, b_offset=6, c_offset=5
    add_strassen(10, 6, 5)
    
    # C2 (4x3): 3 columns, each with 4 trivial 2x2x1 multiplications
    # Column 0 of C2 (col 4 of C):
    # A11 * B2_col0_top: a_offset=0, b_offset=8, c_offset=4
    add_trivial(0, 8, 4)
    # A12 * B2_col0_bottom: a_offset=2, b_offset=10, c_offset=4
    add_trivial(2, 10, 4)
    # A21 * B2_col0_top: a_offset=8, b_offset=8, c_offset=5
    add_trivial(8, 8, 5)
    # A22 * B2_col0_bottom: a_offset=10, b_offset=10, c_offset=5
    add_trivial(10, 10, 5)
    
    # Column 1 of C2 (col 5 of C):
    # A11 * B2_col1_top: a_offset=0, b_offset=12, c_offset=6
    add_trivial(0, 12, 6)
    # A12 * B2_col1_bottom: a_offset=2, b_offset=14, c_offset=6
    add_trivial(2, 14, 6)
    # A21 * B2_col1_top: a_offset=8, b_offset=12, c_offset=7
    add_trivial(8, 12, 7)
    # A22 * B2_col1_bottom: a_offset=10, b_offset=14, c_offset=7
    add_trivial(10, 14, 7)
    
    # Column 2 of C2 (col 6 of C):
    # A11 * B2_col2_top: a_offset=0, b_offset=16, c_offset=8
    add_trivial(0, 16, 8)
    # A12 * B2_col2_bottom: a_offset=2, b_offset=18, c_offset=8
    add_trivial(2, 18, 8)
    # A21 * B2_col2_top: a_offset=8, b_offset=16, c_offset=9
    add_trivial(8, 16, 9)
    # A22 * B2_col2_bottom: a_offset=10, b_offset=18, c_offset=9
    add_trivial(10, 18, 9)
    
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c, (u_indices, u_vals, v_indices, v_vals, w_indices, w_vals) in enumerate(terms):
        for idx, val in zip(u_indices, u_vals):
            U[idx][c] += val
        for idx, val in zip(v_indices, v_vals):
            V[idx][c] += val
        for idx, val in zip(w_indices, w_vals):
            W[idx][c] += val
    
    return {"U": U, "V": V, "W": W}
