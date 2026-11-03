"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 49 decomposition using Strassen's algorithm recursively
    rank = 49
    
    # U: (n*m, rank) = (16, 49)
    # V: (m*p, rank) = (16, 49)
    # W: (p*n, rank) = (16, 49)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term to the decomposition
    # coeffs_A: dict {index: coeff} for A
    # coeffs_B: dict {index: coeff} for B
    # coeffs_C: dict {index: coeff} for C
    def add_term(coeffs_A, coeffs_B, coeffs_C):
        nonlocal col
        for i, c in coeffs_A.items():
            U[i][col] = c
        for j, c in coeffs_B.items():
            V[j][col] = c
        for k, c in coeffs_C.items():
            W[k][col] = c
        col += 1
        
    # Define the 7 outer Strassen steps
    # X_k and Y_k are 2x2 matrices of linear combinations of A and B blocks
    # A blocks: A00 (0:2, 0:2), A01 (0:2, 2:4), A10 (2:4, 0:2), A11 (2:4, 2:4)
    # Indices in A (row-major 4x4):
    # A00: 0, 1, 4, 5
    # A01: 2, 3, 6, 7
    # A10: 8, 9, 12, 13
    # A11: 10, 11, 14, 15
    
    # We represent a 2x2 matrix of linear combinations as a list of 4 dicts
    # mat[0] = top-left, mat[1] = top-right, mat[2] = bottom-left, mat[3] = bottom-right
    
    def make_block_sum(block_indices, signs):
        """
        block_indices: list of 4 block indices (0:A00, 1:A01, 2:A10, 3:A11)
        signs: list of 4 signs
        Returns a list of 4 dicts representing the 2x2 matrix of linear combinations
        """
        # Block 0 (A00) entries: 0, 1, 4, 5
        # Block 1 (A01) entries: 2, 3, 6, 7
        # Block 2 (A10) entries: 8, 9, 12, 13
        # Block 3 (A11) entries: 10, 11, 14, 15
        
        res = [{} for _ in range(4)]
        
        for b_idx, sign in zip(block_indices, signs):
            if sign == 0: continue
            if b_idx == 0:
                entries = [0, 1, 4, 5]
            elif b_idx == 1:
                entries = [2, 3, 6, 7]
            elif b_idx == 2:
                entries = [8, 9, 12, 13]
            elif b_idx == 3:
                entries = [10, 11, 14, 15]
            
            # Map to 2x2 positions
            # 0->0, 1->1, 4->2, 5->3
            pos_map = [0, 1, 2, 3]
            for e, p in zip(entries, pos_map):
                if e in res[p]:
                    res[p][e] += sign
                else:
                    res[p][e] = sign
        return res

    # Outer Strassen Definitions
    # S1 = (A00 + A11)(B00 + B11)
    # S2 = (A10 + A11)B00
    # S3 = A00(B01 - B11)
    # S4 = A11(B10 - B00)
    # S5 = (A00 + A01)B11
    # S6 = (A10 - A00)(B00 + B01)
    # S7 = (A01 - A11)(B10 + B11)
    
    outer_steps = [
        # (X_blocks, X_signs, Y_blocks, Y_signs)
        ([0, 0, 0, 3], [1, 0, 0, 1], [0, 0, 0, 3], [1, 0, 0, 1]), # S1
        ([0, 0, 2, 3], [0, 0, 1, 1], [0, 0, 0, 0], [1, 0, 0, 0]), # S2
        ([0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 3], [0, 1, 0, -1]), # S3
        ([0, 0, 0, 3], [0, 0, 0, 1], [0, 0, 2, 0], [0, 0, 1, -1]), # S4
        ([0, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 3], [0, 0, 0, 1]), # S5
        ([0, 0, 2, 0], [-1, 0, 1, 0], [0, 1, 0, 0], [1, 1, 0, 0]), # S6
        ([0, 1, 0, 3], [0, 1, 0, -1], [0, 0, 2, 0], [0, 0, 1, 1])  # S7
    ]
    
    # Inner Strassen Steps for 2x2 matrices
    # X = [[x00, x01], [x10, x11]], Y = [[y00, y01], [y10, y11]]
    # m1 = (x00 + x11)(y00 + y11)
    # m2 = (x10 + x11)y00
    # m3 = x00(y01 - y11)
    # m4 = x11(y10 - y00)
    # m5 = (x00 + x01)y11
    # m6 = (x10 - x00)(y00 + y01)
    # m7 = (x01 - x11)(y10 + y11)
    
    # Contributions to S (2x2 matrix)
    # S00 = m1 + m4 - m5 + m7
    # S01 = m3 + m5
    # S10 = m2 + m4
    # S11 = m1 - m2 + m3 + m6
    
    inner_steps = [
        # (X_part, Y_part, S_contrib)
        # X_part: list of 4 indices (0,1,2,3) for X entries
        # Y_part: list of 4 indices for Y entries
        # S_contrib: list of 4 indices for S entries with signs
        ([0, 0, 1, 1], [0, 0, 1, 1], [1, 0, 0, 1]), # m1 -> S00(+), S11(+)
        ([0, 0, 2, 2], [0, 0, 0, 0], [0, 0, 1, -1]), # m2 -> S10(+), S11(-)
        ([0, 0, 0, 0], [1, 0, 0, 1], [0, 1, 0, 1]), # m3 -> S01(+), S11(+)
        ([0, 0, 0, 3], [2, 0, 0, 0], [1, 0, 1, 0]), # m4 -> S00(+), S10(+)
        ([0, 0, 1, 0], [0, 0, 0, 3], [-1, 1, 0, 0]), # m5 -> S00(-), S01(+)
        ([0, 0, 2, 0], [0, 1, 0, 0], [0, 0, 0, 1]), # m6 -> S11(+)
        ([0, 1, 0, 3], [2, 0, 0, 1], [1, 0, 0, 0])  # m7 -> S00(+)
    ]
    
    # Mapping from S block entry to C block entry
    # C00 = S1 + S4 - S5 + S7
    # C01 = S3 + S5
    # C10 = S2 + S4
    # C11 = S1 - S2 + S3 + S6
    
    # S_idx (0-6) -> C_block (0-3) -> sign
    s_to_c = {
        0: {0: 1, 3: 1}, # S1
        1: {2: 1, 3: -1}, # S2
        2: {1: 1, 3: 1}, # S3
        3: {0: 1, 2: 1}, # S4
        4: {0: -1, 1: 1}, # S5
        5: {3: 1}, # S6
        6: {0: 1} # S7
    }
    
    # C block 0 (C00) entries: 0, 1, 4, 5
    # C block 1 (C01) entries: 2, 3, 6, 7
    # C block 2 (C10) entries: 8, 9, 12, 13
    # C block 3 (C11) entries: 10, 11, 14, 15
    
    c_block_entries = {
        0: [0, 1, 4, 5],
        1: [2, 3, 6, 7],
        2: [8, 9, 12, 13],
        3: [10, 11, 14, 15]
    }
    
    for s_idx, (x_blocks, x_signs, y_blocks, y_signs) in enumerate(outer_steps):
        X = make_block_sum(x_blocks, x_signs)
        Y = make_block_sum(y_blocks, y_signs)
        
        for i_idx, (x_part, y_part, s_contrib) in enumerate(inner_steps):
            # Construct coeffs_A for this term
            coeffs_A = {}
            for k in range(4):
                if x_part[k] != 0:
                    pos = x_part[k]
                    # X[pos] is a dict of {A_index: coeff}
                    for a_idx, c in X[pos].items():
                        if a_idx in coeffs_A:
                            coeffs_A[a_idx] += c
                        else:
                            coeffs_A[a_idx] = c
            
            # Construct coeffs_B for this term
            coeffs_B = {}
            for k in range(4):
                if y_part[k] != 0:
                    pos = y_part[k]
                    for b_idx, c in Y[pos].items():
                        if b_idx in coeffs_B:
                            coeffs_B[b_idx] += c
                        else:
                            coeffs_B[b_idx] = c
            
            # Construct coeffs_C for this term
            coeffs_C = {}
            for k in range(4):
                if s_contrib[k] != 0:
                    s_pos = k
                    c_block = s_to_c[s_idx].get(s_pos, 0)
                    if c_block != 0:
                        # c_block is the index of the C block (0-3)
                        # s_pos is the position within the S block (0-3)
                        # The position in C block is the same as s_pos
                        c_entry_idx = c_block_entries[c_block - 1][s_pos]
                        if c_entry_idx in coeffs_C:
                            coeffs_C[c_entry_idx] += c_block
                        else:
                            coeffs_C[c_entry_idx] = c_block
            
            add_term(coeffs_A, coeffs_B, coeffs_C)
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
