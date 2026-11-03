"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=6
    # Split A (2x4) into A1 (2x2), A2 (2x2)
    # Split B (4x6) into B_top (2x6), B_bot (2x6)
    # C = A1 @ B_top + A2 @ B_bot
    # B_top has 6 columns. Split into 3 blocks of 2 cols.
    # Each block is 2x2x2 multiplication.
    # Rank of 2x2x2 is 7 (Strassen).
    # Total rank = 2 (A blocks) * 3 (B col blocks) * 7 = 42.
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)]  # 8x42
    V = [[0] * rank for _ in range(m * p)]  # 24x42
    W = [[0] * rank for _ in range(p * n)]  # 12x42
    
    col = 0
    # Strassen constants for 2x2x2
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B22)
    # M4 = A22(B12 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We need to map these to the global indices.
    # A indices: i (0,1), j (0,3)
    # B indices: j (0,3), k (0,5)
    # C indices: k (0,5), i (0,1) -> W index k*n + i
    
    # Loop over A blocks (a_block: 0, 1)
    for a_block in range(2):
        # a_block 0 corresponds to A[0:2, 0:2]
        # a_block 1 corresponds to A[0:2, 2:4]
        # Wait, A is 2x4.
        # A1 is A[:, 0:2], A2 is A[:, 2:4].
        # So a_block 0 -> j in 0,1. a_block 1 -> j in 2,3.
        
        # Loop over B column blocks (b_col_block: 0, 1, 2)
        # b_col_block 0 -> k in 0,1
        # b_col_block 1 -> k in 2,3
        # b_col_block 2 -> k in 4,5
        
        for b_col_block in range(3):
            # Base indices for this 2x2x2 block
            # A rows: 0,1
            # A cols: 2*a_block, 2*a_block+1
            # B rows: 2*a_block, 2*a_block+1 (because B is 4x6, and we split rows corresponding to A cols)
            # B cols: 2*b_col_block, 2*b_col_block+1
            
            a_j0 = 2 * a_block
            a_j1 = 2 * a_block + 1
            b_j0 = 2 * a_block
            b_j1 = 2 * a_block + 1
            b_k0 = 2 * b_col_block
            b_k1 = 2 * b_col_block + 1
            
            # Define the 7 Strassen terms
            # Each term is a sum of products of A and B elements.
            # We add 7 columns to U, V, W.
            
            # Term 1: (A11 + A22)(B11 + B22)
            # A11 is A[0, a_j0], A22 is A[1, a_j1]
            # B11 is B[b_j0, b_k0], B22 is B[b_j1, b_k1]
            
            # We need to represent (A11 + A22) and (B11 + B22)
            # This requires 2 rank-1 decompositions?
            # No, (A11 + A22) is a scalar.
            # The product is (A11 + A22) * (B11 + B22)
            # = A11 B11 + A11 B22 + A22 B11 + A22 B22
            # This is 4 terms.
            # Strassen combines them into 1 multiplication.
            # In the tensor decomposition, we add 1 column for the multiplication result.
            # U entry: coefficient for A elements.
            # V entry: coefficient for B elements.
            # W entry: coefficient for C elements.
            
            # For Term 1:
            # U[0, a_j0] += 1
            # U[1, a_j1] += 1
            # V[b_j0, b_k0] += 1
            # V[b_j1, b_k1] += 1
            # W contributions:
            # C11 += 1, C12 += 0, C21 += 0, C22 += 1
            # C11 index: b_k0 * n + 0
            # C12 index: b_k1 * n + 0
            # C21 index: b_k0 * n + 1
            # C22 index: b_k1 * n + 1
            
            # Wait, Strassen M1 contributes to C11 and C22.
            # C11 = M1 + ...
            # C22 = M1 + ...
            
            # Let's define a helper to add a Strassen term
            def add_strassen_term(term_id, u_coeffs, v_coeffs, w_coeffs):
                nonlocal col
                # u_coeffs: dict (i, j) -> val
                # v_coeffs: dict (j, k) -> val
                # w_coeffs: dict (k, i) -> val
                
                for (i, j), val in u_coeffs.items():
                    U[i * m + j][col] += val
                for (j, k), val in v_coeffs.items():
                    V[j * p + k][col] += val
                for (k, i), val in w_coeffs.items():
                    W[k * n + i][col] += val
                col += 1

            # Term 1: (A11 + A22)(B11 + B22)
            u1 = {(0, a_j0): 1, (1, a_j1): 1}
            v1 = {(b_j0, b_k0): 1, (b_j1, b_k1): 1}
            w1 = {(b_k0, 0): 1, (b_k1, 1): 1}
            add_strassen_term(1, u1, v1, w1)
            
            # Term 2: (A21 + A22)B11
            u2 = {(1, a_j0): 1, (1, a_j1): 1}
            v2 = {(b_j0, b_k0): 1}
            w2 = {(b_k0, 1): 1} # C21
            add_strassen_term(2, u2, v2, w2)
            
            # Term 3: A11(B21 - B22)
            u3 = {(0, a_j0): 1}
            v3 = {(b_j1, b_k0): 1, (b_j1, b_k1): -1}
            w3 = {(b_k1, 0): 1, (b_k0, 1): 1} # C12 += M3, C21 += M3
            # Wait, C12 = M3 + M5. C21 = M2 + M4.
            # M3 contributes to C12 and C21?
            # C12 = M3 + M5
            # C21 = M2 + M4
            # No, M3 is A11(B21-B22).
            # Result is 2x2 matrix.
            # Row 0: M3 contributes to C12?
            # Let's check Strassen formulas.
            # C11 = M1 - M5 + M6 + M7
            # C12 = M3 + M5
            # C21 = M2 + M4
            # C22 = M1 - M2 + M3 + M4  (Wait, standard is M1 - M2 + M3 + M4? No)
            # Standard:
            # C11 = M1 + M4 - M5 + M7
            # C12 = M3 + M5
            # C21 = M2 + M4
            # C22 = M1 - M2 + M3 + M6
            
            # So M3 contributes to C12 and C22.
            w3 = {(b_k1, 0): 1, (b_k1, 1): 1} # C12 (k=b_k1, i=0) and C22 (k=b_k1, i=1)
            # Wait, C12 is col 1, row 0. Index b_k1 * n + 0.
            # C22 is col 1, row 1. Index b_k1 * n + 1.
            add_strassen_term(3, u3, v3, w3)
            
            # Term 4: A22(B12 - B11)
            u4 = {(1, a_j1): 1}
            v4 = {(b_j0, b_k1): 1, (b_j0, b_k0): -1}
            w4 = {(b_k0, 1): 1, (b_k1, 1): 1} # C21 (k=b_k0, i=1) and C22 (k=b_k1, i=1)
            add_strassen_term(4, u4, v4, w4)
            
            # Term 5: (A11 + A12)B22
            u5 = {(0, a_j0): 1, (0, a_j1): 1}
            v5 = {(b_j1, b_k1): 1}
            w5 = {(b_k0, 0): -1, (b_k1, 0): 1} # C11 (k=b_k0, i=0) and C12 (k=b_k1, i=0)
            add_strassen_term(5, u5, v5, w5)
            
            # Term 6: (A21 - A11)(B11 + B12)
            u6 = {(1, a_j0): 1, (0, a_j0): -1}
            v6 = {(b_j0, b_k0): 1, (b_j0, b_k1): 1}
            w6 = {(b_k0, 0): 1, (b_k1, 1): 1} # C11 (k=b_k0, i=0) and C22 (k=b_k1, i=1)
            add_strassen_term(6, u6, v6, w6)
            
            # Term 7: (A12 - A22)(B21 + B22)
            u7 = {(0, a_j1): 1, (1, a_j1): -1}
            v7 = {(b_j1, b_k0): 1, (b_j1, b_k1): 1}
            w7 = {(b_k0, 0): 1, (b_k1, 0): 1} # C11 (k=b_k0, i=0) and C12 (k=b_k1, i=0)
            # Wait, M7 contributes to C11 and C12?
            # C11 = ... + M7
            # C12 = ... + M5 (no M7)
            # C22 = ... (no M7)
            # Let's re-verify Strassen.
            # C11 = M1 + M4 - M5 + M7
            # C12 = M3 + M5
            # C21 = M2 + M4
            # C22 = M1 - M2 + M3 + M6
            # So M7 only contributes to C11.
            w7 = {(b_k0, 0): 1}
            add_strassen_term(7, u7, v7, w7)
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
