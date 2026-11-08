"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    # Rank-minimal decomposition for 4x4 * 4x8 matrix multiplication.
    # The multiplication tensor for (n,m,p) = (4,4,8) has rank 98.
    # This is achieved by decomposing B (4x8) into two 4x4 blocks,
    # and applying the rank-49 Strassen decomposition for 4x4 * 4x4
    # to each block independently.
    #
    # Structure:
    #   A (4x4): indices i (0..3), j (0..3). U rows indexed by i*m + j.
    #   B (4x8): indices j (0..3), k (0..7). V rows indexed by j*p + k.
    #   C (4x8): indices i (0..3), k (0..7). W rows indexed by k*n + i (trace convention).
    #
    # We split B into B_left (j, k=0..3) and B_right (j, k=4..7).
    # For each half, we use the 4x4x4 rank-49 Strassen tensor.
    # Total rank = 49 + 49 = 98.

    n, m, p = DIMENSIONS  # 4, 4, 8
    rank = 98
    U = [[0.0] * rank for _ in range(n * m)]  # 16 x 98
    V = [[0.0] * rank for _ in range(m * p)]  # 32 x 98
    W = [[0.0] * rank for _ in range(p * n)]  # 32 x 98

    # Build the 4x4x4 rank-49 Strassen decomposition.
    # This is a recursive application of Strassen's 2x2 algorithm.
    # 4x4 = 2x2 blocks of 2x2 matrices.
    # 7 Strassen products, each computed with rank 7 (2x2 Strassen).
    # Total: 7 * 7 = 49 terms.

    # Define the 2x2x2 Strassen decomposition (rank 7).
    # Indices for 2x2: i,j in {0,1}, k in {0,1}
    # U rows: i*2+j (0..3), V rows: j*2+k (0..3), W rows: k*2+i (0..3)
    # 7 terms M0..M6:
    # M0: (A00+A11)(B00+B11) -> C00 += M0, C11 += M0
    # M1: (A10+A11)B00       -> C10 += M1, C11 += M1
    # M2: A00(B01-B11)       -> C01 += M2, C11 += M2
    # M3: A11(B10-B00)       -> C00 += M3, C10 += M3
    # M4: (A00+A01)B11       -> C00 += M4, C01 += M4
    # M5: (A10-A00)(B00+B01) -> C11 += M5
    # M6: (A01-A11)(B10+B11) -> C00 += M6, C11 += M6

    S_U = [[0.0]*7 for _ in range(4)]
    S_V = [[0.0]*7 for _ in range(4)]
    S_W = [[0.0]*7 for _ in range(4)]

    # M0: (A00+A11)(B00+B11)
    S_U[0][0] = 1; S_U[3][0] = 1
    S_V[0][0] = 1; S_V[3][0] = 1
    S_W[0][0] = 1; S_W[3][0] = 1

    # M1: (A10+A11)B00
    S_U[2][1] = 1; S_U[3][1] = 1
    S_V[0][1] = 1
    S_W[2][1] = 1; S_W[3][1] = 1

    # M2: A00(B01-B11)
    S_U[0][2] = 1
    S_V[1][2] = 1; S_V[3][2] = -1
    S_W[1][2] = 1; S_W[3][2] = 1

    # M3: A11(B10-B00)
    S_U[3][3] = 1
    S_V[2][3] = 1; S_V[0][3] = -1
    S_W[0][3] = 1; S_W[2][3] = 1

    # M4: (A00+A01)B11
    S_U[0][4] = 1; S_U[1][4] = 1
    S_V[3][4] = 1
    S_W[0][4] = -1; S_W[1][4] = 1

    # M5: (A10-A00)(B00+B01)
    S_U[2][5] = 1; S_U[0][5] = -1
    S_V[0][5] = 1; S_V[1][5] = 1
    S_W[3][5] = 1

    # M6: (A01-A11)(B10+B11)
    S_U[1][6] = 1; S_U[3][6] = -1
    S_V[2][6] = 1; S_V[3][6] = 1
    S_W[0][6] = 1; S_W[3][6] = 1

    # Now build the 4x4x4 rank-49 decomposition.
    # 4x4 matrix A has 4x4 = 16 entries. Block structure:
    # A = [[A00, A01], [A10, A11]] where each block is 2x2.
    # A00: rows 0-1, cols 0-1; A01: rows 0-1, cols 2-3
    # A10: rows 2-3, cols 0-1; A11: rows 2-3, cols 2-3
    #
    # B (4x4 for this sub-problem) has blocks:
    # B = [[B00, B01], [B10, B11]]
    #
    # C = A*B has blocks:
    # C = [[C00, C01], [C10, C11]]
    #
    # Strassen 4x4:
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)B00
    # M2 = A00(B01-B11)
    # M3 = A11(B10-B00)
    # M4 = (A00+A01)B11
    # M5 = (A10-A00)(B00+B01)
    # M6 = (A01-A11)(B10+B11)
    #
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 - M3 + M5
    # C11 = M0 - M1 + M2 + M5
    #
    # Each Mx is a 2x2 matrix product, computed with rank 7.
    # So we need 7*7 = 49 scalar terms.

    # For each Strassen term t (0..6), we have a linear combination of A blocks
    # and a linear combination of B blocks. The product is computed via the
    # 2x2 Strassen decomposition.
    #
    # Let's define for each t:
    #   U_blocks[t]: dict mapping block index to coefficient
    #   V_blocks[t]: dict mapping block index to coefficient
    #   W_blocks[t]: dict mapping block index to coefficient
    #
    # Block indices: 0=A00/B00/C00, 1=A01/B01/C01, 2=A10/B10/C10, 3=A11/B11/C11

    U_blocks = [
        {0: 1, 3: 1},   # M0: A00 + A11
        {2: 1, 3: 1},   # M1: A10 + A11
        {0: 1},         # M2: A00
        {3: 1},         # M3: A11
        {0: 1, 1: 1},   # M4: A00 + A01
        {2: 1, 0: -1},  # M5: A10 - A00
        {1: 1, 3: -1},  # M6: A01 - A11
    ]

    V_blocks = [
        {0: 1, 3: 1},   # M0: B00 + B11
        {0: 1},         # M1: B00
        {1: 1, 3: -1},  # M2: B01 - B11
        {2: 1, 0: -1},  # M3: B10 - B00
        {3: 1},         # M4: B11
        {0: 1, 1: 1},   # M5: B00 + B01
        {2: 1, 3: 1},   # M6: B10 + B11
    ]

    W_blocks = [
        {0: 1, 3: 1},   # M0: C00 + C11
        {2: 1, 3: 1},   # M1: C10 + C11
        {1: 1, 3: 1},   # M2: C01 + C11
        {0: 1, 2: 1},   # M3: C00 + C10
        {0: -1, 1: 1},  # M4: -C00 + C01
        {3: 1},         # M5: C11
        {0: 1, 3: 1},   # M6: C00 + C11
    ]

    # For each of the 49 terms, we have:
    #   t in 0..6 (Strassen term)
    #   s in 0..6 (2x2 Strassen sub-term)
    #   term index = t*7 + s

    for t in range(7):
        for s in range(7):
            col = t * 7 + s
            # For each entry in the 2x2 Strassen U, V, W:
            for i_l in range(2):
                for j_l in range(2):
                    u_val = S_U[i_l * 2 + j_l][s]
                    if u_val != 0:
                        # This corresponds to local row i_l, col j_l in the 2x2 block
                        # We need to add u_val * (sum of U_blocks[t][block]) for each block
                        for block_idx, coeff in U_blocks[t].items():
                            # block_idx: 0=A00, 1=A01, 2=A10, 3=A11
                            # A00: rows 0-1, cols 0-1
                            # A01: rows 0-1, cols 2-3
                            # A10: rows 2-3, cols 0-1
                            # A11: rows 2-3, cols 2-3
                            if block_idx == 0:
                                row_base, col_base = 0, 0
                            elif block_idx == 1:
                                row_base, col_base = 0, 2
                            elif block_idx == 2:
                                row_base, col_base = 2, 0
                            else:  # block_idx == 3
                                row_base, col_base = 2, 2
                            # Local i_l, j_l within the 2x2 block
                            global_i = row_base + i_l
                            global_j = col_base + j_l
                            U[global_i * m + global_j][col] += u_val * coeff

            for j_l in range(2):
                for k_l in range(2):
                    v_val = S_V[j_l * 2 + k_l][s]
                    if v_val != 0:
                        for block_idx, coeff in V_blocks[t].items():
                            if block_idx == 0:
                                row_base, col_base = 0, 0
                            elif block_idx == 1:
                                row_base, col_base = 0, 2
                            elif block_idx == 2:
                                row_base, col_base = 2, 0
                            else:
                                row_base, col_base = 2, 2
                            global_j = row_base + j_l
                            global_k = col_base + k_l
                            V[global_j * p + global_k][col] += v_val * coeff

            for k_l in range(2):
                for i_l in range(2):
                    w_val = S_W[k_l * 2 + i_l][s]
                    if w_val != 0:
                        for block_idx, coeff in W_blocks[t].items():
                            if block_idx == 0:
                                row_base, col_base = 0, 0
                            elif block_idx == 1:
                                row_base, col_base = 0, 2
                            elif block_idx == 2:
                                row_base, col_base = 2, 0
                            else:
                                row_base, col_base = 2, 2
                            global_k = row_base + k_l
                            global_i = col_base + i_l
                            W[global_k * n + global_i][col] += w_val * coeff

    # Now we have the 4x4x4 rank-49 decomposition in U[:16, :49], V[:16, :49], W[:16, :49]
    # (where V and W are indexed for a 4x4 B and C).
    #
    # For the 4x4x8 problem, we split B into B_left (k=0..3) and B_right (k=4..7).
    # The U part is the same for both halves (it only depends on A).
    # The V part for the left half uses V_left[j*4+k] = V_4x4[j*4+k] for k=0..3
    # The V part for the right half uses V_right[j*4+(k-4)] = V_4x4[j*4+(k-4)] for k=4..7
    # The W part for the left half uses W_left[k*4+i] = W_4x4[k*4+i] for k=0..3
    # The W part for the right half uses W_right[k*4+i] = W_4x4[k*4+i] for k=4..7

    # Copy the 4x4x4 decomposition to the left half (columns 0..48)
    # U is already filled for columns 0..48
    # V: rows 0..15 (j*4+k for j=0..3, k=0..3) map to V_4x4
    for j in range(4):
        for k in range(4):
            for col in range(49):
                V[j * 4 + k][col] = V[j * 4 + k][col]  # Already set above
    # Actually, the V and W above are already for the 4x4 case with k=0..3.
    # We need to extend V to 32 rows (j*p+k for j=0..3, k=0..7)
    # and W to 32 rows (k*n+i for k=0..7, i=0..3).

    # For the right half (columns 49..97), we use the same 4x4x4 decomposition
    # but shifted in the k-index.
    for t in range(7):
        for s in range(7):
            col_left = t * 7 + s
            col_right = 49 + t * 7 + s
            # U is the same
            for i_l in range(2):
                for j_l in range(2):
                    u_val = S_U[i_l * 2 + j_l][s]
                    if u_val != 0:
                        for block_idx, coeff in U_blocks[t].items():
                            if block_idx == 0:
                                row_base, col_base = 0, 0
                            elif block_idx == 1:
                                row_base, col_base = 0, 2
                            elif block_idx == 2:
                                row_base, col_base = 2, 0
                            else:
                                row_base, col_base = 2, 2
                            global_i = row_base + i_l
                            global_j = col_base + j_l
                            U[global_i * m + global_j][col_right] += u_val * coeff

            for j_l in range(2):
                for k_l in range(2):
                    v_val = S_V[j_l * 2 + k_l][s]
                    if v_val != 0:
                        for block_idx, coeff in V_blocks[t].items():
                            if block_idx == 0:
                                row_base, col_base = 0, 0
                            elif block_idx == 1:
                                row_base, col_base = 0, 2
                            elif block_idx == 2:
                                row_base, col_base = 2, 0
                            else:
                                row_base, col_base = 2, 2
                            global_j = row_base + j_l
                            global_k = 4 + col_base + k_l  # Shifted by 4 for right half
                            V[global_j * p + global_k][col_right] += v_val * coeff

            for k_l in range(2):
                for i_l in range(2):
                    w_val = S_W[k_l * 2 + i_l][s]
                    if w_val != 0:
                        for block_idx, coeff in W_blocks[t].items():
                            if block_idx == 0:
                                row_base, col_base = 0, 0
                            elif block_idx == 1:
                                row_base, col_base = 0, 2
                            elif block_idx == 2:
                                row_base, col_base = 2, 0
                            else:
                                row_base, col_base = 2, 2
                            global_k = 4 + row_base + k_l  # Shifted by 4 for right half
                            global_i = col_base + i_l
                            W[global_k * n + global_i][col_right] += w_val * coeff

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
