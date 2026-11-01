"""
Rank-minimal tensor decomposition for 2x4 by 4x6 matrix multiplication.
Uses a block-Strassen construction (rank 42) instead of schoolbook (rank 48).
"""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 6

    # Strassen 2x2x2 decomposition:
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h), M2 = (c+d)e, M3 = a(f-h), M4 = d(g-e),
    # M5 = (a+b)h, M6 = (c-a)(e+f), M7 = (b-d)(g+h)
    # C = [[M1+M4-M5+M7, M3+M5], [M2+M4, M1-M2+M3+M6]]

    # U_A: 2x7 (rows: a, b; cols: M1..M7)
    # V_B: 2x7 (rows: e, f, g, h mapped to 4 rows of B for 2x2x2)
    # W_C: 2x7 (rows: c11, c12, c21, c22 mapped to 4 rows of C)

    # For a single 2x2x2 block:
    # U_A rows (a, b):
    #   M1: a -> 1, M2: a -> 0, M3: a -> 1, M4: a -> 0, M5: a -> 1, M6: a -> -1, M7: a -> 0
    #   M1: b -> 0, M2: b -> 0, M3: b -> 0, M4: b -> 0, M5: b -> 1, M6: b -> 0, M7: b -> 1
    # V_B rows (e, f, g, h):
    #   M1: e -> 1, M2: e -> 1, M3: e -> 0, M4: e -> -1, M5: e -> 0, M6: e -> 1, M7: e -> 0
    #   M1: f -> 0, M2: f -> 0, M3: f -> 1, M4: f -> 0, M5: f -> 0, M6: f -> 1, M7: f -> 0
    #   M1: g -> 0, M2: g -> 0, M3: g -> 0, M4: g -> 1, M5: g -> 0, M6: g -> 0, M7: g -> 1
    #   M1: h -> 1, M2: h -> 0, M3: h -> -1, M4: h -> 0, M5: h -> 1, M6: h -> 0, M7: h -> 1
    # W_C rows (c11, c12, c21, c22):
    #   M1: c11 -> 1, c12 -> 0, c21 -> 0, c22 -> 1
    #   M2: c11 -> 0, c12 -> 0, c21 -> 1, c22 -> -1
    #   M3: c11 -> 0, c12 -> 1, c21 -> 0, c22 -> 1
    #   M4: c11 -> 1, c12 -> 0, c21 -> 1, c22 -> 0
    #   M5: c11 -> -1, c12 -> 1, c21 -> 0, c22 -> 0
    #   M6: c11 -> 0, c12 -> 0, c21 -> 0, c22 -> 1
    #   M7: c11 -> 1, c12 -> 0, c21 -> 0, c22 -> 0

    U_A_base = [
        [1, 0, 1, 0, 1, -1, 0],   # a
        [0, 0, 0, 0, 1, 0, 1]     # b
    ]
    V_B_base = [
        [1, 1, 0, -1, 0, 1, 0],   # e
        [0, 0, 1, 0, 0, 1, 0],    # f
        [0, 0, 0, 1, 0, 0, 1],    # g
        [1, 0, -1, 0, 1, 0, 1]    # h
    ]
    W_C_base = [
        [1, 0, 0, 1, -1, 0, 1],   # c11
        [0, 0, 1, 0, 1, 0, 0],    # c12
        [0, 1, 0, 1, 0, 0, 0],    # c21
        [1, -1, 1, 0, 0, 1, 0]    # c22
    ]

    # Now we need to build the full decomposition for 2x4 by 4x6.
    # A is 2x4: rows i=0,1; cols j=0,1,2,3
    # B is 4x6: rows j=0,1,2,3; cols k=0,1,2,3,4,5
    # C is 6x2: rows k=0,1,2,3,4,5; cols i=0,1

    # We partition B's rows into two groups of 2: B_top (rows 0,1), B_bot (rows 2,3)
    # A has 4 columns: j=0,1 (for B_top) and j=2,3 (for B_bot)
    # B_top has 6 columns, B_bot has 6 columns.

    # For each (A_sub, B_sub) pair, we use Strassen 2x2x2 on the 2x2 blocks.
    # A_sub is 2x2 (rows 0,1 of A; cols j0,j0+1)
    # B_sub is 2x6 (rows j0,j0+1 of B; all 6 cols)
    # We further split B_sub's 6 columns into three groups of 2: cols 0-1, 2-3, 4-5
    # Each 2x2x2 Strassen block gives 7 terms.
    # Total blocks: 2 (A subgroups) * 3 (B col groups) = 6 blocks, each rank 7.
    # Total rank = 6 * 7 = 42.

    rank = 42
    U = [[0] * rank for _ in range(n * m)]   # 8 x 42
    V = [[0] * rank for _ in range(m * p)]   # 24 x 42
    W = [[0] * rank for _ in range(p * n)]   # 12 x 42

    block_idx = 0
    for a_group in range(2):  # A cols: 0-1 or 2-3
        a_col_offset = a_group * 2  # 0 or 2
        for b_col_group in range(3):  # B cols: 0-1, 2-3, 4-5
            b_col_offset = b_col_group * 2  # 0, 2, or 4

            # This block uses A rows 0,1 and cols a_col_offset, a_col_offset+1
            # B rows a_col_offset, a_col_offset+1 (these are the j indices)
            # Wait: A is 2x4, B is 4x6. The intermediate dimension is m=4.
            # A[i,j] * B[j,k] -> C[k,i]
            # A has columns j=0,1,2,3. B has rows j=0,1,2,3.
            # So for a given j pair (j0, j0+1), A cols are j0,j0+1 and B rows are j0,j0+1.

            j0 = a_col_offset  # This is the j index, 0 or 2

            # A_sub: rows i=0,1; cols j=j0, j0+1 -> A[i, j0], A[i, j0+1]
            # B_sub: rows j=j0, j0+1; cols k=b_col_offset, b_col_offset+1
            # C_sub: rows k=b_col_offset, b_col_offset+1; cols i=0,1

            # U_A_base maps (a, b) where a=A[i=0], b=A[i=1] for the 2x2 block
            # But we need to map to global A rows: A[i, j] for i=0,1 and j=j0, j0+1
            # Global A row index: i * m + j = i * 4 + j

            # V_B_base maps (e, f, g, h) where e=B[j0, k0], f=B[j0, k0+1], g=B[j0+1, k0], h=B[j0+1, k0+1]
            # Global B row index: j * p + k = j * 6 + k

            # W_C_base maps (c11, c12, c21, c22) where:
            # c11 = C[k0, i=0], c12 = C[k0, i=1], c21 = C[k0+1, i=0], c22 = C[k0+1, i=1]
            # Global C row index: k * n + i = k * 2 + i
            # k0 = b_col_offset, k0+1 = b_col_offset + 1

            k0 = b_col_offset

            # Fill U for this block
            for t in range(7):
                col = block_idx + t
                # U rows: A[i, j] for i=0,1 and j=j0, j0+1
                # U_A_base[0][t] corresponds to A[i=0, j=j0] -> global row 0*4 + j0 = j0
                # U_A_base[1][t] corresponds to A[i=1, j=j0+1] -> global row 1*4 + (j0+1) = 4 + j0 + 1
                # Wait, U_A_base is for the 2x2 A block:
                #   row 0 of U_A_base = coefficient for A[0,0] in the 2x2 block
                #   row 1 of U_A_base = coefficient for A[0,1] in the 2x2 block?
                # No. Let me re-examine.
                # In Strassen, A = [[a, b], [c, d]] where:
                #   a = A[0,0], b = A[0,1], c = A[1,0], d = A[1,1]
                # U_A_base was defined for (a, b) only? No, it should be for all 4 elements.
                # Let me redefine properly.

            # Actually, U_A_base should have 4 rows: a, b, c, d
            # Let me redefine:
            # a = A[0,0], b = A[0,1], c = A[1,0], d = A[1,1]
            # M1 = (a+d)(e+h): a->1, b->0, c->0, d->1
            # M2 = (c+d)e: a->0, b->0, c->1, d->1
            # M3 = a(f-h): a->1, b->0, c->0, d->0
            # M4 = d(g-e): a->0, b->0, c->0, d->1
            # M5 = (a+b)h: a->1, b->1, c->0, d->0
            # M6 = (c-a)(e+f): a->-1, b->0, c->1, d->0
            # M7 = (b-d)(g+h): a->0, b->1, c->0, d->-1

            # I'll just rebuild the base matrices properly below.

    # Let me restart the construction with correct base matrices.
    rank = 42
    U = [[0] * rank for _ in range(n * m)]   # 8 x 42
    V = [[0] * rank for _ in range(m * p)]   # 24 x 42
    W = [[0] * rank for _ in range(p * n)]   # 12 x 42

    # Correct Strassen base matrices (4x7 each for U_A, V_B, W_C)
    # U_A: rows (a, b, c, d) = (A[0,0], A[0,1], A[1,0], A[1,1])
    U_A_base = [
        [1, 0, 1, 0, 1, -1, 0],   # a
        [0, 0, 0, 0, 1, 0, 1],    # b
        [0, 1, 0, 0, 0, 1, 0],    # c
        [1, 1, 0, 1, 0, 0, -1]    # d
    ]
    # V_B: rows (e, f, g, h) = (B[0,0], B[0,1], B[1,0], B[1,1])
    V_B_base = [
        [1, 1, 0, -1, 0, 1, 0],   # e
        [0, 0, 1, 0, 0, 1, 0],    # f
        [0, 0, 0, 1, 0, 0, 1],    # g
        [1, 0, -1, 0, 1, 0, 1]    # h
    ]
    # W_C: rows (c11, c12, c21, c22) = (C[0,0], C[0,1], C[1,0], C[1,1])
    W_C_base = [
        [1, 0, 0, 1, -1, 0, 1],   # c11
        [0, 0, 1, 0, 1, 0, 0],    # c12
        [0, 1, 0, 1, 0, 0, 0],    # c21
        [1, -1, 1, 0, 0, 1, 0]    # c22
    ]

    block_idx = 0
    for a_group in range(2):  # j0 = 0 or 2
        j0 = a_group * 2
        for b_col_group in range(3):  # k0 = 0, 2, or 4
            k0 = b_col_group * 2

            # A_sub elements: A[i, j] for i in {0,1}, j in {j0, j0+1}
            # Global A row: i * 4 + j
            # a = A[0, j0] -> row 0*4 + j0 = j0
            # b = A[0, j0+1] -> row 0*4 + j0+1 = j0+1
            # c = A[1, j0] -> row 1*4 + j0 = 4 + j0
            # d = A[1, j0+1] -> row 1*4 + j0+1 = 4 + j0 + 1

            a_rows = [j0, j0 + 1, 4 + j0, 4 + j0 + 1]  # rows for a, b, c, d

            # B_sub elements: B[j, k] for j in {j0, j0+1}, k in {k0, k0+1}
            # Global B row: j * 6 + k
            # e = B[j0, k0] -> row j0*6 + k0
            # f = B[j0, k0+1] -> row j0*6 + k0+1
            # g = B[j0+1, k0] -> row (j0+1)*6 + k0
            # h = B[j0+1, k0+1] -> row (j0+1)*6 + k0+1

            b_rows = [j0*6 + k0, j0*6 + k0+1, (j0+1)*6 + k0, (j0+1)*6 + k0+1]

            # C_sub elements: C[k, i] for k in {k0, k0+1}, i in {0,1}
            # Global C row: k * 2 + i
            # c11 = C[k0, 0] -> row k0*2 + 0 = k0
            # c12 = C[k0, 1] -> row k0*2 + 1 = k0+1
            # c21 = C[k0+1, 0] -> row (k0+1)*2 + 0 = k0+2
            # c22 = C[k0+1, 1] -> row (k0+1)*2 + 1 = k0+3

            c_rows = [k0, k0+1, k0+2, k0+3]

            for t in range(7):
                col = block_idx + t
                # Fill U
                for r in range(4):
                    U[a_rows[r]][col] = U_A_base[r][t]
                # Fill V
                for r in range(4):
                    V[b_rows[r]][col] = V_B_base[r][t]
                # Fill W
                for r in range(4):
                    W[c_rows[r]][col] = W_C_base[r][t]

            block_idx += 7

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
