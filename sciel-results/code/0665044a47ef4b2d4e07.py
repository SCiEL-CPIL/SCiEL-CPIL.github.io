"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 4, 6

    # Strassen decomposition for 2x2x2
    # U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-f)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    #
    # C[0,0] = m1 + m4 - m5 + m7
    # C[0,1] = m3 + m5
    # C[1,0] = m2 + m4
    # C[1,1] = m1 - m2 + m3 + m6
    #
    # Let's verify the W indices. W[k*2+i, c].
    # For C[0,0]: i=0, k=0 -> W[0, c]
    # For C[0,1]: i=1, k=0 -> W[2, c] (k*2+i = 0*2+1 = 1? No, k is col index of B, i is row index of A, output C[i, k].
    # Wait, the tensor convention is W[k*n+i, c].
    # For 2x2x2: n=2, p=2.
    # C[i,k] gets contribution from rank-1 term c.
    # W[k*2+i, c] = coefficient.
    # C[0,0]: i=0, k=0. W[0*2+0, c] = W[0, c].
    # C[0,1]: i=0, k=1. W[1*2+0, c] = W[2, c].
    # C[1,0]: i=1, k=0. W[0*2+1, c] = W[1, c].
    # C[1,1]: i=1, k=1. W[1*2+1, c] = W[3, c].
    #
    # U[i*2+j, c]. i in {0,1}, j in {0,1}.
    # V[j*2+k, c]. j in {0,1}, k in {0,1}.
    #
    # m1 = (a+d)(e+h).
    # a = A[0,0] -> U[0,0] = 1. d = A[1,1] -> U[1,1] = 1.
    # e = B[0,0] -> V[0,0] = 1. h = B[1,1] -> V[1,1] = 1.
    # W: C[0,0] += m1 -> W[0] += 1.
    #     C[0,1] += m1 -> W[2] += 1.
    #     C[1,0] += m1 -> W[1] += 1.
    #     C[1,1] += m1 -> W[3] += 1.
    # So W for m1 is [1, 1, 1, 1].
    #
    # m2 = (c+d)e.
    # c = A[1,0] -> U[1,0] = 1. d = A[1,1] -> U[1,1] = 1.
    # e = B[0,0] -> V[0,0] = 1.
    # W: C[1,0] += m2 -> W[1] += 1.
    #     C[1,1] -= m2 -> W[3] -= 1.
    # W for m2 is [0, 1, 0, -1].
    #
    # m3 = a(f-h).
    # a = A[0,0] -> U[0,0] = 1.
    # f = B[0,1] -> V[0,1] = 1. h = B[1,1] -> V[1,1] = -1.
    # W: C[0,1] += m3 -> W[2] += 1.
    #     C[1,1] += m3 -> W[3] += 1.
    # W for m3 is [0, 0, 1, 1].
    #
    # m4 = d(g-f).
    # d = A[1,1] -> U[1,1] = 1.
    # g = B[1,0] -> V[1,0] = 1. f = B[0,1] -> V[0,1] = -1.
    # W: C[0,0] += m4 -> W[0] += 1.
    #     C[1,0] += m4 -> W[1] += 1.
    # W for m4 is [1, 1, 0, 0].
    #
    # m5 = (a+b)h.
    # a = A[0,0] -> U[0,0] = 1. b = A[0,1] -> U[0,1] = 1.
    # h = B[1,1] -> V[1,1] = 1.
    # W: C[0,0] -= m5 -> W[0] -= 1.
    #     C[0,1] += m5 -> W[2] += 1.
    # W for m5 is [-1, 0, 1, 0].
    #
    # m6 = (c-a)(e+f).
    # c = A[1,0] -> U[1,0] = 1. a = A[0,0] -> U[0,0] = -1.
    # e = B[0,0] -> V[0,0] = 1. f = B[0,1] -> V[0,1] = 1.
    # W: C[1,1] += m6 -> W[3] += 1.
    # W for m6 is [0, 0, 0, 1].
    #
    # m7 = (b-d)(g+h).
    # b = A[0,1] -> U[0,1] = 1. d = A[1,1] -> U[1,1] = -1.
    # g = B[1,0] -> V[1,0] = 1. h = B[1,1] -> V[1,1] = 1.
    # W: C[0,0] += m7 -> W[0] += 1.
    # W for m7 is [1, 0, 0, 0].

    strassen_U = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [1, 1, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 1, -1]
    ]
    strassen_V = [
        [1, 1, 1, -1, 0, 1, 1],
        [0, 0, -1, 1, 1, 1, 1]
    ]
    # Wait, V indices are j*2+k.
    # j=0, k=0 -> V[0]
    # j=0, k=1 -> V[1]
    # j=1, k=0 -> V[2]
    # j=1, k=1 -> V[3]
    #
    # m1: e(V[0]), h(V[3])
    # m2: e(V[0])
    # m3: f(V[1]), h(V[3])
    # m4: g(V[2]), f(V[1])
    # m5: h(V[3])
    # m6: e(V[0]), f(V[1])
    # m7: g(V[2]), h(V[3])
    #
    strassen_V = [
        [1, 1, 0, 0, 0, 1, 0],  # e
        [0, 0, 1, -1, 0, 1, 0], # f
        [0, 0, 0, 1, 0, 0, 1],  # g
        [1, 0, -1, 0, 1, 0, 1]  # h
    ]
    strassen_W = [
        [1, 0, 0, 1, -1, 0, 1], # C[0,0]
        [0, 1, 0, 1, 0, 0, 0],  # C[1,0]
        [1, 0, 1, 0, 1, 0, 0],  # C[0,1]
        [1, -1, 1, 0, 0, 1, 0]  # C[1,1]
    ]

    # Schoolbook for 1x2x2. Rank 4.
    # A has 1 row (i=0), 2 cols (j=0,1).
    # B has 2 rows (j=0,1), 2 cols (k=0,1).
    # C has 1 row (i=0), 2 cols (k=0,1).
    # U indices i*2+j -> 0, 1.
    # V indices j*2+k -> 0, 1, 2, 3.
    # W indices k*2+i -> 0, 1.
    #
    # C[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0]
    # C[0,1] = A[0,0]B[0,1] + A[0,1]B[1,1]
    #
    # Term 1: A[0,0]B[0,0]C[0,0]. U[0]=1, V[0]=1, W[0]=1.
    # Term 2: A[0,1]B[1,0]C[0,0]. U[1]=1, V[2]=1, W[0]=1.
    # Term 3: A[0,0]B[0,1]C[0,1]. U[0]=1, V[1]=1, W[1]=1.
    # Term 4: A[0,1]B[1,1]C[0,1]. U[1]=1, V[3]=1, W[1]=1.
    schoolbook_1x2x2_U = [
        [1, 0, 1, 0],
        [0, 1, 0, 1]
    ]
    schoolbook_1x2x2_V = [
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0], # Wait, V[2] is j=1, k=0.
        [0, 0, 0, 1]
    ]
    # Let's re-verify V indices.
    # V[j*2+k].
    # Term 1: B[0,0] -> j=0, k=0 -> V[0].
    # Term 2: B[1,0] -> j=1, k=0 -> V[2].
    # Term 3: B[0,1] -> j=0, k=1 -> V[1].
    # Term 4: B[1,1] -> j=1, k=1 -> V[3].
    schoolbook_1x2x2_V = [
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]
    # W indices k*2+i. i=0.
    # Term 1: C[0,0] -> k=0 -> W[0].
    # Term 2: C[0,0] -> k=0 -> W[0].
    # Term 3: C[0,1] -> k=1 -> W[1].
    # Term 4: C[0,1] -> k=1 -> W[1].
    schoolbook_1x2x2_W = [
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]

    # 3x2x2 decomposition. Rank 11.
    # A is 3x2. B is 2x2. C is 3x2.
    # Rows 0,1 of A use Strassen (rank 7).
    # Row 2 of A uses Schoolbook (rank 4).
    # Total rank 11.
    #
    # U indices i*2+j. i in {0,1,2}, j in {0,1}. 6 entries.
    # V indices j*2+k. j in {0,1}, k in {0,1}. 4 entries.
    # W indices k*2+i. k in {0,1}, i in {0,1,2}. 6 entries.
    #
    # Strassen part (rows 0,1):
    # U_strassen (2x7) -> embed into U_3x2x2 (6x11).
    # U_strassen[i*2+j, c] -> U_3x2x2[i*2+j, c] for i=0,1.
    # V_strassen (4x7) -> V_3x2x2 (4x11).
    # W_strassen (4x7) -> W_3x2x2 (6x11).
    # W_strassen[k*2+i, c] -> W_3x2x2[k*2+i, c] for i=0,1.
    #
    # Schoolbook part (row 2):
    # U_school (2x4) -> embed into U_3x2x2.
    # U_school[i*2+j, c] -> U_3x2x2[(2+i)*2+j, c+7] = U_3x2x2[4+i*2+j, c+7]. Since i=0, U[4+j, c+7].
    # V_school (4x4) -> V_3x2x2.
    # V_school[j*2+k, c] -> V_3x2x2[j*2+k, c+7].
    # W_school (2x4) -> W_3x2x2.
    # W_school[k*2+i, c] -> W_3x2x2[k*2+(2+i), c+7] = W_3x2x2[k*2+2+i, c+7]. Since i=0, W[k*2+2, c+7].

    U_3x2x2 = [[0]*11 for _ in range(6)]
    V_3x2x2 = [[0]*11 for _ in range(4)]
    W_3x2x2 = [[0]*11 for _ in range(6)]

    # Embed Strassen
    for i in range(2):
        for j in range(2):
            for c in range(7):
                U_3x2x2[i*2+j][c] += strassen_U[i*2+j][c]
    for j in range(2):
        for k in range(2):
            for c in range(7):
                V_3x2x2[j*2+k][c] += strassen_V[j*2+k][c]
    for k in range(2):
        for i in range(2):
            for c in range(7):
                W_3x2x2[k*2+i][c] += strassen_W[k*2+i][c]

    # Embed Schoolbook
    for i in range(1):
        for j in range(2):
            for c in range(4):
                U_3x2x2[(2+i)*2+j][c+7] += schoolbook_1x2x2_U[i*2+j][c]
    for j in range(2):
        for k in range(2):
            for c in range(4):
                V_3x2x2[j*2+k][c+7] += schoolbook_1x2x2_V[j*2+k][c]
    for k in range(2):
        for i in range(1):
            for c in range(4):
                W_3x2x2[k*2+(2+i)][c+7] += schoolbook_1x2x2_W[k*2+i][c]

    # Now assemble 3x4x6 from 3x2x2 blocks.
    # A is 3x4. Split into A_L (3x2) and A_R (3x2).
    # B is 4x6. Split into B_T (2x6) and B_B (2x6).
    # B_T and B_B each split into 3 blocks of 2 cols.
    # For each block j in 0,1,2:
    #   C[:, 2j:2j+2] = A_L @ B_T[:, 2j:2j+2] + A_R @ B_B[:, 2j:2j+2]
    #
    # Each term is a 3x2x2 multiplication.
    # We have 6 such multiplications (2 A-blocks * 3 B-blocks).
    # Each has rank 11. Total rank 66.
    #
    # U indices for 3x4x6: i*4+j. i in {0,1,2}, j in {0,1,2,3}.
    # V indices: j*6+k. j in {0,1,2,3}, k in {0,1,2,3,4,5}.
    # W indices: k*3+i. k in {0,1,2,3,4,5}, i in {0,1,2}.
    #
    # For block j in 0,1,2 (corresponding to B cols 2j, 2j+1):
    #   A_L (i, j in {0,1}) maps to U[i*4 + j, c].
    #   A_R (i, j in {2,3}) maps to U[i*4 + (j-2), c] = U[i*4 + j, c] where j in {2,3}.
    #   Actually, A_L cols are 0,1. A_R cols are 2,3.
    #   In 3x2x2, j is 0,1.
    #   For A_L: U_3x2x2[i*2 + j_local, c] -> U[i*4 + j_local, c].
    #   For A_R: U_3x2x2[i*2 + j_local, c] -> U[i*4 + (2+j_local), c].
    #
    #   B_T rows are 0,1. B_B rows are 2,3.
    #   B_T block j: rows 0,1, cols 2j, 2j+1.
    #   B_B block j: rows 2,3, cols 2j, 2j+1.
    #   In 3x2x2, j (row of B) is 0,1. k (col of B) is 0,1.
    #   For B_T: V_3x2x2[j_local*2 + k_local, c] -> V[j_local*6 + (2j + k_local), c].
    #   For B_B: V_3x2x2[j_local*2 + k_local, c] -> V[(2+j_local)*6 + (2j + k_local), c].
    #
    #   C block j: rows i in {0,1,2}, cols 2j, 2j+1.
    #   In 3x2x2, k (col of C) is 0,1. i (row of C) is 0,1,2.
    #   W_3x2x2[k_local*2 + i, c] -> W[(2j + k_local)*3 + i, c].

    rank_total = 6 * 11
    U = [[0]*rank_total for _ in range(n*m)]
    V = [[0]*rank_total for _ in range(m*p)]
    W = [[0]*rank_total for _ in range(p*n)]

    col_offset = 0
    for b_j in range(3):
        for a_block in range(2): # 0: A_L, 1: A_R
            for c in range(11):
                # U
                for i in range(3):
                    for j_local in range(2):
                        u_val = U_3x2x2[i*2 + j_local][c]
                        if u_val != 0:
                            j_global = j_local if a_block == 0 else 2 + j_local
                            U[i*4 + j_global][col_offset + c] += u_val
                # V
                for j_local in range(2):
                    for k_local in range(2):
                        v_val = V_3x2x2[j_local*2 + k_local][c]
                        if v_val != 0:
                            j_global = j_local if a_block == 0 else 2 + j_local
                            k_global = 2*b_j + k_local
                            V[j_global*6 + k_global][col_offset + c] += v_val
                # W
                for k_local in range(2):
                    for i in range(3):
                        w_val = W_3x2x2[k_local*2 + i][c]
                        if w_val != 0:
                            k_global = 2*b_j + k_local
                            W[k_global*3 + i][col_offset + c] += w_val
            col_offset += 11

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
