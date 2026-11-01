"""Matrix multiplication 2x2x7 via block-diagonalization with Strassen for 2x2x2 blocks."""

def construct():
    n, m, p = 2, 2, 7
    # Strassen rank-7 coefficients for 2x2x2
    # M0 = A00*B00, M1 = (A00+A11)*B11, M2 = (A10-A00)*B01,
    # M3 = A11*(B01+B10), M4 = (A00+A01)*B10, M5 = (A10+A11)*(B00-B10),
    # M6 = (A01-A11)*(B10+B11)
    # C00 = M0 + M3 - M5 - M6, C01 = M4 + M6, C10 = M2 + M5, C11 = M1 + M3 - M2 - M4

    # U coefficients: which A_ij contributes to each M (as 4-vector [A00, A01, A10, A11])
    strassen_U = [
        [1, 0, 0, 0],  # M0
        [1, 0, 0, 1],  # M1
        [0, 0, 1, 0],  # M2
        [0, 0, 0, 1],  # M3
        [1, 1, 0, 0],  # M4
        [0, 0, 1, 1],  # M5
        [0, 1, 0, -1], # M6
    ]

    # V coefficients: which B_jk contributes to each M (as 4-vector [B00, B01, B10, B11])
    strassen_V = [
        [1, 0, 0, 0],  # M0
        [0, 0, 0, 1],  # M1
        [0, 1, 0, 0],  # M2
        [0, 1, 1, 0],  # M3
        [0, 0, 1, 0],  # M4
        [1, 0, -1, 0], # M5
        [0, 0, 1, 1],  # M6
    ]

    # W coefficients: which C_ik each M contributes to (as 4-vector [C00, C01, C10, C11])
    strassen_W = [
        [1, 0, 0, 0],  # M0
        [0, 0, 0, 1],  # M1
        [0, 0, 1, 0],  # M2
        [1, 0, 0, 1],  # M3
        [0, 1, 0, 0],  # M4
        [-1, 0, 1, 0], # M5
        [-1, 1, 0, 0], # M6
    ]

    # Total rank: 3 blocks of Strassen (7 each) + 1 block of schoolbook (4) = 25
    rank = 3 * 7 + 4
    U = [[0] * rank for _ in range(n * m)]   # 4 x 25
    V = [[0] * rank for _ in range(m * p)]   # 14 x 25
    W = [[0] * rank for _ in range(p * n)]   # 14 x 25

    col = 0
    # Process three 2-column blocks using Strassen
    for block in range(3):
        k_offset = block * 2  # global k starts at k_offset
        for s in range(7):
            # Fill U: rows are i*m+j = i*2+j
            for ij in range(4):
                i = ij // 2
                j = ij % 2
                U[i * m + j][col] = strassen_U[s][ij]

            # Fill V: rows are j*p+k
            for jk in range(4):
                j = jk // 2
                k_local = jk % 2
                k_global = k_offset + k_local
                V[j * p + k_global][col] = strassen_V[s][jk]

            # Fill W: rows are k*n+i
            for ki in range(4):
                k_local = ki // 2
                i = ki % 2
                k_global = k_offset + k_local
                W[k_global * n + i][col] = strassen_W[s][ki]

            col += 1

    # Process the last 1-column block (k=6) using schoolbook rank-4
    k_offset = 6
    # Schoolbook: C[i,k] = sum_j A[i,j] * B[j,k]
    # For a single k, 4 terms: (i=0,j=0), (i=0,j=1), (i=1,j=0), (i=1,j=1)
    schoolbook_terms = [
        (0, 0),  # A00 * B06
        (0, 1),  # A01 * B16
        (1, 0),  # A10 * B06
        (1, 1),  # A11 * B16
    ]
    for (i, j) in schoolbook_terms:
        # U: A[i,j]
        U[i * m + j][col] = 1
        # V: B[j, k_offset]
        V[j * p + k_offset][col] = 1
        # W: C[i, k_offset]
        W[k_offset * n + i][col] = 1
        col += 1

    return {"U": U, "V": V, "W": W}
