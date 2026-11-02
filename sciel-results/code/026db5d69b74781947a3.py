def construct():
    n, m, p = 2, 2, 7
    # Rank 24: 3 blocks of Strassen (rank 7) + 1 block of rank-3 for 2x1 multiplication
    rank = 24

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    # Strassen coefficients for 2x2 * 2x2
    # A indices: 0:A00, 1:A01, 2:A10, 3:A11
    strassen_U = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]

    # C indices in block: 0:C00, 1:C01, 2:C10, 3:C11
    # Global W index = k*n + i where k is column in C, i is row in C
    # For block b, k is 2b or 2b+1
    strassen_W_local = [
        {0: 1, 3: 1},
        {2: 1, 3: -1},
        {1: 1, 3: 1},
        {0: 1, 2: 1},
        {0: -1, 1: 1},
        {3: 1},
        {0: 1}
    ]

    # B indices in block: 0:B00, 1:B01, 2:B10, 3:B11
    # Global V index = j*p + k where j is row in B, k is column in B
    # For block b, k is 2b or 2b+1
    strassen_V_local = [
        {0: 1, 3: 1},
        {0: 1},
        {1: 1, 3: -1},
        {2: 1, 0: -1},
        {3: 1},
        {0: 1, 1: 1},
        {2: 1, 3: 1}
    ]

    # Process 3 Strassen blocks (columns 0-5 of B/C)
    for b in range(3):
        for r in range(7):
            # U
            for i in range(4):
                U[i][col] += strassen_U[r][i]

            # V
            for loc_idx, coeff in strassen_V_local[r].items():
                if loc_idx == 0:
                    v_idx = 0 * p + (2 * b)
                elif loc_idx == 1:
                    v_idx = 0 * p + (2 * b + 1)
                elif loc_idx == 2:
                    v_idx = 1 * p + (2 * b)
                elif loc_idx == 3:
                    v_idx = 1 * p + (2 * b + 1)
                V[v_idx][col] += coeff

            # W
            for loc_idx, coeff in strassen_W_local[r].items():
                if loc_idx == 0:
                    w_idx = (2 * b) * n + 0
                elif loc_idx == 1:
                    w_idx = (2 * b + 1) * n + 0
                elif loc_idx == 2:
                    w_idx = (2 * b) * n + 1
                elif loc_idx == 3:
                    w_idx = (2 * b + 1) * n + 1
                W[w_idx][col] += coeff

            col += 1

    # Rank-3 decomposition for 2x2 * 2x1 (last column, col 6)
    # A is 2x2: A00, A01, A10, A11
    # B is 2x1: B0, B1
    # C is 2x1: C0 = A00*B0 + A01*B1, C1 = A10*B0 + A11*B1
    #
    # We need 3 bilinear forms:
    # M1 = A00 * B0
    # M2 = A01 * B1
    # M3 = A10 * B0
    # M4 = A11 * B1
    # But we can do it in 3:
    # Let's use:
    # M1 = A00 * B0
    # M2 = A01 * B1
    # M3 = (A10 + A11) * (B0 + B1)
    # Then C0 = M1 + M2
    # C1 = M3 - M1 - M2 + A10*B0 + A11*B1 - (A10+B11)*(B0+B1) ... no this doesn't work directly.
    #
    # Actually, 2x2 * 2x1 has 4 terms. Can we do it in 3?
    # The tensor has 4 components. The minimum rank is 4 unless there's structure.
    # Let me reconsider. The rank of [2,2,1] is actually 4 (schoolbook).
    # Wait, no. Let me think again.
    # C0 = A00*B0 + A01*B1
    # C1 = A10*B0 + A11*B1
    # These are 4 independent bilinear terms. The rank is 4.
    #
    # So we can't reduce the last block to rank 3. We need rank 4.
    # Total rank = 3*7 + 4 = 25.
    #
    # But wait, the task says "rank-minimal". Let me verify if rank 24 is possible.
    # The total number of terms is 28 (2x2*2x7 = 2*2*7 = 28).
    # Using 3 Strassen blocks (rank 7 each) handles 6 columns, leaving 1 column.
    # The last column is a 2x2 * 2x1 multiplication, which has rank 4.
    # So minimum rank is 3*7 + 4 = 25.
    #
    # However, maybe we can share some terms between blocks?
    # The Strassen decomposition is local to each 2x2*2x2 block.
    # Since the blocks are independent (different columns of B/C), there's no sharing possible.
    #
    # So rank 25 is indeed minimal. Let me just use rank 25.

    # Reset and use rank 25
    rank = 25
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0

    for b in range(3):
        for r in range(7):
            for i in range(4):
                U[i][col] += strassen_U[r][i]

            for loc_idx, coeff in strassen_V_local[r].items():
                if loc_idx == 0:
                    v_idx = 0 * p + (2 * b)
                elif loc_idx == 1:
                    v_idx = 0 * p + (2 * b + 1)
                elif loc_idx == 2:
                    v_idx = 1 * p + (2 * b)
                elif loc_idx == 3:
                    v_idx = 1 * p + (2 * b + 1)
                V[v_idx][col] += coeff

            for loc_idx, coeff in strassen_W_local[r].items():
                if loc_idx == 0:
                    w_idx = (2 * b) * n + 0
                elif loc_idx == 1:
                    w_idx = (2 * b + 1) * n + 0
                elif loc_idx == 2:
                    w_idx = (2 * b) * n + 1
                elif loc_idx == 3:
                    w_idx = (2 * b + 1) * n + 1
                W[w_idx][col] += coeff

            col += 1

    # Schoolbook for last block (col 6)
    schoolbook_U = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]

    schoolbook_V_local = [
        {0: 1},
        {1: 1},
        {0: 1},
        {1: 1}
    ]

    schoolbook_W_local = [
        {0: 1},
        {0: 1},
        {1: 1},
        {1: 1}
    ]

    for r in range(4):
        for i in range(4):
            U[i][col] += schoolbook_U[r][i]

        for loc_idx, coeff in schoolbook_V_local[r].items():
            if loc_idx == 0:
                v_idx = 0 * p + 6
            elif loc_idx == 1:
                v_idx = 1 * p + 6
            V[v_idx][col] += coeff

        for loc_idx, coeff in schoolbook_W_local[r].items():
            if loc_idx == 0:
                w_idx = 6 * n + 0
            elif loc_idx == 1:
                w_idx = 6 * n + 1
            W[w_idx][col] += coeff

        col += 1

    return {"U": U, "V": V, "W": W}
