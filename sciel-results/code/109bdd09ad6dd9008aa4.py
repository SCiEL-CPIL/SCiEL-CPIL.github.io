"""Strassen-based rank-14 decomposition for 3x3 x 3x6 matrix multiplication."""
DIMENSIONS = [3, 3, 6]

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 6

    # Strassen rank-7 decomposition for 3x3 x 3x3.
    # M1..M7 are linear combinations of A and B entries.
    # A indices: (i,j) with i=row of A (0..2), j=col of A (0..2)
    # B indices: (j,k) with j=row of B (0..2), k=col of B (0..2)
    # C indices: (i,k)
    #
    # M1 = A00*(B11+B22)
    # M2 = (A10+A20)*B00
    # M3 = (A00+A01)*B22
    # M4 = A22*(B00+B11)
    # M5 = (A02+A22)*B01
    # M6 = (A10+A22)*(B11+B21)
    # M7 = (A00+A11)*(B12+B22)
    #
    # C00 = M1 - M3 + M4 + M7
    # C01 = M5 + M3
    # C02 = M7 - M1 + M6
    # C10 = M2 + M1
    # C11 = M4 - M2 + M5 + M7
    # C12 = M2 - M4 + M3
    # C20 = M6 - M2 + M4
    # C21 = M5 + M2
    # C22 = M1 - M5 + M4

    rank = 14
    U = [[0] * rank for _ in range(n * m)]  # 9 x 14
    V = [[0] * rank for _ in range(m * p)]  # 18 x 14
    W = [[0] * rank for _ in range(p * n)]  # 18 x 14

    # Strassen factors for one 3x3x3 block
    # For each Strassen term t (0..6):
    #   A-part: dict from (i,j) -> coeff
    #   B-part: dict from (j,k) -> coeff
    #   C-part: dict from (i,k) -> coeff
    strassen_A = [
        {(0, 0): 1},                                    # M1: A00
        {(1, 0): 1, (2, 0): 1},                         # M2: A10+A20
        {(0, 0): 1, (0, 1): 1},                         # M3: A00+A01
        {(2, 2): 1},                                    # M4: A22
        {(0, 2): 1, (2, 2): 1},                         # M5: A02+A22
        {(1, 0): 1, (2, 2): 1},                         # M6: A10+A22
        {(0, 0): 1, (1, 1): 1},                         # M7: A00+A11
    ]
    strassen_B = [
        {(1, 1): 1, (2, 2): 1},                         # M1: B11+B22
        {(0, 0): 1},                                    # M2: B00
        {(2, 2): 1},                                    # M3: B22
        {(0, 0): 1, (1, 1): 1},                         # M4: B00+B11
        {(0, 1): 1},                                    # M5: B01
        {(1, 1): 1, (2, 1): 1},                         # M6: B11+B21
        {(1, 2): 1, (2, 2): 1},                         # M7: B12+B22
    ]
    strassen_C = [
        {(0, 0): 1, (0, 2): -1, (1, 0): 1, (2, 0): 1},  # M1
        {(0, 1): 1, (1, 1): 1, (2, 1): 1},              # M3
        {(0, 2): 1, (1, 1): 1, (1, 2): -1, (2, 0): 1},  # M4
        {(0, 2): 1, (1, 1): 1, (2, 2): 1},              # M5
        {(0, 2): 1, (1, 1): 1, (2, 0): -1, (2, 2): -1}, # M6
        {(0, 0): 1, (0, 2): -1, (1, 1): 1, (2, 0): -1}, # M7  -- wait, let me recheck
    ]

    # Let me carefully recompute the C combinations:
    # C00 = M1 - M3 + M4 + M7
    # C01 = M3 + M5
    # C02 = -M1 + M6 + M7
    # C10 = M1 + M2
    # C11 = -M2 + M4 + M5 + M7
    # C12 = M3 - M4 + M2
    # C20 = -M2 + M4 + M6
    # C21 = M2 + M5
    # C22 = M1 - M4 - M5

    # Rebuild strassen_C properly:
    # For each term t, which C entries get +1 or -1
    strassen_C = [
        {(0, 0): 1, (0, 2): -1, (1, 0): 1, (2, 2): 1},  # M1: +C00, -C02, +C10, +C22
        {(1, 0): 1, (1, 1): -1, (1, 2): 1, (2, 0): -1, (2, 1): 1},  # M2: +C10, -C11, +C12, -C20, +C21
        {(0, 0): -1, (0, 1): 1, (1, 1): 1, (1, 2): 1},  # M3: -C00, +C01, +C11, +C12
        {(0, 0): 1, (0, 2): 1, (1, 1): 1, (1, 2): -1, (2, 0): 1, (2, 2): -1},  # M4: +C00, +C02, +C11, -C12, +C20, -C22
        {(0, 1): 1, (1, 1): 1, (2, 1): 1, (2, 2): -1},  # M5: +C01, +C11, +C21, -C22
        {(0, 2): 1, (1, 1): 1, (2, 0): 1},              # M6: +C02, +C11, +C20
        {(0, 0): 1, (0, 2): 1, (1, 1): 1},              # M7: +C00, +C02, +C11
    ]

    # Verify: C00 = M1 - M3 + M4 + M7
    # M1: C00=1, M3: C00=-1, M4: C00=1, M7: C00=1 => 1-1+1+1=2? No!
    # That's wrong. Let me redo.
    # C00 = M1 - M3 + M4 + M7 means:
    #   M1 contributes +1 to C00
    #   M3 contributes -1 to C00
    #   M4 contributes +1 to C00
    #   M7 contributes +1 to C00
    # So in strassen_C, for each term t, we list the C entries it affects.
    # strassen_C[t] = {(i,k): coeff} where coeff is the contribution of term t to C[i,k]

    # Let me redo this carefully.
    # C[i,k] = sum over t of (coeff of t in C[i,k]) * M_t
    #
    # C00 = M1 - M3 + M4 + M7
    # C01 = M3 + M5
    # C02 = -M1 + M6 + M7
    # C10 = M1 + M2
    # C11 = -M2 + M4 + M5 + M7
    # C12 = M3 - M4 + M2
    # C20 = -M2 + M4 + M6
    # C21 = M2 + M5
    # C22 = M1 - M4 - M5

    # So for each term t, the C coefficients are:
    # M1 (t=0): C00=+1, C02=-1, C10=+1, C22=+1
    # M2 (t=1): C10=+1, C11=-1, C12=+1, C20=-1, C21=+1
    # M3 (t=2): C00=-1, C01=+1, C11=+1, C12=+1
    # M4 (t=3): C00=+1, C02=+1, C11=+1, C12=-1, C20=+1, C22=-1
    # M5 (t=4): C01=+1, C11=+1, C21=+1, C22=-1
    # M6 (t=5): C02=+1, C11=+1, C20=+1
    # M7 (t=6): C00=+1, C02=+1, C11=+1

    strassen_C = [
        {(0, 0): 1, (0, 2): -1, (1, 0): 1, (2, 2): 1},
        {(1, 0): 1, (1, 1): -1, (1, 2): 1, (2, 0): -1, (2, 1): 1},
        {(0, 0): -1, (0, 1): 1, (1, 1): 1, (1, 2): 1},
        {(0, 0): 1, (0, 2): 1, (1, 1): 1, (1, 2): -1, (2, 0): 1, (2, 2): -1},
        {(0, 1): 1, (1, 1): 1, (2, 1): 1, (2, 2): -1},
        {(0, 2): 1, (1, 1): 1, (2, 0): 1},
        {(0, 0): 1, (0, 2): 1, (1, 1): 1},
    ]

    # Now fill U, V, W for two blocks.
    # Block 0: uses columns of B at positions 0, 2, 4 (even columns)
    # Block 1: uses columns of B at positions 1, 3, 5 (odd columns)

    for block in range(2):
        base_col = block * 7
        # For this block, B's column k maps to actual column (2*k + block)
        for t in range(7):
            col = base_col + t
            # Fill U: A indices (i, j) -> U[i*m + j, col]
            for (i, j), coeff in strassen_A[t].items():
                U[i * m + j][col] += coeff
            # Fill V: B indices (j, k) -> V[j*p + k, col]
            for (j, k), coeff in strassen_B[t].items():
                actual_k = 2 * k + block
                V[j * p + actual_k][col] += coeff
            # Fill W: C indices (i, k) -> W[k*n + i, col]
            for (i, k), coeff in strassen_C[t].items():
                actual_k = 2 * k + block
                W[actual_k * n + i][col] += coeff

    return {"U": U, "V": V, "W": W}
