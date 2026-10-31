"""Recursive Strassen 4x4x4 matrix multiplication tensor decomposition (rank 49)."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 4
    rank = 49
    U = [[0.0] * rank for _ in range(n * m)]  # 16 x 49
    V = [[0.0] * rank for _ in range(m * p)]  # 16 x 49
    W = [[0.0] * rank for _ in range(p * n)]  # 16 x 49

    # Outer Strassen: 4x4 split into 2x2 blocks of size 2.
    # Outer step o defines linear combinations of A-blocks (4 of them) and B-blocks (4 of them).
    # Each outer step produces a 2x2 block product P_o = A'_o * B'_o.
    # Then each P_o is computed via inner Strassen (2x2x2) with 7 scalar multiplications.

    # Outer Strassen 2x2 block scheme:
    # P_0: A00+A11, B00+B11
    # P_1: A10+A11, B00
    # P_2: A00, B01-B11
    # P_3: A11, B10-B00
    # P_4: A00+A01, B11
    # P_5: A10-A00, B00+B01
    # P_6: A01-A11, B10+B11
    # Block index (r,c) for r,c in {0,1} maps to sub-indices:
    # A block (r,c) covers A[r*2:(r+1)*2, c*2:(c+1)*2]
    # B block (r,c) covers B[r*2:(r+1)*2, c*2:(c+1)*2]
    # C block (r,c) covers C[r*2:(r+1)*2, c*2:(c+1)*2]

    # Outer A coefficients: for each outer step o, which A-blocks and with what coefficient
    # A-block (r,c) row-index base: r*2, col-index base: c*2 -> A index = (r*2+u)*4 + (c*2+v) for u,v in {0,1}
    outer_A = [
        {(0,0): 1, (1,1): 1},    # P0
        {(1,0): 1, (1,1): 1},    # P1
        {(0,0): 1},               # P2
        {(1,1): 1},               # P3
        {(0,0): 1, (0,1): 1},    # P4
        {(1,0): 1, (0,0): -1},   # P5
        {(0,1): 1, (1,1): -1},   # P6
    ]
    outer_B = [
        {(0,0): 1, (1,1): 1},    # P0
        {(0,0): 1},               # P1
        {(0,1): 1, (1,1): -1},   # P2
        {(1,0): 1, (0,0): -1},   # P3
        {(1,1): 1},               # P4
        {(0,0): 1, (0,1): 1},    # P5
        {(1,0): 1, (1,1): 1},    # P6
    ]

    # Output reconstruction: C_block(R,C) = sum_o wRC_o * P_o
    # C00 = P0 - P4 + P6
    # C01 = P2 + P4
    # C10 = P1 + P3
    # C11 = P0 - P1 + P5 + P6
    outer_W = {
        (0,0): {0: 1, 4: -1, 6: 1},
        (0,1): {2: 1, 4: 1},
        (1,0): {1: 1, 3: 1},
        (1,1): {0: 1, 1: -1, 5: 1, 6: 1},
    }

    # Inner Strassen 2x2x2 scheme (for computing 2x2 = 2x2 * 2x2):
    # Inner step i: M_i = (linear combo of A' entries) * (linear combo of B' entries)
    # M0: (a00+a11)*(b00+b11)
    # M1: (a10+a11)*b00
    # M2: a00*(b01-b11)
    # M3: a11*(b10-b00)
    # M4: (a00+a01)*b11
    # M5: (a10-a00)*(b00+b01)
    # M6: (a01-a11)*(b10+b11)
    # Output: C00 = M0 - M4 + M6, C01 = M2 + M4, C10 = M1 + M3, C11 = M0 - M1 + M5 + M6

    inner_A = [
        {(0,0): 1, (1,1): 1},    # M0
        {(1,0): 1, (1,1): 1},    # M1
        {(0,0): 1},               # M2
        {(1,1): 1},               # M3
        {(0,0): 1, (0,1): 1},    # M4
        {(1,0): 1, (0,0): -1},   # M5
        {(0,1): 1, (1,1): -1},   # M6
    ]
    inner_B = [
        {(0,0): 1, (1,1): 1},    # M0
        {(0,0): 1},               # M1
        {(0,1): 1, (1,1): -1},   # M2
        {(1,0): 1, (0,0): -1},   # M3
        {(1,1): 1},               # M4
        {(0,0): 1, (0,1): 1},    # M5
        {(1,0): 1, (1,1): 1},    # M6
    ]

    inner_W = {
        (0,0): {0: 1, 4: -1, 6: 1},
        (0,1): {2: 1, 4: 1},
        (1,0): {1: 1, 3: 1},
        (1,1): {0: 1, 1: -1, 5: 1, 6: 1},
    }

    # Build U, V, W:
    # For outer step o and inner step i, rank column r = o*7 + i.
    # U: A index (row a, col b) -> A[a*4+b]. The A' entry at inner position (u,v) corresponds to
    #   A-block (R,C) where outer step o uses A-block (R,C) with coefficient outer_A[o][(R,C)],
    #   and within that block, inner position (u,v) is at A[(R*2+u), (C*2+v)] = index (R*2+u)*4 + (C*2+v).
    for o in range(7):
        for i in range(7):
            r = o * 7 + i
            # U: for each (u,v) in inner_A[i], the A' entry (u,v) is a linear combo of A-blocks
            for (u, v), coeff_inner in inner_A[i].items():
                # A' (u,v) = sum over A-blocks (R,C) used in outer step o
                for (R, C), coeff_outer in outer_A[o].items():
                    a_row = R * 2 + u
                    a_col = C * 2 + v
                    a_idx = a_row * m + a_col  # = a_row * 4 + a_col
                    U[a_idx][r] += coeff_outer * coeff_inner

            # V: for each (u,v) in inner_B[i], B' entry (u,v)
            for (u, v), coeff_inner in inner_B[i].items():
                for (R, C), coeff_outer in outer_B[o].items():
                    b_row = R * 2 + u
                    b_col = C * 2 + v
                    b_idx = b_row * p + b_col  # = b_row * 4 + b_col
                    V[b_idx][r] += coeff_outer * coeff_inner

            # W: C output at position (a_out, b_out) where a_out in [0,4), b_out in [0,4)
            # C[a_out, b_out] is in C-block (a_out//2, b_out//2) at inner position (a_out%2, b_out%2)
            for R in range(2):
                for C in range(2):
                    for u in range(2):
                        for v in range(2):
                            a_out = R * 2 + u
                            b_out = C * 2 + v
                            # C-block (R,C) at inner pos (u,v) = sum_o outer_W[(R,C)][o] * P_o
                            # P_o at inner pos (u,v) = sum_i inner_W[(u,v)][i] * M_i
                            # For this specific (o,i): contribution = outer_W[(R,C)][o] * inner_W[(u,v)][i]
                            w_rc = outer_W.get((R, C), {}).get(o, 0)
                            w_iv = inner_W.get((u, v), {}).get(i, 0)
                            if w_rc != 0 and w_iv != 0:
                                # W index: k*n + i where k=b_out, i=a_out (trace convention W[k*n+i, r])
                                w_idx = b_out * n + a_out
                                W[w_idx][r] += w_rc * w_iv

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
