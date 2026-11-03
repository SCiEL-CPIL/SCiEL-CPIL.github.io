import time

def construct():
    start_time = time.time()
    deadline = start_time + 100.0

    n, m, p = 4, 4, 5

    # Strategy: Use Strassen for 4x4x4 part (rank 49) + schoolbook for 4x4x1 part (rank 16)
    # Total rank = 65

    rank = 65
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    def add_rank1_term(u_lf, v_lf, w_lf, col):
        """Add a rank-1 term where u_lf, v_lf, w_lf are lists of (index, coeff) tuples."""
        for idx, c in u_lf:
            if c != 0:
                U[idx][col] += c
        for idx, c in v_lf:
            if c != 0:
                V[idx][col] += c
        for idx, c in w_lf:
            if c != 0:
                W[idx][col] += c

    # Strassen 2x2x2 rank-7 decomposition
    # For matrices X (2x2), Y (2x2), Z = X*Y (2x2)
    # X indices: x00, x01, x10, x11
    # Y indices: y00, y01, y10, y11
    # Z indices: z00, z01, z10, z11
    #
    # p1 = (x00+x11)(y00+y11) -> z00: +1, z11: +1
    # p2 = (x10+x11)y00       -> z10: +1, z11: -1
    # p3 = x00(y10-y11)       -> z10: +1, z01: +1
    # p4 = x11(y01-y00)       -> z01: +1, z00: -1
    # p5 = (x00+x01)y11       -> z00: +1, z11: +1
    # p6 = (x10-x00)(y00+y01) -> z00: +1, z10: +1
    # p7 = (x01-x11)(y10+y11) -> z11: +1, z01: -1

    # Each term is: u_form (dict local_idx->coeff), v_form, w_form
    # Local indices: 0=x00, 1=x01, 2=x10, 3=x11 for X
    #                0=y00, 1=y01, 2=y10, 3=y11 for Y
    #                0=z00, 1=z01, 2=z10, 3=z11 for Z

    s2x2_terms = [
        # p1: (x00+x11)(y00+y11) -> z00+1, z11+1
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # p2: (x10+x11)y00 -> z10+1, z11-1
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # p3: x00(y10-y11) -> z10+1, z01+1
        ({0: 1}, {2: 1, 3: -1}, {2: 1, 1: 1}),
        # p4: x11(y01-y00) -> z01+1, z00-1
        ({3: 1}, {1: 1, 0: -1}, {1: 1, 0: -1}),
        # p5: (x00+x01)y11 -> z00+1, z11+1
        ({0: 1, 1: 1}, {3: 1}, {0: 1, 3: 1}),
        # p6: (x10-x00)(y00+y01) -> z00+1, z10+1
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {0: 1, 2: 1}),
        # p7: (x01-x11)(y10+y11) -> z11+1, z01-1
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {3: 1, 1: -1}),
    ]

    # A block indices (i*m+j):
    # A00: (0,0),(0,1),(1,0),(1,1) -> [0, 1, 4, 5]
    # A01: (0,2),(0,3),(1,2),(1,3) -> [2, 3, 6, 7]
    # A10: (2,0),(2,1),(3,0),(3,1) -> [8, 9, 12, 13]
    # A11: (2,2),(2,3),(3,2),(3,3) -> [10, 11, 14, 15]

    A00 = [0, 1, 4, 5]
    A01 = [2, 3, 6, 7]
    A10 = [8, 9, 12, 13]
    A11 = [10, 11, 14, 15]

    # B block indices (j*p+k):
    # B00: (0,0),(0,1),(1,0),(1,1) -> [0, 1, 5, 6]
    # B01: (0,2),(0,3),(1,2),(1,3) -> [2, 3, 7, 8]
    # B10: (2,0),(2,1),(3,0),(3,1) -> [10, 11, 15, 16]
    # B11: (2,2),(2,3),(3,2),(3,3) -> [12, 13, 17, 18]

    B00 = [0, 1, 5, 6]
    B01 = [2, 3, 7, 8]
    B10 = [10, 11, 15, 16]
    B11 = [12, 13, 17, 18]

    # C block indices (k*n+i):
    # C00: (0,0),(0,1),(1,0),(1,1) -> [0, 1, 4, 5]
    # C01: (0,2),(0,3),(1,2),(1,3) -> [2, 3, 6, 7]
    # C10: (2,0),(2,1),(3,0),(3,1) -> [8, 9, 12, 13]
    # C11: (2,2),(2,3),(3,2),(3,3) -> [10, 11, 14, 15]

    C00 = [0, 1, 4, 5]
    C01 = [2, 3, 6, 7]
    C10 = [8, 9, 12, 13]
    C11 = [10, 11, 14, 15]

    # Define the 7 Strassen products with their A, B, C block combinations
    # Each product: (X_lfs, Y_lfs, Z_lfs) where each is a list of 4 elements,
    # each element is a list of (global_index, coeff) tuples

    # P1 = (A00+A11)*(B00+B11) -> contributes to C00, C01, C10, C11
    P1_X = [
        [(A00[0], 1), (A11[0], 1)],
        [(A00[1], 1), (A11[1], 1)],
        [(A00[2], 1), (A11[2], 1)],
        [(A00[3], 1), (A11[3], 1)]
    ]
    P1_Y = [
        [(B00[0], 1), (B11[0], 1)],
        [(B00[1], 1), (B11[1], 1)],
        [(B00[2], 1), (B11[2], 1)],
        [(B00[3], 1), (B11[3], 1)]
    ]
    P1_Z = [
        [(C00[0], 1)],
        [(C01[0], 1)],
        [(C10[0], 1)],
        [(C11[0], 1)]
    ]
    # Wait, P1 contributes to all 4 C blocks with coefficient +1
    # But each C block has 4 entries, and Z has 4 entries
    # So Z00->C00, Z01->C01, Z10->C10, Z11->C11
    # Each Z entry maps to one C block entry
    P1_Z = [
        [(C00[0], 1), (C00[1], 0), (C00[2], 0), (C00[3], 0)],  # This is wrong
    ]

    # Let me redo this. Z00 maps to C00[0], Z01 maps to C01[0], etc.
    # Actually no. Z is a 2x2 matrix, and each block of C is 2x2.
    # Z00 -> C00[0], Z01 -> C01[0], Z10 -> C10[0], Z11 -> C11[0]
    # But that's only one entry per block. We need Z00->C00[0], and also...
    
    # I think I'm confusing myself. Let me reconsider.
    
    # When we say P1 = (A00+A11)*(B00+B11), this is a 2x2 * 2x2 = 2x2 product.
    # The result is a 2x2 matrix. This 2x2 matrix contributes to...
    # Actually, it contributes to ALL four C blocks? No.
    
    # Let me re-read the Strassen formula:
    # C00 = P1 - P4 + P5 + P6
    # C01 = P3 + P4
    # C10 = P2 + P3
    # C11 = P1 - P2 + P5 + P7
    
    # So P1 is a 2x2 matrix that adds to C00 and C11.
    # Specifically, P1[0,0] adds to C00[0,0], P1[0,1] adds to C01[0,0]? No.
    
    # I think the issue is that each Pi is a 2x2 matrix, and when we say
    # "C00 = P1 - P4 + P5 + P6", we mean element-wise addition of 2x2 matrices.
    # So P1[0,0] -> C00[0,0], P1[0,1] -> C00[0,1], P1[1,0] -> C00[1,0], P1[1,1] -> C00[1,1]
    
    # So for P1, which contributes to C00 and C11:
    # Z00 -> C00[0], Z01 -> C00[1], Z10 -> C00[2], Z11 -> C00[3]
    # AND Z00 -> C11[0], Z01 -> C11[1], Z10 -> C11[2], Z11 -> C11[3]
    
    # So P1_Z should have both C00 and C11 entries
    P1_Z = [
        [(C00[0], 1), (C11[0], 1)],
        [(C00[1], 1), (C11[1], 1)],
        [(C00[2], 1), (C11[2], 1)],
        [(C00[3], 1), (C11[3], 1)]
    ]

    # P2 = (A10+A11)*B00 -> contributes to C10 (+1) and C11 (-1)
    P2_X = [
        [(A10[0], 1), (A11[0], 1)],
        [(A10[1], 1), (A11[1], 1)],
        [(A10[2], 1), (A11[2], 1)],
        [(A10[3], 1), (A11[3], 1)]
    ]
    P2_Y = [
        [(B00[0], 1)],
        [(B00[1], 1)],
        [(B00[2], 1)],
        [(B00[3], 1)]
    ]
    P2_Z = [
        [(C10[0], 1), (C11[0], -1)],
        [(C10[1], 1), (C11[1], -1)],
        [(C10[2], 1), (C11[2], -1)],
        [(C10[3], 1), (C11[3], -1)]
    ]

    # P3 = A00*(B10-B11) -> contributes to C01 (+1) and C10 (+1)
    P3_X = [
        [(A00[0], 1)],
        [(A00[1], 1)],
        [(A00[2], 1)],
        [(A00[3], 1)]
    ]
    P3_Y = [
        [(B10[0], 1), (B11[0], -1)],
        [(B10[1], 1), (B11[1], -1)],
        [(B10[2], 1), (B11[2], -1)],
        [(B10[3], 1), (B11[3], -1)]
    ]
    P3_Z = [
        [(C01[0], 1), (C10[0], 1)],
        [(C01[1], 1), (C10[1], 1)],
        [(C01[2], 1), (C10[2], 1)],
        [(C01[3], 1), (C10[3], 1)]
    ]

    # P4 = A11*(B01-B00) -> contributes to C00 (-1) and C01 (+1)
    P4_X = [
        [(A11[0], 1)],
        [(A11[1], 1)],
        [(A11[2], 1)],
        [(A11[3], 1)]
    ]
    P4_Y = [
        [(B01[0], 1), (B00[0], -1)],
        [(B01[1], 1), (B00[1], -1)],
        [(B01[2], 1), (B00[2], -1)],
        [(B01[3], 1), (B00[3], -1)]
    ]
    P4_Z = [
        [(C00[0], -1), (C01[0], 1)],
        [(C00[1], -1), (C01[1], 1)],
        [(C00[2], -1), (C01[2], 1)],
        [(C00[3], -1), (C01[3], 1)]
    ]

    # P5 = (A00+A01)*B11 -> contributes to C00 (+1) and C11 (+1)
    P5_X = [
        [(A00[0], 1), (A01[0], 1)],
        [(A00[1], 1), (A01[1], 1)],
        [(A00[2], 1), (A01[2], 1)],
        [(A00[3], 1), (A01[3], 1)]
    ]
    P5_Y = [
        [(B11[0], 1)],
        [(B11[1], 1)],
        [(B11[2], 1)],
        [(B11[3], 1)]
    ]
    P5_Z = [
        [(C00[0], 1), (C11[0], 1)],
        [(C00[1], 1), (C11[1], 1)],
        [(C00[2], 1), (C11[2], 1)],
        [(C00[3], 1), (C11[3], 1)]
    ]

    # P6 = (A10-A00)*(B00+B01) -> contributes to C00 (+1) and C10 (+1)
    P6_X = [
        [(A10[0], 1), (A00[0], -1)],
        [(A10[1], 1), (A00[1], -1)],
        [(A10[2], 1), (A00[2], -1)],
        [(A10[3], 1), (A00[3], -1)]
    ]
    P6_Y = [
        [(B00[0], 1), (B01[0], 1)],
        [(B00[1], 1), (B01[1], 1)],
        [(B00[2], 1), (B01[2], 1)],
        [(B00[3], 1), (B01[3], 1)]
    ]
    P6_Z = [
        [(C00[0], 1), (C10[0], 1)],
        [(C00[1], 1), (C10[1], 1)],
        [(C00[2], 1), (C10[2], 1)],
        [(C00[3], 1), (C10[3], 1)]
    ]

    # P7 = (A01-A11)*(B10+B11) -> contributes to C11 (+1) and C01 (-1)
    P7_X = [
        [(A01[0], 1), (A11[0], -1)],
        [(A01[1], 1), (A11[1], -1)],
        [(A01[2], 1), (A11[2], -1)],
        [(A01[3], 1), (A11[3], -1)]
    ]
    P7_Y = [
        [(B10[0], 1), (B11[0], 1)],
        [(B10[1], 1), (B11[1], 1)],
        [(B10[2], 1), (B11[2], 1)],
        [(B10[3], 1), (B11[3], 1)]
    ]
    P7_Z = [
        [(C11[0], 1), (C01[0], -1)],
        [(C11[1], 1), (C01[1], -1)],
        [(C11[2], 1), (C01[2], -1)],
        [(C11[3], 1), (C01[3], -1)]
    ]

    products = [
        (P1_X, P1_Y, P1_Z),
        (P2_X, P2_Y, P2_Z),
        (P3_X, P3_Y, P3_Z),
        (P4_X, P4_Y, P4_Z),
        (P5_X, P5_Y, P5_Z),
        (P6_X, P6_Y, P6_Z),
        (P7_X, P7_Y, P7_Z)
    ]

    for prod_idx, (X_lfs, Y_lfs, Z_lfs) in enumerate(products):
        for term_idx, (u_form, v_form, w_form) in enumerate(s2x2_terms):
            col = prod_idx * 7 + term_idx
            u_lf = []
            for local_idx, coeff in u_form.items():
                for idx, c in X_lfs[local_idx]:
                    u_lf.append((idx, c * coeff))
            
            v_lf = []
            for local_idx, coeff in v_form.items():
                for idx, c in Y_lfs[local_idx]:
                    v_lf.append((idx, c * coeff))
                    
            w_lf = []
            for local_idx, coeff in w_form.items():
                for idx, c in Z_lfs[local_idx]:
                    w_lf.append((idx, c * coeff))
                    
            add_rank1_term(u_lf, v_lf, w_lf, col)

    # Part 2: 4x4x1 Schoolbook
    # C[i, 4] = sum_j A[i, j] B[j, 4]
    for i in range(4):
        for j in range(4):
            col = 49 + i * 4 + j
            U[i * 4 + j][col] = 1
            V[j * 5 + 4][col] = 1
            W[4 * 4 + i][col] = 1

    return {"U": U, "V": V, "W": W}
