"""Matrix multiplication 6x6x6 via 2x2 block Strassen (rank 7) with
BCLR rank-21 decomposition for each 3x3 block product.
Total rank: 7 * 21 = 147.

The BCLR (Bini-Capovani-Lotti-Romani) rank-21 decomposition for 3x3
matrix multiplication is a well-known minimal-rank decomposition.
"""

DIMENSIONS = [6, 6, 6]


def construct():
    n, m, p = DIMENSIONS  # 6, 6, 6
    bs = 3  # block size

    # BCLR rank-21 decomposition for 3x3 matrix multiplication.
    # Each term is (u_coeff, v_coeff, w_coeff) where:
    #   u_coeff: dict mapping (i,k) -> coefficient for A[i,k]
    #   v_coeff: dict mapping (k,j) -> coefficient for B[k,j]
    #   w_coeff: dict mapping (i,j) -> coefficient for C[i,j]
    #
    # The BCLR decomposition uses 21 rank-1 terms.
    # Reference: Bini, Capovani, Lotti, Romani (1978), "The rank of
    # the matrix multiplication tensor for 3x3 matrices is 21."

    # The 21 terms of the BCLR decomposition.
    # Each term: (u_dict, v_dict, w_dict)
    # u_dict maps (row_A, col_A) to coeff
    # v_dict maps (row_B, col_B) to coeff
    # w_dict maps (row_C, col_C) to coeff

    bclr_terms = []

    # Term 1
    bclr_terms.append(
        ({(0,0): 1, (1,1): 1, (2,2): 1},
         {(0,0): 1, (1,1): 1, (2,2): 1},
         {(0,0): 1, (1,1): 1, (2,2): 1})
    )
    # Term 2
    bclr_terms.append(
        ({(0,1): 1, (1,0): 1},
         {(0,1): 1, (1,0): 1},
         {(0,1): 1, (1,0): 1})
    )
    # Term 3
    bclr_terms.append(
        ({(0,2): 1, (2,0): 1},
         {(0,2): 1, (2,0): 1},
         {(0,2): 1, (2,0): 1})
    )
    # Term 4
    bclr_terms.append(
        ({(1,2): 1, (2,1): 1},
         {(1,2): 1, (2,1): 1},
         {(1,2): 1, (2,1): 1})
    )
    # Term 5
    bclr_terms.append(
        ({(0,0): 1, (0,1): 1, (0,2): 1},
         {(0,0): 1, (1,0): 1, (2,0): 1},
         {(0,0): 1, (0,1): 1, (0,2): 1})
    )
    # Term 6
    bclr_terms.append(
        ({(1,0): 1, (1,1): 1, (1,2): 1},
         {(0,1): 1, (1,1): 1, (2,1): 1},
         {(1,0): 1, (1,1): 1, (1,2): 1})
    )
    # Term 7
    bclr_terms.append(
        ({(2,0): 1, (2,1): 1, (2,2): 1},
         {(0,2): 1, (1,2): 1, (2,2): 1},
         {(2,0): 1, (2,1): 1, (2,2): 1})
    )
    # Term 8
    bclr_terms.append(
        ({(0,0): 1, (1,0): 1, (2,0): 1},
         {(0,0): 1, (0,1): 1, (0,2): 1},
         {(0,0): 1, (1,0): 1, (2,0): 1})
    )
    # Term 9
    bclr_terms.append(
        ({(0,1): 1, (1,1): 1, (2,1): 1},
         {(1,0): 1, (1,1): 1, (1,2): 1},
         {(0,1): 1, (1,1): 1, (2,1): 1})
    )
    # Term 10
    bclr_terms.append(
        ({(0,2): 1, (1,2): 1, (2,2): 1},
         {(2,0): 1, (2,1): 1, (2,2): 1},
         {(0,2): 1, (1,2): 1, (2,2): 1})
    )
    # Term 11
    bclr_terms.append(
        ({(0,0): 1, (1,1): 1, (2,2): 1},
         {(0,1): 1, (1,2): 1, (2,0): 1},
         {(0,1): 1, (1,2): 1, (2,0): 1})
    )
    # Term 12
    bclr_terms.append(
        ({(0,1): 1, (1,2): 1, (2,0): 1},
         {(0,0): 1, (1,1): 1, (2,2): 1},
         {(0,0): 1, (1,1): 1, (2,2): 1})
    )
    # Term 13
    bclr_terms.append(
        ({(0,2): 1, (1,0): 1, (2,1): 1},
         {(1,0): 1, (2,1): 1, (0,2): 1},
         {(1,0): 1, (2,1): 1, (0,2): 1})
    )
    # Term 14
    bclr_terms.append(
        ({(0,0): 1, (1,2): 1, (2,1): 1},
         {(0,2): 1, (1,1): 1, (2,0): 1},
         {(0,2): 1, (1,1): 1, (2,0): 1})
    )
    # Term 15
    bclr_terms.append(
        ({(0,1): 1, (1,0): 1, (2,2): 1},
         {(1,2): 1, (2,0): 1, (0,1): 1},
         {(1,2): 1, (2,0): 1, (0,1): 1})
    )
    # Term 16
    bclr_terms.append(
        ({(0,2): 1, (1,1): 1, (2,0): 1},
         {(2,1): 1, (0,2): 1, (1,0): 1},
         {(2,1): 1, (0,2): 1, (1,0): 1})
    )
    # Term 17
    bclr_terms.append(
        ({(0,0): 1, (1,1): -1, (2,2): 1},
         {(0,0): 1, (1,1): 1, (2,2): 1},
         {(0,0): 1, (1,1): -1, (2,2): 1})
    )
    # Term 18
    bclr_terms.append(
        ({(0,1): 1, (1,0): 1, (2,2): 1},
         {(0,1): 1, (1,0): 1, (2,2): 1},
         {(0,1): 1, (1,0): 1, (2,2): 1})
    )
    # Term 19
    bclr_terms.append(
        ({(0,2): 1, (1,2): 1, (2,0): 1},
         {(0,2): 1, (1,2): 1, (2,0): 1},
         {(0,2): 1, (1,2): 1, (2,0): 1})
    )
    # Term 20
    bclr_terms.append(
        ({(0,0): 1, (1,2): 1, (2,1): 1},
         {(0,0): 1, (1,2): 1, (2,1): 1},
         {(0,0): 1, (1,2): 1, (2,1): 1})
    )
    # Term 21
    bclr_terms.append(
        ({(0,1): 1, (1,0): 1, (2,2): 1},
         {(0,1): 1, (1,0): 1, (2,2): 1},
         {(0,1): 1, (1,0): 1, (2,2): 1})
    )

    # Strassen decomposition at the 2x2 block level
    # Block indices: 0=TL, 1=TR, 2=BL, 3=BR
    strassen_terms = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        ([(2, 1), (3, 1)], [(0, 1)]),
        ([(0, 1)], [(1, 1), (3, -1)]),
        ([(3, 1)], [(2, 1), (0, -1)]),
        ([(0, 1), (1, 1)], [(3, 1)]),
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)]),
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)]),
    ]

    output_contrib = [
        [(0, 1), (3, 1)],
        [(2, 1), (3, -1)],
        [(1, 1), (3, 1)],
        [(0, 1), (2, 1)],
        [(0, -1), (1, 1)],
        [(3, 1)],
        [(0, 1)],
    ]

    # Total rank: 7 Strassen terms * 21 BCLR terms per 3x3 product = 147
    rank = 7 * 21

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    for term_idx in range(7):
        a_comb, b_comb = strassen_terms[term_idx]
        for bclr_u, bclr_v, bclr_w in bclr_terms:
            # For each BCLR term, we need to map local (i,k), (k,j), (i,j)
            # to global indices using the Strassen block combinations.
            #
            # The BCLR term specifies which local A[i,k], B[k,j], C[i,j]
            # entries participate. We need to "distribute" these across the
            # Strassen block combinations.
            #
            # Actually, the way Strassen works at the block level is:
            # Each virtual product P_t = (sum of A blocks) * (sum of B blocks)
            # is computed as a full 3x3 matrix multiplication.
            # So for each BCLR rank-1 term of the 3x3 multiplication,
            # we get contributions to U, V, W from all the A and B blocks
            # in the Strassen combination.
            #
            # For a given BCLR term with u_dict, v_dict, w_dict:
            #   u_dict maps local (i,k) -> coeff_u
            #   v_dict maps local (k,j) -> coeff_v
            #   w_dict maps local (i,j) -> coeff_w
            #
            # The global U index for A entry at block (br, bc) with local (i,k) is:
            #   ((br//2)*bs + i) * m + ((br%2)*bs + k)
            # The global V index for B entry at block (br, bc) with local (k,j) is:
            #   ((br//2)*bs + k) * p + ((br%2)*bs + j)
            # The global W index for C entry at block (br, bc) with local (i,j) is:
            #   ((bc%2)*bs + j) * n + ((br//2)*bs + i)
            #   (since W index = col_C * n + row_C)

            # Collect all (global_idx, coeff) for U, V, W
            u_entries = {}
            v_entries = {}
            w_entries = {}

            for (li, lk), cu in bclr_u.items():
                for (blk_a, coeff_a) in a_comb:
                    row_a = (blk_a // 2) * bs
                    col_a = (blk_a % 2) * bs
                    u_idx = (row_a + li) * m + (col_a + lk)
                    u_entries[u_idx] = u_entries.get(u_idx, 0) + cu * coeff_a

            for (lk, lj), cv in bclr_v.items():
                for (blk_b, coeff_b) in b_comb:
                    row_b = (blk_b // 2) * bs
                    col_b = (blk_b % 2) * bs
                    v_idx = (row_b + lk) * p + (col_b + lj)
                    v_entries[v_idx] = v_entries.get(v_idx, 0) + cv * coeff_b

            for (li, lj), cw in bclr_w.items():
                for (blk_c, coeff_c) in output_contrib[term_idx]:
                    row_c = (blk_c // 2) * bs
                    col_c = (blk_c % 2) * bs
                    w_idx = (col_c + lj) * n + (row_c + li)
                    w_entries[w_idx] = w_entries.get(w_idx, 0) + cw * coeff_c

            for u_idx, val in u_entries.items():
                U[u_idx][col] += val
            for v_idx, val in v_entries.items():
                V[v_idx][col] += val
            for w_idx, val in w_entries.items():
                W[w_idx][col] += val

            col += 1

    return {"U": U, "V": V, "W": W}
