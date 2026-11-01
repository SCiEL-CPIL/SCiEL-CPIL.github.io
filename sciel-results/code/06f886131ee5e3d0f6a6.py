"""Deterministic Kronecker-product decomposition for the 2x5 x 5x6 tensor."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS

    # Build the two sub-problems:
    #   S1: 2 x 1 x 6  -> rank 12 (trivial outer-product)
    #   S2: 2 x 4 x 6  -> use two 2x2x6 Strassen blocks -> rank 2 * 7 * 3 = 42
    # Total rank = 12 + 42 = 54

    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    # ---- Sub-problem S1: 2 x 1 x 6 ----
    # A[:, 0] * B[0, :]
    for i in range(n):
        for k in range(p):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1

    # ---- Sub-problem S2: 2 x 4 x 6 ----
    # Split the 4 columns of A into two groups of 2: j in {1,2} and j in {3,4}
    # Split the 6 rows of B into three groups of 2: k in {0,1}, {2,3}, {4,5}
    # Each 2x2x2 block uses 7 Strassen terms

    # Strassen terms for 2x2x2:
    # Each term is (u_list, v_list, w_list) where each list contains (i, j, val) tuples
    # Local indices: A has (i,j) in {0,1}x{0,1}, B has (j,k) in {0,1}x{0,1}, C has (i,k) in {0,1}x{0,1}

    def strassen_2x2x2_terms():
        return [
            # T1: (a+d)(e+h) -> C00 += 1, C11 += 1
            ([(0, 0, 1), (1, 1, 1)], [(0, 0, 1), (1, 1, 1)], [(0, 0, 1), (1, 1, 1)]),
            # T2: (c+d)e -> C10 += 1, C11 -= 1
            ([(1, 0, 1), (1, 1, 1)], [(0, 0, 1)], [(1, 0, 1), (1, 1, -1)]),
            # T3: a(f-h) -> C01 += 1, C11 += 1
            ([(0, 0, 1)], [(0, 1, 1), (1, 1, -1)], [(0, 1, 1), (1, 1, 1)]),
            # T4: d(g-e) -> C00 += 1, C10 += 1
            ([(1, 1, 1)], [(1, 0, 1), (0, 0, -1)], [(0, 0, 1), (1, 0, 1)]),
            # T5: (a+b)h -> C00 -= 1, C01 += 1
            ([(0, 0, 1), (0, 1, 1)], [(1, 1, 1)], [(0, 0, -1), (0, 1, 1)]),
            # T6: (c-a)(e+f) -> C11 += 1
            ([(1, 0, 1), (0, 0, -1)], [(0, 0, 1), (0, 1, 1)], [(1, 1, 1)]),
            # T7: (b-d)(g+h) -> C00 += 1
            ([(0, 1, 1), (1, 1, -1)], [(1, 0, 1), (1, 1, 1)], [(0, 0, 1)]),
        ]

    terms = strassen_2x2x2_terms()

    # For each of the two j-blocks (j_base = 1 or 3)
    for j_block in range(2):
        j_base = 1 + j_block * 2
        # For each of the three k-blocks (k_base = 0, 2, 4)
        for k_block in range(3):
            k_base = k_block * 2
            # Apply all 7 Strassen terms
            for u_list, v_list, w_list in terms:
                for (li, lj, val) in u_list:
                    gi = li
                    gj = j_base + lj
                    U[gi * m + gj][col] += val
                for (lj, lk, val) in v_list:
                    gj = j_base + lj
                    gk = k_base + lk
                    V[gj * p + gk][col] += val
                for (li, lk, val) in w_list:
                    gi = li
                    gk = k_base + lk
                    W[gk * n + gi][col] += val
                col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
