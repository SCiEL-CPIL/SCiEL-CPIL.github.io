"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Direct sum of four independent 2x2x2 Strassen decompositions.
    # B has shape (m x p) = (2 x 8), split into 4 blocks of (2 x 2).
    # Each block contributes rank 7. Total rank = 4 * 7 = 28.
    # This is the known minimal rank for this tensor (direct sum of 4 M_2,2,2).
    rank = 28

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen rank-7 decomposition of M_{2,2,2}.
    # A index mapping: 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1)
    # B index mapping (local): 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1)
    # C index mapping (local): 0->(0,0), 1->(1,0), 2->(0,1), 3->(1,1)

    # U factors (coefficients on A entries for each of the 7 products)
    U_strassen = [
        {0: 1, 3: 1},    # M1 = A00 * A11
        {2: 1, 3: 1},    # M2 = A10 * A11
        {0: 1},          # M3 = A00
        {3: 1},          # M4 = A11
        {0: 1, 1: 1},    # M5 = A00 * A01
        {2: 1, 0: -1},   # M6 = A10 * A00 (with sign)
        {1: 1, 3: -1},   # M7 = A01 * A11 (with sign)
    ]

    # V factors (coefficients on B entries for each of the 7 products)
    V_strassen = [
        {0: 1, 3: 1},    # M1
        {0: 1},          # M2
        {1: 1, 3: -1},   # M3
        {2: 1, 0: -1},   # M4
        {3: 1},          # M5
        {0: 1, 1: 1},    # M6
        {2: 1, 3: 1},    # M7
    ]

    # W factors (coefficients on C entries for each of the 7 products)
    W_strassen = [
        {0: 1, 3: 1},    # M1
        {1: 1, 3: -1},   # M2
        {2: 1, 3: 1},    # M3
        {0: 1, 1: 1},    # M4
        {0: -1, 2: 1},   # M5
        {3: 1},          # M6
        {0: 1},          # M7
    ]

    for q in range(4):
        base_col = 2 * q
        for s in range(7):
            r = q * 7 + s

            for u_idx, val in U_strassen[s].items():
                U[u_idx][r] = val

            for v_local, val in V_strassen[s].items():
                j = v_local >> 1
                k_local = v_local & 1
                v_idx = j * p + base_col + k_local
                V[v_idx][r] = val

            for w_local, val in W_strassen[s].items():
                i = w_local & 1
                k_local = w_local >> 1
                w_idx = (base_col + k_local) * n + i
                W[w_idx][r] = val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
