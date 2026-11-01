"""Rank-minimal tensor decomposition for 2x2 * 2x6 matrix multiplication.

Uses Strassen's algorithm independently on each of the three 2x2 blocks of the
right-hand (2x6) matrix, giving rank 3*7 = 21 (below the schoolbook rank 24).
"""
DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS          # n=2 (rows of A), m=2 (cols of A / rows of B), p=6 (cols of B)
    rank = 3 * 7                  # three 2x2 blocks, each via Strassen (rank 7)
    U = [[0] * rank for _ in range(n * m)]   # rows indexed by (i,j): i*m + j
    V = [[0] * rank for _ in range(m * p)]   # rows indexed by (j,k): j*p + k
    W = [[0] * rank for _ in range(p * n)]   # rows indexed by (k,i): k*n + i  (W[k*n+i, r])

    # Strassen 7-product coefficient tables.
    # U coefficients over A entries (i,j) with index i*m + j.
    # V coefficients over B entries of one 2x2 block: B00,B01,B10,B11.
    # W coefficients over C entries of one 2x2 block: C00,C01,C10,C11.
    U_tabs = [
        {0: 1},                                  # M1: A00
        {2: 1},                                  # M2: A10
        {0: 1, 2: 1},                            # M3: A00+A10
        {3: 1},                                  # M4: A11
        {0: 1, 3: 1},                            # M5: A00+A11
        {2: 1, 3: 1},                            # M6: A11+A10
        {0: 1, 3: 1},                            # M7: A00+A11
    ]
    # B entry local indices: 0=B00, 1=B01, 2=B10, 3=B11
    V_tabs = [
        {0: 1, 3: 1},   # M1: B00+B11
        {0: 1, 1: 1},   # M2: B00+B01
        {3: 1},         # M3: B11
        {0: 1, 3: 1},   # M4: B00+B11
        {1: 1, 2: 1},   # M5: B01+B10
        {0: 1, 2: 1},   # M6: B00+B10
        {3: 1, 2: 1},   # M7: B11+B10
    ]
    # C entry local indices: 0=C00, 1=C01, 2=C10, 3=C11
    W_tabs = [
        {0: 1, 3: 1},    # M1: C00+1, C11+1
        {2: 1, 3: -1},   # M2: C10+1, C11-1
        {0: -1, 2: 1},   # M3: C00-1, C10+1
        {0: 1, 1: 1},    # M4: C00+1, C01+1
        {1: 1, 3: 1},    # M5: C01+1, C11+1
        {3: 1},          # M6: C11+1
        {0: 1, 3: -1},   # M7: C00+1, C11-1
    ]

    for b in range(3):            # block b uses B columns 2b, 2b+1 and C columns 2b, 2b+1
        for s in range(7):        # Strassen product s
            r = b * 7 + s
            # U (independent of block)
            for idx, val in U_tabs[s].items():
                U[idx][r] = val
            # V: map local B index to global (j,k)
            for loc, val in V_tabs[s].items():
                jj = loc // 2      # 0 for B0x, 1 for B1x
                kk = 2 * b + (loc % 2)
                V[jj * p + kk][r] = val
            # W: map local C index to global (k,i) -> row k*n + i
            for loc, val in W_tabs[s].items():
                ii = loc // 2      # 0 for C0x, 1 for C1x
                kk = 2 * b + (loc % 2)
                W[kk * n + ii][r] = val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
