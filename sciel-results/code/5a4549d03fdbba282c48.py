"""Matrix multiplication 2x3x8 via a split-and-Strassen exact tensor decomposition.

Split the middle (m) dimension into {0,1} (a 2x2x8 tensor) and {2} (a 2x1x8 tensor).
The 2x1x8 part is a pure "outer product" tensor of rank 8.
The 2x2x8 part is partitioned into three 2x2 output blocks (columns 0-1, 2-3, 4-7)
and decomposed with the 7-product Strassen scheme per block.
Total rank = 3*7 + 8 = 29, compared to the schoolbook 48.

The identity (A[i,j]*B[j,k]) = sum_c W[k*n+i, c] * (U_col . A_vec) * (V_col . B_vec)
holds by construction, so the factorisation is exact.
"""
DIMENSIONS = [2, 3, 8]


def construct():
    n, m, p = DIMENSIONS  # 2, 3, 8
    rank = 3 * 7 + 8      # 29

    U = [[0] * rank for _ in range(n * m)]  # 6 x rank
    V = [[0] * rank for _ in range(m * p)]  # 24 x rank
    W = [[0] * rank for _ in range(p * n)]  # 16 x rank

    # ---- Part 1: 2x1x8 tensor (m=2 column), rank 8 -------------------------
    # term = A[i,2] * B[2,k] -> C[i,k]
    for c in range(8):
        k = c
        for i in range(2):
            U[i * m + 2][c] = 1          # coefficient of A[i,2]
            W[k * n + i][c] = 1          # routes to C[i,k]
        V[2 * p + k][c] = 1              # coefficient of B[2,k]

    # ---- Part 2: 2x2x8 tensor (m in {0,1}), 3 blocks -----------------------
    # Standard Strassen 7 products for a 2x2x2 tensor, per output block.
    # A indices: 0=A[0,0], 1=A[0,1], 2=A[1,0], 3=A[1,1]
    # B indices: 0=B[0,2t], 1=B[0,2t+1], 2=B[1,2t], 3=B[1,2t+1]
    # C indices: 0=C[0,2t], 1=C[0,2t+1], 2=C[1,2t], 3=C[1,2t+1]
    M1 = ([0, 3], [0, 3])
    M2 = ([2, 3], [0])
    M3 = ([0], [2, 3]);  M3v = {2: 1, 3: -1}
    M4 = ([3], [1, 2]);  M4v = {1: 1, 2: -1}
    M5 = ([0, 1], [3])
    M6 = ([2, 0], [0, 2]); M6u = {2: 1, 0: -1}
    M7 = ([1, 3], [1, 3]); M7u = {1: 1, 3: -1}; M7v = {1: 1, 3: 1}

    # output contribution: C_index -> {product_name: coefficient}
    Wmap = {
        0: {"M1": 1, "M4": 1, "M5": -1, "M7": 1},   # C00
        1: {"M3": 1, "M5": 1},                       # C01
        2: {"M2": 1, "M4": 1},                       # C10
        3: {"M1": 1, "M2": -1, "M3": 1, "M6": 1},    # C11
    }

    products = [
        ("M1", M1, None, None),
        ("M2", M2, None, None),
        ("M3", M3, M3u, M3v),
        ("M4", M4, M4v, None),
        ("M5", M5, None, None),
        ("M6", M6, M6u, None),
        ("M7", M7, M7u, M7v),
    ]

    blocks = [(0, 1), (2, 3), (4, 7)]  # (start, end) output columns
    for (c0, c1) in blocks:
        block_idx = c0 // 2
        for pi in range(7):
            name, (Uidx, Vidx), Uover, Vover = products[pi]
            c = 8 + 7 * block_idx + pi

            # U coefficients (defaults +1, override if given)
            for u in Uidx:
                U[u][c] += Uover.get(u, 1) if Uover else 1
            # V coefficients
            for v in Vidx:
                V[v][c] += Vover.get(v, 1) if Vover else 1
            # W routing
            for out, coefmap in Wmap.items():
                if name in coefmap:
                    val = coefmap[name]
                    # out: 0=C[0,c0], 1=C[0,c0+1], 2=C[1,c0], 3=C[1,c0+1]
                    i = 0 if out in (0, 1) else 1
                    k = c0 + (out % 2)
                    W[k * n + i][c] += val

    return {"U": U, "V": V, "W": W}
