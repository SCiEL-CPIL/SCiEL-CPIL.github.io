"""Program for matrix multiplication tensor decomposition: 2x2 by 2x16."""
DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 16
    num_blocks = p // 2  # 8 blocks
    rank = 7 * num_blocks  # 56

    # Build U, V, W as flat lists for speed, then reshape
    # U: (n*m) x rank = 4 x 56
    # V: (m*p) x rank = 32 x 56
    # W: (p*n) x rank = 32 x 56

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen's 7 multiplications for 2x2 matrices
    # Each entry: (U_indices, V_indices, W_indices)
    # U_indices: list of (row, coeff) where row = i*m + j
    # V_indices: list of (row, coeff) where row = j*p + k
    # W_indices: list of (row, coeff) where row = k*n + i

    # M1 = (a00+a11)(b00+b11)
    #   U: a00 -> row 0, a11 -> row 3
    #   V: b00 -> row 0, b11 -> row 17
    #   W: c00 -> row 0, c11 -> row 3
    strassen = [
        # (U_entries, V_entries, W_entries)
        # M1
        ([(0, 1), (3, 1)], [(0, 1), (17, 1)], [(0, 1), (3, 1)]),
        # M2 = (a10+a11)(b00)
        #   U: a10 -> row 2, a11 -> row 3
        #   V: b00 -> row 0
        #   W: c10 -> row 2, c11 -> row 3 with -1
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
        # M3 = a00(b01-b11)
        #   U: a00 -> row 0
        #   V: b01 -> row 1, b11 -> row 17 with -1
        #   W: c01 -> row 1, c11 -> row 3
        ([(0, 1)], [(1, 1), (17, -1)], [(1, 1), (3, 1)]),
        # M4 = a11(b10-b00)
        #   U: a11 -> row 3
        #   V: b10 -> row 16, b00 -> row 0 with -1
        #   W: c00 -> row 0, c10 -> row 2
        ([(3, 1)], [(16, 1), (0, -1)], [(0, 1), (2, 1)]),
        # M5 = (a00+a01)(b11)
        #   U: a00 -> row 0, a01 -> row 1
        #   V: b11 -> row 17
        #   W: c00 -> row 0 with -1, c01 -> row 1
        ([(0, 1), (1, 1)], [(17, 1)], [(0, -1), (1, 1)]),
        # M6 = (a10-a00)(b00+b01)
        #   U: a10 -> row 2, a00 -> row 0 with -1
        #   V: b00 -> row 0, b01 -> row 1
        #   W: c11 -> row 3
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        # M7 = (a01-a11)(b10+b11)
        #   U: a01 -> row 1, a11 -> row 3 with -1
        #   V: b10 -> row 16, b11 -> row 17
        #   W: c00 -> row 0
        ([(1, 1), (3, -1)], [(16, 1), (17, 1)], [(0, 1)]),
    ]

    for t in range(num_blocks):
        for s in range(7):
            col = t * 7 + s
            u_entries, v_entries, w_entries = strassen[s]

            for (urow, coeff) in u_entries:
                U[urow][col] = coeff

            for (vrow, coeff) in v_entries:
                # vrow is in the local 2x2 block: local j in {0,1}, local k in {0,1}
                # Global row = j * p + (2*t + k)
                # vrow = j*16 + k_local where k_local in {0,1}
                # But our strassen entries already use global-ish rows for a single 2x2 block
                # Actually we need to map: in block t, the B submatrix rows are 2t and 2t+1
                # V row = j * p + (2*t + k_local)
                # The vrow in strassen is for a single 2x2 B: row 0=j=0,k=0; row 1=j=0,k=1; row 16=j=1,k=0; row 17=j=1,k=1
                # So we need to offset by 2*t for the k dimension
                # Parse: vrow = j*16 + k where j in {0,1}, k in {0,1}
                j_local = vrow // 16
                k_local = vrow % 16
                global_row = j_local * p + (2 * t + k_local)
                V[global_row][col] = coeff

            for (wrow, coeff) in w_entries:
                # wrow = k_local * n + i where k_local in {0,1}, i in {0,1}
                # Global: row = (2*t + k_local) * n + i
                k_local = wrow // n
                i_local = wrow % n
                global_row = (2 * t + k_local) * n + i_local
                W[global_row][col] = coeff

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
