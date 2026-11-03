"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen coefficients for M1..M7
    # U coeffs for X
    # V coeffs for Y
    # W coeffs for Z (00, 01, 10, 11)
    
    # M1: (x00+x11)(y00+y11) -> z00+1, z11+1
    # M2: (x10+x11)y00 -> z10+1, z11-1
    # M3: x00(y10-y11) -> z01+1, z11+1
    # M4: x11(y01-y00) -> z00+1, z10+1
    # M5: (x00+x10)y11 -> z00-1, z01+1
    # M6: (x11-x00)(y00+y10) -> z11+1
    # M7: (x01-x11)(y11+y10) -> z00+1
    
    strassen_terms = [
        # (u_coeffs, v_coeffs, w_coeffs)
        # u: [x00, x01, x10, x11]
        # v: [y00, y01, y10, y11]
        # w: [z00, z01, z10, z11]
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),  # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),  # M2
        ([1, 0, 0, 0], [0, 0, 1, -1], [0, 1, 0, 1]),  # M3
        ([0, 0, 0, 1], [0, 1, -1, 0], [1, 0, 1, 0]),  # M4
        ([1, 0, 1, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),  # M5
        ([1, 0, 0, -1], [1, 0, 1, 0], [0, 0, 0, 1]),  # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),  # M7
    ]

    # Define index mappings for the 4 blocks
    # Block 1: A_L, B_TL, C_TL
    # Block 2: A_R, B_BL, C_TL
    # Block 3: A_L, B_TR, C_TR
    # Block 4: A_R, B_BR, C_TR

    # A indices (row, col) -> global index i*m + j
    # A_L: (0,0)->0, (0,1)->1, (1,0)->4, (1,1)->5
    # A_R: (0,0)->2, (0,1)->3, (1,0)->6, (1,1)->7
    A_idx_L = [0, 1, 4, 5]
    A_idx_R = [2, 3, 6, 7]

    # B indices (row, col) -> global index j*p + k
    # B_TL: (0,0)->0, (0,1)->1, (1,0)->4, (1,1)->5
    # B_TR: (0,0)->2, (0,1)->3, (1,0)->6, (1,1)->7
    # B_BL: (0,0)->8, (0,1)->9, (1,0)->12, (1,1)->13
    # B_BR: (0,0)->10, (0,1)->11, (1,0)->14, (1,1)->15
    B_idx_TL = [0, 1, 4, 5]
    B_idx_TR = [2, 3, 6, 7]
    B_idx_BL = [8, 9, 12, 13]
    B_idx_BR = [10, 11, 14, 15]

    # C indices (row, col) -> global index k*n + i
    # C_TL: (0,0)->0, (0,1)->2, (1,0)->1, (1,1)->3
    # C_TR: (0,0)->4, (0,1)->6, (1,0)->5, (1,1)->7
    C_idx_TL = [0, 2, 1, 3]
    C_idx_TR = [4, 6, 5, 7]

    blocks = [
        (A_idx_L, B_idx_TL, C_idx_TL, 0),   # P1
        (A_idx_R, B_idx_BL, C_idx_TL, 7),   # P2
        (A_idx_L, B_idx_TR, C_idx_TR, 14),  # P3
        (A_idx_R, B_idx_BR, C_idx_TR, 21),  # P4
    ]

    for A_idx, B_idx, C_idx, offset in blocks:
        for t_idx, (u_c, v_c, w_c) in enumerate(strassen_terms):
            col = offset + t_idx
            for local_idx, val in enumerate(u_c):
                if val != 0:
                    U[A_idx[local_idx]][col] += val
            for local_idx, val in enumerate(v_c):
                if val != 0:
                    V[B_idx[local_idx]][col] += val
            for local_idx, val in enumerate(w_c):
                if val != 0:
                    W[C_idx[local_idx]][col] += val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
