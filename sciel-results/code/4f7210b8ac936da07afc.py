"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 42 decomposition using 6 Strassen blocks
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 matrices
    # X = [[x00, x01], [x10, x11]], Y = [[y00, y01], [y10, y11]]
    # M1 = (x00+x11)(y00+y11)
    # M2 = (x10+x11)y00
    # M3 = x00(y10-y11)
    # M4 = x11(y01-y00)
    # M5 = (x00+x01)y11
    # M6 = (x10-x00)(y00+y10)
    # M7 = (x01-x11)(y11+y01)
    
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6
    
    # Define the 6 blocks
    # A is 2x4. A_L is cols 0-1, A_R is cols 2-3.
    # B is 4x6. B_T is rows 0-1, B_B is rows 2-3.
    # B_T is split into 3 2x2 blocks: cols 0-1, 2-3, 4-5.
    # B_B is split into 3 2x2 blocks: cols 0-1, 2-3, 4-5.
    
    # Block definitions:
    # (a_row_offset, a_col_offset, b_row_offset, b_col_offset)
    # A_L: a_row=0, a_col=0. A_R: a_row=0, a_col=2.
    # B_T: b_row=0. B_B: b_row=2.
    # Cols for B blocks: 0, 2, 4.
    
    blocks = [
        # (A_L, B_T1)
        (0, 0, 0, 0),
        # (A_L, B_T2)
        (0, 0, 0, 2),
        # (A_L, B_T3)
        (0, 0, 0, 4),
        # (A_R, B_B1)
        (0, 2, 2, 0),
        # (A_R, B_B2)
        (0, 2, 2, 2),
        # (A_R, B_B3)
        (0, 2, 2, 4),
    ]
    
    # Strassen steps definition
    # Each step is (u_coeffs, v_coeffs, w_coeffs)
    # u_coeffs: dict mapping local (r,c) to coeff
    # v_coeffs: dict mapping local (r,c) to coeff
    # w_coeffs: dict mapping local (r,c) to coeff
    
    # Local indices for 2x2:
    # (0,0), (0,1), (1,0), (1,1)
    
    steps = [
        # M1
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # M2
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1}),
        # M3
        ({(0,0):1}, {(1,0):1, (1,1):-1}, {(0,1):1}),
        # M4
        ({(1,1):1}, {(0,1):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        # M5
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        # M6
        ({(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}, {(1,1):1}),
        # M7
        ({(0,1):1, (1,1):-1}, {(1,1):1, (0,1):1}, {(0,0):1}),
    ]
    
    col = 0
    for b_idx, (a_r, a_c, b_r, b_c) in enumerate(blocks):
        for s_idx, (u_map, v_map, w_map) in enumerate(steps):
            # Fill U
            for (lr, lc), val in u_map.items():
                i = a_r + lr
                j = a_c + lc
                U[i * m + j][col] = val
            
            # Fill V
            for (lr, lc), val in v_map.items():
                j = b_r + lr
                k = b_c + lc
                V[j * p + k][col] = val
            
            # Fill W
            # C index: i (row), k (col)
            # W index: k * n + i
            for (lr, lc), val in w_map.items():
                i = a_r + lr # C row is same as A row
                k = b_c + lc # C col is same as B col offset + local col
                W[k * n + i][col] = val
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
