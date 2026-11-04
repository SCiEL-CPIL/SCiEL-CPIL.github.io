import numpy as np
import time

DIMENSIONS = [4, 4, 6]
_DEADLINE = 120.0

def construct():
    t_start = time.time()
    n, m, p = DIMENSIONS  # n=4, m=4, p=6
    
    # Strassen coefficients for 2x2x2
    # alpha[s][r] where s is the Strassen product index (0..6), r is the 2x2 block index (0..3)
    # r: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    alpha = [
        {0: 1, 1: 0, 2: 0, 3: 1},
        {0: 0, 1: 0, 2: 1, 3: 1},
        {0: 1, 1: 0, 2: 0, 3: 0},
        {0: 0, 1: 0, 2: 0, 3: 1},
        {0: 1, 1: 1, 2: 0, 3: 0},
        {0: -1, 1: 0, 2: 1, 3: 0},
        {0: 0, 1: 1, 2: 0, 3: -1}
    ]
    beta = [
        {0: 1, 1: 0, 2: 0, 3: 1},
        {0: 1, 1: 0, 2: 0, 3: 0},
        {0: 0, 1: 1, 2: 0, 3: -1},
        {0: -1, 1: 0, 2: 1, 3: 0},
        {0: 0, 1: 0, 2: 0, 3: 1},
        {0: 1, 1: 1, 2: 0, 3: 0},
        {0: 0, 1: 0, 2: 1, 3: 1}
    ]
    # recon[s] contains tuples (a, b, gamma) for the reconstruction of the 2x2 result
    # C_ab = sum gamma * M_s
    recon = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, -1)],
        [(0, 1, 1), (1, 1, 1)],
        [(0, 0, 1), (1, 0, 1)],
        [(0, 0, -1), (0, 1, 1)],
        [(1, 1, 1)],
        [(0, 0, 1)]
    ]

    R = 77
    U = np.zeros((16, R))
    V = np.zeros((24, R))
    W = np.zeros((24, R))

    col = 0
    
    # Part 1: 4x4x4 Strassen (Rank 49)
    # This handles the first 4 columns of B (and thus first 4 columns of C)
    for s in range(7):
        for si in range(7):
            # U part: A is 4x4. Blocks A_ar,br where ar,br in {0,1}
            # Index in U: (row, col) of 4x4 matrix.
            # Block (ar, br) corresponds to rows ar*2..ar*2+1, cols br*2..br*2+1
            # alpha[s][r] is coeff for block r of A.
            # r: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
            # So block (r//2, r%2)
            
            # V part: B is 4x4. Blocks B_ac,bc
            # Index in V: (row, col) of 4x4 matrix.
            
            # W part: C is 4x4. Blocks C_ab
            # Index in W: (row, col) of 4x4 matrix.
            
            # Iterate over all non-zero combinations
            for a in range(4):
                if alpha[si][a] != 0:
                    a_r, a_c = a // 2, a % 2
                    for r in range(4):
                        if alpha[s][r] != 0:
                            r_r, r_c = r // 2, r % 2
                            # U index: row = r_r*2 + a_r, col = r_c*2 + a_c
                            u_idx = (r_r * 2 + a_r) * 4 + (r_c * 2 + a_c)
                            U[u_idx, col] += alpha[si][a] * alpha[s][r]
            
            for b in range(4):
                if beta[si][b] != 0:
                    b_r, b_c = b // 2, b % 2
                    for c in range(4):
                        if beta[s][c] != 0:
                            c_r, c_c = c // 2, c % 2
                            # V index: row = c_r*2 + b_r, col = c_c*2 + b_c
                            v_idx = (c_r * 2 + b_r) * 4 + (c_c * 2 + b_c)
                            V[v_idx, col] += beta[si][b] * beta[s][c]
            
            for a, b, gamma in recon[s]:
                for za, zb, zgamma in recon[si]:
                    # W index: row = zb*2 + a, col = za*2 + b
                    # Note: recon[s] gives C_ab = sum gamma M_s.
                    # Here a is row of C block, b is col of C block.
                    # So C row index = b*2 + a? No.
                    # C is 4x4. Block (a,b) is at rows a*2.., cols b*2..
                    # Wait, standard Strassen:
                    # C_11 = M1 + M4 - M5 - M7
                    # C_12 = M2 + M3
                    # C_21 = M4 + M6
                    # C_22 = M1 - M3 + M5 + M7
                    # My recon list:
                    # s=0: (0,0,1), (1,1,1) -> C_00 += M0, C_11 += M0
                    # This matches M1 (if 0-indexed M0 is M1).
                    # So a is row block index, b is col block index.
                    # W index: row = a*2 + za, col = b*2 + zb
                    w_idx = (a * 2 + za) * 4 + (b * 2 + zb)
                    W[w_idx, col] += gamma * zgamma
            
            col += 1

    # Part 2: 4x4x2 Strassen hybrid (Rank 28)
    # This handles the last 2 columns of B (cols 4,5) and thus last 2 cols of C (cols 4,5)
    # B is 4x2. We can view it as two 2x2 blocks vertically?
    # B = [B_top; B_bottom] where B_top, B_bottom are 2x2.
    # A is 4x4.
    # C = A B = [A_top B_top + A_bottom B_bottom; ...]
    # Wait, C is 4x2.
    # C_top (2x2) = A_top_left B_top + A_top_right B_bottom
    # C_bottom (2x2) = A_bottom_left B_top + A_bottom_right B_bottom
    
    # This requires 4 Strassen multiplications:
    # 1. A_top_left * B_top
    # 2. A_top_right * B_bottom
    # 3. A_bottom_left * B_top
    # 4. A_bottom_right * B_bottom
    
    # Each is 2x2x2 (Rank 7). Total 28.
    
    # We map these to the global indices.
    # U (A 4x4):
    #   A_top_left: rows 0-1, cols 0-1
    #   A_top_right: rows 0-1, cols 2-3
    #   A_bottom_left: rows 2-3, cols 0-1
    #   A_bottom_right: rows 2-3, cols 2-3
    # V (B 4x6):
    #   B_top: rows 0-1, cols 4-5 (since B is 4x6, last 2 cols are 4,5)
    #   B_bottom: rows 2-3, cols 4-5
    # W (C 4x6):
    #   C_top: rows 0-1, cols 4-5
    #   C_bottom: rows 2-3, cols 4-5
    
    # We iterate over the 4 blocks of A and 2 blocks of B.
    # Block indices for A:
    # 0: TL (0,0)
    # 1: TR (0,1)
    # 2: BL (1,0)
    # 3: BR (1,1)
    # Block indices for B (in the 4x2 submatrix):
    # 0: Top (0) -> rows 0-1
    # 1: Bottom (1) -> rows 2-3
    
    # We need to compute:
    # C_TL (2x2) = A_TL B_T + A_TR B_B
    # C_BL (2x2) = A_BL B_T + A_BR B_B
    
    # Let's generate the 28 terms.
    
    # Helper to add a 2x2x2 Strassen block
    def add_strassen_block(u_row_offset, u_col_offset, v_row_offset, v_col_offset, w_row_offset, w_col_offset, coeff=1):
        nonlocal col
        for s in range(7):
            # U
            for r in range(4):
                if alpha[s][r] != 0:
                    r_r, r_c = r // 2, r % 2
                    u_idx = (u_row_offset + r_r) * 4 + (u_col_offset + r_c)
                    U[u_idx, col] += coeff * alpha[s][r]
            # V
            for c in range(4):
                if beta[s][c] != 0:
                    c_r, c_c = c // 2, c % 2
                    v_idx = (v_row_offset + c_r) * 6 + (v_col_offset + c_c)
                    V[v_idx, col] += coeff * beta[s][c]
            # W
            for a, b, gamma in recon[s]:
                w_idx = (w_row_offset + a) * 4 + (w_col_offset + b)
                W[w_idx, col] += coeff * gamma
            col += 1

    # 1. A_TL * B_T -> contributes to C_TL
    add_strassen_block(0, 0, 0, 4, 0, 4, 1)
    # 2. A_TR * B_B -> contributes to C_TL
    add_strassen_block(0, 2, 2, 4, 0, 4, 1)
    # 3. A_BL * B_T -> contributes to C_BL
    add_strassen_block(2, 0, 0, 4, 2, 4, 1)
    # 4. A_BR * B_B -> contributes to C_BL
    add_strassen_block(2, 2, 2, 4, 2, 4, 1)

    # Check for linear dependencies (optional, but good for safety)
    # In this specific construction, the 49 and 28 terms are on disjoint support in W?
    # Part 1 W supports cols 0-3.
    # Part 2 W supports cols 4-5.
    # So they are linearly independent. Rank is exactly 77.

    def to_list(mat):
        return [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in mat]

    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
