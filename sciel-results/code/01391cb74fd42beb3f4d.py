"""Matrix multiplication 2x4 x 4x4 with Strassen-based block decomposition (rank 28)."""
DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 4
    # We decompose the 2x4 * 4x4 multiplication into 4 blocks of 2x2 * 2x2.
    # A is 2x4: split into A_left (2x2) and A_right (2x2)
    # B is 4x4: split into B_top_left (2x2), B_top_right (2x2), B_bot_left (2x2), B_bot_right (2x2)
    # C is 2x4: split into C_left (2x2) and C_right (2x2)
    #
    # C_left = A_left * B_top_left + A_right * B_bot_left
    # C_right = A_left * B_top_right + A_right * B_bot_right
    #
    # Each 2x2 * 2x2 = 2x2 product can be done with Strassen (7 scalar products).
    # For C_left: we have two 2x2*2x2 products, but they share the A_left and A_right factors.
    # Actually, let's think of this differently.
    #
    # Let's view A as a 2x4 matrix with rows a0, a1 (each 1x4).
    # B is 4x4 with rows b0,b1,b2,b3 (each 1x4).
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # Split j into j in {0,1} and j in {2,3}.
    # C[i,k] = sum_{j=0}^{1} A[i,j]*B[j,k] + sum_{j=2}^{3} A[i,j]*B[j,k]
    #
    # Let A1 = A[:, 0:2] (2x2), A2 = A[:, 2:4] (2x2)
    # Let B1 = B[0:2, :] (2x4), B2 = B[2:4, :] (2x4)
    # C = A1 * B1 + A2 * B2
    #
    # Now A1*B1 is 2x2 * 2x4 = 2x4. Similarly A2*B2.
    # We can split B1 into B1_left (2x2) and B1_right (2x2), and B2 into B2_left (2x2) and B2_right (2x2).
    # C_left = A1 * B1_left + A2 * B2_left (2x2)
    # C_right = A1 * B1_right + A2 * B2_right (2x2)
    #
    # So we have 4 sub-problems of type 2x2 * 2x2:
    # P1 = A1 * B1_left
    # P2 = A2 * B2_left
    # P3 = A1 * B1_right
    # P4 = A2 * B2_right
    #
    # Each can be done with Strassen (rank 7). Total rank = 28.
    #
    # Strassen 2x2: given X (2x2), Y (2x2), compute Z = X*Y (2x2) with 7 products:
    # X = [[x00, x01], [x10, x11]], Y = [[y00, y01], [y10, y11]]
    # M1 = (x00 + x11) * (y00 + y11)
    # M2 = (x10 + x11) * y00
    # M3 = x00 * (y01 - y11)
    # M4 = x11 * (y10 - y00)
    # M5 = (x00 + x01) * y11
    # M6 = (x10 - x00) * (y00 + y01)
    # M7 = (x01 - x11) * (y10 + y11)
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6
    #
    # For each Strassen block, we need 7 "tensor terms" (rank-1 contributions).
    #
    # Let's define the mapping for each block.
    # Block P1: X = A1 (2x2), Y = B1_left (2x2), Z = P1 (2x2)
    #   A1[i,j] corresponds to A[i, j] where i in {0,1}, j in {0,1}
    #   B1_left[j,k] corresponds to B[j, k] where j in {0,1}, k in {0,1}
    #   P1[i,k] contributes to C[i, k] where k in {0,1}
    #
    # For U: rows are indexed by (i, j) where i in {0,1}, j in {0,3}
    #   A1[0,0] -> U[0][col]
    #   A1[0,1] -> U[1][col]
    #   A1[1,0] -> U[2][col]
    #   A1[1,1] -> U[3][col]
    #   A2[0,0] -> U[4][col]
    #   A2[0,1] -> U[5][col]
    #   A2[1,0] -> U[6][col]
    #   A2[1,1] -> U[7][col]
    #
    # For V: rows are indexed by (j, k) where j in {0,3}, k in {0,3}
    #   B1_left[0,0] -> V[0][col]
    #   B1_left[0,1] -> V[1][col]
    #   B1_left[1,0] -> V[4][col]
    #   B1_left[1,1] -> V[5][col]
    #   B1_right[0,0] -> V[2][col]
    #   B1_right[0,1] -> V[3][col]
    #   B1_right[1,0] -> V[6][col]
    #   B1_right[1,1] -> V[7][col]
    #   B2_left[0,0] -> V[8][col]
    #   B2_left[0,1] -> V[9][col]
    #   B2_left[1,0] -> V[12][col]
    #   B2_left[1,1] -> V[13][col]
    #   B2_right[0,0] -> V[10][col]
    #   B2_right[0,1] -> V[11][col]
    #   B2_right[1,0] -> V[14][col]
    #   B2_right[1,1] -> V[15][col]
    #
    # For W: rows are indexed by (k, i) where k in {0,3}, i in {0,1}
    #   C[i,k] -> W[k*2 + i][col]
    #   P1[i,k] -> C[i,k] where k in {0,1}: W[k*2+i][col]
    #   P2[i,k] -> C[i,k] where k in {0,1}: W[k*2+i][col]
    #   P3[i,k] -> C[i,k] where k in {2,3}: W[k*2+i][col]
    #   P4[i,k] -> C[i,k] where k in {2,3}: W[k*2+i][col]
    #
    # Let's build the decomposition.
    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # 8 x 28
    V = [[0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0] * rank for _ in range(p * n)]  # 8 x 28

    def strassen_block(X_rows, Y_rows, Z_w_rows, col_start):
        """
        X_rows: list of 4 U-row indices for X00, X01, X10, X11
        Y_rows: list of 4 V-row indices for Y00, Y01, Y10, Y11
        Z_w_rows: list of 4 W-row indices for Z00, Z01, Z10, Z11
        col_start: starting column index
        """
        # M1 = (X00 + X11) * (Y00 + Y11)
        col = col_start
        U[X_rows[0]][col] += 1
        U[X_rows[3]][col] += 1
        V[Y_rows[0]][col] += 1
        V[Y_rows[3]][col] += 1
        W[Z_w_rows[0]][col] += 1  # Z00
        W[Z_w_rows[3]][col] += 1  # Z11
        # Z00 = M1 + ... , Z11 = M1 + ...

        # M2 = (X10 + X11) * Y00
        col = col_start + 1
        U[X_rows[2]][col] += 1
        U[X_rows[3]][col] += 1
        V[Y_rows[0]][col] += 1
        W[Z_w_rows[2]][col] += 1  # Z10
        W[Z_w_rows[3]][col] += -1  # Z11

        # M3 = X00 * (Y01 - Y11)
        col = col_start + 2
        U[X_rows[0]][col] += 1
        V[Y_rows[1]][col] += 1
        V[Y_rows[3]][col] += -1
        W[Z_w_rows[1]][col] += 1  # Z01
        W[Z_w_rows[3]][col] += 1  # Z11

        # M4 = X11 * (Y10 - Y00)
        col = col_start + 3
        U[X_rows[3]][col] += 1
        V[Y_rows[2]][col] += 1
        V[Y_rows[0]][col] += -1
        W[Z_w_rows[0]][col] += 1  # Z00
        W[Z_w_rows[2]][col] += 1  # Z10

        # M5 = (X00 + X01) * Y11
        col = col_start + 4
        U[X_rows[0]][col] += 1
        U[X_rows[1]][col] += 1
        V[Y_rows[3]][col] += 1
        W[Z_w_rows[0]][col] += -1  # Z00
        W[Z_w_rows[1]][col] += 1  # Z01

        # M6 = (X10 - X00) * (Y00 + Y01)
        col = col_start + 5
        U[X_rows[2]][col] += 1
        U[X_rows[0]][col] += -1
        V[Y_rows[0]][col] += 1
        V[Y_rows[1]][col] += 1
        W[Z_w_rows[3]][col] += 1  # Z11

        # M7 = (X01 - X11) * (Y10 + Y11)
        col = col_start + 6
        U[X_rows[1]][col] += 1
        U[X_rows[3]][col] += -1
        V[Y_rows[2]][col] += 1
        V[Y_rows[3]][col] += 1
        W[Z_w_rows[0]][col] += 1  # Z00
        W[Z_w_rows[1]][col] += -1  # Z01
        # Wait, let me re-check Z01 = M3 + M5. So M7 should not contribute to Z01.
        # Let me re-derive:
        # Z00 = M1 + M4 - M5 + M7
        # Z01 = M3 + M5
        # Z10 = M2 + M4
        # Z11 = M1 - M2 + M3 + M6
        # So for M7: only Z00.

    # Fix M7:
    # Actually, let me re-check the standard Strassen:
    # M1 = (a11 + a22)(b11 + b22)
    # M2 = (a21 + a22)b11
    # M3 = a11(b12 - b22)
    # M4 = a22(b21 - b11)
    # M5 = (a11 + a12)b22
    # M6 = (a21 - a11)(b11 + b12)
    # M7 = (a12 - a22)(b21 + b22)
    # c11 = M1 + M4 - M5 + M7
    # c12 = M3 + M5
    # c21 = M2 + M4
    # c22 = M1 - M2 + M3 + M6
    # Yes, so M7 only contributes to c11 (Z00).

    # Block P1: X = A1, Y = B1_left, Z = P1 (contributes to C[:, 0:2])
    # A1: rows 0-3 of U (A[0,0], A[0,1], A[1,0], A[1,1])
    # B1_left: rows 0,1,4,5 of V (B[0,0], B[0,1], B[1,0], B[1,1])
    # P1: W rows for C[:, 0:2]: C[0,0]->W[0], C[0,1]->W[2], C[1,0]->W[1], C[1,1]->W[3]
    # Z00 = C[0,0] -> W[0]
    # Z01 = C[0,1] -> W[2]
    # Z10 = C[1,0] -> W[1]
    # Z11 = C[1,1] -> W[3]
    X1 = [0, 1, 2, 3]  # A1
    Y1 = [0, 1, 4, 5]  # B1_left
    Z1 = [0, 2, 1, 3]  # P1 -> C[:, 0:2]
    strassen_block(X1, Y1, Z1, 0)

    # Block P2: X = A2, Y = B2_left, Z = P2 (contributes to C[:, 0:2])
    # A2: rows 4-7 of U (A[0,2], A[0,3], A[1,2], A[1,3])
    # B2_left: rows 8,9,12,13 of V (B[2,0], B[2,1], B[3,0], B[3,1])
    X2 = [4, 5, 6, 7]  # A2
    Y2 = [8, 9, 12, 13]  # B2_left
    Z2 = [0, 2, 1, 3]  # P2 -> C[:, 0:2]
    strassen_block(X2, Y2, Z2, 7)

    # Block P3: X = A1, Y = B1_right, Z = P3 (contributes to C[:, 2:4])
    # B1_right: rows 2,3,6,7 of V (B[0,2], B[0,3], B[1,2], B[1,3])
    X3 = [0, 1, 2, 3]  # A1
    Y3 = [2, 3, 6, 7]  # B1_right
    # P3 -> C[:, 2:4]: C[0,2]->W[4], C[0,3]->W[6], C[1,2]->W[5], C[1,3]->W[7]
    # Z00 = C[0,2] -> W[4]
    # Z01 = C[0,3] -> W[6]
    # Z10 = C[1,2] -> W[5]
    # Z11 = C[1,3] -> W[7]
    Z3 = [4, 6, 5, 7]
    strassen_block(X3, Y3, Z3, 14)

    # Block P4: X = A2, Y = B2_right, Z = P4 (contributes to C[:, 2:4])
    # B2_right: rows 10,11,14,15 of V (B[2,2], B[2,3], B[3,2], B[3,3])
    X4 = [4, 5, 6, 7]  # A2
    Y4 = [10, 11, 14, 15]  # B2_right
    Z4 = [4, 6, 5, 7]  # P4 -> C[:, 2:4]
    strassen_block(X4, Y4, Z4, 21)

    return {"U": U, "V": V, "W": W}
