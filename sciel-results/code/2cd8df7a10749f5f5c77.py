"""Matrix multiplication 3,4,4 - rank-44 decomposition using Strassen on 2x2 blocks."""

def construct():
    n, m, p = 3, 4, 4
    rank = 44
    U = [[0] * rank for _ in range(n * m)]  # 12 x 44
    V = [[0] * rank for _ in range(m * p)]  # 16 x 44
    W = [[0] * rank for _ in range(p * n)]  # 12 x 44

    col = 0

    # For each pair of A-rows (i) and B-block, apply Strassen to the 1x2 @ 2x2
    # Actually, let me think more carefully.
    #
    # C[i][j] = sum_{k=0}^{3} A[i][k] * B[k][j]
    #
    # Split k into {0,1} and {2,3}, split j into {0,1} and {2,3}:
    # C[i][0:2] = A[i][0:2] @ B[0:2, 0:2] + A[i][2:4] @ B[2:4, 0:2]
    # C[i][2:4] = A[i][0:2] @ B[0:2, 2:4] + A[i][2:4] @ B[2:4, 2:4]
    #
    # For fixed i, consider the 1x4 @ 4x4 multiplication.
    # Split into 4 sub-problems of 1x2 @ 2x2.
    # Each 1x2 @ 2x2 has rank 4 (schoolbook), total 16 per row, 48 total.
    #
    # To save: use Strassen-like approach on pairs of 2x2 blocks.
    # For fixed i:
    #   Let a1 = A[i][0], a2 = A[i][1], a3 = A[i][2], a4 = A[i][3]
    #   C[i][0] = a1*B[0][0] + a2*B[1][0] + a3*B[2][0] + a4*B[3][0]
    #   C[i][1] = a1*B[0][1] + a2*B[1][1] + a3*B[2][1] + a4*B[3][1]
    #   C[i][2] = a1*B[0][2] + a2*B[1][2] + a3*B[2][2] + a4*B[3][2]
    #   C[i][3] = a1*B[0][3] + a2*B[1][3] + a3*B[2][3] + a4*B[3][3]
    #
    # Group: 
    #   C_L = [C[i][0], C[i][1]] = [a1, a2] @ B_TL + [a3, a4] @ B_BL
    #   C_R = [C[i][2], C[i][3]] = [a1, a2] @ B_TR + [a3, a4] @ B_BR
    #
    # where B_TL = B[0:2, 0:2], B_BL = B[2:4, 0:2], etc.
    #
    # For 1x2 @ 2x2, schoolbook rank is 4. But we can use a trick:
    # [c1, c2] = [a1, a2] @ [[b11, b12], [b21, b22]]
    # c1 = a1*b11 + a2*b21
    # c2 = a1*b12 + a2*b22
    #
    # 4 products: a1*b11, a2*b21, a1*b12, a2*b22
    # Can we do it in 3? No, rank of 1x2x2x2 tensor is 4.
    #
    # But we have 4 such sub-problems per row (2 for C_L, 2 for C_R).
    # Total 16 per row, 48 for 3 rows.
    #
    # Key insight from inspirations: combine the two halves differently.
    # For C_L: [c1, c2] = [a1, a2] @ B_TL + [a3, a4] @ B_BL
    # This is a 1x4 @ 4x2 problem, rank 8 schoolbook.
    # Can we do better? 
    #
    # Actually, let me use the approach from inspiration 1:
    # Split into 1x4x4 (rank 16 schoolbook) + 2x4x4 (rank 28 using Strassen on 2x2 blocks)
    # Total: 16 + 28 = 44.
    #
    # For the 2x4x4 part (rows 1,2 of A):
    # C[1:3, :] = A[1:3, :] @ B  (2x4 @ 4x4)
    # Split A[1:3,:] into columns {0,1} and {2,3}:
    # C[1:3, 0:2] = A[1:3, 0:2] @ B[0:2, 0:2] + A[1:3, 2:4] @ B[2:4, 0:2]
    # C[1:3, 2:4] = A[1:3, 0:2] @ B[0:2, 2:4] + A[1:3, 2:4] @ B[2:4, 2:4]
    #
    # Each 2x2 @ 2x2 can be done with Strassen (rank 7 instead of 4).
    # 4 such blocks: 4 * 7 = 28.
    #
    # For the 1x4x4 part (row 0 of A):
    # C[0, :] = A[0, :] @ B  (1x4 @ 4x4)
    # Schoolbook rank: 16.
    #
    # Total: 16 + 28 = 44.

    # Part 1: Row 0 of A, schoolbook (16 products)
    for k in range(4):
        for j in range(4):
            U[0 * m + k][col] = 1  # A[0][k]
            V[k * p + j][col] = 1  # B[k][j]
            W[j * n + 0][col] = 1  # C[0][j], W index = j*n + i = j*3 + 0
            col += 1

    # Part 2: Rows 1,2 of A, Strassen on 2x2 blocks (28 products)
    # A_sub = A[1:3, :] (2x4), split into A_left = A[1:3, 0:2] (2x2), A_right = A[1:3, 2:4] (2x2)
    # B split into 2x2 blocks: B_TL=B[0:2,0:2], B_TR=B[0:2,2:4], B_BL=B[2:4,0:2], B_BR=B[2:4,2:4]
    # C_sub = C[1:3, :] (2x4), split into C_left = C[1:3, 0:2] (2x2), C_right = C[1:3, 2:4] (2x2)
    # C_left = A_left @ B_TL + A_right @ B_BL
    # C_right = A_left @ B_TR + A_right @ B_BR

    # Strassen for 2x2 @ 2x2:
    # Given X (2x2) and Y (2x2), compute Z = X @ Y with 7 products:
    # m1 = (X00+X11)*(Y00+Y11)
    # m2 = (X10+X11)*Y00
    # m3 = X00*(Y01-Y11)
    # m4 = X11*Y11
    # m5 = (X00+X01)*Y11
    # m6 = (X10-X00)*(Y00+Y01)
    # m7 = (X01-X11)*(Y10+Y11)
    # Z00 = m1 + m4 - m5 + m7
    # Z01 = m3 + m5
    # Z10 = m2 + m4
    # Z11 = m1 - m2 + m3 + m6

    # We need to compute C_left and C_right, each being sum of two 2x2@2x2 products.
    # C_left = A_left @ B_TL + A_right @ B_BL
    # C_right = A_left @ B_TR + A_right @ B_BR

    # We can use Strassen for each 2x2@2x2, giving 4*7=28 products.
    # But we can share some computations. Let me just do 4 separate Strassen calls.

    # Helper: add a Strassen block for X @ Y contributing to Z at given indices
    def add_strassen_block(X_i, Y_i, Z_i, sign=1):
        """
        X_i: list of 4 indices into U (for the 2x2 matrix X, row-major: [00,01,10,11])
        Y_i: list of 4 indices into V (for the 2x2 matrix Y, row-major: [00,01,10,11])
        Z_i: list of 4 indices into W (for the 2x2 matrix Z, row-major: [00,01,10,11])
        sign: multiply all W coefficients by this sign
        """
        nonlocal col
        X00, X01, X10, X11 = X_i
        Y00, Y01, Y10, Y11 = Y_i
        Z00, Z01, Z10, Z11 = Z_i

        # m1 = (X00+X11)*(Y00+Y11)
        U[X00][col] = sign
        U[X11][col] = sign
        V[Y00][col] = sign
        V[Y11][col] = sign
        W[Z00][col] = sign
        W[Z11][col] = sign
        col += 1

        # m2 = (X10+X11)*Y00
        U[X10][col] = sign
        U[X11][col] = sign
        V[Y00][col] = sign
        W[Z10][col] = sign
        col += 1

        # m3 = X00*(Y01-Y11)
        U[X00][col] = sign
        V[Y01][col] = sign
        V[Y11][col] = -sign
        W[Z01][col] = sign
        W[Z11][col] = sign
        col += 1

        # m4 = X11*Y11
        U[X11][col] = sign
        V[Y11][col] = sign
        W[Z00][col] = sign
        W[Z10][col] = sign
        col += 1

        # m5 = (X00+X01)*Y11
        U[X00][col] = sign
        U[X01][col] = sign
        V[Y11][col] = sign
        W[Z00][col] = -sign
        W[Z01][col] = sign
        col += 1

        # m6 = (X10-X00)*(Y00+Y01)
        U[X10][col] = sign
        U[X00][col] = -sign
        V[Y00][col] = sign
        V[Y01][col] = sign
        W[Z11][col] = sign
        col += 1

        # m7 = (X01-X11)*(Y10+Y11)
        U[X01][col] = sign
        U[X11][col] = -sign
        V[Y10][col] = sign
        V[Y11][col] = sign
        W[Z00][col] = sign
        col += 1

    # A_left indices (rows 1,2 of A, cols 0,1): A[1][0]=U[4], A[1][1]=U[5], A[2][0]=U[8], A[2][1]=U[9]
    # A_right indices (rows 1,2 of A, cols 2,3): A[1][2]=U[6], A[1][3]=U[7], A[2][2]=U[10], A[2][3]=U[11]
    #
    # B_TL indices: B[0][0]=V[0], B[0][1]=V[1], B[1][0]=V[4], B[1][1]=V[5]
    # B_TR indices: B[0][2]=V[2], B[0][3]=V[3], B[1][2]=V[6], B[1][3]=V[7]
    # B_BL indices: B[2][0]=V[8], B[2][1]=V[9], B[3][0]=V[12], B[3][1]=V[13]
    # B_BR indices: B[2][2]=V[10], B[2][3]=V[11], B[3][2]=V[14], B[3][3]=V[15]
    #
    # C_left indices (rows 1,2 of C, cols 0,1): C[1][0]=W[0*3+1]=W[1], C[1][1]=W[1*3+1]=W[4], C[2][0]=W[0*3+2]=W[2], C[2][1]=W[1*3+2]=W[5]
    # C_right indices (rows 1,2 of C, cols 2,3): C[1][2]=W[2*3+1]=W[7], C[1][3]=W[3*3+1]=W[10], C[2][2]=W[2*3+2]=W[8], C[2][3]=W[3*3+2]=W[11]

    # X00=A[1][0]=U[1*4+0]=U[4], X01=A[1][1]=U[5], X10=A[2][0]=U[8], X11=A[2][1]=U[9]
    A_left = [4, 5, 8, 9]
    # X00=A[1][2]=U[6], X01=A[1][3]=U[7], X10=A[2][2]=U[10], X11=A[2][3]=U[11]
    A_right = [6, 7, 10, 11]

    # B_TL: [0, 1, 4, 5], B_TR: [2, 3, 6, 7], B_BL: [8, 9, 12, 13], B_BR: [10, 11, 14, 15]
    B_TL = [0, 1, 4, 5]
    B_TR = [2, 3, 6, 7]
    B_BL = [8, 9, 12, 13]
    B_BR = [10, 11, 14, 15]

    # C_left: C[1][0]=W[1], C[1][1]=W[4], C[2][0]=W[2], C[2][1]=W[5]
    C_left = [1, 4, 2, 5]
    # C_right: C[1][2]=W[7], C[1][3]=W[10], C[2][2]=W[8], C[2][3]=W[11]
    C_right = [7, 10, 8, 11]

    # C_left = A_left @ B_TL + A_right @ B_BL
    add_strassen_block(A_left, B_TL, C_left, 1)
    add_strassen_block(A_right, B_BL, C_left, 1)

    # C_right = A_left @ B_TR + A_right @ B_BR
    add_strassen_block(A_left, B_TR, C_right, 1)
    add_strassen_block(A_right, B_BR, C_right, 1)

    return {"U": U, "V": V, "W": W}
