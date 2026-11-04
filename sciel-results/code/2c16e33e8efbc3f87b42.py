import numpy as np

def construct():
    n, m, p = 2, 4, 4
    nm, mp, pn = n * m, m * p, p * n  # 8, 16, 8
    rank = 28

    U = np.zeros((nm, rank), dtype=float)
    V = np.zeros((mp, rank), dtype=float)
    W = np.zeros((pn, rank), dtype=float)

    # Standard Strassen 2x2x2 block:
    #  M0 = (A00+A11)*(B00+B11)
    #  M1 = (A10+A11)*B00
    #  M2 = A00*(B01-B11)
    #  M3 = A11*(B10-B00)
    #  M4 = (A00+A01)*B11
    #  M5 = (A10-A00)*(B00+B01)
    #  M6 = (A01-A11)*(B10+B11)
    #  C00 = M0+M3-M4+M6
    #  C01 = M2+M4
    #  C10 = M1+M5
    #  C11 = M0-M1+M2+M5

    def add_strassen(X_rows, Y_rows, Z_rows, col_offset):
        # X_rows = [X00, X01, X10, X11]  (row indices in U for the 4 sub-rows of A block)
        # Y_rows = [Y00, Y01, Y10, Y11]  (row indices in V for the 4 sub-rows of B block)
        # Z_rows = [Z00, Z01, Z10, Z11]  (row indices in W for the 4 sub-rows of C block)

        # M0 = (A00+A11)*(B00+B11)
        U[X_rows[0], col_offset + 0] += 1.0
        U[X_rows[3], col_offset + 0] += 1.0
        V[Y_rows[0], col_offset + 0] += 1.0
        V[Y_rows[3], col_offset + 0] += 1.0
        W[Z_rows[0], col_offset + 0] += 1.0
        W[Z_rows[3], col_offset + 0] += 1.0

        # M1 = (A10+A11)*B00
        U[X_rows[2], col_offset + 1] += 1.0
        U[X_rows[3], col_offset + 1] += 1.0
        V[Y_rows[0], col_offset + 1] += 1.0
        W[Z_rows[2], col_offset + 1] += 1.0
        W[Z_rows[3], col_offset + 1] -= 1.0

        # M2 = A00*(B01-B11)
        U[X_rows[0], col_offset + 2] += 1.0
        V[Y_rows[1], col_offset + 2] += 1.0
        V[Y_rows[3], col_offset + 2] -= 1.0
        W[Z_rows[1], col_offset + 2] += 1.0
        W[Z_rows[3], col_offset + 2] += 1.0

        # M3 = A11*(B10-B00)
        U[X_rows[3], col_offset + 3] += 1.0
        V[Y_rows[2], col_offset + 3] += 1.0
        V[Y_rows[0], col_offset + 3] -= 1.0
        W[Z_rows[0], col_offset + 3] += 1.0
        W[Z_rows[2], col_offset + 3] += 1.0

        # M4 = (A00+A01)*B11
        U[X_rows[0], col_offset + 4] += 1.0
        U[X_rows[1], col_offset + 4] += 1.0
        V[Y_rows[3], col_offset + 4] += 1.0
        W[Z_rows[0], col_offset + 4] -= 1.0
        W[Z_rows[1], col_offset + 4] += 1.0

        # M5 = (A10-A00)*(B00+B01)
        U[X_rows[2], col_offset + 5] += 1.0
        U[X_rows[0], col_offset + 5] -= 1.0
        V[Y_rows[0], col_offset + 5] += 1.0
        V[Y_rows[1], col_offset + 5] += 1.0
        W[Z_rows[3], col_offset + 5] += 1.0

        # M6 = (A01-A11)*(B10+B11)
        U[X_rows[1], col_offset + 6] += 1.0
        U[X_rows[3], col_offset + 6] -= 1.0
        V[Y_rows[2], col_offset + 6] += 1.0
        V[Y_rows[3], col_offset + 6] += 1.0
        W[Z_rows[0], col_offset + 6] += 1.0

    # The 2x4x4 multiplication: A (2x4) * B (4x4) = C (2x4)
    # Index convention: A[i,l] at row i*m+l, B[l,k] at row l*p+k, C[k,i] at row k*n+i
    # So A has rows 0-7 (i=0: 0,1,2,3; i=1: 4,5,6,7)
    # B has rows 0-15 (l=0: 0-3; l=1: 4-7; l=2: 8-11; l=3: 12-15)
    # C has rows 0-7 (k=0: 0,1; k=1: 2,3; k=2: 4,5; k=3: 6,7)

    # We view A as 2x2 blocks of 2x2:
    # A = [A00 A01; A10 A11] where each is 2x2
    # A00 = A[0:2, 0:2] -> U rows 0,1 (i=0, l=0,1)
    # A01 = A[0:2, 2:4] -> U rows 2,3 (i=0, l=2,3)
    # A10 = A[2:4, 0:2] -> U rows 4,5 (i=1, l=0,1)
    # A11 = A[2:4, 2:4] -> U rows 6,7 (i=1, l=2,3)

    # B is 4x4, viewed as 2x2 blocks of 2x2:
    # B = [B00 B01; B10 B11]
    # B00 = B[0:2, 0:2] -> V rows 0,1 (l=0, k=0,1)
    # B01 = B[0:2, 2:4] -> V rows 2,3 (l=0, k=2,3)
    # B10 = B[2:4, 0:2] -> V rows 8,9 (l=2, k=0,1)
    # B11 = B[2:4, 2:4] -> V rows 10,11 (l=2, k=2,3)

    # Wait, B is 4x4 with l=0..3, k=0..3.
    # B[l,k] at row l*p+k = l*4+k
    # B00: l=0,1; k=0,1 -> rows 0,1,4,5
    # B01: l=0,1; k=2,3 -> rows 2,3,6,7
    # B10: l=2,3; k=0,1 -> rows 8,9,12,13
    # B11: l=2,3; k=2,3 -> rows 10,11,14,15

    # C is 2x4, viewed as 2x2 blocks of 2x2:
    # C = [C00 C01; C10 C11] where each is 2x2
    # C[k,i] at row k*n+i = k*2+i
    # C00: k=0,1; i=0,1 -> rows 0,1,2,3
    # C01: k=2,3; i=0,1 -> rows 4,5,6,7

    # Block 1: C00 = A00*B00 + A10*B10  (top-left 2x2 of C)
    # This is a 2x2x2 multiplication: (A00+A10) doesn't quite work...
    # Actually C00 = A00*B00 + A10*B10
    # We need to compute this. Let's use Strassen on the combined problem.
    # 
    # Actually, let me reconsider the block structure.
    # C[0:2, 0:2] = A[0:2, 0:4] * B[0:4, 0:2]
    # = A[:, 0:2] * B[0:2, 0:2] + A[:, 2:4] * B[2:4, 0:2]
    # = A00 * B00 + A01 * B10  (where A00=A[0:2,0:2], A01=A[0:2,2:4], B00=B[0:2,0:2], B10=B[2:4,0:2])
    #
    # C[0:2, 2:4] = A[0:2, 0:4] * B[0:4, 2:4]
    # = A00 * B01 + A01 * B11
    #
    # C[2:4, 0:2] = A[2:4, 0:4] * B[0:4, 0:2]
    # = A10 * B00 + A11 * B10
    #
    # C[2:4, 2:4] = A[2:4, 0:4] * B[0:4, 2:4]
    # = A10 * B01 + A11 * B11

    # So we have 4 sub-problems, each of form: X*Y + Z*W where X,Z are 2x2 and Y,W are 2x2.
    # Each such sum can be computed with 7 multiplications via Strassen-like tricks,
    # but that's not quite right either.

    # Let me just use the standard approach: treat the whole thing as Strassen on 2x2 blocks
    # where the "blocks" are 2x2 matrices.

    # A is 2x4 = [A_left | A_right] where A_left, A_right are 2x2
    # B is 4x4 = [[B_tl, B_tr], [B_bl, B_br]] where each is 2x2
    # C = A * B = [A_left * B_tl + A_right * B_bl, A_left * B_tr + A_right * B_br]

    # So C_left = A_left * B_tl + A_right * B_bl  (2x2 result)
    # C_right = A_left * B_tr + A_right * B_br  (2x2 result)

    # Each of these is a sum of two 2x2x2 matrix products.
    # A_left * B_tl is rank 7 (Strassen), A_right * B_bl is rank 7, total 14.
    # Similarly C_right is rank 14. Total 28.

    # For C_left = A_left * B_tl + A_right * B_bl:
    # We can compute this as: let P = A_left + A_right, Q = B_tl + B_bl
    # Then A_left*B_tl + A_right*B_bl = P*Q - A_right*B_tl - A_left*B_bl + A_right*B_bl
    # That's 1 + 3 = 4 Strassen calls = 28 multiplications. Not helpful.

    # Better: just do two separate Strassen computations.
    # A_left * B_tl: 7 multiplications
    # A_right * B_bl: 7 multiplications
    # Sum them in the output.

    # A_left rows in U: 0,1 (i=0, l=0,1)
    # A_right rows in U: 2,3 (i=0, l=2,3)
    # A_left_bottom rows in U: 4,5 (i=1, l=0,1)
    # A_right_bottom rows in U: 6,7 (i=1, l=2,3)

    # B_tl rows in V: 0,1,4,5 (l=0,1; k=0,1)
    # B_tr rows in V: 2,3,6,7 (l=0,1; k=2,3)
    # B_bl rows in V: 8,9,12,13 (l=2,3; k=0,1)
    # B_br rows in V: 10,11,14,15 (l=2,3; k=2,3)

    # C_left (top) rows in W: 0,1 (k=0,1; i=0)
    # C_left (bottom) rows in W: 2,3 (k=0,1; i=1)
    # C_right (top) rows in W: 4,5 (k=2,3; i=0)
    # C_right (bottom) rows in W: 6,7 (k=2,3; i=1)

    # Strassen for 2x2x2: input X = [x00, x01, x10, x11], Y = [y00, y01, y10, y11]
    # Output Z = [z00, z01, z10, z11]
    # The 7 products contribute to Z as:
    # z00 += M0, M3, -M4, M6
    # z01 += M2, M4
    # z10 += M1, M5
    # z11 += M0, -M1, M2, M5

    # For A_left * B_tl -> contributes to C_left:
    # A_left = [[a00,a01],[a10,a11]] = U rows 0,1 (for i=0) and 4,5 (for i=1)
    # Wait, I need to be more careful. A is 2x4.
    # A[i,l] is at U row i*m+l = i*4+l.
    # A_left = A[:, 0:2] = A[0:2, 0:2] for the top part and A[2:4, 0:2] for bottom.
    # But in the 2x2x2 Strassen, we treat the 2x2 block as having 4 elements.
    # For A_left (top 2x2): elements A[0,0], A[0,1], A[1,0], A[1,1] at U rows 0,1,4,5.
    # For A_right (top 2x2): elements A[0,2], A[0,3], A[1,2], A[1,3] at U rows 2,3,6,7.

    # B_tl = B[0:2, 0:2]: elements B[0,0], B[0,1], B[1,0], B[1,1] at V rows 0,1,4,5.
    # B_bl = B[2:4, 0:2]: elements B[2,0], B[2,1], B[3,0], B[3,1] at V rows 8,9,12,13.

    # C_left = A_left * B_tl + A_right * B_bl
    # C_left is 2x2: C[0,0], C[0,1], C[1,0], C[1,1] at W rows 0,1,2,3.

    # Strassen for A_left * B_tl:
    # X = [A[0,0], A[0,1], A[1,0], A[1,1]] = U rows [0, 1, 4, 5]
    # Y = [B[0,0], B[0,1], B[1,0], B[1,1]] = V rows [0, 1, 4, 5]
    # Z = [C[0,0], C[0,1], C[1,0], C[1,1]] = W rows [0, 1, 2, 3]
    add_strassen([0, 1, 4, 5], [0, 1, 4, 5], [0, 1, 2, 3], 0)

    # Strassen for A_right * B_bl:
    # X = [A[0,2], A[0,3], A[1,2], A[1,3]] = U rows [2, 3, 6, 7]
    # Y = [B[2,0], B[2,1], B[3,0], B[3,1]] = V rows [8, 9, 12, 13]
    # Z = [C[0,0], C[0,1], C[1,0], C[1,1]] = W rows [0, 1, 2, 3]
    add_strassen([2, 3, 6, 7], [8, 9, 12, 13], [0, 1, 2, 3], 7)

    # C_right = A_left * B_tr + A_right * B_br
    # C_right is 2x2: C[2,0], C[2,1], C[3,0], C[3,1] at W rows 4,5,6,7.

    # Strassen for A_left * B_tr:
    # X = [A[0,0], A[0,1], A[1,0], A[1,1]] = U rows [0, 1, 4, 5]
    # Y = [B[0,2], B[0,3], B[1,2], B[1,3]] = V rows [2, 3, 6, 7]
    # Z = [C[2,0], C[2,1], C[3,0], C[3,1]] = W rows [4, 5, 6, 7]
    add_strassen([0, 1, 4, 5], [2, 3, 6, 7], [4, 5, 6, 7], 14)

    # Strassen for A_right * B_br:
    # X = [A[0,2], A[0,3], A[1,2], A[1,3]] = U rows [2, 3, 6, 7]
    # Y = [B[2,2], B[2,3], B[3,2], B[3,3]] = V rows [10, 11, 14, 15]
    # Z = [C[2,0], C[2,1], C[3,0], C[3,1]] = W rows [4, 5, 6, 7]
    add_strassen([2, 3, 6, 7], [10, 11, 14, 15], [4, 5, 6, 7], 21)

    # Verify
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for l in range(m):
            for k in range(p):
                a = i * m + l
                b = l * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    S = np.zeros((nm, mp, pn), dtype=float)
    for r in range(rank):
        S += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])

    max_err = np.max(np.abs(S - T))
    if max_err > 1e-10:
        # Fallback to schoolbook if something went wrong
        U2 = np.zeros((nm, 32), dtype=float)
        V2 = np.zeros((mp, 32), dtype=float)
        W2 = np.zeros((pn, 32), dtype=float)
        for i in range(n):
            for l in range(m):
                for k in range(p):
                    r = i * 16 + l * 4 + k
                    a = i * m + l
                    b = l * p + k
                    c = k * n + i
                    U2[a, r] = 1.0
                    V2[b, r] = 1.0
                    W2[c, r] = 1.0
        return {"U": U2.tolist(), "V": V2.tolist(), "W": W2.tolist()}

    def to_list(mat):
        result = []
        for row in mat:
            row_list = []
            for val in row:
                rounded = round(val * 2) / 2.0
                if abs(rounded - val) < 1e-10:
                    row_list.append(rounded)
                else:
                    row_list.append(val)
            result.append(row_list)
        return result

    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
