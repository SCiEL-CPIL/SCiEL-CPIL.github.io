"""Matrix multiplication 3x4 * 4x8 with rank-88 decomposition using Strassen for 2x2 blocks."""
import numpy as np
import time

def construct():
    n, m, p = 3, 4, 8
    deadline = time.time() + 140  # safety margin before 150s kill

    # Decomposition strategy:
    # Split A (3x4) into blocks:
    #   A_top (2x4): rows 0-1
    #   A_bot (1x4): row 2
    # Split B (4x8) into blocks:
    #   B_top (2x8): rows 0-1  
    #   B_bot (2x8): rows 2-3
    #
    # C = A@B where:
    #   C[0:2, :] = A_top @ B
    #   C[2, :] = A_bot @ B
    #
    # For A_top @ B (2x4 * 4x8):
    #   Split into 2x2 column blocks of B
    #   Each 2x4 * 4x2 subproblem can use Strassen-like techniques
    #
    # Actually, let's use the direct Strassen approach:
    # A = [[A11, A12], [A21, A22]] where A11,A12 are 2x2, A21,A22 are 1x2
    # B = [[B11, B12, B13, B14], [B21, B22, B23, B24]] where Bjk are 2x2
    #
    # C11 = A11*B11 + A12*B21 (each 2x2*2x2, use Strassen: 7+7=14)
    # C12 = A11*B12 + A12*B22 (14)
    # C13 = A11*B13 + A12*B23 (14)
    # C14 = A11*B14 + A12*B24 (14)
    # C21 = A21*B11 + A22*B21 (1x2*2x2 each, schoolbook: 4+4=8)
    # C22 = A21*B12 + A22*B22 (8)
    # C23 = A21*B13 + A22*B23 (8)
    # C24 = A21*B14 + A22*B24 (8)
    # Total: 4*14 + 4*8 = 56 + 32 = 88

    rank_total = 88
    U = [[0] * rank_total for _ in range(n * m)]  # 12 x 88
    V = [[0] * rank_total for _ in range(m * p)]  # 32 x 88
    W = [[0] * rank_total for _ in range(p * n)]  # 24 x 88

    col = 0

    # Strassen terms for 2x2 * 2x2
    strassen_terms = [
        # M1: (A00+A11)(B00+B11) -> C00, C11
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # M2: (A10+A11)B00 -> C10, C11(-)
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        # M3: A00(B01-B11) -> C01, C10
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,0):1}),
        # M4: A11(B10-B00) -> C10, C00
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(1,0):1, (0,0):1}),
        # M5: (A00+A01)B11 -> C00(-), C01
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        # M6: (A10-A00)(B00+B01) -> C11
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        # M7: (A01-A11)(B10+B11) -> C00, C11
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
    ]

    def add_strassen(a_rb, a_cb, b_rb, b_cb, c_rb, c_cb):
        """Add 7 Strassen terms for 2x2*2x2 at given block positions."""
        nonlocal col
        for a_coeffs, b_coeffs, c_coeffs in strassen_terms:
            for (ar, ac), coeff in a_coeffs.items():
                U[(a_rb + ar) * m + (a_cb + ac)][col] += coeff
            for (br, bc), coeff in b_coeffs.items():
                V[(b_rb + br) * p + (b_cb + bc)][col] += coeff
            for (cr, cc), coeff in c_coeffs.items():
                W[(c_cb + cc) * n + (c_rb + cr)][col] += coeff
            col += 1

    def add_1x2x2(a_rb, a_cb, b_rb, b_cb, c_rb, c_cb):
        """Add schoolbook terms for (1x2)*(2x2) product."""
        nonlocal col
        terms = [
            ({(0,0):1}, {(0,0):1}, {(0,0):1}),  # A00*B00 -> C00
            ({(0,1):1}, {(1,0):1}, {(0,0):1}),  # A01*B10 -> C00
            ({(0,0):1}, {(0,1):1}, {(0,1):1}),  # A00*B01 -> C01
            ({(0,1):1}, {(1,1):1}, {(0,1):1}),  # A01*B11 -> C01
        ]
        for a_coeffs, b_coeffs, c_coeffs in terms:
            for (ar, ac), coeff in a_coeffs.items():
                U[(a_rb + ar) * m + (a_cb + ac)][col] += coeff
            for (br, bc), coeff in b_coeffs.items():
                V[(b_rb + br) * p + (b_cb + bc)][col] += coeff
            for (cr, cc), coeff in c_coeffs.items():
                W[(c_cb + cc) * n + (c_rb + cr)][col] += coeff
            col += 1

    # Build C1 row (rows 0-1 of C):
    # C11: cols 0-1
    add_strassen(0, 0, 0, 0, 0, 0)  # A11*B11
    add_strassen(0, 2, 2, 0, 0, 0)  # A12*B21
    # C12: cols 2-3
    add_strassen(0, 0, 0, 2, 0, 2)  # A11*B12
    add_strassen(0, 2, 2, 2, 0, 2)  # A12*B22
    # C13: cols 4-5
    add_strassen(0, 0, 0, 4, 0, 4)  # A11*B13
    add_strassen(0, 2, 2, 4, 0, 4)  # A12*B23
    # C14: cols 6-7
    add_strassen(0, 0, 0, 6, 0, 6)  # A11*B14
    add_strassen(0, 2, 2, 6, 0, 6)  # A12*B24

    # Build C2 row (row 2 of C):
    # C21: cols 0-1
    add_1x2x2(2, 0, 0, 0, 2, 0)  # A21*B11
    add_1x2x2(2, 2, 2, 0, 2, 0)  # A22*B21
    # C22: cols 2-3
    add_1x2x2(2, 0, 0, 2, 2, 2)  # A21*B12
    add_1x2x2(2, 2, 2, 2, 2, 2)  # A22*B22
    # C23: cols 4-5
    add_1x2x2(2, 0, 0, 4, 2, 4)  # A21*B13
    add_1x2x2(2, 2, 2, 4, 2, 4)  # A22*B23
    # C24: cols 6-7
    add_1x2x2(2, 0, 0, 6, 2, 6)  # A21*B14
    add_1x2x2(2, 2, 2, 6, 2, 6)  # A22*B24

    # Verify
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)

    for i in range(n):
        for k in range(p):
            for j in range(m):
                s = np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i])
                if abs(s - 1.0) > 1e-9:
                    # Fallback to schoolbook
                    rank_sb = n * m * p
                    U_f = [[0] * rank_sb for _ in range(n*m)]
                    V_f = [[0] * rank_sb for _ in range(m*p)]
                    W_f = [[0] * rank_sb for _ in range(p*n)]
                    c = 0
                    for ii in range(n):
                        for jj in range(m):
                            for kk in range(p):
                                U_f[ii*m+jj][c] = 1
                                V_f[jj*p+kk][c] = 1
                                W_f[kk*n+ii][c] = 1
                                c += 1
                    return {"U": U_f, "V": V_f, "W": W_f}

    return {"U": U, "V": V, "W": W}
