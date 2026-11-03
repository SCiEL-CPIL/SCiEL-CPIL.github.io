"""Tensor decomposition for 3x3 * 3x5 matrix multiplication.
Uses Strassen on 2x2x2 blocks plus schoolbook for the remaining terms.
"""
DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 5
    rank = 43
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    def strassen_2x2x2(i0, j0, k0):
        """Place 7 Strassen terms for a 2x2x2 block starting at (i0,j0,k0).
        Returns next column index."""
        col = 0
        # Strassen terms for 2x2 matrices:
        # M1 = A00*(B01-B11)  -> C01
        # M2 = (A00+A10)*B11  -> C01, C11
        # M3 = (A10-A00)*(B00+B10) -> C10
        # M4 = A11*(B01-B10)  -> C00, C10
        # M5 = (A01-A11)*(B01+B11) -> C00, C11
        # M6 = (A00+A11)*(B00+B11) -> C00, C01, C10, C11
        # M7 = (A01-A00)*(B10+B11) -> C10, C11

        # We'll use the standard 7-term Strassen:
        # M1 = A00 * (B01 - B11)
        # M2 = (A00 + A10) * B11
        # M3 = (A10 - A00) * (B00 + B10)
        # M4 = A11 * (B01 - B10)
        # M5 = (A01 - A11) * (B01 + B11)
        # M6 = (A00 + A11) * (B00 + B11)
        # M7 = (A01 - A00) * (B10 + B11)
        #
        # C00 = M6 + M4 - M2 + M5
        # C01 = M1 + M6
        # C10 = M2 + M3
        # C11 = M1 - M4 + M3 + M7

        terms = [
            # (U_dict, V_dict, W_dict) for each term
            # M1 = A00 * (B01 - B11)
            ({(0, 0): 1}, {(0, 1): 1, (1, 1): -1}, {(0, 1): 1}),
            # M2 = (A00 + A10) * B11
            ({(0, 0): 1, (1, 0): 1}, {(1, 1): 1}, {(0, 1): -1, (1, 1): 1}),
            # M3 = (A10 - A00) * (B00 + B10)
            ({(1, 0): 1, (0, 0): -1}, {(0, 0): 1, (1, 0): 1}, {(1, 0): 1}),
            # M4 = A11 * (B01 - B10)
            ({(1, 1): 1}, {(0, 1): 1, (1, 0): -1}, {(0, 0): 1, (1, 0): -1}),
            # M5 = (A01 - A11) * (B01 + B11)
            ({(0, 1): 1, (1, 1): -1}, {(0, 1): 1, (1, 1): 1}, {(0, 0): 1, (1, 1): -1}),
            # M6 = (A00 + A11) * (B00 + B11)
            ({(0, 0): 1, (1, 1): 1}, {(0, 0): 1, (1, 1): 1},
             {(0, 0): 1, (0, 1): 1, (1, 0): 1, (1, 1): 1}),
            # M7 = (A01 - A00) * (B10 + B11)
            ({(0, 1): 1, (0, 0): -1}, {(1, 0): 1, (1, 1): 1}, {(1, 0): 1, (1, 1): 1}),
        ]

        c = 0
        for u_d, v_d, w_d in terms:
            for (di, dj), coeff in u_d.items():
                U[(i0 + di) * m + (j0 + dj)][col + c] += coeff
            for (dj, dk), coeff in v_d.items():
                V[(j0 + dj) * p + (k0 + dk)][col + c] += coeff
            for (dk, di), coeff in w_d.items():
                W[(k0 + dk) * n + (i0 + di)][col + c] += coeff
            c += 1
        return c

    col = 0
    # Block (i=0,1), (j=0,1), (k=0,1)
    col += strassen_2x2x2(0, 0, 0)
    # Block (i=0,1), (j=0,1), (k=2,3)
    col += strassen_2x2x2(0, 0, 2)

    # Schoolbook for remaining (i,j,k) triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i < 2 and j < 2 and k < 4:
                    continue  # covered by Strassen blocks
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    assert col == rank, f"Expected rank {rank}, got {col}"
    return {"U": U, "V": V, "W": W}
