"""Block-diagonal Strassen decomposition for 3x3 x 3x6 matrix multiplication (rank 14)."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    # Partition the 6 columns of the second factor into two groups of 3:
    #   B = [B1 | B2],  B1 = B[:, 0:3], B2 = B[:, 3:6]
    # Then C = A*B = A1*B1 + A2*B2, where A1 = A2 = A (same 3x3 left factor).
    # Each 3x3 x 3x3 block is handled by the Strassen rank-7 decomposition.
    # Total rank = 7 + 7 = 14.
    n, m, p = DIMENSIONS

    # Strassen coefficients: M_r = (a + c_i * a_i)(b + c_j * b_j), i.e.
    # M_r = a*b + a*b_j + c_i*a*b + c_i*a*b_j.
    # (c_i, c_j) pairs:
    strassen = [
        (1, 0),  # M1
        (1, 1),  # M2
        (0, 1),  # M3
        (-1, 0), # M4
        (0, -1), # M5
        (1, -1), # M6
        (-1, 1), # M7
    ]

    # W row index for the (k, i) entry of a 3x3 block: k*n + i = 3k + i.
    def w_idx(k, i):
        return 3 * k + i

    # Block 1 (B1): columns 0..2 of B. V row = j*p + k = 3j + k (k in 0..2).
    # Block 2 (B2): columns 3..5 of B. V row = j*p + k = 3j + (k+3) = 3j + k + 3 (k in 0..2).
    # W row for block 2: same (k, i) -> 3k + i, since C's column = k.
    #   (B1 contributes to C[:, 0:3], B2 to C[:, 3:6]; but C[k,i] for block 2
    #    is at column k+3 of the full 3x6 C, and W uses the full output index.)
    #
    # IMPORTANT: W row index in the full problem is k_full*n + i, where k_full is
    # the column index of the 3x6 output C. For block 1, k_full = k (0..2).
    # For block 2, k_full = k + 3 (3..5).
    # So W index for block 2 = (k+3)*n + i = 3k + 9 + i.

    rank = 14
    U = [[0] * rank for _ in range(n * m)]   # 9 x 14
    V = [[0] * rank for _ in range(m * p)]   # 18 x 14
    W = [[0] * rank for _ in range(p * n)]   # 18 x 14

    for block in range(2):
        base = block * 7
        for r, (ci, cj) in enumerate(strassen):
            col = base + r
            for i in range(3):
                for j in range(3):
                    a = U[i * m + j][col]
                    a += 1
                    a += ci * i
                    a += cj * j
                    U[i * m + j][col] = a
                    for k in range(3):
                        b = V[j * p + k + block * 3][col]
                        b += 1
                        b += cj * k
                        V[j * p + k + block * 3][col] = b
                        wrow = (k + block * 3) * n + i
                        W[wrow][col] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
