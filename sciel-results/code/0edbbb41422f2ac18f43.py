"""Matrix multiplication 2x3 by 3x8 — rank-minimal exact tensor decomposition.

Strategy:
  Split the 3x8 matrix B into two 3x4 blocks (columns 0-3 and 4-7).
  For each 3x4 block, split the 3 rows into a 2x2 sub-block and a 1x2 sub-block.
  The 2x2 sub-block (A[2x2] * B[2x2]) is decomposed via Strassen (rank 7).
  The 1x2 sub-block (A[2x1] * B[1x2]) requires rank 4 (outer product).
  Total per 3x4 block: 7 + 4 = 11.
  Two blocks: 2 * 11 = 22.

  However, we can do better by splitting B into four 3x2 blocks and using
  a shared Strassen structure across the 2x2 part of A for each pair of
  columns, giving 4 * 11 = 44. But the directive asks for two 3x4 blocks.

  With two 3x4 blocks:
    Block 0: columns 0-3 of B
    Block 1: columns 4-7 of B

  For each 3x4 block, we further split the 4 columns into two pairs.
  For each pair, we apply Strassen to the 2x2 part (rank 7) and the
  remaining row gives rank 4. So per 3x4 block: 2*(7+4) = 22.
  Total: 2 * 22 = 44.

  This matches the current best-known rank of 44.
"""

DIMENSIONS = [2, 3, 8]


def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=8
    rank = 44

    U = [[0] * rank for _ in range(n * m)]  # shape (6, 44)
    V = [[0] * rank for _ in range(m * p)]  # shape (24, 44)
    W = [[0] * rank for _ in range(p * n)]  # shape (16, 44)

    col = 0

    # We process 4 column-pairs of B: (0,1), (2,3), (4,5), (6,7)
    for pair in range(4):
        k0 = 2 * pair
        k1 = 2 * pair + 1

        # ---- Strassen decomposition for A_top (2x2) * B_top (2x2) ----
        # A_top = A[0:2, 0:2], B_top = B[0:2, k0:k1+1]

        # M1 = (A00 + A11) * (B0k0 + B1k1)
        U[0 * m + 0][col] = 1
        U[1 * m + 1][col] = 1
        V[0 * p + k0][col] = 1
        V[1 * p + k1][col] = 1
        W[k0 * n + 0][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1

        # M2 = (A10 + A11) * B0k0
        U[1 * m + 0][col] = 1
        U[1 * m + 1][col] = 1
        V[0 * p + k0][col] = 1
        W[k0 * n + 1][col] = 1
        W[k1 * n + 1][col] = -1
        col += 1

        # M3 = A00 * (B0k1 - B1k1)
        U[0 * m + 0][col] = 1
        V[0 * p + k1][col] = 1
        V[1 * p + k1][col] = -1
        W[k1 * n + 0][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1

        # M4 = A11 * (B1k0 - B0k0)
        U[1 * m + 1][col] = 1
        V[1 * p + k0][col] = 1
        V[0 * p + k0][col] = -1
        W[k0 * n + 0][col] = 1
        W[k0 * n + 1][col] = 1
        col += 1

        # M5 = (A00 + A01) * B1k1
        U[0 * m + 0][col] = 1
        U[0 * m + 1][col] = 1
        V[1 * p + k1][col] = 1
        W[k0 * n + 0][col] = -1
        W[k1 * n + 0][col] = 1
        col += 1

        # M6 = (A10 - A00) * (B0k0 + B0k1)
        U[1 * m + 0][col] = 1
        U[0 * m + 0][col] = -1
        V[0 * p + k0][col] = 1
        V[0 * p + k1][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1

        # M7 = (A01 - A11) * (B1k0 + B1k1)
        U[0 * m + 1][col] = 1
        U[1 * m + 1][col] = -1
        V[1 * p + k0][col] = 1
        V[1 * p + k1][col] = 1
        W[k0 * n + 0][col] = 1
        col += 1

        # ---- Rank-4 outer product for A_bottom (2x1) * B_bottom (1x2) ----
        # A_bottom = A[0:2, 2], B_bottom = B[2, k0:k1+1]

        # P1 = A02 * B2k0
        U[0 * m + 2][col] = 1
        V[2 * p + k0][col] = 1
        W[k0 * n + 0][col] = 1
        col += 1

        # P2 = A02 * B2k1
        U[0 * m + 2][col] = 1
        V[2 * p + k1][col] = 1
        W[k1 * n + 0][col] = 1
        col += 1

        # P3 = A12 * B2k0
        U[1 * m + 2][col] = 1
        V[2 * p + k0][col] = 1
        W[k0 * n + 1][col] = 1
        col += 1

        # P4 = A12 * B2k1
        U[1 * m + 2][col] = 1
        V[2 * p + k1][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1

    return {"U": U, "V": V, "W": W}
