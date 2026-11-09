"""Strassen-based rank-56 decomposition for 2x4x7 matrix multiplication."""
import time

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 7
    deadline = time.time() + 110  # safety margin before 120s limit

    rank = n * m * p  # 56

    U = [[0] * rank for _ in range(n * m)]  # 8 x 56
    V = [[0] * rank for _ in range(m * p)]  # 28 x 56
    W = [[0] * rank for _ in range(p * n)]  # 14 x 56

    col = 0

    # Strassen for 2x2 x 2x2 -> 2x2
    # Blocks for A (n x m): i in {0,1}, j in {0,1,2,3}
    # Blocks for B (m x p): j in {0,1,2,3}, k in {0,1,2,3}
    # We split j into two 2-element blocks and k into two 2-element blocks
    # giving 4 sub-problems of 2x2 x 2x2 -> 2x2

    # Strassen multipliers for 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A00+A01)(B10-B11)
    # M3 = (A10+A11)(B00-B01)
    # M4 = A00*B00
    # M5 = A11*B11
    # M6 = (A01-A11)(B10+B11)
    # M7 = (A10-A00)(B00+B10)
    #
    # C00 = M1 + M4 - M5 - M6
    # C01 = M2 + M3
    # C10 = M3 + M5
    # C11 = M1 - M2 + M4

    def strassen_block(i0, j0, k0, start_col):
        """Add Strassen terms for block A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+2]."""
        nonlocal col
        terms = 0
        # M1 = (A00+A11)(B00+B11)
        U[i0 * m + (j0)] [start_col + 0] = 1
        U[(i0+1) * m + (j0+1)] [start_col + 0] = 1
        V[(j0) * p + (k0)] [start_col + 0] = 1
        V[(j0+1) * p + (k0+1)] [start_col + 0] = 1
        # C00 gets +1, C11 gets +1
        W[(k0) * n + i0] [start_col + 0] = 1
        W[(k0+1) * n + (i0+1)] [start_col + 0] = 1
        terms += 1

        # M2 = (A00+A01)(B10-B11)
        U[i0 * m + (j0)] [start_col + 1] = 1
        U[i0 * m + (j0+1)] [start_col + 1] = 1
        V[(j0+1) * p + (k0)] [start_col + 1] = 1
        V[(j0+1) * p + (k0+1)] [start_col + 1] = -1
        # C01 gets +1
        W[(k0+1) * n + i0] [start_col + 1] = 1
        terms += 1

        # M3 = (A10+A11)(B00-B01)
        U[(i0+1) * m + (j0)] [start_col + 2] = 1
        U[(i0+1) * m + (j0+1)] [start_col + 2] = 1
        V[(j0) * p + (k0)] [start_col + 2] = 1
        V[(j0) * p + (k0+1)] [start_col + 2] = -1
        # C01 gets +1, C10 gets +1
        W[(k0+1) * n + i0] [start_col + 2] = 1
        W[(k0) * n + (i0+1)] [start_col + 2] = 1
        terms += 1

        # M4 = A00*B00
        U[i0 * m + (j0)] [start_col + 3] = 1
        V[(j0) * p + (k0)] [start_col + 3] = 1
        # C00 gets +1, C11 gets +1
        W[(k0) * n + i0] [start_col + 3] = 1
        W[(k0+1) * n + (i0+1)] [start_col + 3] = 1
        terms += 1

        # M5 = A11*B11
        U[(i0+1) * m + (j0+1)] [start_col + 4] = 1
        V[(j0+1) * p + (k0+1)] [start_col + 4] = 1
        # C00 gets -1, C10 gets +1
        W[(k0) * n + i0] [start_col + 4] = -1
        W[(k0) * n + (i0+1)] [start_col + 4] = 1
        terms += 1

        # M6 = (A01-A11)(B10+B11)
        U[i0 * m + (j0+1)] [start_col + 5] = 1
        U[(i0+1) * m + (j0+1)] [start_col + 5] = -1
        V[(j0+1) * p + (k0)] [start_col + 5] = 1
        V[(j0+1) * p + (k0+1)] [start_col + 5] = 1
        # C00 gets -1
        W[(k0) * n + i0] [start_col + 5] = -1
        terms += 1

        # M7 = (A10-A00)(B00+B10)
        U[(i0+1) * m + (j0)] [start_col + 6] = 1
        U[i0 * m + (j0)] [start_col + 6] = -1
        V[(j0) * p + (k0)] [start_col + 6] = 1
        V[(j0+1) * p + (k0)] [start_col + 6] = 1
        # C01 gets -1, C11 gets +1
        W[(k0+1) * n + i0] [start_col + 6] = -1
        W[(k0+1) * n + (i0+1)] [start_col + 6] = 1
        terms += 1

        return start_col + terms

    # Four 2x2 x 2x2 blocks covering j in {0,2} and k in {0,2}
    # Block (i0=0, j0=0, k0=0)
    col = strassen_block(0, 0, 0, 0)
    # Block (i0=0, j0=0, k0=2)
    col = strassen_block(0, 0, 2, col)
    # Block (i0=0, j0=2, k0=0)
    col = strassen_block(0, 2, 0, col)
    # Block (i0=0, j0=2, k0=2)
    col = strassen_block(0, 2, 2, col)

    # Now handle the remaining columns k=4,5,6 with schoolbook
    for i in range(n):
        for j in range(m):
            for k in range(4, p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    assert col == rank, f"Expected {rank} terms, got {col}"

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
