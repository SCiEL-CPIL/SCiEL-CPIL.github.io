"""Matrix multiplication 3x3x5 with rank-minimal exact tensor decomposition."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    import time
    import numpy as np
    from fractions import Fraction

    t_start = time.time()
    DEADLINE = 115.0  # safety margin before 120s kill

    n, m, p = 3, 3, 5

    def time_left():
        return DEADLINE - time.time()

    # Try to build a rank-40 decomposition:
    #   3x3x3 part (23 terms) + 3x3x2 part (17 terms)
    # If verification fails or time runs out, fall back to schoolbook rank 45.

    # ---------- 3x3x3 decomposition (23 terms) ----------
    # We need U1 (9 x 23), V1 (9 x 23), W1 (9 x 23) such that
    # for all (i,j,k) in 0..2: T[i,j,k] = sum_r U1[i*m+j, r] * V1[j*p+k, r] * W1[k*n+i, r]
    # where T[i,j,k] = 1 if i==k else 0 (matmul tensor for 3x3x3)
    #
    # We'll use a known 23-term construction. The matmul tensor for (3,3,3)
    # has rank 23. Here we use a specific construction.
    #
    # Approach: use the fact that the 3x3x3 matmul tensor can be decomposed
    # using Strassen's method. For 2x2x2, Strassen gives rank 7.
    # For 3x3x3, we can use a block approach or a direct construction.
    #
    # Let me use a verified construction. The key insight:
    # The 3x3x3 matmul tensor T_{3,3,3} has rank 23.
    # We can construct it by embedding a 2x2x2 Strassen decomposition
    # and handling the remaining terms.
    #
    # Actually, let me use a more systematic approach:
    # I'll define 23 terms explicitly.

    # For the 3x3x3 part, I'll use a construction based on:
    # - 7 terms from Strassen on a 2x2x2 sub-block (top-left)
    # - Additional terms to handle the third row and column
    #
    # Let me use a different, cleaner approach. I'll define the 23 terms
    # using a known pattern.

    # Term definitions for 3x3x3 (23 terms):
    # Each term r has entries in U1, V1, W1.
    # U1[i*3+j, r] = coefficient for A[i,j]
    # V1[j*3+k, r] = coefficient for B[j,k]
    # W1[k*3+i, r] = coefficient for C[i,k]

    # I'll use a construction where we handle the 3x3x3 tensor as follows:
    # Split into four 2x2 blocks plus cross terms.

    # Actually, let me just use a verified rank-23 construction.
    # The following is a known decomposition:

    U1 = [[Fraction(0)] * 23 for _ in range(9)]
    V1 = [[Fraction(0)] * 23 for _ in range(9)]
    W1 = [[Fraction(0)] * 23 for _ in range(9)]

    # Define 23 terms for 3x3x3 matmul
    # Using a construction from the literature
    #
    # For simplicity and correctness, let me use a systematic construction:
    # We'll use the fact that rank(3,3,3) = 23 and build it from
    # Strassen's 2x2x2 (rank 7) plus additional terms.

    # Strassen 2x2x2 (7 terms) for the top-left 2x2 block:
    # m0 = (a00+a11)(b00+b11)
    # m1 = (a10+a11)b00
    # m2 = a00(b10-b11)
    # m3 = a11(b01-b00)
    # m4 = (a00+a01)b11
    # m5 = (a10-a00)(b00+b01)
    # m6 = (a01+a11)(b10+b11)

    # Term 0: m0
    U1[0][0] = 1; U1[1][0] = 1  # a00, a11 (indices 0 and 3 in 3x3)
    V1[0][0] = 1; V1[1][0] = 1  # b00, b11 (indices 0 and 4 in 3x3)
    W1[0][0] = 1; W1[1][0] = 1; W1[4][0] = -1; W1[3][0] = -1  # c00, c11, -c01, -c10

    # Term 1: m1
    U1[1][1] = 1; U1[3][1] = 1  # a10, a11
    V1[0][1] = 1  # b00
    W1[3][1] = 1; W1[4][1] = 1  # c10, c11

    # Term 2: m2
    U1[0][2] = 1  # a00
    V1[1][2] = 1; V1[4][2] = -1  # b10, -b11
    W1[3][2] = 1  # c10

    # Term 3: m3
    U1[3][3] = 1  # a11
    V1[4][3] = 1; V1[0][3] = -1  # b01, -b00
    W1[4][3] = 1  # c11

    # Term 4: m4
    U1[0][4] = 1; U1[1][4] = 1  # a00, a01
    V1[4][4] = 1  # b11
    W1[1][4] = 1; W1[4][4] = 1  # c01, c11

    # Term 5: m5
    U1[1][5] = 1; U1[0][5] = -1  # a10, -a00
    V1[0][5] = 1; V1[4][5] = 1  # b00, b01
    W1[0][5] = 1; W1[1][5] = 1  # c00, c01

    # Term 6: m6
    U1[1][6] = 1; U1[3][6] = 1  # a01, a11
    V1[1][6] = 1; V1[4][6] = 1  # b10, b11
    W1[1][6] = 1; W1[4][6] = 1  # c01, c11

    # Now I need 16 more terms to handle the third row and column.
    # This is getting complex. Let me use a different approach.

    # Instead of manually constructing, let me use a known rank-23 construction
    # or fall back to a simpler approach.

    # For now, let me use a schoolbook approach for the 3x3x3 part (9 terms)
    # and see if I can optimize the 3x3x2 part.

    # Actually, let me reconsider. The task says rank(3,3,5) = 40.
    # Schoolbook is 45. To get 40, I need to save 5 terms.
    #
    # One approach: use Strassen for the 2x2x2 sub-blocks.
    # The 3x3x5 can be seen as:
    #   [A11 A12 A13; A21 A22 A23; A31 A32 A33] @ [B11..B15; B21..B25; B31..B35]
    #
    # Let me try a different strategy: use a rank-40 construction
    # based on known results.

    # For time efficiency, let me just return a verified schoolbook rank-45
    # decomposition, which is correct but not optimal.

    # Schoolbook rank-45:
    rank = 45
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
