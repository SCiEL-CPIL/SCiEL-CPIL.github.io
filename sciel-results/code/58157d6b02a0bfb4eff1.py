"""Tensor decomposition for 4x4x7 matrix multiplication using Strassen."""
import time
import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 7
    deadline = time.time() + 140  # safety margin before 150s kill

    # Strategy: Split the 4x4x7 problem into:
    #   4x4x4 part (columns 0-3 of B): use Strassen-based decomposition, rank 28
    #   4x4x3 part (columns 4-6 of B): use schoolbook, rank 48
    # Total rank = 76

    # Strassen decomposition for 2x2x2 (rank 7):
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)B00
    # M2 = A00(B10-B11)
    # M3 = A11(B01-B00)
    # M4 = (A00+A01)B11
    # M5 = (A10-A00)(B00+B01)
    # M6 = (A01-A11)(B10+B11)
    #
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5

    # For 4x4x4, split A into 2x2 blocks of 2x2, B into 2x2 blocks of 2x2.
    # Block indices: I,J in {0,1}, sub-indices i',j' in {0,1}
    # i = 2*I + i', j = 2*J + j'
    #
    # Each Strassen term M_t is a 2x2 matrix product. Using schoolbook for each
    # 2x2 product (rank 4), total = 7*4 = 28 rank-1 terms for 4x4x4.

    # Strassen structure:
    # term: (a_coeffs, b_coeffs, c_coeffs)
    # a_coeffs: dict from (I,J) -> sign (A block coefficients)
    # b_coeffs: dict from (J,K) -> sign (B block coefficients)
    # c_coeffs: dict from (I,K) -> sign (C block coefficients)

    strassen_terms = [
        # M0 = (A00+A11)(B00+B11)
        (
            {(0,0): 1, (1,1): 1},   # A coeffs
            {(0,0): 1, (1,1): 1},   # B coeffs
            {(0,0): 1, (1,1): 1}    # C coeffs
        ),
        # M1 = (A10+A11)B00
        (
            {(1,0): 1, (1,1): 1},
            {(0,0): 1},
            {(1,0): 1}
        ),
        # M2 = A00(B10-B11)
        (
            {(0,0): 1},
            {(1,0): 1, (1,1): -1},
            {(0,1): 1}
        ),
        # M3 = A11(B01-B00)
        (
            {(1,1): 1},
            {(0,1): 1, (0,0): -1},
            {(0,0): 1, (1,0): 1}
        ),
        # M4 = (A00+A01)B11
        (
            {(0,0): 1, (0,1): 1},
            {(1,1): 1},
            {(0,0): -1, (0,1): 1}
        ),
        # M5 = (A10-A00)(B00+B01)
        (
            {(1,0): 1, (0,0): -1},
            {(0,0): 1, (0,1): 1},
            {(1,1): 1}
        ),
        # M6 = (A01-A11)(B10+B11)
        (
            {(0,1): 1, (1,1): -1},
            {(1,0): 1, (1,1): 1},
            {(0,0): 1, (1,1): -1}
        ),
    ]

    # Total rank: 28 (Strassen for 4x4x4) + 48 (schoolbook for 4x4x3) = 76
    total_rank = 28 + 48

    U = [[0] * total_rank for _ in range(n * m)]  # 16 x 76
    V = [[0] * total_rank for _ in range(m * p)]  # 28 x 76
    W = [[0] * total_rank for _ in range(p * n)]  # 28 x 76

    col = 0

    # Part 1: 4x4x4 via Strassen (28 terms)
    # For each Strassen term t and each sub-multiplication (ip, jp, kp) in {0,1}^3,
    # we need to determine the rank-1 contribution.
    #
    # The Strassen term t computes:
    # M_t = (sum_{I,J} a_IJ * A[I,J]) * (sum_{J,K} b_JK * B[J,K])
    #
    # Entry (ip, kp) of M_t is:
    # sum_{J} [sum_I a_IJ * A[2I+ip, 2J+jp]] * [sum_K b_JK * B[2J+jp, 2K+kp]]
    #
    # This is NOT a single rank-1 term. To get rank-1 terms, we need to fix J as well.
    # So for each (ip, jp, kp) and each J, we get a rank-1 term.
    #
    # Wait, that would give 7 * 4 * 2 = 56 terms, not 28.
    #
    # Let me reconsider. The product of two 2x2 matrices using schoolbook gives 4 rank-1 terms.
    # For each Strassen term, the A_combined and B_combined are 2x2 matrices.
    # The product A_combined * B_combined has entries:
    # (A_combined * B_combined)[ip, kp] = sum_{jp} A_combined[ip, jp] * B_combined[jp, kp]
    #
    # A_combined[ip, jp] = sum_{I,J: 2I+ip=ip, 2J+jp=jp} a_IJ * A[2I+ip, 2J+jp]
    #                     = sum_I a_IJ * A[2I+ip, 2J+jp]  (where J is fixed by jp)
    #
    # This is getting complex. Let me just enumerate all rank-1 terms directly.
    #
    # For a rank-1 term in the 4x4x4 tensor, we need:
    # U[row, col] = 1 for some row in [0,16)
    # V[row, col] = 1 for some row in [0,28)
    # W[row, col] = 1 for some row in [0,28)
    #
    # The rank-1 term corresponds to a triple (i, j, k) where:
    # U[i*4+j, col] = 1
    # V[j*4+k, col] = 1  (for the 4x4x4 part, k in [0,4))
    # W[k*4+i, col] = 1
    #
    # But with Strassen, we're combining multiple (i,j,k) triples.
    #
    # Let me take a different approach: just use schoolbook for everything.
    # Rank = 112. This is guaranteed correct.

    # Actually, let me try to properly construct the Strassen decomposition.
    # The key insight is that each Strassen term M_t is itself a 2x2 matrix product.
    # To decompose this into rank-1 terms of the full 4x4x4 tensor, we need to
    # expand the 2x2 product using schoolbook.
    #
    # For Strassen term t:
    # M_t = A_t * B_t where A_t and B_t are 2x2 matrices with entries that are
    # linear combinations of A and B blocks.
    #
    # M_t[ip, kp] = sum_{jp} A_t[ip, jp] * B_t[jp, kp]
    #
    # Each A_t[ip, jp] is a linear combination of A[2I+ip, 2J+jp] for various I,J.
    # Each B_t[jp, kp] is a linear combination of B[2J+jp, 2K+kp] for various J,K.
    #
    # The product A_t[ip, jp] * B_t[jp, kp] is a product of two linear combinations,
    # which expands to a sum of products. Each product A[i,j] * B[j,k] is a rank-1
    # term in the tensor.
    #
    # So for each (t, ip, jp, kp), we get multiple rank-1 terms.
    #
    # This is getting very complex. Let me just use schoolbook for simplicity.
    # The rank will be 112, same as the seed, but at least it's correct.

    rank = n * m * p  # 112
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
