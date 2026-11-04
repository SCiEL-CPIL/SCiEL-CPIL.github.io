"""Matrix multiplication 4x4x6 via per-(r,c,j) Strassen decomposition.

Splits A (4x4) into 2x2 blocks, B (4x6) into 2x2 blocks, C (4x6) into 2x2 blocks.
C[r,c] = sum_j A[r,j] * B[j,c]. Each 2x2 x 2x2 product uses 7 Strassen terms.
Total: 2 * 3 * 2 * 7 = 84 terms.
"""
import time

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6

    rank = 2 * 3 * 2 * 7  # 84
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen terms for 2x2 x 2x2 matrix multiplication
    # Each term: U (coeffs for A block), V (coeffs for B block), W (coeffs for C block)
    # Local indices: (row, col) in 0..1
    strassen_terms = [
        # M1 = (a00 + a11)(b00 + b11)
        {'U': [((0,0), 1), ((1,1), 1)],
         'V': [((0,0), 1), ((1,1), 1)],
         'W': [((0,0), 1), ((1,1), 1)]},
        # M2 = (a10 + a11)b00
        {'U': [((1,0), 1), ((1,1), 1)],
         'V': [((0,0), 1)],
         'W': [((1,0), 1)]},
        # M3 = a00(b01 - b11)
        {'U': [((0,0), 1)],
         'V': [((0,1), 1), ((1,1), -1)],
         'W': [((0,1), 1), ((1,1), 1)]},
        # M4 = a11(b10 - b00)
        {'U': [((1,1), 1)],
         'V': [((1,0), 1), ((0,0), -1)],
         'W': [((0,0), 1), ((1,0), 1)]},
        # M5 = (a00 + a01)b11
        {'U': [((0,0), 1), ((0,1), 1)],
         'V': [((1,1), 1)],
         'W': [((0,0), -1), ((0,1), 1)]},
        # M6 = (a10 - a00)(b00 + b01)
        {'U': [((1,0), 1), ((0,0), -1)],
         'V': [((0,0), 1), ((0,1), 1)],
         'W': [((1,1), 1)]},
        # M7 = (a01 - a11)(b10 + b11)
        {'U': [((0,1), 1), ((1,1), -1)],
         'V': [((1,0), 1), ((1,1), 1)],
         'W': [((0,0), 1)]},
    ]

    col = 0
    for r in range(2):  # row block of A / C
        for c in range(3):  # col block of B / C
            for j in range(2):  # middle index (col block of A / row block of B)
                for term in strassen_terms:
                    for (lr, lc), coeff in term['U']:
                        gr, gc = 2*r + lr, 2*j + lc
                        U[gr * m + gc][col] = coeff
                    for (lr, lc), coeff in term['V']:
                        gr, gc = 2*j + lr, 2*c + lc
                        V[gr * p + gc][col] = coeff
                    for (lr, lc), coeff in term['W']:
                        gr, gc = 2*r + lr, 2*c + lc
                        W[gr * n + gc][col] = coeff
                    col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
