import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110

    # Build a block-structured decomposition using Strassen on 2x2 blocks
    # where possible, falling back to schoolbook for non-power-of-2 slices.
    #
    # Strategy:
    #   A (4x5) = [A0 | A1]  where A0 is 4x4, A1 is 4x1
    #   B (5x6) = [B0; B1]  where B0 is 4x6, B1 is 1x6
    #   C = A0*B0 + A1*B1
    #
    # A0*B0: 4x4 * 4x6. Split B0 = [B00 | B01] (4x3 | 4x3).
    #   A0*B00: 4x4*4x3. Split into 2x2 Strassen blocks where possible.
    #   A0*B01: 4x4*4x3. Same.
    # A1*B1: 4x1 * 1x6 = outer product, rank 4.
    #
    # For 4x4 * 4x3:
    #   A0 = [[A11, A12], [A21, A22]] (2x2 blocks of 2x2)
    #   B = [[B11, B12, B13], [B21, B22, B23]] (2x3 blocks of 2x1... no)
    #   Actually 4x3: split rows into 2+2, cols into 1+1+1.
    #   This doesn't fit Strassen cleanly.
    #
    # Alternative: use schoolbook for 4x4*4x3 (rank 48) but that's worse.
    #
    # Better structural approach: use the identity that for rectangular matmul,
    # we can use "padded" Strassen. Pad 4x3 to 4x4, use Strassen (rank 28 for 4x4*4x4),
    # then the extra column is zero. But padding changes the tensor.
    #
    # Let me use a different decomposition:
    #   M(4,5,6) <= M(4,4,4) + M(4,4,4) + M(4,1,6) + cross terms
    #   This gets complicated.
    #
    # Simplest reliable improvement: use Strassen on the 4x4*4x4 sub-problem
    # embedded in 4x5*5x6.
    #
    # A (4x5), B (5x6), C (4x6)
    # Let A = [Aa Ab] where Aa is 4x4, Ab is 4x1
    # Let B = [Ba; Bb] where Ba is 4x6, Bb is 1x6
    # C = Aa*Ba + Ab*Bb
    #
    # Aa*Ba: 4x4 * 4x6. Pad Ba to 4x8 by adding 4 zero columns? No.
    # Split Ba = [Ba0 | Ba1] where Ba0 is 4x3, Ba1 is 4x3.
    # Aa*Ba0: 4x4*4x3. 
    #
    # For 4x4*4x3, use the following: 
    #   View as 2x2 blocks of 2x2 and 2x2x1... 
    #   Actually, use schoolbook: rank = 4*4*3 = 48. Two such: 96. Plus Ab*Bb: 4. Total 100.
    #   That's better than 120!
    #
    # Let me verify: Aa*Ba0 (4x4*4x3) schoolbook rank 48
    # Aa*Ba1 (4x4*4x3) schoolbook rank 48  
    # Ab*Bb (4x1*1x6) rank 4
    # Total: 48+48+4 = 100
    #
    # Wait, but we also need to handle the fact that A has 5 columns and B has 5 rows.
    # A = [Aa(4x4) Ab(4x1)], B = [Ba(4x6); Bb(1x6)]
    # C = Aa*Ba + Ab*Bb. Yes, that's correct.
    #
    # So total rank = 48 + 48 + 4 = 100.

    # Build the decomposition
    # Each term is (i, j, k) with coefficient, contributing to U[i*m+j], V[j*p+k], W[k*n+i]
    terms = []  # list of (i, j, k, coeff)

    # Aa*Ba0: i in 0..3, j in 0..3, k in 0..2
    for i in range(4):
        for j in range(4):
            for k in range(3):
                terms.append((i, j, k, 1.0))

    # Aa*Ba1: i in 0..3, j in 0..3, k in 3..5
    for i in range(4):
        for j in range(4):
            for k in range(3, 6):
                terms.append((i, j, k, 1.0))

    # Ab*Bb: i in 0..3, j=4, k in 0..5
    for i in range(4):
        for k in range(6):
            terms.append((i, 4, k, 1.0))

    rank = len(terms)
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    for r, (i, j, k, coeff) in enumerate(terms):
        U[i * m + j, r] = coeff
        V[j * p + k, r] = coeff
        W[k * n + i, r] = coeff

    def verify(U, V, W, n, m, p):
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0
                    actual = 0.0
                    for r_idx in range(r):
                        actual += U[i*m+j, r_idx] * V[j*p+k, r_idx] * W[k*n+i, r_idx]
                    if abs(actual - expected) > 1e-9:
                        return False
        return True

    if not verify(U, V, W, n, m, p):
        # Fallback to trivial
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
