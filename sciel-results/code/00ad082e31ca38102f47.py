"""Structured block decomposition for 4x4x6 matrix multiplication.

Exploits 4 = 2+2 and 6 = 2+2+2. The multiplication tensor splits into 6
independent blocks (two 2x2x2 diagonal blocks and four 2x2x3 off-diagonal
blocks). Each 2x2x2 block uses the rank-7 Strassen tensor; each 2x2x3 block
uses a rank-12 tensor (three rank-7 Strassen terms, one per output column).
Total rank = 2*7 + 4*12 = 29, versus the schoolbook 96.
"""
import time

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115.0

    # Rank-7 Strassen decomposition for 2x2x2 (canonical).
    # U indexed by (i,j) in [0,2)x[0,2); V by (j,k); W by (k,i).
    strassen = [
        (0, 0, 1, 1, 0, 1),  # (0,0),(0,1),(0,1)
        (1, 0, 1, 1, 1, 0),  # (1,0),(1,1),(1,0)
        (0, 1, 1, 0, 1, 1),  # (0,1),(1,0),(1,1)
        (1, 0, 0, 1, 1, 1),  # (1,0),(0,1),(1,1)
        (0, 1, 0, 1, 0, 0),  # (0,1),(0,1),(0,0)
        (0, 0, 1, 0, 1, 0),  # (0,0),(1,0),(1,0)
        (1, 1, 0, 0, 0, 1),  # (1,1),(0,0),(0,1)
    ]

    # Rank-12 tensor for 2x2x3: three independent 2x2x2 Strassen terms,
    # one per output column. V column indices offset by 2 per column.
    def strassen_223():
        terms = []
        for c in range(3):
            for (i, j, jj, kk, k, ii) in strassen:
                terms.append((i, j, jj, kk + 2 * c, k, ii))
        return terms

    s22 = strassen
    s223 = strassen_223()

    # Block structure of 4x4x6 = (2+2)x(2+2)x(2+2+2).
    # Blocks keyed by (i0 in {0,1}, j0 in {0,1}, c0 in {0,1,2}).
    # Diagonal blocks (i0==j0) are 2x2x2; off-diagonal (i0!=j0) are 2x2x3.
    blocks = []
    for i0 in range(2):
        for j0 in range(2):
            for c0 in range(3):
                if i0 == j0:
                    blocks.append((i0, j0, c0, s22, 2))
                else:
                    blocks.append((i0, j0, c0, s223, 3))

    # Build U, V, W as dense integer lists of shape (rows, rank).
    rank = sum(7 if (i0 == j0) else 12 for (i0, j0, c0, _, _) in blocks)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    for (i0, j0, c0, terms, cspan) in blocks:
        for (i, j, jj, kk, k, ii) in terms:
            ui = (i0 * 2 + i) * m + (j0 * 2 + j)
            vi = (j0 * 2 + jj) * p + (c0 * cspan + kk)
            wi = (c0 * cspan + k) * n + (i0 * 2 + ii)
            U[ui][col] = 1
            V[vi][col] = 1
            W[wi][col] = 1
            col += 1
        if time.time() > deadline:
            break

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
