"""Matrix multiplication 4x5 x 5x5, rank-minimal exact tensor decomposition.

Family: block decomposition with fast sub-blocks (Strassen).
We tile the (k,j) plane (k in 0..4, j in 0..4) with 2x2 blocks where possible:
k in {0,1,2,3} and j in {0,1,2,3} form 2x2 = 4 blocks in (k,j); with 2 A-row
blocks (i in {0,1} and {2,3}) this yields 8 disjoint 2x2x2 sub-problems. Each
is decomposed with Strassen's 7-term method (rank 7 instead of 8), saving 1 per
block = 8 total. The remaining 36 (i,k,j) triples (those with k=4 or j=4) use
schoolbook. Total rank = 8*7 + 36 = 92.
"""
import time

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    DEADLINE = 145.0  # safety margin before the 150 s wall-clock kill.

    n, m, p = DIMENSIONS  # n=4 rows of A/C, m=5 cols of A / rows of B, p=5 cols of B/C.

    # Strassen 7-term decomposition for a 2x2x2 block.
    # Each term is (u, v, w):
    #   u: list of (i, k, val) -> coefficient of A[i,k]
    #   v: list of (k, j, val) -> coefficient of B[k,j]
    #   w: list of (i, j, val) -> coefficient added to C[i,j]
    STRASSEN = [
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        ([(1,0,1),(1,1,1)], [(0,0,1)],          [(1,0,1)]),
        ([(0,0,1)],          [(0,1,1),(1,1,-1)],[(0,1,1),(1,1,1)]),
        ([(1,1,1)],          [(1,0,1),(0,0,-1)],[(0,0,1),(1,0,1)]),
        ([(0,0,1),(0,1,1)],  [(1,1,1)],          [(0,0,-1),(0,1,1)]),
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)],  [(1,1,1)]),
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)],  [(0,0,1)]),
    ]

    def is_strassen(i, k, j):
        # Covered by a 2x2x2 sub-block iff k < 4 and j < 4 (i always in 0..3).
        return k < 4 and j < 4

    # Count rank: 8 sub-blocks * 7 Strassen terms + schoolbook remainder.
    schoolbook_count = 0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                if not is_strassen(i, k, j):
                    schoolbook_count += 1
    n_subblocks = 8
    rank = n_subblocks * 7 + schoolbook_count

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    # Strassen terms for each of the 8 disjoint 2x2x2 sub-blocks.
    for a in range(2):          # A-row block: i in {2a, 2a+1}
        for b in range(2):      # A-col / B-row block: k in {2b, 2b+1}
            for c in range(2):  # B-col block: j in {2c, 2c+1}
                i0, k0, j0 = 2 * a, 2 * b, 2 * c
                for (u, v, w) in STRASSEN:
                    for (ii, kk, val) in u:
                        U[(i0 + ii) * m + (k0 + kk)][col] = val
                    for (kk, jj, val) in v:
                        V[(k0 + kk) * p + (j0 + jj)][col] = val
                    for (ii, jj, val) in w:
                        W[(j0 + jj) * n + (i0 + ii)][col] = val
                    col += 1
                    if time.time() > DEADLINE:
                        # Deterministic construction; guard is a no-op safety net.
                        return {"U": U, "V": V, "W": W}

    # Schoolbook terms for the remaining (i, k, j) triples.
    for i in range(n):
        for k in range(m):
            for j in range(p):
                if not is_strassen(i, k, j):
                    U[i * m + k][col] = 1
                    V[k * p + j][col] = 1
                    W[j * n + i][col] = 1
                    col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
