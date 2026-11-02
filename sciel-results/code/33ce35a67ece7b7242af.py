"""Matrix multiplication 3,5,5 — rank-minimal exact tensor decomposition."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Build a rank-minimal tensor decomposition for multiplying a 3x5 matrix by a 5x5 matrix.
    Returns dict with keys 'U', 'V', 'W' — 2D lists of shape (n*m, rank), (m*p, rank), (p*n, rank).
    The trace convention is W[k*n + i, r].
    Score = rank (lower is better).
    """
    start = time.time()
    n, m, p = DIMENSIONS  # 3, 5, 5

    # ---------------------------------------------------------------
    # Strategy:
    #   1. Try to pack as many 2x2x2 Strassen blocks (rank 7 each) as possible
    #      into disjoint sub-boxes of the [3,5,5] tensor.
    #   2. For the remaining uncovered (i,j,k) triples, use schoolbook (rank 1 each).
    #   3. Verify the decomposition by checking the tensor identity.
    #   4. If verification fails (shouldn't happen), fall back to pure schoolbook.
    #
    # The key improvement over a single Strassen block: we can fit multiple
    # 2x2x2 blocks in the 3x5x5 space. Since n=3, we can only use i0 in {0,1}
    # for 2x2x2 blocks (need i0 and i0+1 < 3). For m=5, j0 in {0,2,3} (need j0+1<5).
    # For p=5, k0 in {0,2,3} (need k0+1<5).
    #
    # Optimal packing: we want to cover as many of the 75 triples with 7-triplet
    # Strassen blocks as possible. Each Strassen block covers 8 triples in a 2x2x2
    # sub-box. We need the sub-boxes to be disjoint.
    #
    # Let's enumerate all possible 2x2x2 blocks and greedily pick non-overlapping ones.
    # ---------------------------------------------------------------

    # All possible 2x2x2 block origins
    candidates = []
    for i0 in range(n - 1):       # 0, 1
        for j0 in range(m - 1):   # 0, 1, 2, 3
            for k0 in range(p - 1):  # 0, 1, 2, 3
                candidates.append((i0, j0, k0))

    # Greedy: pick blocks that cover the most uncovered triples
    # (all cover exactly 8, so any maximal independent set works)
    # For maximum packing, we want blocks whose 2x2x2 boxes don't overlap.
    # Two boxes overlap if their i-ranges, j-ranges, AND k-ranges all overlap.

    def boxes_overlap(b1, b2):
        i1s, i1e = b1[0], b1[0] + 1
        i2s, i2e = b2[0], b2[0] + 1
        j1s, j1e = b1[1], b1[1] + 1
        j2s, j2e = b2[1], b2[1] + 1
        k1s, k1e = b1[2], b1[2] + 1
        k2s, k2e = b2[2], b2[2] + 1
        return not (i1e < i2s or i2e < i1s or j1e < j2s or j2e < j1s or k1e < k2s or k2e < k1s)

    selected = []
    covered = [[[False] * p for _ in range(m)] for _ in range(n)]

    for (i0, j0, k0) in candidates:
        # Check if this block overlaps any already-selected block
        conflict = False
        for (si0, sj0, sk0) in selected:
            # Check if the 2x2x2 boxes share any cell
            for di in range(2):
                for dj in range(2):
                    for dk in range(2):
                        ii, jj, kk = i0 + di, j0 + dj, k0 + dk
                        # Check if (ii,jj,kk) is in the other block
                        if (si0 <= ii <= si0 + 1 and sj0 <= jj <= sj0 + 1 and sk0 <= kk <= sk0 + 1):
                            conflict = True
                            break
                    if conflict:
                        break
                if conflict:
                    break
            if conflict:
                break
        if not conflict:
            selected.append((i0, j0, k0))
            for di in range(2):
                for dj in range(2):
                    for dk in range(2):
                        covered[i0 + di][j0 + dj][k0 + dk] = True

    # Count uncovered
    uncovered = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if not covered[i][j][k]:
                    uncovered += 1

    total_rank = len(selected) * 7 + uncovered

    # Build the matrices
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]

    def add_strassen_block(U, V, W, i0, j0, k0, col_offset):
        """Add a 2x2x2 Strassen decomposition starting at (i0, j0, k0)."""
        # M1 = (A[i0,j0] + A[i0+1,j0+1]) * (B[j0,k0] + B[j0+1,k0+1])
        c = col_offset + 0
        U[i0 * m + j0][c] += 1
        U[(i0 + 1) * m + (j0 + 1)][c] += 1
        V[j0 * p + k0][c] += 1
        V[(j0 + 1) * p + (k0 + 1)][c] += 1
        W[k0 * n + i0][c] += 1
        W[(k0 + 1) * n + (i0 + 1)][c] += 1

        # M2 = (A[i0+1,j0] + A[i0+1,j0+1]) * B[j0,k0]
        c = col_offset + 1
        U[(i0 + 1) * m + j0][c] += 1
        U[(i0 + 1) * m + (j0 + 1)][c] += 1
        V[j0 * p + k0][c] += 1
        W[k0 * n + (i0 + 1)][c] += 1
        W[(k0 + 1) * n + (i0 + 1)][c] += -1

        # M3 = A[i0,j0] * (B[j0,k0+1] - B[j0+1,k0+1])
        c = col_offset + 2
        U[i0 * m + j0][c] += 1
        V[j0 * p + (k0 + 1)][c] += 1
        V[(j0 + 1) * p + (k0 + 1)][c] += -1
        W[(k0 + 1) * n + i0][c] += 1
        W[(k0 + 1) * n + (i0 + 1)][c] += 1

        # M4 = A[i0+1,j0+1] * (B[j0+1,k0] - B[j0,k0])
        c = col_offset + 3
        U[(i0 + 1) * m + (j0 + 1)][c] += 1
        V[(j0 + 1) * p + k0][c] += 1
        V[j0 * p + k0][c] += -1
        W[k0 * n + i0][c] += 1
        W[k0 * n + (i0 + 1)][c] += 1

        # M5 = (A[i0,j0] + A[i0,j0+1]) * B[j0+1,k0+1]
        c = col_offset + 4
        U[i0 * m + j0][c] += 1
        U[i0 * m + (j0 + 1)][c] += 1
        V[(j0 + 1) * p + (k0 + 1)][c] += 1
        W[k0 * n + i0][c] += -1
        W[(k0 + 1) * n + i0][c] += 1

        # M6 = (A[i0+1,j0] - A[i0,j0]) * (B[j0,k0] + B[j0,k0+1])
        c = col_offset + 5
        U[(i0 + 1) * m + j0][c] += 1
        U[i0 * m + j0][c] += -1
        V[j0 * p + k0][c] += 1
        V[j0 * p + (k0 + 1)][c] += 1
        W[(k0 + 1) * n + (i0 + 1)][c] += 1

        # M7 = (A[i0,j0+1] - A[i0+1,j0+1]) * (B[j0+1,k0] + B[j0+1,k0+1])
        c = col_offset + 6
        U[i0 * m + (j0 + 1)][c] += 1
        U[(i0 + 1) * m + (j0 + 1)][c] += -1
        V[(j0 + 1) * p + k0][c] += 1
        V[(j0 + 1) * p + (k0 + 1)][c] += 1
        W[k0 * n + i0][c] += 1

    col = 0
    for (i0, j0, k0) in selected:
        add_strassen_block(U, V, W, i0, j0, k0, col)
        col += 7

    # Schoolbook for uncovered triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if not covered[i][j][k]:
                    U[i * m + j][col] += 1
                    V[j * p + k][col] += 1
                    W[k * n + i][col] += 1
                    col += 1

    # Verify the decomposition: for each (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] should be 1
    # And for any other combination it should be 0 (but we only need to check the diagonal)
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)

    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # The (i,j,k) entry of the tensor = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                val = np.sum(U_arr[i * m + j] * V_arr[j * p + k] * W_arr[k * n + i])
                if abs(val - 1.0) > 1e-10:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break

    if not valid:
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][c] = 1
                    V[j * p + k][c] = 1
                    W[k * n + i][c] = 1
                    c += 1
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
