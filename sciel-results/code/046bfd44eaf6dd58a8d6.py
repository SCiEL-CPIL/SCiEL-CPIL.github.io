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
    deadline = start + 130.0  # safety margin before 150s kill
    n, m, p = DIMENSIONS  # 3, 5, 5

    # For n=3, all Strassen blocks must have i0=0 (since i0+1 < 3 requires i0 <= 1,
    # but i0=1 would cover i={1,2}, leaving i=0 uncovered by any block).
    # Actually i0 can be 0 or 1, but blocks with i0=0 cover i={0,1} and i0=1 covers i={1,2}.
    # To maximize coverage, we want to use i0=0 blocks (covering i=0,1) since i=2 can't be
    # in any 2x2x2 block with another i value that's also in a block... wait, i0=1 covers {1,2}.
    # But we can't have overlapping blocks, so if we use i0=0, we can't use i0=1 for the same
    # j,k region. Let's just find the max set of non-overlapping 2x2x2 blocks.
    
    # Find maximum set of non-overlapping 2x2x2 blocks
    # A block at (i0,j0,k0) occupies i in [i0,i0+1], j in [j0,j0+1], k in [k0,k0+1]
    # Two blocks overlap if their i-ranges, j-ranges, and k-ranges all overlap.
    
    candidates = []
    for i0 in range(n - 1):
        for j0 in range(m - 1):
            for k0 in range(p - 1):
                candidates.append((i0, j0, k0))
    
    # Try all subsets is too expensive, but the space is small:
    # i0 in {0,1}, j0 in {0,1,2,3}, k0 in {0,1,2,3} => 2*4*4 = 32 candidates
    # Use greedy with multiple orderings, then verify optimality by checking known bound.
    
    def boxes_disjoint(b1, b2):
        """Check if two 2x2x2 boxes are disjoint (no shared cell)."""
        # i ranges: [b1[0], b1[0]+1] and [b2[0], b2[0]+1]
        if not (b1[0] + 1 < b2[0] or b2[0] + 1 < b1[0]):
            # i ranges overlap
            # j ranges: [b1[1], b1[1]+1] and [b2[1], b2[1]+1]
            if not (b1[1] + 1 < b2[1] or b2[1] + 1 < b1[1]):
                # j ranges overlap
                # k ranges: [b1[2], b1[2]+1] and [b2[2], b2[2]+1]
                if not (b1[2] + 1 < b2[2] or b2[2] + 1 < b1[2]):
                    # all three ranges overlap -> boxes share a cell
                    return False
        return True
    
    # Greedy: try different orderings to find good packings
    best_blocks = []
    
    import itertools
    orderings = [
        candidates,
        list(reversed(candidates)),
        sorted(candidates, key=lambda x: (x[1], x[2], x[0])),
        sorted(candidates, key=lambda x: (x[2], x[1], x[0])),
        sorted(candidates, key=lambda x: (x[0], x[1], x[2])),
    ]
    
    for ordering in orderings:
        selected = []
        for block in ordering:
            if all(boxes_disjoint(block, sel) for sel in selected):
                selected.append(block)
        if len(selected) > len(best_blocks):
            best_blocks = selected
    
    # The maximum should be 4 (proven by grid packing in 5x5 for j,k with i constraint)
    # Verify: 4 blocks cover 32 triples, remaining 43 use schoolbook
    # Total rank: 4*7 + 43 = 71
    
    blocks = best_blocks
    num_blocks = len(blocks)
    
    # Count covered triples
    covered = [[[False] * p for _ in range(m)] for _ in range(n)]
    for i0, j0, k0 in blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    covered[i0 + di][j0 + dj][k0 + dk] = True
    
    uncovered_count = sum(1 for i in range(n) for j in range(m) for k in range(p) if not covered[i][j][k])
    total_rank = num_blocks * 7 + uncovered_count
    
    # Build the matrices
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    def add_strassen_block(i0, j0, k0, col_offset):
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
    for i0, j0, k0 in blocks:
        add_strassen_block(i0, j0, k0, col)
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

    # Verify the decomposition
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)

    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
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

    if time.time() > deadline:
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
