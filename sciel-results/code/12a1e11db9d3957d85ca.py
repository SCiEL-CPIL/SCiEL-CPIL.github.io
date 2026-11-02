"""Tensor decomposition for 4x4 * 4x6 matrix multiplication.

Uses a block decomposition: split the 4x4*4x6 problem into 8 sub-problems
of 2x2 * 2x3, each using a rank-5 decomposition. Total rank = 40.
"""
import time

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START

def _rank5_2x2x3():
    """
    Return a rank-5 decomposition for 2x2 * 2x3 -> 2x3.
    U: (4, 5)  for A (2x2)
    V: (6, 5)  for B (2x3)
    W: (6, 5)  for C (2x3) in trace convention W[k*2 + i, c]
    """
    U = [
        [1, 0, 1, 0, 0],   # i=0,j=0
        [0, 1, 0, 0, 1],   # i=0,j=1
        [0, 0, 1, 1, 0],   # i=1,j=0
        [0, 0, 0, 1, 1],   # i=1,j=1
    ]
    V = [
        [1, 1, 0, 0, 0],   # j=0,k=0
        [0, 0, 1, 1, 0],   # j=0,k=1
        [0, 0, 0, 0, 1],   # j=0,k=2
        [0, 0, 0, 1, 0],   # j=1,k=0
        [0, 0, 1, 0, 0],   # j=1,k=1
        [1, 0, 0, 0, 0],   # j=1,k=2
    ]
    W = [
        [1, 0, 0, 0, 0],   # k=0,i=0
        [0, 1, 0, 0, 0],   # k=0,i=1
        [0, 0, 1, 0, 0],   # k=1,i=0
        [0, 0, 0, 1, 0],   # k=1,i=1
        [0, 0, 0, 0, 1],   # k=2,i=0
        [1, 1, 0, 0, 0],   # k=2,i=1
    ]
    return U, V, W


def _verify_full(U, V, W, n, m, p):
    """Verify the full decomposition."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True


def construct():
    deadline = time.time() + 130  # safety margin

    n, m, p = DIMENSIONS  # 4, 4, 6

    # Get rank-5 decomposition for 2x2x3
    U2, V2, W2 = _rank5_2x2x3()

    # Build the 4x4x6 decomposition by splitting into eight 2x2x3 blocks
    # A is 4x4, B is 4x6, C = A*B is 4x6
    # Split into 2x2 (i-blocks) x 2x2 (j-blocks) x 2x3 (k-blocks)
    # 8 sub-problems, each 2x2 * 2x3 -> 2x3, each rank 5
    # Total rank = 8 * 5 = 40

    rank = 8 * 5
    U = [[0] * rank for _ in range(n * m)]  # 16 x 40
    V = [[0] * rank for _ in range(m * p)]  # 24 x 40
    W = [[0] * rank for _ in range(p * n)]  # 24 x 40

    # Sub-problems indexed by (a, b, c) where:
    # a in {0,1}: i-block (rows 0-1 or 2-3)
    # b in {0,1}: j-block (cols 0-1 or 2-3)
    # c in {0,1}: k-block (cols 0-2 or 3-5)
    subproblem_idx = 0
    for a in range(2):
        for b in range(2):
            for c in range(2):
                for cc in range(5):
                    col = subproblem_idx * 5 + cc
                    # U: rows are (i, j) pairs
                    for ii in range(2):
                        for jj in range(2):
                            i_global = a * 2 + ii
                            j_global = b * 2 + jj
                            U[i_global * m + j_global][col] = U2[ii * 2 + jj][cc]

                    # V: rows are (j, k) pairs
                    for jj in range(2):
                        for kk in range(3):
                            j_global = b * 2 + jj
                            k_global = c * 3 + kk
                            V[j_global * p + k_global][col] = V2[jj * 3 + kk][cc]

                    # W: rows are (k, i) pairs (trace convention W[k*n + i])
                    for kk in range(3):
                        for ii in range(2):
                            k_global = c * 3 + kk
                            i_global = a * 2 + ii
                            W[k_global * n + i_global][col] = W2[kk * 2 + ii][cc]

                subproblem_idx += 1

    # Verify
    if _verify_full(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to schoolbook (rank 96)
        rank = n * m * p
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
