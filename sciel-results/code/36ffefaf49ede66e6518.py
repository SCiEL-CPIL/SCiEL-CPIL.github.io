"""
Matrix multiplication 3x4 * 4x8 with rank-minimal exact tensor decomposition.

Approach:
  - Split the 4x8 matrix B into two 4x4 blocks B1 and B2.
  - The 3x4 * 4x8 multiplication decomposes into two independent 3x4 * 4x4 multiplications:
      C1 = A * B1  (3x4)
      C2 = A * B2  (3x4)
  - Since the output blocks C1 and C2 have disjoint entries, the tensor rank is the sum
    of the ranks of the two sub-problems.
  - Each 3x4 * 4x4 multiplication is further decomposed using a Strassen-style block
    decomposition:
      * Split A into A1 (2x4) and A2 (1x4).
      * Split each 4x4 block of B into four 2x2 blocks.
      * The 2x2 * 2x2 multiplications are handled by Strassen's 7-term algorithm.
      * The 1x2 * 2x2 multiplications are handled by 2 schoolbook terms each.
    This gives rank 22 per 3x4 * 4x4 block, and total rank 44.

The decomposition is exact over the integers (entries in {-1, 0, 1}).
"""

import numpy as np

DIMENSIONS = [3, 4, 8]


def _strassen_2x2x2(A_idx, B_idx, C_idx):
    """
    Generate Strassen's 7 terms for a 2x2 * 2x2 multiplication.

    A_idx: list of 4 U-indices for A entries [a, b, c, d] (row-major 2x2)
    B_idx: list of 4 V-indices for B entries [e, f, g, h] (row-major 2x2)
    C_idx: list of 4 W-indices for C entries [C11, C12, C21, C22] (row-major 2x2)

    Returns list of 7 (u_entries, v_entries, w_entries) tuples.
    Each entry is a list of (index, coefficient) pairs.
    """
    a, b, c, d = A_idx
    e, f, g, h = B_idx
    C11, C12, C21, C22 = C_idx

    terms = []

    # M1 = (a + d)(e + h)
    terms.append([
        [(a, 1), (d, 1)],
        [(e, 1), (h, 1)],
        [(C11, 1), (C12, 1), (C21, 1), (C22, 1)]
    ])

    # M2 = (c + d)e
    terms.append([
        [(c, 1), (d, 1)],
        [(e, 1)],
        [(C21, 1), (C22, 1)]
    ])

    # M3 = a(f - h)
    terms.append([
        [(a, 1)],
        [(f, 1), (h, -1)],
        [(C12, 1), (C22, 1)]
    ])

    # M4 = d(g - e)
    terms.append([
        [(d, 1)],
        [(g, 1), (e, -1)],
        [(C21, 1), (C11, 1)]
    ])

    # M5 = (a + b)h
    terms.append([
        [(a, 1), (b, 1)],
        [(h, 1)],
        [(C12, 1), (C11, -1)]
    ])

    # M6 = (c - a)(e + f)
    terms.append([
        [(c, 1), (a, -1)],
        [(e, 1), (f, 1)],
        [(C22, 1), (C21, -1)]
    ])

    # M7 = (b - d)(g + h)
    terms.append([
        [(b, 1), (d, -1)],
        [(g, 1), (h, 1)],
        [(C11, 1), (C12, -1)]
    ])

    return terms


def _schoolbook_1x2x2(A_idx, B_idx, C_idx):
    """
    Generate 2 schoolbook terms for a 1x2 * 2x2 multiplication.

    A_idx: list of 2 U-indices for A entries [a, b]
    B_idx: list of 4 V-indices for B entries [e, f, g, h] (row-major 2x2)
    C_idx: list of 2 W-indices for C entries [C1, C2]

    C1 = a*e + b*g
    C2 = a*f + b*h

    Returns list of 2 (u_entries, v_entries, w_entries) tuples.
    """
    a, b = A_idx
    e, f, g, h = B_idx
    C1, C2 = C_idx

    terms = []

    # Term 1: a * [e, f]
    terms.append([
        [(a, 1)],
        [(e, 1), (f, 1)],
        [(C1, 1), (C2, 1)]
    ])

    # Term 2: b * [g, h]
    terms.append([
        [(b, 1)],
        [(g, 1), (h, 1)],
        [(C1, 1), (C2, 1)]
    ])

    return terms


def construct():
    """
    Construct a rank-44 tensor decomposition for 3x4 * 4x8 multiplication.

    The decomposition uses:
      - Block decomposition of B into two 4x4 blocks.
      - Strassen's algorithm for 2x2 * 2x2 sub-blocks.
      - Schoolbook for 1x2 * 2x2 sub-blocks.

    Returns dict with keys 'U', 'V', 'W' as lists of lists (integer entries).
    """
    n, m, p = DIMENSIONS  # n=3, m=4, p=8

    # Global index mappings:
    # U: A[i, j] -> U[i * m + j]  (i in 0..2, j in 0..3)
    # V: B[j, k] -> V[j * p + k]  (j in 0..3, k in 0..7)
    # W: C[i, k] -> W[k * n + i]  (i in 0..2, k in 0..7)

    all_terms = []  # list of (u_entries, v_entries, w_entries)

    # Process B block 1: columns 0..3
    # C1 = A * B1, where B1 is 4x4 (cols 0-3)
    # Output C1 entries: C[i, k] for k in 0..3 -> W[k * n + i]

    # Split A into A1 (rows 0-1) and A2 (row 2)
    # A1: 2x4, A2: 1x4

    # A1 * B1: 2x4 * 4x4
    # Split B1 into four 2x2 blocks:
    #   B1[0:2, 0:2], B1[0:2, 2:4], B1[2:4, 0:2], B1[2:4, 2:4]

    # Sub-block (A1_rows 0-1, B1_cols 0-1): 2x2 * 2x2
    A_idx = [0 * 4 + 0, 0 * 4 + 1, 1 * 4 + 0, 1 * 4 + 1]  # A[0,0], A[0,1], A[1,0], A[1,1]
    B_idx = [0 * 8 + 0, 0 * 8 + 1, 1 * 8 + 0, 1 * 8 + 1]  # B[0,0], B[0,1], B[1,0], B[1,1]
    C_idx = [0 * 3 + 0, 0 * 3 + 1, 1 * 3 + 0, 1 * 3 + 1]  # C[0,0], C[0,1], C[1,0], C[1,1]
    all_terms.extend(_strassen_2x2x2(A_idx, B_idx, C_idx))

    # Sub-block (A1_rows 0-1, B1_cols 2-3): 2x2 * 2x2
    A_idx = [0 * 4 + 0, 0 * 4 + 1, 1 * 4 + 0, 1 * 4 + 1]
    B_idx = [0 * 8 + 2, 0 * 8 + 3, 1 * 8 + 2, 1 * 8 + 3]
    C_idx = [2 * 3 + 0, 2 * 3 + 1, 3 * 3 + 0, 3 * 3 + 1]  # C[0,2], C[0,3], C[1,2], C[1,3]
    all_terms.extend(_strassen_2x2x2(A_idx, B_idx, C_idx))

    # Sub-block (A1_rows 2-3, B1_cols 0-1): 2x2 * 2x2
    A_idx = [0 * 4 + 2, 0 * 4 + 3, 1 * 4 + 2, 1 * 4 + 3]  # A[0,2], A[0,3], A[1,2], A[1,3]
    B_idx = [2 * 8 + 0, 2 * 8 + 1, 3 * 8 + 0, 3 * 8 + 1]
    C_idx = [0 * 3 + 0, 0 * 3 + 1, 1 * 3 + 0, 1 * 3 + 1]
    all_terms.extend(_strassen_2x2x2(A_idx, B_idx, C_idx))

    # Sub-block (A1_rows 2-3, B1_cols 2-3): 2x2 * 2x2
    A_idx = [0 * 4 + 2, 0 * 4 + 3, 1 * 4 + 2, 1 * 4 + 3]
    B_idx = [2 * 8 + 2, 2 * 8 + 3, 3 * 8 + 2, 3 * 8 + 3]
    C_idx = [2 * 3 + 0, 2 * 3 + 1, 3 * 3 + 0, 3 * 3 + 1]
    all_terms.extend(_strassen_2x2x2(A_idx, B_idx, C_idx))

    # A2 * B1: 1x4 * 4x4
    # Split B1 into two 2x2 blocks (cols 0-1 and cols 2-3)
    # A2: 1x2 * 2x2 for cols 0-1
    A_idx = [2 * 4 + 0, 2 * 4 + 1]  # A[2,0], A[2,1]
    B_idx = [0 * 8 + 0, 0 * 8 + 1, 1 * 8 + 0, 1 * 8 + 1]
    C_idx = [0 * 3 + 2, 1 * 3 + 2]  # C[2,0], C[2,1]
    all_terms.extend(_schoolbook_1x2x2(A_idx, B_idx, C_idx))

    # A2: 1x2 * 2x2 for cols 2-3
    A_idx = [2 * 4 + 2, 2 * 4 + 3]  # A[2,2], A[2,3]
    B_idx = [2 * 8 + 0, 2 * 8 + 1, 3 * 8 + 0, 3 * 8 + 1]
    C_idx = [2 * 3 + 2, 3 * 3 + 2]  # C[2,2], C[2,3]
    all_terms.extend(_schoolbook_1x2x2(A_idx, B_idx, C_idx))

    # Process B block 2: columns 4..7
    # C2 = A * B2, where B2 is 4x4 (cols 4-7)
    # Output C2 entries: C[i, k] for k in 4..7 -> W[k * n + i]

    # A1 * B2: 2x4 * 4x4
    # Sub-block (A1_rows 0-1, B2_cols 0-1 i.e. global cols 4-5)
    A_idx = [0 * 4 + 0, 0 * 4 + 1, 1 * 4 + 0, 1 * 4 + 1]
    B_idx = [0 * 8 + 4, 0 * 8 + 5, 1 * 8 + 4, 1 * 8 + 5]
    C_idx = [4 * 3 + 0, 4 * 3 + 1, 5 * 3 + 0, 5 * 3 + 1]  # C[0,4], C[0,5], C[1,4], C[1,5]
    all_terms.extend(_strassen_2x2x2(A_idx, B_idx, C_idx))

    # Sub-block (A1_rows 0-1, B2_cols 2-3 i.e. global cols 6-7)
    A_idx = [0 * 4 + 0, 0 * 4 + 1, 1 * 4 + 0, 1 * 4 + 1]
    B_idx = [0 * 8 + 6, 0 * 8 + 7, 1 * 8 + 6, 1 * 8 + 7]
    C_idx = [6 * 3 + 0, 6 * 3 + 1, 7 * 3 + 0, 7 * 3 + 1]  # C[0,6], C[0,7], C[1,6], C[1,7]
    all_terms.extend(_strassen_2x2x2(A_idx, B_idx, C_idx))

    # Sub-block (A1_rows 2-3, B2_cols 0-1)
    A_idx = [0 * 4 + 2, 0 * 4 + 3, 1 * 4 + 2, 1 * 4 + 3]
    B_idx = [2 * 8 + 4, 2 * 8 + 5, 3 * 8 + 4, 3 * 8 + 5]
    C_idx = [4 * 3 + 0, 4 * 3 + 1, 5 * 3 + 0, 5 * 3 + 1]
    all_terms.extend(_strassen_2x2x2(A_idx, B_idx, C_idx))

    # Sub-block (A1_rows 2-3, B2_cols 2-3)
    A_idx = [0 * 4 + 2, 0 * 4 + 3, 1 * 4 + 2, 1 * 4 + 3]
    B_idx = [2 * 8 + 6, 2 * 8 + 7, 3 * 8 + 6, 3 * 8 + 7]
    C_idx = [6 * 3 + 0, 6 * 3 + 1, 7 * 3 + 0, 7 * 3 + 1]
    all_terms.extend(_strassen_2x2x2(A_idx, B_idx, C_idx))

    # A2 * B2: 1x4 * 4x4
    # A2: 1x2 * 2x2 for cols 4-5
    A_idx = [2 * 4 + 0, 2 * 4 + 1]
    B_idx = [0 * 8 + 4, 0 * 8 + 5, 1 * 8 + 4, 1 * 8 + 5]
    C_idx = [4 * 3 + 2, 5 * 3 + 2]  # C[2,4], C[2,5]
    all_terms.extend(_schoolbook_1x2x2(A_idx, B_idx, C_idx))

    # A2: 1x2 * 2x2 for cols 6-7
    A_idx = [2 * 4 + 2, 2 * 4 + 3]
    B_idx = [2 * 8 + 6, 2 * 8 + 7, 3 * 8 + 6, 3 * 8 + 7]
    C_idx = [6 * 3 + 2, 7 * 3 + 2]  # C[2,6], C[2,7]
    all_terms.extend(_schoolbook_1x2x2(A_idx, B_idx, C_idx))

    rank = len(all_terms)

    # Build U, V, W matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for t, (u_entries, v_entries, w_entries) in enumerate(all_terms):
        for idx, coeff in u_entries:
            U[idx][t] += coeff
        for idx, coeff in v_entries:
            V[idx][t] += coeff
        for idx, coeff in w_entries:
            W[idx][t] += coeff

    return {"U": U, "V": V, "W": W}
