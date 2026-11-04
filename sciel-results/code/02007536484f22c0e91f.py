"""
Strategy:
Decompose 2x3x8 as direct sum of three blocks: 2x3x3 + 2x3x3 + 2x3x2.
Each 2x3x3 block decomposes as (2x2x3) + (2x1x3):
  - 2x2x3 = (2x2x2 Strassen rank 7) + (2x2x1 rank 4) = rank 11
  - 2x1x3 = rank 3
  Total per 2x3x3: rank 14
The 2x3x2 block decomposes as (2x2x2 Strassen rank 7) + (2x1x2 rank 2) = rank 9
Grand total: 14 + 14 + 9 = rank 37
"""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 140.0

    def _elapsed_ok():
        return (time.time() - t0) < deadline

    n, m, p = 2, 3, 8

    def _schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    U[i * m + a][col] = 1
                    V[a * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def _validate(candidate):
        U = candidate["U"]
        V = candidate["V"]
        W = candidate["W"]
        rank = len(U[0])
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    acc = Fraction(0)
                    u_row = U[i * m + a]
                    v_row = V[a * p + k]
                    w_row = W[k * n + i]
                    for c in range(rank):
                        if u_row[c] == 0 or v_row[c] == 0 or w_row[c] == 0:
                            continue
                        acc += Fraction(u_row[c]) * Fraction(v_row[c]) * Fraction(w_row[c])
                    if acc != 1:
                        return False
        return True

    # Strassen 2x2x2 decomposition (rank 7)
    S_U = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    S_V = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    S_W = [
        [1, 0, 0, 1],
        [0, 1, 0, -1],
        [0, 0, 1, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]

    # Build a 2x2x2 Strassen block at given offsets
    def _add_strassen_block(U, V, W, i_off, a_off, k_off, col_start, num_cols):
        """Add a 2x2x2 Strassen block. i_off, a_off, k_off are offsets into the full matrices."""
        for r in range(7):
            for li in range(2):
                for la in range(2):
                    u_val = S_U[r][li * 2 + la]
                    if u_val != 0:
                        U[(i_off + li) * m + (a_off + la)][col_start + r] += u_val
            for la in range(2):
                for lk in range(2):
                    v_val = S_V[r][la * 2 + lk]
                    if v_val != 0:
                        V[(a_off + la) * p + (k_off + lk)][col_start + r] += v_val
            for lk in range(2):
                for li in range(2):
                    w_val = S_W[r][lk * 2 + li]
                    if w_val != 0:
                        W[(k_off + lk) * n + (i_off + li)][col_start + r] += w_val
        return col_start + 7

    # Build a 2x2x1 schoolbook block (rank 4)
    def _add_2x2x1_block(U, V, W, i_off, a_off, k_off, col_start):
        for li in range(2):
            for la in range(2):
                U[(i_off + li) * m + (a_off + la)][col_start] = 1
                V[(a_off + la) * p + k_off][col_start] = 1
                W[k_off * n + (i_off + li)][col_start] = 1
                col_start += 1
        return col_start

    # Build a 2x1xk schoolbook block (rank k)
    def _add_2x1xk_block(U, V, W, i_off, a_off, k_off, col_start, k_count):
        for li in range(2):
            for lk in range(k_count):
                U[(i_off + li) * m + a_off][col_start] = 1
                V[a_off * p + (k_off + lk)][col_start] = 1
                W[(k_off + lk) * n + (i_off + li)][col_start] = 1
                col_start += 1
        return col_start

    # Build a 1x1xk block (rank k)
    def _add_1x1xk_block(U, V, W, i_idx, a_idx, k_off, col_start, k_count):
        for lk in range(k_count):
            U[i_idx * m + a_idx][col_start] = 1
            V[a_idx * p + (k_off + lk)][col_start] = 1
            W[(k_off + lk) * n + i_idx][col_start] = 1
            col_start += 1
        return col_start

    # Total rank: 14 + 14 + 9 = 37
    rank = 37
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    # Block 1: 2x3x3 (k=0,1,2)
    # Part A: 2x2x3 (a=0,1; k=0,1,2)
    #   Sub-part A1: 2x2x2 Strassen (a=0,1; k=0,1) -> rank 7
    col = _add_strassen_block(U, V, W, 0, 0, 0, col, 7)
    #   Sub-part A2: 2x2x1 schoolbook (a=0,1; k=2) -> rank 4
    col = _add_2x2x1_block(U, V, W, 0, 0, 2, col)
    # Part B: 2x1x3 (a=2; k=0,1,2) -> rank 6
    col = _add_2x1xk_block(U, V, W, 0, 2, 0, col, 3)
    if not _elapsed_ok():
        return _schoolbook()

    # Block 2: 2x3x3 (k=3,4,5)
    # Part A: 2x2x3 (a=0,1; k=3,4,5)
    #   Sub-part A1: 2x2x2 Strassen (a=0,1; k=3,4) -> rank 7
    col = _add_strassen_block(U, V, W, 0, 0, 3, col, 7)
    #   Sub-part A2: 2x2x1 schoolbook (a=0,1; k=5) -> rank 4
    col = _add_2x2x1_block(U, V, W, 0, 0, 5, col)
    # Part B: 2x1x3 (a=2; k=3,4,5) -> rank 6
    col = _add_2x1xk_block(U, V, W, 0, 2, 3, col, 3)
    if not _elapsed_ok():
        return _schoolbook()

    # Block 3: 2x3x2 (k=6,7)
    # Part A: 2x2x2 Strassen (a=0,1; k=6,7) -> rank 7
    col = _add_strassen_block(U, V, W, 0, 0, 6, col, 7)
    # Part B: 2x1x2 (a=2; k=6,7) -> rank 4
    col = _add_2x1xk_block(U, V, W, 0, 2, 6, col, 2)
    if not _elapsed_ok():
        return _schoolbook()

    # Verify col == rank
    assert col == rank, f"Column mismatch: col={col}, rank={rank}"

    candidate = {"U": U, "V": V, "W": W}

    if _validate(candidate):
        return candidate
    else:
        # Fallback to schoolbook if validation fails
        return _schoolbook()

# EVOLVE-BLOCK-END
