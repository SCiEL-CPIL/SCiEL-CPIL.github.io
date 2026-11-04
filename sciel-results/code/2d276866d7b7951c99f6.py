"""Matrix multiplication 2x3x8 with rank-44 Strassen-based decomposition,
with an attempt at linear-algebraic compression to reduce rank below 44.

Strategy:
1. Build the standard rank-44 decomposition (4 Strassen blocks for 2x2x8 + 16 schoolbook terms for 2x1x8).
2. Attempt to find linear dependencies among the rank-1 terms by checking if any column of the combined
   factor matrices is in the span of others (over rationals). If a dependency is found, eliminate the
   redundant term to get a lower rank.
3. Validate the result exactly.
"""
import time
from fractions import Fraction
import numpy as np

def construct():
    t0 = time.time()
    deadline = t0 + 130.0

    n, m, p = 2, 3, 8

    def _elapsed_ok():
        return (time.time() - t0) < deadline

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
                    acc = 0
                    for c in range(rank):
                        acc += U[i * m + a][c] * V[a * p + k][c] * W[k * n + i][c]
                    if acc != 1:
                        return False
        return True

    # Strassen 2x2x2 coefficients
    S_U = [
        [1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 0, 1],
        [1, 1, 0, 0], [-1, 0, 1, 0], [0, 1, 0, -1]
    ]
    S_V = [
        [1, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, -1], [-1, 0, 1, 0],
        [0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]
    ]
    S_W = [
        [1, 0, 0, 1], [0, 1, 0, -1], [0, 0, 1, 1], [1, 1, 0, 0],
        [-1, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]
    ]

    def _verify_strassen():
        for i in range(2):
            for a in range(2):
                for k in range(2):
                    acc = 0
                    for r in range(7):
                        u_val = S_U[r][i * 2 + a]
                        v_val = S_V[r][a * 2 + k]
                        w_val = S_W[r][k * 2 + i]
                        acc += u_val * v_val * w_val
                    if acc != 1:
                        return False
        return True

    if not _verify_strassen():
        return _schoolbook()

    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    # Part 1: A1 (2x2) * B1 (2x8), split B1 into 4 blocks of 2 columns
    for block_idx in range(4):
        k0 = block_idx * 2
        for r in range(7):
            for li in range(2):
                for la in range(2):
                    u_val = S_U[r][li * 2 + la]
                    if u_val == 0:
                        continue
                    u_idx = li * m + la
                    U[u_idx][col] += u_val

            for la in range(2):
                for lk in range(2):
                    v_val = S_V[r][la * 2 + lk]
                    if v_val == 0:
                        continue
                    v_idx = la * p + (k0 + lk)
                    V[v_idx][col] += v_val

            for lk in range(2):
                for li in range(2):
                    w_val = S_W[r][lk * 2 + li]
                    if w_val == 0:
                        continue
                    w_idx = (k0 + lk) * n + li
                    W[w_idx][col] += w_val

            col += 1
            if not _elapsed_ok():
                return _schoolbook()

    # Part 2: A2 (2x1) * B2 (1x8)
    for i in range(2):
        for k in range(8):
            u_idx = i * m + 2
            v_idx = 2 * p + k
            w_idx = k * n + i
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1
            if not _elapsed_ok():
                return _schoolbook()

    candidate = {"U": U, "V": V, "W": W}

    if not _validate(candidate):
        return _schoolbook()

    # Attempt linear algebraic compression: check if any column is linearly dependent
    # on the others in the combined factor space.
    # We check if removing column j still allows the decomposition to hold.
    # This is equivalent to checking if the column is in the span of the others
    # in a way that preserves the tensor structure.

    # A simpler check: try removing each column and see if the remaining columns
    # can still sum to the correct values. This is expensive but rank is only 44.

    best_rank = rank
    best_candidate = candidate

    # Try removing each column one at a time
    for j in range(rank):
        if not _elapsed_ok():
            break

        # Create a copy without column j
        new_rank = rank - 1
        new_U = [[0] * new_rank for _ in range(n * m)]
        new_V = [[0] * new_rank for _ in range(m * p)]
        new_W = [[0] * new_rank for _ in range(p * n)]

        for row in range(n * m):
            for c in range(rank):
                if c == j:
                    continue
                new_col = c if c < j else c - 1
                new_U[row][new_col] = U[row][c]

        for row in range(m * p):
            for c in range(rank):
                if c == j:
                    continue
                new_col = c if c < j else c - 1
                new_V[row][new_col] = V[row][c]

        for row in range(p * n):
            for c in range(rank):
                if c == j:
                    continue
                new_col = c if c < j else c - 1
                new_W[row][new_col] = W[row][c]

        new_candidate = {"U": new_U, "V": new_V, "W": new_W}

        if _validate(new_candidate):
            best_rank = new_rank
            best_candidate = new_candidate
            # Try to remove another column from the new candidate
            # (greedy approach)
            rank = new_rank
            U = new_U
            V = new_V
            W = new_W
            candidate = new_candidate

    if best_rank < 44:
        return best_candidate
    else:
        return candidate
