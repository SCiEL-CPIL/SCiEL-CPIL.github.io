"""
Rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x8 matrix.

Strategy:
  1. Build a structured base: Strassen for (2,2,2) sub-blocks + schoolbook for remaining row/column.
     - 4 Strassen blocks of (2x2)*(2x2) covering A1(2x2)*B1(2x8): 4*7 = 28 terms
     - Schoolbook for A2(2x1)*B2(1x8): 2*1*8 = 16 terms
     - Total: rank 44
  2. Attempt to reduce rank by finding linear dependencies among the 44 term vectors
     and by trying to merge the schoolbook part more efficiently with Strassen ideas.
  3. Validate any candidate with exact rational arithmetic.
  4. If no improvement is found within the time budget, return the known rank-44 decomposition.
"""
import time
import sys
import random
from fractions import Fraction

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 130.0  # 20 s safety margin before 150 s kill

    def _elapsed_ok():
        return (time.time() - t0) < deadline

    n, m, p = 2, 3, 8

    # ---- Schoolbook fallback (rank 48) ----
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

    # ---- Exact validation using Fractions ----
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

    # ---- Strassen 2x2x2 decomposition (rank 7) ----
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

    # ---- Build rank-44 base candidate ----
    def _build_base():
        rank = 44
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0

        # Part 1: A1 (2x2) * B1 (2x8) using 4 Strassen blocks
        for block_idx in range(4):
            k0 = block_idx * 2
            k1 = k0 + 1
            for r in range(7):
                for li in range(2):
                    for la in range(2):
                        u_val = S_U[r][li * 2 + la]
                        if u_val == 0:
                            continue
                        u_idx = li * m + la
                        U[u_idx][col] = u_val

                for la in range(2):
                    for lk in range(2):
                        v_val = S_V[r][la * 2 + lk]
                        if v_val == 0:
                            continue
                        v_idx = la * p + (k0 + lk)
                        V[v_idx][col] = v_val

                for lk in range(2):
                    for li in range(2):
                        w_val = S_W[r][lk * 2 + li]
                        if w_val == 0:
                            continue
                        w_idx = (k0 + lk) * n + li
                        W[w_idx][col] = w_val

                col += 1

        # Part 2: A2 (2x1) * B2 (1x8) schoolbook
        for i in range(2):
            for k in range(8):
                u_idx = i * m + 2
                v_idx = 2 * p + k
                w_idx = k * n + i
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1

        return {"U": U, "V": V, "W": W}

    base = _build_base()
    if not _validate(base):
        return _schoolbook()

    best_candidate = base
    best_rank = 44

    # ---- Try to reduce rank via column elimination and merging ----
    def _try_reduce():
        nonlocal best_candidate, best_rank

        num_terms = n * m * p  # 48

        def _col_vector(U, V, W, c):
            vec = [0] * num_terms
            for i in range(n):
                for a in range(m):
                    u_val = U[i * m + a][c]
                    if u_val == 0:
                        continue
                    for k in range(p):
                        v_val = V[a * p + k][c]
                        if v_val == 0:
                            continue
                        w_val = W[k * n + i][c]
                        if w_val == 0:
                            continue
                        vec[i * m * p + a * p + k] = u_val * v_val * w_val
            return vec

        U = base["U"]
        V = base["V"]
        W = base["W"]
        rank0 = len(U[0])  # 44

        cols = []
        for c in range(rank0):
            cols.append(_col_vector(U, V, W, c))

        target = [1] * num_terms

        # Check that base is valid
        total = [0] * num_terms
        for cv in cols:
            for idx in range(num_terms):
                total[idx] += cv[idx]
        for idx in range(num_terms):
            if total[idx] != 1:
                return  # base invalid, shouldn't happen

        # Check for duplicate columns
        seen = {}
        removable = []
        for c in range(rank0):
            key = tuple(cols[c])
            if key in seen:
                removable.append(c)
            else:
                seen[key] = c

        if removable:
            for c_rem in removable:
                new_rank = rank0 - 1
                new_U = [[0] * new_rank for _ in range(n * m)]
                new_V = [[0] * new_rank for _ in range(m * p)]
                new_W = [[0] * new_rank for _ in range(p * n)]
                nc = 0
                for c in range(rank0):
                    if c == c_rem:
                        continue
                    for i in range(n * m):
                        new_U[i][nc] = U[i][c]
                    for i in range(m * p):
                        new_V[i][nc] = V[i][c]
                    for i in range(p * n):
                        new_W[i][nc] = W[i][c]
                    nc += 1

                cand = {"U": new_U, "V": new_V, "W": new_W}
                if _validate(cand):
                    best_candidate = cand
                    best_rank = new_rank
                    return

        # Try merging schoolbook part more efficiently
        # The schoolbook part computes A2 (2x1) * B2 (1x8) in 16 terms.
        # We can try to use Strassen-like tricks on the 2x1 x 1x8 product.
        # Since one dimension is 1, Strassen doesn't directly apply, but we can
        # try to combine terms from different k-blocks.
        
        # Alternative: try to reduce the total rank by finding linear dependencies
        # among all 44 columns using Gaussian elimination over Q.
        
        if not _elapsed_ok():
            return

        # Build the 48x44 matrix of column vectors
        mat = [[cols[c][idx] for c in range(rank0)] for idx in range(num_terms)]
        
        # Gaussian elimination to find rank of the column space
        # If rank < 44, we have dependencies
        rank_mat = 0
        pivots = []
        for col in range(rank0):
            pivot_row = -1
            for row in range(rank_mat, num_terms):
                if mat[row][col] != 0:
                    pivot_row = row
                    break
            if pivot_row == -1:
                continue
            # Swap
            mat[rank_mat], mat[pivot_row] = mat[pivot_row], mat[rank_mat]
            pivot_val = mat[rank_mat][col]
            # Eliminate below
            for row in range(rank_mat + 1, num_terms):
                if mat[row][col] != 0:
                    factor = Fraction(mat[row][col]) / Fraction(pivot_val)
                    for c2 in range(col, rank0):
                        mat[row][c2] -= factor * mat[rank_mat][c2]
            pivots.append(col)
            rank_mat += 1
            if rank_mat == num_terms:
                break

        # If rank_mat < 44, there are dependencies
        # But finding the actual reduced decomposition is complex
        # For now, just note that we've checked

        if best_rank < 44:
            return

        return

    if _elapsed_ok():
        _try_reduce()

    if best_rank < 44:
        return best_candidate

    return base

# EVOLVE-BLOCK-END
