"""Rank-minimal exact tensor decomposition for 2x3 * 3x8 matrix multiplication.

Design (kept from the current program's family):
  * Base decomposition rank 44 = 4 * Strassen(2x2x2, rank 7) for the (2x2)x(2x8)
    block + schoolbook rank 16 for the (2x1)x(1x8) block.
  * Exact integer validation (no Fractions needed since all entries are integers).
  * Rank-reduction search within the same family:
        - exact duplicate-column removal,
        - single-column linear-dependency elimination (solve residual over the
          remaining columns, then round to integer/half-integer and validate),
        - two-column simultaneous elimination (solve 2x system, round, validate).
  * If nothing better is found in the time budget, return the rank-44 base.
"""
import time
import sys
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 125.0  # explicit guard: 150s kill, keep safety margin

    def _ok():
        return (time.time() - t0) < deadline

    n, m, p = 2, 3, 8
    num_terms = n * m * p  # 48

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

    # ---- Exact integer validation: sum_c U*V*W == 1 for every (i,a,k) ----
    def _validate(candidate):
        U = candidate["U"]
        V = candidate["V"]
        W = candidate["W"]
        rank = len(U[0])
        for i in range(n):
            for a in range(m):
                u_row = U[i * m + a]
                for k in range(p):
                    v_row = V[a * p + k]
                    w_row = W[k * n + i]
                    acc = 0
                    for c in range(rank):
                        uv = u_row[c] * v_row[c]
                        if uv:
                            acc += uv * w_row[c]
                    if acc != 1:
                        return False
        return True

    # ---- Strassen 2x2x2 (rank 7) ----
    S_U = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1],
    ]
    S_V = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
    ]
    S_W = [
        [1, 0, 0, 1],
        [0, 1, 0, -1],
        [0, 0, 1, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0],
    ]

    def _verify_strassen():
        for i in range(2):
            for a in range(2):
                for k in range(2):
                    acc = 0
                    for r in range(7):
                        acc += S_U[r][i * 2 + a] * S_V[r][a * 2 + k] * S_W[r][k * 2 + i]
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
            for r in range(7):
                for li in range(2):
                    for la in range(2):
                        u_val = S_U[r][li * 2 + la]
                        if u_val:
                            U[li * m + la][col] = u_val
                for la in range(2):
                    for lk in range(2):
                        v_val = S_V[r][la * 2 + lk]
                        if v_val:
                            V[la * p + (k0 + lk)][col] = v_val
                for lk in range(2):
                    for li in range(2):
                        w_val = S_W[r][lk * 2 + li]
                        if w_val:
                            W[(k0 + lk) * n + li][col] = w_val
                col += 1

        # Part 2: A2 (2x1) * B2 (1x8) schoolbook
        for i in range(2):
            for k in range(8):
                U[i * m + 2][col] = 1
                V[2 * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

        return {"U": U, "V": V, "W": W}

    base = _build_base()
    if not _validate(base):
        return _schoolbook()

    best_candidate = base
    best_rank = 44

    # ---- Tensor-space representation of each column ----
    def _col_vector(U, V, W, c):
        vec = [0] * num_terms
        for i in range(n):
            for a in range(m):
                uv = U[i * m + a][c]
                if not uv:
                    continue
                for k in range(p):
                    v_val = V[a * p + k][c]
                    if not v_val:
                        continue
                    w_val = W[k * n + i][c]
                    if not w_val:
                        continue
                    vec[i * m * p + a * p + k] = uv * v_val * w_val
        return vec

    U, V, W = base["U"], base["V"], base["W"]
    rank0 = len(U[0])  # 44
    cols = [_col_vector(U, V, W, c) for c in range(rank0)]

    # ---- (a) Exact duplicate-column removal ----
    seen = {}
    removable = []
    for c in range(rank0):
        key = tuple(cols[c])
        if key in seen:
            removable.append(c)
        else:
            seen[key] = c

    for c_rem in removable:
        if not _ok():
            break
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
            return best_candidate

    # ---- (b) Single-column linear-dependency elimination ----
    # For each column c, the residual to absorb is -cols[c]. Try to express it as a
    # linear combination of the other columns; round the weights to half-integers and
    # validate the resulting (rank-1) decomposition.
    def _round_half_exact(fr):
        """Round a Fraction to the nearest half-integer (multiple of 1/2)."""
        two = fr * 2
        # floor of two
        if two >= 0:
            fl = two.numerator // two.denominator
        else:
            fl = -((-two.numerator) // two.denominator)
        # Check if we should round up
        frac = two - fl
        if frac >= Fraction(1, 2):
            fl = fl + 1
        return Fraction(fl, 2)

    def _try_eliminate(c):
        nonlocal best_candidate, best_rank
        kept = [c2 for c2 in range(rank0) if c2 != c]
        # Build matrix A (num_terms x len(kept)) of kept column vectors
        mrows = num_terms
        ncols = len(kept)
        A = np.array([[cols[c2][r] for c2 in kept] for r in range(mrows)], dtype=object)
        b = np.array([-x for x in cols[c]], dtype=object)

        # Gaussian elimination (fractional) to solve A x = b.
        # Augmented matrix with row-reduction; track pivot columns.
        mat = np.column_stack([A, b.reshape(-1, 1)])
        pivots = []
        rank_r = 0
        for col in range(ncols):
            # find pivot
            piv = -1
            for r in range(rank_r, mrows):
                if mat[r, col] != 0:
                    piv = r
                    break
            if piv < 0:
                continue
            mat[[rank_r, piv]] = mat[[piv, rank_r]]
            pivots.append(col)
            pv = mat[rank_r, col]
            inv = Fraction(1) / pv
            mat[rank_r, col:] = mat[rank_r, col:] * inv
            for r in range(mrows):
                if r == rank_r:
                    continue
                f = mat[r, col]
                if f:
                    mat[r, col:] = mat[r, col:] - f * mat[rank_r, col:]
            rank_r += 1

        # Consistency check: rows that are zero in A must be zero in b.
        for r in range(rank_r, mrows):
            if mat[r, ncols] != 0:
                return False  # inconsistent: column c is not in the span -> cannot remove

        # Particular solution: free (non-pivot) columns = 0.
        x = [Fraction(0)] * ncols
        for r in range(rank_r):
            x[pivots[r]] = mat[r, ncols]

        # Round to nearest half-integer and validate.
        x_round = [_round_half_exact(xj) for xj in x]

        # Build candidate: kept columns unchanged, removed column c's contribution
        # replaced by adding x_round[j] * kept[j] into kept[j] (i.e. fold residual in).
        new_rank = ncols
        new_U = [[0] * new_rank for _ in range(n * m)]
        new_V = [[0] * new_rank for _ in range(m * p)]
        new_W = [[0] * new_rank for _ in range(p * n)]
        for j, c2 in enumerate(kept):
            wgt = x_round[j]
            for i in range(n * m):
                new_U[i][j] = U[i][c2] + wgt * U[i][c]
            for i in range(m * p):
                new_V[i][j] = V[i][c2] + wgt * V[i][c]
            for i in range(p * n):
                new_W[i][j] = W[i][c2] + wgt * W[i][c]

        cand = {"U": new_U, "V": new_V, "W": new_W}
        # Ensure entries are half-integers (they are, since base is integer and wgt is half-int).
        if _validate(cand):
            best_candidate = cand
            best_rank = new_rank
            return True
        return False

    def _try_eliminate_pair(c1, c2):
        """Try to simultaneously eliminate columns c1 and c2."""
        nonlocal best_candidate, best_rank
        kept = [c3 for c3 in range(rank0) if c3 != c1 and c3 != c2]
        mrows = num_terms
        ncols = len(kept)
        
        # Build matrix A (num_terms x ncols) of kept column vectors
        A = [[cols[c3][r] for c3 in kept] for r in range(mrows)]
        
        # Residual target b = -(cols[c1] + cols[c2])
        b = [-(cols[c1][r] + cols[c2][r]) for r in range(mrows)]
        
        # Gaussian elimination (fractional) to solve A x = b.
        mat = [row[:] + [b[r]] for r, row in enumerate(A)]
        pivots = []
        rank_r = 0
        for col in range(ncols):
            piv = -1
            for r in range(rank_r, mrows):
                if mat[r][col] != 0:
                    piv = r
                    break
            if piv < 0:
                continue
            mat[rank_r], mat[piv] = mat[piv], mat[rank_r]
            pivots.append(col)
            pv = mat[rank_r][col]
            inv = Fraction(1) / pv
            for cc in range(col, ncols + 1):
                mat[rank_r][cc] *= inv
            for r in range(mrows):
                if r == rank_r:
                    continue
                f = mat[r][col]
                if f:
                    for cc in range(col, ncols + 1):
                        mat[r][cc] -= f * mat[rank_r][cc]
            rank_r += 1

        # Consistency check
        for r in range(rank_r, mrows):
            if mat[r][ncols] != 0:
                return False

        # Particular solution
        x = [Fraction(0)] * ncols
        for r in range(rank_r):
            x[pivots[r]] = mat[r][ncols]

        # Round to nearest half-integer
        x_round = [_round_half_exact(xj) for xj in x]

        # Build candidate
        new_rank = ncols
        new_U = [[0] * new_rank for _ in range(n * m)]
        new_V = [[0] * new_rank for _ in range(m * p)]
        new_W = [[0] * new_rank for _ in range(p * n)]
        for j, c3 in enumerate(kept):
            wgt = x_round[j]
            for i in range(n * m):
                new_U[i][j] = U[i][c3] + wgt * (U[i][c1] + U[i][c2])
            for i in range(m * p):
                new_V[i][j] = V[i][c3] + wgt * (V[i][c1] + V[i][c2])
            for i in range(p * n):
                new_W[i][j] = W[i][c3] + wgt * (W[i][c1] + W[i][c2])

        cand = {"U": new_U, "V": new_V, "W": new_W}
        if _validate(cand):
            best_candidate = cand
            best_rank = new_rank
            return True
        return False

    # Try elimination on columns that are most likely dependent first.
    # Strassen columns (0-27) are more entangled; schoolbook columns (28-43) are isolated.
    # Try Strassen columns first, then schoolbook.
    order = list(range(rank0))
    random.seed(1234)
    random.shuffle(order)
    
    # Single-column elimination
    for c in order:
        if not _ok():
            break
        if _try_eliminate(c):
            return best_candidate

    # Two-column elimination (try pairs of Strassen columns first)
    strassen_cols = list(range(28))
    schoolbook_cols = list(range(28, rank0))
    
    # Try pairs within Strassen columns
    for i in range(len(strassen_cols)):
        if not _ok():
            break
        for j in range(i + 1, len(strassen_cols)):
            if not _ok():
                break
            c1, c2 = strassen_cols[i], strassen_cols[j]
            if _try_eliminate_pair(c1, c2):
                return best_candidate

    # Try pairs within schoolbook columns
    for i in range(len(schoolbook_cols)):
        if not _ok():
            break
        for j in range(i + 1, len(schoolbook_cols)):
            if not _ok():
                break
            c1, c2 = schoolbook_cols[i], schoolbook_cols[j]
            if _try_eliminate_pair(c1, c2):
                return best_candidate

    # Try mixed pairs (Strassen + schoolbook)
    for c1 in strassen_cols:
        if not _ok():
            break
        for c2 in schoolbook_cols:
            if not _ok():
                break
            if _try_eliminate_pair(c1, c2):
                return best_candidate

    if best_rank < 44:
        return best_candidate
    return base

# EVOLVE-BLOCK-END
