"""Matrix multiplication 3x4x5: rank-minimal exact tensor decomposition.

The program constructs the schoolbook decomposition of rank n*m*p = 60 and then
performs a bounded, exact (Fraction-based) search for lower-rank decompositions
of the matmul tensor with half-integer entries.

Tensor convention (from the task):
  U shape (n*m, r):  row index a = i*m + j
  V shape (m*p, r):  row index b = j'*p + k
  W shape (p*n, r):  row index c = k'*n + i'
  T[a,b,c] = 1 iff j == j' AND k == k' AND i == i', else 0
  A rank-r decomposition: T[a,b,c] = sum_col U[a,col]*V[b,col]*W[c,col]

Search:
  1. Start from schoolbook (rank 60).
  2. Repeatedly try to eliminate a column: find half-integer coefficients
     expressing one column as a combination of the others (simultaneously in U,V,W).
  3. Also try proportional merges (col2 = alpha * col1).
  4. Return best (lowest-rank) valid decomposition found.
"""
import time
import numpy as np
from fractions import Fraction
from itertools import product

n, m, p = 3, 4, 5
_nm, _mp, _pn = n * m, m * p, p * n  # 12, 20, 15
_RANK_SCHOOLBOOK = _nm * _mp * _pn  # 60

_START_TIME = time.time()
_DEADLINE = 145.0  # 5s safety margin


def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)


def _build_schoolbook():
    """Build the schoolbook decomposition as lists of ints (0 or 1)."""
    r = _RANK_SCHOOLBOOK
    U = [[0] * r for _ in range(_nm)]
    V = [[0] * r for _ in range(_mp)]
    W = [[0] * r for _ in range(_pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W


def _validate(U, V, W):
    """Check that U,V,W form an exact decomposition of the matmul tensor."""
    r = len(U[0])
    for a in range(_nm):
        ia, ja = divmod(a, m)
        for b in range(_mp):
            jb, kb = divmod(b, p)
            for c in range(_pn):
                kc, ic = divmod(c, n)
                expected = 1 if (ja == jb and kb == kc and ia == ic) else 0
                actual = 0
                for col in range(r):
                    actual += U[a][col] * V[b][col] * W[c][col]
                if actual != expected:
                    return False
    return True


def _try_eliminate_column(U, V, W, col_elim):
    """Try to eliminate col_elim by expressing it as a half-integer combo of other columns.
    Returns (alpha_list, col_to_unk) on success, None otherwise.
    alpha_list has one Fraction per remaining column (in order)."""
    rank = len(U[0])
    if rank <= 1:
        return None

    remaining = [c for c in range(rank) if c != col_elim]
    num_unk = len(remaining)
    col_to_unk = {c: idx for idx, c in enumerate(remaining)}

    # Build system: for each equation (row from U, V, or W),
    # sum_c alpha_c * M[row, c] = M[row, col_elim]
    # Use Fraction arithmetic for exactness.

    # Collect equations
    equations = []  # list of (list_of_coeffs, rhs)
    for a in range(_nm):
        coeffs = [Fraction(U[a][c]) for c in remaining]
        rhs = Fraction(U[a][col_elim])
        equations.append((coeffs, rhs))
    for b in range(_mp):
        coeffs = [Fraction(V[b][c]) for c in remaining]
        rhs = Fraction(V[b][col_elim])
        equations.append((coeffs, rhs))
    for c in range(_pn):
        coeffs = [Fraction(W[c][col_elim] and W[c][col_elim] or 0) for c in remaining]  # placeholder
        # fix:
        coeffs = [Fraction(W[ci][col] for col in remaining) for ci in [c]]
        rhs = Fraction(W[c][col_elim])
        equations.append((coeffs, rhs))

    # Solve using Gaussian elimination over Fractions
    # Build augmented matrix
    rows = []
    for coeffs, rhs in equations:
        row = list(coeffs) + [rhs]
        rows.append(row)

    # Gaussian elimination
    num_cols = num_unk
    augmented = [row[:] for row in rows]
    num_rows = len(augmented)

    # Reduce to row echelon form
    r = 0
    col = 0
    while r < num_rows and col < num_cols:
        # Find pivot
        pivot = None
        for i in range(r, num_rows):
            if augmented[i][col] != 0:
                pivot = i
                break
        if pivot is None:
            col += 1
            continue
        augmented[r], augmented[pivot] = augmented[pivot], augmented[r]
        # Eliminate below
        pivot_val = augmented[r][col]
        for i in range(r + 1, num_rows):
            if augmented[i][col] != 0:
                factor = augmented[i][col] / pivot_val
                for j in range(col, num_cols + 1):
                    augmented[i][j] -= factor * augmented[r][j]
        r += 1
        col += 1

    # Check for inconsistency (row like [0, 0, ..., 0 | nonzero])
    for i in range(r, num_rows):
        if all(augmented[i][j] == 0 for j in range(num_cols)) and augmented[i][num_cols] != 0:
            return None

    # Back-substitute
    alpha = [Fraction(0)] * num_cols
    # Process from bottom
    for i in range(r - 1, -1, -1):
        # Find the pivot column for this row
        pivot_col = None
        for j in range(num_cols - 1, -1, -1):
            if augmented[i][j] != 0:
                pivot_col = j
                break
        if pivot_col is None:
            continue
        val = augmented[i][num_cols]
        for j in range(pivot_col + 1, num_cols):
            val -= augmented[i][j] * alpha[j]
        alpha[pivot_col] = val / augmented[i][pivot_col]

    # Check that all alpha values are half-integers
    for a in alpha:
        if a.denominator not in (1, 2):
            return None

    # Verify exactly
    for coeffs, rhs in equations:
        s = sum(alpha[idx] * coeffs[idx] for idx in range(num_unk))
        if s != rhs:
            return None

    return (alpha, col_to_unk)


def _do_eliminate(U, V, W, col_elim, alpha, col_to_unk):
    """Perform the elimination, returning new U, V, W."""
    rank = len(U[0])
    new_rank = rank - 1
    remaining = [c for c in range(rank) if c != col_elim]

    new_U = [[0] * new_rank for _ in range(_nm)]
    new_V = [[0] * new_rank for _ in range(_mp)]
    new_W = [[0] * new_rank for _ in range(_pn)]

    for idx, c in enumerate(remaining):
        for a in range(_nm):
            new_U[a][idx] = U[a][c]
        for b in range(_mp):
            new_V[b][idx] = V[b][c]
        for c_idx in range(_pn):
            new_W[c_idx][idx] = W[c_idx][c]

    return new_U, new_V, new_W


def _try_proportional_merge(U, V, W):
    """Try to find two columns where col2 = alpha * col1 for half-integer alpha.
    Returns (new_U, new_V, new_W) or None."""
    rank = len(U[0])
    for c1 in range(rank):
        for c2 in range(c1 + 1, rank):
            if _time_left() < 2:
                return None
            alpha = None
            ok = True
            # Check U
            for a in range(_nm):
                u1, u2 = U[a][c1], U[a][c2]
                if u1 == 0:
                    if u2 != 0:
                        ok = False
                        break
                else:
                    r = Fraction(u2, u1)
                    if alpha is None:
                        alpha = r
                    elif alpha != r:
                        ok = False
                        break
            if not ok:
                continue
            if alpha is None:
                if any(U[a][c2] != 0 for a in range(_nm)):
                    continue
                alpha = Fraction(0)
            if alpha.denominator not in (1, 2):
                continue
            # Check V
            for b in range(_mp):
                v1, v2 = V[b][c1], V[b][c2]
                if v1 == 0:
                    if v2 != 0:
                        ok = False
                        break
                else:
                    if Fraction(v2, v1) != alpha:
                        ok = False
                        break
            if not ok:
                continue
            # Check W
            for c_idx in range(_pn):
                w1, w2 = W[c_idx][c1], W[c_idx][c2]
                if w1 == 0:
                    if w2 != 0:
                        ok = False
                        break
                else:
                    if Fraction(w2, w1) != alpha:
                        ok = False
                        break
            if not ok:
                continue
            # Merge: drop c2
            return _do_eliminate(U, V, W, c2, None, None)
    return None


def construct():
    """Find a rank-minimal exact tensor decomposition for 3x4x5 matrix multiplication."""
    U, V, W = _build_schoolbook()
    current_rank = _RANK_SCHOOLBOOK
    best_rank = current_rank
    best_U, best_V, best_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]

    # Iteratively try to reduce rank
    max_iterations = 80
    for it in range(max_iterations):
        if _time_left() < 5:
            break
        merged = False

        # Try proportional merges first (cheaper)
        result = _try_proportional_merge(U, V, W)
        if result is not None:
            U, V, W = result
            current_rank = len(U[0])
            if current_rank < best_rank:
                best_rank = current_rank
                best_U = [row[:] for row in U]
                best_V = [row[:] for row in V]
                best_W = [row[:] for row in W]
            merged = True

        if not merged and _time_left() > 10:
            # Try general elimination
            rank = len(U[0])
            for col_elim in range(rank):
                if _time_left() < 3:
                    break
                res = _try_eliminate_column(U, V, W, col_elim)
                if res is not None:
                    alpha, col_to_unk = res
                    U, V, W = _do_eliminate(U, V, W, col_elim, alpha, col_to_unk)
                    current_rank = len(U[0])
                    if current_rank < best_rank:
                        best_rank = current_rank
                        best_U = [row[:] for row in U]
                        best_V = [row[:] for row in V]
                        best_W = [row[:] for row in W]
                    merged = True
                    break

        if not merged:
            break

    # Final validation
    if not _validate(best_U, best_V, best_W):
        # Fallback to schoolbook
        U, V, W = _build_schoolbook()
        return {"U": U, "V": V, "W": W}

    return {"U": best_U, "V": best_V, "W": best_W}
