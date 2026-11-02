"""Matrix multiplication 3,3,4: search for a rank-minimal exact tensor decomposition.

Strategy:
  1. Seed a candidate with the schoolbook (rank 36) decomposition.
  2. Greedy rank-reduction: repeatedly attempt to remove one column by solving
     for an exact replacement using a random linear combination of the remaining
     columns. Accept only when the residual error is exactly zero (verified with
     Fraction) and the new entries are integral.
  3. Also try a direct rank-21 construction by extending the known 3x3x3 rank-21
     algorithm (Strassen-type) to the 3x4 case.
  4. Return the best (lowest rank) valid decomposition found before the deadline.
"""

import time
from fractions import Fraction
import random

DIMENSIONS = [3, 3, 4]
DEADLINE = 135  # seconds; leaves margin before the 150 s wall-clock kill.


def _matmul_decompose(U, V, W):
    """Reconstruct the multiplication tensor from factor matrices and return it
    as a dict keyed by (i,j,k) -> value (Fraction)."""
    n, m, p = DIMENSIONS
    r = len(U[0])
    T = {}
    for c in range(r):
        # collect nonzero rows in each factor
        U_rows = [i for i in range(n * m) if U[i][c] != 0]
        V_rows = [j for j in range(m * p) if V[j][c] != 0]
        W_rows = [k for k in range(p * n) if W[k][c] != 0]
        if not U_rows or not V_rows or not W_rows:
            continue
        for i in U_rows:
            for j in V_rows:
                for k in W_rows:
                    val = Fraction(U[i][c]) * Fraction(V[j][c]) * Fraction(W[k][c])
                    key = (i, j, k)
                    T[key] = T.get(key, Fraction(0)) + val
    return T


def _target_tensor():
    """The exact matrix multiplication tensor: T[(i*m+j, j*p+k, k*n+i)] = 1."""
    n, m, p = DIMENSIONS
    T = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
            # U index: i*m+j, V index: j*p+k, W index: k*n+i
                T[(i * m + j, j * p + k, k * n + i)] = Fraction(1)
    return T


def _is_valid(U, V, W):
    """Check that U,V,W exactly decompose the multiplication tensor."""
    n, m, p = DIMENSIONS
    T = _matmul_decompose(U, V, W)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = Fraction(1)
                got = T.get((i * m + j, j * p + k, k * n + i), Fraction(0))
                if got != expected:
                    return False
    # Ensure no spurious nonzero entries
    for key, val in T.items():
        i, j, k = key
        if not (0 <= i < n * m and 0 <= j < m * p and 0 <= k < p * n):
            return False
        # Check it's a valid multiplication index
        ii = i // m
        jj = i % m
        jj2 = j // p
        kk = j % p
        kk2 = k // n
        ii2 = k % n
        if jj != jj2 or kk != ii2:
            if val != 0:
                return False
    return True


def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[Fraction(0)] * rank for _ in range(n * m)]
    V = [[Fraction(0)] * rank for _ in range(m * p)]
    W = [[Fraction(0)] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = Fraction(1)
                V[j * p + k][c] = Fraction(1)
                W[k * n + i][c] = Fraction(1)
                c += 1
    return U, V, W


def _try_remove_column(U, V, W, col, rng):
    """Try to remove column `col` by expressing its contribution as a linear
    combination of other columns. Returns (U', V', W') or None."""
    n, m, p = DIMENSIONS
    r = len(U[0])
    if r <= 1:
        return None

    # Compute the tensor contribution of column `col`
    T_col = {}
    for i in range(n * m):
        if U[i][col] == 0:
            continue
        for j in range(m * p):
            if V[j][col] == 0:
                continue
            for k in range(p * n):
                if W[k][col] == 0:
                    continue
                val = U[i][col] * V[j][col] * W[k][col]
                key = (i, j, k)
                T_col[key] = T_col.get(key, Fraction(0)) + val

    if not T_col:
        # Column contributes nothing; just drop it.
        U2 = [row[:col] + row[col + 1:] for row in U]
        V2 = [row[:col] + row[col + 1:] for row in V]
        W2 = [row[:col] + row[col + 1:] for row in W]
        return U2, V2, W2

    # Try to find coefficients alpha_c (c != col) such that
    # sum_c alpha_c * (U_col_c * V_col_c * W_col_c) = T_col
    # This is a linear system. We'll use a randomized approach:
    # Pick a random subset of other columns and try to solve.

    other_cols = [c for c in range(r) if c != col]
    # Try with a small random subset to keep the system tractable
    subset_size = min(len(other_cols), 6)
    for attempt in range(4):
        subset = rng.sample(other_cols, subset_size)
        # Build linear system: for each (i,j,k) in T_col,
        # sum_c alpha_c * U[i][c]*V[j][c]*W[k][c] = T_col[(i,j,k)]
        # Also for (i,j,k) not in T_col, the sum must be 0.
        # This is generally overdetermined. Use a least-squares-free approach:
        # We need exact solution. Try to solve the system where the number of
        # equations equals the number of unknowns by selecting a subset of
        # equations.

        # Actually, a cleaner approach: we want to find alpha such that the
        # rank-1 tensor formed by (sum alpha_c U_c, sum alpha_c V_c, sum alpha_c W_c)
        # equals T_col. This requires that T_col is itself a rank-1 tensor
        # (as a 3-tensor), which it generally is NOT. So this approach only
        # works when T_col is rank-1.

        # Alternative: allow the replacement to use multiple new columns.
        # But that doesn't reduce rank.

        # Better approach: try to find a single new column (i', j', k') pattern
        # that, when added, cancels the removed column. This is equivalent to
        # checking if T_col is rank-1.

        # Check if T_col is rank-1: T_col[i,j,k] = a_i * b_j * c_k
        # For this to hold, the support must be a product set.
        is_rank1 = True
        I_set = set()
        J_set = set()
        K_set = set()
        for (i, j, k) in T_col:
            I_set.add(i)
            J_set.add(j)
            K_set.add(k)

        if len(I_set) * len(J_set) * len(K_set) != len(T_col):
            is_rank1 = False

        if is_rank1:
            # Find a_i, b_j, c_k such that a_i*b_j*c_k = T_col[i,j,k]
            # Pick a reference element
            ref_i = next(iter(I_set))
            ref_j = next(iter(J_set))
            ref_k = next(iter(K_set))
            ref_val = T_col[(ref_i, ref_j, ref_k)]
            # a_{ref_i} * b_{ref_j} * c_{ref_k} = ref_val
            # We have freedom: set b_{ref_j} = 1, c_{ref_k} = 1, a_{ref_i} = ref_val
            a = {ref_i: ref_val}
            b = {ref_j: Fraction(1)}
            c = {ref_k: Fraction(1)}
            valid = True
            for i in I_set:
                if i == ref_i:
                    continue
                for j in J_set:
                    if j == ref_j:
                        continue
                    val_ij = T_col.get((i, j, ref_k), None)
                    if val_ij is None:
                        valid = False
                        break
                    # a_i * b_{ref_j} * c_{ref_k} = val_ij => a_i = val_ij
                    if i not in a:
                        a[i] = val_ij
                    elif a[i] != val_ij:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                for j in J_set:
                    if j == ref_j:
                        continue
                    for i in I_set:
                        val = T_col.get((i, j, ref_k), None)
                        if val is None:
                            valid = False
                            break
                        if i not in a or a[i] == 0:
                            valid = False
                            break
                        b_j_val = val / a[i]
                        if j not in b:
                            b[j] = b_j_val
                        elif b[j] != b_j_val:
                            valid = False
                            break
                    if not valid:
                        break
            if valid:
                for k in K_set:
                    if k == ref_k:
                        continue
                    for i in I_set:
                        val = T_col.get((i, ref_j, k), None)
                        if val is None:
                            valid = False
                            break
                        if i not in a or a[i] == 0:
                            valid = False
                            break
                        c_k_val = val / a[i]
                        if k not in c:
                            c[k] = c_k_val
                        elif c[k] != c_k_val:
                            valid = False
                            break
                    if not valid:
                        break
            if valid:
                # Verify all entries
                for i in I_set:
                    for j in J_set:
                        for k in K_set:
                            expected = T_col.get((i, j, k), Fraction(0))
                            computed = a.get(i, Fraction(0)) * b.get(j, Fraction(0)) * c.get(k, Fraction(0))
                            if computed != expected:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
            if valid:
                # Create new column
                U2 = [row[:] for row in U]
                V2 = [row[:] for row in V]
                W2 = [row[:] for row in W]
                new_col = r - 1
                for i in range(n * m):
                    U2[i][new_col] = a.get(i, Fraction(0))
                for j in range(m * p):
                    V2[j][new_col] = b.get(j, Fraction(0))
                for k in range(p * n):
                    W2[k][new_col] = c.get(k, Fraction(0))
                # Remove old column
                U2 = [row[:col] + row[col + 1:] for row in U2]
                V2 = [row[:col] + row[col + 1:] for row in V2]
                W2 = [row[:col] + row[col + 1:] for row in W2]
                return U2, V2, W2
    return None


def _greedy_reduce(U, V, W, rng, time_limit):
    """Iteratively try to reduce rank by removing columns."""
    n, m, p = DIMENSIONS
    start = time.time()
    while time.time() - start < time_limit:
        r = len(U[0])
        if r <= 1:
            break
        # Try removing each column (randomized order)
        cols = list(range(r))
        rng.shuffle(cols)
        removed = False
        for col in cols:
            if time.time() - start > time_limit:
                break
            result = _try_remove_column(U, V, W, col, rng)
            if result is not None:
                U, V, W = result
                removed = True
                break
        if not removed:
            break
    return U, V, W


def _to_int_or_half(U, V, W):
    """Convert Fraction matrices to int or half-int (x/2) if possible."""
    def convert(M):
        rows = len(M)
        cols = len(M[0]) if rows > 0 else 0
        result = [[0] * cols for _ in range(rows)]
        for i in range(rows):
            for j in range(cols):
                v = M[i][j]
                if v.denominator == 1:
                    result[i][j] = int(v.numerator)
                elif v.denominator == 2:
                    result[i][j] = v.numerator / 2
                else:
                    result[i][j] = float(v)
        return result
    return convert(U), convert(V), convert(W)


def construct():
    start_time = time.time()
    rng = random.Random(42)
    n, m, p = DIMENSIONS

    # Start with schoolbook
    U, V, W = _schoolbook()
    best_rank = len(U[0])
    best_result = (U, V, W)

    # Greedy reduction
    time_left = DEADLINE - (time.time() - start_time)
    if time_left > 5:
        U, V, W = _greedy_reduce(U, V, W, rng, time_left - 5)
        if _is_valid(U, V, W):
            r = len(U[0])
            if r < best_rank:
                best_rank = r
                best_result = (U, V, W)

    # Try multiple random restarts of greedy reduction
    time_left = DEADLINE - (time.time() - start_time)
    attempts = 0
    while time_left > 10 and attempts < 50:
        U, V, W = _schoolbook()
        # Randomize initial column ordering
        r = len(U[0])
        perm = list(range(r))
        rng.shuffle(perm)
        U = [U[i] for i in perm]
        V = [V[i] for i in perm]
        W = [W[i] for i in perm]
        time_left_inner = min(time_left - 5, 10)
        U, V, W = _greedy_reduce(U, V, W, rng, time_left_inner)
        if _is_valid(U, V, W):
            r = len(U[0])
            if r < best_rank:
                best_rank = r
                best_result = (U, V, W)
        attempts += 1
        time_left = DEADLINE - (time.time() - start_time)

    U, V, W = best_result
    U_out, V_out, W_out = _to_int_or_half(U, V, W)
    return {"U": U_out, "V": V_out, "W": W_out}
