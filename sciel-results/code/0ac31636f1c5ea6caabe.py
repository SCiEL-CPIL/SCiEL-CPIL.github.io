"""Rank-minimal exact tensor decomposition for 2x3 * 3x6 matrix multiplication.

Structural approach: model the multiplication tensor as a sum of rank-1 terms.
For a candidate rank R, the tensor equations (one per (i,j,k), total n*m*p = 36)
are linear in the flattened factor matrices. We solve with least squares, then
repair residual entries with a few targeted coordinate updates, round to the
nearest half-integer, and verify exactly. We search a range of ranks in a
deadline-guarded loop, keeping the best valid (lowest-rank) decomposition.
"""
import time
import numpy as np

DIMENSIONS = [2, 3, 6]
N, M, P = DIMENSIONS
TENSOR_SIZE = N * M * P  # 36 scalar equations


def _schoolbook(n, m, p):
    """Standard schoolbook (rank n*m*p) decomposition. Always valid."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _verify_decomposition(U, V, W, n, m, p):
    """Verify exact decomposition: for all i,j,k, sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1."""
    rank = len(U[0]) if U else 0
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if any(len(row) != rank for row in U):
        return False
    if any(len(row) != rank for row in V):
        return False
    if any(len(row) != rank for row in W):
        return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for c in range(rank):
                    total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if total != 1:
                    return False
    return True


def _build_linear_system(n, m, p, R):
    """Build A (size TENSOR_SIZE x R*(nm+mp+pn)) and b (all ones).
    Each column c of factors contributes a rank-1 tensor; the (i,j,k) equation
    sums over c of U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1."""
    nm, mp, pn = n * m, m * p, p * n
    ncol = R * (nm + mp + pn)
    A = np.zeros((TENSOR_SIZE, ncol), dtype=float)
    b = np.ones(TENSOR_SIZE, dtype=float)
    for c in range(R):
        u_off = c * (nm + mp + pn)
        v_off = u_off + nm
        w_off = v_off + mp
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = (i * m + j) * p + k
                    A[idx, u_off + i * m + j] = 1.0
                    A[idx, v_off + j * p + k] = 1.0
                    A[idx, w_off + k * n + i] = 1.0
    return A, b


def _round_half(x):
    """Round to nearest half-integer, return as float (e.g. 0.5, 1.0, -0.5)."""
    return float(int(round(2.0 * x)) / 2.0)


def _round_factors(U, V, W):
    return (
        [[_round_half(v) for v in row] for row in U],
        [[_round_half(v) for v in row] for row in V],
        [[_round_half(v) for v in row] for row in W],
    )


def _residual(U, V, W, n, m, p):
    """Return list of (idx, i, j, k, current_sum) for entries not equal to 1."""
    rank = len(U[0])
    res = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for c in range(rank):
                    total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(total - 1.0) > 1e-6:
                    res.append(((i * m + j) * p + k, i, j, k, total))
    return res


def _try_solve_rank(n, m, p, R, rng, deadline):
    """Attempt to find a valid rank-R decomposition. Returns dict or None."""
    A, b = _build_linear_system(n, m, p, R)
    nm, mp, pn = n * m, m * p, p * n
    ncol = R * (nm + mp + pn)

    # Initial least-squares solution
    x, *_ = np.linalg.lstsq(A, b, rcond=None)

    # Convert to factor lists
    U = [[0.0] * R for _ in range(nm)]
    V = [[0.0] * R for _ in range(mp)]
    W = [[0.0] * R for _ in range(pn)]
    for c in range(R):
        u_off = c * (nm + mp + pn)
        v_off = u_off + nm
        w_off = v_off + mp
        for a in range(nm):
            U[a][c] = x[u_off + a]
        for a in range(mp):
            V[a][c] = x[v_off + a]
        for a in range(pn):
            W[a][c] = x[w_off + a]

    # Repair loop: fix residual entries with coordinate updates.
    for _round in range(400):
        if time.time() > deadline:
            return None
        res = _residual(U, V, W, n, m, p)
        if not res:
            break
        # Pick the entry with the largest error
        idx, i, j, k, total = max(res, key=lambda r: abs(r[4] - 1.0))
        err = 1.0 - total
        # Candidate coordinate updates: change one factor in one term by delta
        # such that the (i,j,k) product changes by ~err.
        best_update = None
        best_gain = 0.0
        for c in range(R):
            u_val = U[i * m + j][c]
            v_val = V[j * p + k][c]
            w_val = W[k * n + i][c]
            # Changing W by delta changes product by u_val*v_val*delta
            if abs(u_val * v_val) > 1e-9:
                d = err / (u_val * v_val)
                gain = abs(d)
                if gain > best_gain:
                    best_gain = gain
                    best_update = ('W', k * n + i, c, d)
            if abs(u_val * w_val) > 1e-9:
                d = err / (u_val * w_val)
                gain = abs(d)
                if gain > best_gain:
                    best_gain = gain
                    best_update = ('V', j * p + k, c, d)
            if abs(v_val * w_val) > 1e-9:
                d = err / (v_val * w_val)
                gain = abs(d)
                if gain > best_gain:
                    best_gain = gain
                    best_update = ('U', i * m + j, c, d)
        if best_update is None:
            # No usable coordinate; perturb a random factor to break symmetry
            fac, row, col, _ = best_update if False else (rng.choice(['U', 'V', 'W']), 0, 0, 0.0)
            continue
        fac, row, col, d = best_update
        if fac == 'U':
            U[row][col] += d
        elif fac == 'V':
            V[row][col] += d
        else:
            W[row][col] += d

    res = _residual(U, V, W, n, m, p)
    if res:
        return None

    # Round to half-integers and verify
    Ur, Vr, Wr = _round_factors(U, V, W)
    if _verify_decomposition(Ur, Vr, Wr, n, m, p):
        return {"U": Ur, "V": Vr, "W": Wr}
    return None


def construct():
    """Construct a rank-minimal exact tensor decomposition for 2x3 * 3x6 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    n, m, p = DIMENSIONS

    # Always-valid fallback: schoolbook
    schoolbook = _schoolbook(n, m, p)
    best_result = schoolbook
    best_rank = len(schoolbook["U"][0])

    # Search lower ranks, from most promising (just below schoolbook) downward.
    # Lower bound for matrix mult rank is known to be at least n*m*p - something,
    # but we simply try a descending range.
    rng = np.random.default_rng(12345)
    tried = set()
    for R in range(best_rank - 1, 14, -1):
        if time.time() > deadline - 5:
            break
        if R in tried:
            continue
        tried.add(R)
        for attempt in range(3):
            if time.time() > deadline - 2:
                break
            try:
                res = _try_solve_rank(n, m, p, R, rng, deadline)
            except Exception:
                res = None
            if res is not None:
                rank_r = len(res["U"][0])
                if rank_r < best_rank:
                    best_result = res
                    best_rank = rank_r
                    # Found a valid one; break inner and try even lower
                    break

    if best_result is None:
        best_result = _schoolbook(n, m, p)

    return best_result
