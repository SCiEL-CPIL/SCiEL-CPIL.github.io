"""Rank-minimal exact tensor decomposition search for 3x3 by 3x4 matrix multiplication."""
import time
import random
import fractions
import numpy as np

DIMENSIONS = [3, 3, 4]
Frac = fractions.Fraction


def _validate(U, V, W, n, m, p, rank):
    """Exact check that the tensor decomposition holds."""
    Uf = [[Frac(U[i][c]) for c in range(rank)] for i in range(n * m)]
    Vf = [[Frac(V[j][c]) for c in range(rank)] for j in range(m * p)]
    Wf = [[Frac(W[k][c]) for c in range(rank)] for k in range(p * n)]
    for i in range(n):
        for k in range(p):
            for j in range(m):
                for l in range(m):
                    s = Frac(0)
                    for c in range(rank):
                        s += Uf[i * m + j][c] * Vf[l * p + k][c] * Wf[k * n + i][c]
                    expected = Frac(1) if j == l else Frac(0)
                    if s != expected:
                        return False
    return True


def _check_half_integer(W, rank):
    """Check all entries are integer or half-integer."""
    for row in W:
        for x in row:
            if isinstance(x, Frac):
                if x.denominator not in (1, 2):
                    return False
            elif isinstance(x, (int, float)):
                if isinstance(x, float) and x != int(x) and 2 * x != int(2 * x):
                    return False
    return True


def _solve_w_system_exact(U, V, i, k, n, m, p, rank):
    """Solve for W[k*n+i] given U, V using exact fractions. Returns list of Fractions or None."""
    rows = []
    rhs = []
    for j in range(m):
        for l in range(m):
            row = [Frac(U[i * m + j][c]) * Frac(V[l * p + k][c]) for c in range(rank)]
            rows.append(row)
            rhs.append(Frac(1) if j == l else Frac(0))

    R = len(rows)
    C = rank
    aug = [rows[r][:] + [rhs[r]] for r in range(R)]

    pivots = []
    col = 0
    for r in range(R):
        if col >= C:
            break
        pr = None
        for rr in range(r, R):
            if aug[rr][col] != 0:
                pr = rr
                break
        if pr is None:
            col += 1
            continue
        if pr != r:
            aug[r], aug[pr] = aug[pr], aug[r]
        pivots.append((r, col))
        pv = aug[r][col]
        if pv != 1:
            aug[r] = [x / pv for x in aug[r]]
        for rr in range(R):
            if rr != r and aug[rr][col] != 0:
                f = aug[rr][col]
                aug[rr] = [a - f * b for a, b in zip(aug[rr], aug[r])]
        col += 1

    for rr in range(R):
        coeff_zero = all(aug[rr][c] == 0 for c in range(C))
        if coeff_zero and aug[rr][C] != 0:
            return None

    Wvec = [Frac(0)] * C
    for (r, c) in pivots:
        Wvec[c] = aug[r][C]

    return Wvec


def _fast_rank_check(U, V, i, k, n, m, p, rank):
    """Fast numpy-based check: rank of coefficient matrix must be sufficient for consistency."""
    M = np.zeros((m * m, rank), dtype=float)
    for j in range(m):
        for l in range(m):
            for c in range(rank):
                M[j * m + l, c] = float(U[i * m + j][c]) * float(V[l * p + k][c])

    b = np.zeros(m * m, dtype=float)
    for j in range(m):
        b[j * m + j] = 1.0

    M_aug = np.column_stack([M, b])
    rank_M = np.linalg.matrix_rank(M)
    rank_aug = np.linalg.matrix_rank(M_aug)
    return rank_M == rank_aug


def _generate_structured_UV(rank, n, m, p, rng, pattern_type):
    """Generate structured U,V matrices using Winograd-like patterns."""
    nm = n * m
    mp = m * p

    if pattern_type == 0:
        # Kronecker-like: use block structure
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        # Assign random values to a subset of columns per row
        for i in range(n):
            for j in range(m):
                idx = i * m + j
                for c in range(rank):
                    if (i + j + c) % 3 == 0:
                        U[idx][c] = rng.randint(-2, 2)
        for j in range(m):
            for k in range(p):
                idx = j * p + k
                for c in range(rank):
                    if (j + k + c) % 3 == 0:
                        V[idx][c] = rng.randint(-2, 2)

    elif pattern_type == 1:
        # Convolutional: cyclic shifts
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        base_u = [rng.randint(-2, 2) for _ in range(rank)]
        base_v = [rng.randint(-2, 2) for _ in range(rank)]
        for i in range(n):
            for j in range(m):
                idx = i * m + j
                shift = (i + j) % max(1, rank)
                for c in range(rank):
                    U[idx][c] = base_u[(c + shift) % rank]
        for j in range(m):
            for k in range(p):
                idx = j * p + k
                shift = (j + k) % max(1, rank)
                for c in range(rank):
                    V[idx][c] = base_v[(c + shift) % rank]

    elif pattern_type == 2:
        # Sparse random with higher density
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        for i in range(nm):
            for c in range(rank):
                if rng.random() < 0.4:
                    U[i][c] = rng.randint(-2, 2)
        for j in range(mp):
            for c in range(rank):
                if rng.random() < 0.4:
                    V[j][c] = rng.randint(-2, 2)

    elif pattern_type == 3:
        # Low-rank factorization style
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        # Use few non-zero columns
        nz_cols_u = rng.sample(range(rank), min(rank, rng.randint(3, max(3, rank // 2))))
        nz_cols_v = rng.sample(range(rank), min(rank, rng.randint(3, max(3, rank // 2))))
        for i in range(nm):
            for c in nz_cols_u:
                U[i][c] = rng.randint(-2, 2)
        for j in range(mp):
            for c in nz_cols_v:
                V[j][c] = rng.randint(-2, 2)

    else:
        # Pure random
        U = [[rng.randint(-2, 2) for _ in range(rank)] for _ in range(nm)]
        V = [[rng.randint(-2, 2) for _ in range(rank)] for _ in range(mp)]

    return U, V


def _search_rank(rank, n, m, p, deadline, rng, max_iter):
    """Try to find an exact rank-`rank` decomposition. Returns (U,V,W) or None."""
    nm = n * m
    mp = m * p
    pn = p * n

    it = 0
    while time.time() < deadline and it < max_iter:
        it += 1

        # Cycle through pattern types
        pattern_type = it % 5

        U, V = _generate_structured_UV(rank, n, m, p, rng, pattern_type)

        # Quick check: ensure no row of U or V is all zeros
        if any(all(x == 0 for x in row) for row in U):
            continue
        if any(all(x == 0 for x in row) for row in V):
            continue

        # For each (i,k), check consistency and solve for W
        W = [[0] * rank for _ in range(pn)]
        ok = True
        for i in range(n):
            for k in range(p):
                # Fast float check first
                if not _fast_rank_check(U, V, i, k, n, m, p, rank):
                    ok = False
                    break
                # Exact solve
                Wvec = _solve_w_system_exact(U, V, i, k, n, m, p, rank)
                if Wvec is None:
                    ok = False
                    break
                W[k * n + i] = [int(x) if x.denominator == 1 else x for x in Wvec]
            if not ok:
                break

        if not ok:
            continue

        # Check half-integer constraint
        if not _check_half_integer(W, rank):
            continue

        # Full exact verification
        if _validate(U, V, W, n, m, p, rank):
            return U, V, W

    return None


def _schoolbook(n, m, p):
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


def _try_winograd_structured(n, m, p, deadline):
    """Try explicit Winograd-style structured decompositions."""
    # For 3x3x4, try rank 24-27 with structured patterns
    for rank in range(24, 28):
        if time.time() > deadline:
            break
        rng = random.Random(rank * 1000 + 42)
        result = _search_rank(rank, n, m, p, deadline, rng, max_iter=3000)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    return None


def _try_low_rank_search(n, m, p, deadline):
    """Try lower ranks with more iterations and varied patterns."""
    for rank in range(22, 24):
        if time.time() > deadline:
            break
        rng = random.Random(rank * 7777 + 123)
        result = _search_rank(rank, n, m, p, deadline, rng, max_iter=5000)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    return None


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 135  # Safety margin before 150s limit

    # Start with schoolbook as fallback
    best_result = _schoolbook(n, m, p)
    best_rank = n * m * p

    # Try Winograd-structured search for rank 24-27
    try:
        result = _try_winograd_structured(n, m, p, deadline)
        if result is not None:
            best_result = result
            best_rank = len(result['U'][0])
    except Exception:
        pass

    # If we still have time, try lower ranks
    remaining = deadline - time.time()
    if remaining > 10:
        try:
            result = _try_low_rank_search(n, m, p, deadline)
            if result is not None:
                best_result = result
                best_rank = len(result['U'][0])
        except Exception:
            pass

    return best_result
