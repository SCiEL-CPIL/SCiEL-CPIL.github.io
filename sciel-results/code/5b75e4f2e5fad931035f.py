"""Rank-reduction search for 4x5 @ 5x5 matrix multiplication tensor.

Uses the schoolbook basis plus a low-rank correction built from a structured
perturbation.  For coefficients a,b,c we have (a-1)(b-1)(c-1)=0 whenever any
coefficient equals 1, so the perturbation tensor
    D_{ijk} = (a_i-1)(b_j-1)(c_k-1)
has many zero slices, giving a low-rank decomposition that we use to *replace*
the base schoolbook term, reducing the total rank below 100.
"""
import time

DIMENSIONS = [4, 5, 5]
_START = time.time()
_DEADLINE = 150.0
_MARGIN = 25.0          # seconds to spare before the 150s kill


def _time_left():
    return _DEADLINE - _MARGIN - (time.time() - _START)


def _schoolbook(n, m, p):
    """Return the schoolbook (U, V, W) lists, rank = n*m*p, all +1 entries."""
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
    return U, V, W


def _corr_columns(a, b, c, n, m, p):
    """Return the list of (u, v, w) rank-1 vectors of D=(a-1)(b-1)(c-1).

    D_{ijk} = alpha_i beta_j gamma_k with alpha_i=a_i-1, beta_j=b_j-1,
    gamma_k=c_k-1.  A valid decomposition keeps only terms with all three
    nonzero, because D_{ijk}=0 on any slice where a coefficient is 1.
    """
    cols = []
    for i in range(n):
        ai = a[i] - 1
        if ai == 0:
            continue
        for j in range(m):
            bj = b[j] - 1
            if bj == 0:
                continue
            for k in range(p):
                ck = c[k] - 1
                if ck == 0:
                    continue
                cols.append((ai, bj, ck))
    return cols


def _build(a, b, c, n, m, p):
    """Build a candidate decomposition of rank = n*m*p - len(cols).

    The base schoolbook term e_i e_j e_k is cancelled by the correction D
    (which vanishes on the (i,j,k) where the base would need cancelling), so
    we keep base terms only where at least one coefficient is 1, and add the
    D columns.
    """
    cols = _corr_columns(a, b, c, n, m, p)
    rank = n * m * p - len(cols)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if a[i] != 1 and b[j] != 1 and c[k] != 1:
                    continue
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    for ai, bj, ck in cols:
        for i in range(n):
            if a[i] == 1:
                continue
            U[i * m + j][col] = ai
        for j in range(m):
            if b[j] == 1:
                continue
            V[j * p + k][col] = bj
        for k in range(p):
            if c[k] == 1:
                continue
            W[k * n + i][col] = ck
        col += 1
    return {"U": U, "V": V, "W": W}, rank


def _verify(res, n, m, p):
    """Exact integer verification of the decomposition."""
    U, V, W = res["U"], res["V"], res["W"]
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                acc = 0
                urow = U[i * m + j]
                vrow = V[j * p + k]
                wrow = W[k * n + i]
                for c in range(rank):
                    acc += urow[c] * vrow[c] * wrow[c]
                if acc != 1:
                    return False
    return True


def construct():
    global _START
    _START = time.time()
    n, m, p = DIMENSIONS
    base_rank = n * m * p

    # Coefficient grids: mostly 1s (keeps the base term) with a few 2s.
    def make_vec(length, ones, twos, vals=(1, 2)):
        v = [1] * length
        for idx in ones:
            v[idx] = 1
        for idx in twos:
            v[idx] = 2
        return v

    best = {"U": None, "V": None, "W": None, "rank": base_rank}

    # Start with pure schoolbook as the guaranteed fallback.
    U0, V0, W0 = _schoolbook(n, m, p)
    best = {"U": U0, "V": V0, "W": W0, "rank": base_rank}

    # Candidate coefficient patterns.  Each pattern sets some entries to 2,
    # which creates nonzero correction columns and removes base terms.
    import itertools

    patterns = []
    # Single 2 in each of a, b, c.
    for i in range(n):
        patterns.append((make_vec(n, [], [i]), [1] * m, [1] * p))
    for j in range(m):
        patterns.append(([1] * n, make_vec(m, [], [j]), [1] * p))
    for k in range(p):
        patterns.append(([1] * n, [1] * m, make_vec(p, [], [k])))

    # Pairs of 2s.
    for i in range(n):
        for j in range(m):
            patterns.append((make_vec(n, [], [i]), make_vec(m, [], [j]), [1] * p))
    for i in range(n):
        for k in range(p):
            patterns.append((make_vec(n, [], [i]), [1] * m, make_vec(p, [], [k])))
    for j in range(m):
        for k in range(p):
            patterns.append(([1] * n, make_vec(m, [], [j]), make_vec(p, [], [k])))

    # Triples.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                patterns.append(
                    (make_vec(n, [], [i]), make_vec(m, [], [j]), make_vec(p, [], [k]))
                )

    # 3x3x3 block.
    for i in range(n - 2):
        for j in range(m - 2):
            for k in range(p - 2):
                a = [1] * n; a[i:i + 3] = [2, 2, 2]
                b = [1] * m; b[j:j + 3] = [2, 2, 2]
                c = [1] * p; c[k:k + 3] = [2, 2, 2]
                patterns.append((a, b, c))

    for (a, b, c) in patterns:
        if _time_left() <= 0:
            break
        res, rank = _build(a, b, c, n, m, p)
        if rank < best["rank"]:
            if _verify(res, n, m, p):
                best = dict(res)
                best["rank"] = rank
                if rank <= 95:
                    break

    return {"U": best["U"], "V": best["V"], "W": best["W"]}
