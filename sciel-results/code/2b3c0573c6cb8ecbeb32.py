"""Packing 50 equal circles: contact-graph relaxation + sparse distance-constraint solver."""
import time
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

N = 50
HALF = 0.5
DEADLINE = time.time() + 135.0   # safety margin before the 150 s kill


def _dist2(P):
    d = P[:, None, :] - P[None, :, :]
    return np.einsum('ijk,ijk->ij', d, d)


def _score(P):
    """Largest common radius (circles disjoint and inside the unit square)."""
    P = np.asarray(P)
    r_min = HALF - np.max(np.abs(P))
    if r_min <= 0:
        return 0.0
    d2 = _dist2(P)
    np.fill_diagonal(d2, np.inf)
    return 0.5 * min(r_min, np.sqrt(d2.min()))


def _hex(m):
    """Centres of an m x m hexagonal lattice, rescaled to the unit square."""
    a = 1.0
    b = a * np.sqrt(3.0)
    pts = []
    for row in range(m):
        for col in range(m):
            x = col * a + (0.5 if row % 2 else 0.0)
            y = row * b
            pts.append((x, y))
    pts = np.array(pts)
    pts = pts - pts.mean(axis=0)
    pts = pts / np.max(np.abs(pts)) * 0.9
    return pts


def _jittered_grid(m, rng):
    ticks = (np.arange(m) + 0.5) / m - 0.5
    g = np.array([(x, y) for y in ticks for x in ticks])[:N]
    return g + 0.05 * rng.standard_normal((N, 2))


def _random(rng):
    return rng.uniform(-0.42, 0.42, (N, 2))


def _lattice(rng):
    pts = _hex(7)
    return pts + 0.03 * rng.standard_normal((N, 2))


def _relax(P, r, rng, iters=250, step=0.02, k=6):
    """Contact-graph relaxation: sparse distance-constraint solve + overlap/boundary push."""
    P = np.array(P, dtype=float)
    for it in range(iters):
        d2 = _dist2(P)
        d = np.sqrt(d2)
        np.fill_diagonal(d, np.inf)
        # sparse neighbour graph
        idx = np.argpartition(d, k, axis=1)[:, :k]
        rows = []
        cols = []
        for i in range(N):
            for j in idx[i]:
                if j != i:
                    rows.append(i)
                    cols.append(int(j))
        rows = np.array(rows)
        cols = np.array(cols)
        d_ij = d[rows, cols]
        d_ij = np.where(d_ij < 1e-9, 1e-9, d_ij)
        u = (P[rows] - P[cols]) / d_ij[:, None]

        # pull overlapping pairs apart, push near contacts to exactly 2r
        f = np.zeros((len(rows), 2))
        overlap = d_ij < 2.0 * r
        near = (d_ij >= 2.0 * r) & (d_ij < 2.2 * r)
        f[overlap] = (2.0 * r - d_ij[overlap])[:, None] * u[overlap]
        f[near] = (2.0 * r - d_ij[near])[:, None] * u[near] * 0.5

        # sparse A^T A solve
        A = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(N, N))
        b = np.zeros((N, 2))
        b[rows] += f
        b[cols] -= f
        M = A.T @ A
        try:
            delta = spsolve(M, b)
        except Exception:
            delta = np.zeros((N, 2))
        P = P + step * delta

        # hard de-overlap (iterative push)
        for _ in range(4):
            d2 = _dist2(P)
            d = np.sqrt(d2)
            np.fill_diagonal(d, np.inf)
            for i in range(N):
                for j in range(i + 1, N):
                    dd = d[i, j]
                    if dd < 2.0 * r - 1e-9:
                        if dd < 1e-9:
                            P[i, 0] += 1e-3
                            P[j, 0] -= 1e-3
                        else:
                            u = (P[i] - P[j]) / dd
                            push = 0.5 * (2.0 * r - dd)
                            P[i] += push * u
                            P[j] -= push * u

        # boundary
        P = np.clip(P, -HALF + r, HALF - r)

        # small random kick to escape local minima
        if it % 20 == 19:
            P += 0.01 * rng.standard_normal((N, 2))
            P = np.clip(P, -HALF + r, HALF - r)

    # final exact de-overlap / boundary cleanup
    for _ in range(60):
        changed = False
        d2 = _dist2(P)
        d = np.sqrt(d2)
        np.fill_diagonal(d, np.inf)
        for i in range(N):
            for j in range(i + 1, N):
                dd = d[i, j]
                if dd < 2.0 * r - 1e-9:
                    changed = True
                    if dd < 1e-9:
                        P[i, 0] += 1e-3
                    else:
                        u = (P[i] - P[j]) / dd
                        push = 0.5 * (2.0 * r - dd)
                        P[i] += push * u
                        P[j] -= push * u
        P = np.clip(P, -HALF + r, HALF - r)
        if not changed:
            break
    return P


def _feasible(P, r, rng):
    """Try to make all circles radius r; return (P, max_overlap)."""
    P = _relax(P, r, rng, iters=180)
    d2 = _dist2(P)
    d = np.sqrt(d2)
    np.fill_diagonal(d, np.inf)
    return P, (2.0 * r - d.min())


def _solve_for_radius(P, r, rng):
    """Bisection on radius around a starting radius; returns (best_r, best_P)."""
    # find an upper bound that is infeasible
    hi = r * 1.05
    while hi < 0.12:
        _, ov = _feasible(P.copy(), hi, rng)
        if ov > 1e-9:
            break
        hi *= 1.15
    lo = r * 0.9
    _, ov = _feasible(P.copy(), lo, rng)
    if ov <= 1e-9:
        lo = r
    best_r = lo
    best_P = P.copy()
    for _ in range(8):
        if time.time() > DEADLINE:
            break
        mid = 0.5 * (lo + hi)
        Pm, ov = _feasible(P.copy(), mid, rng)
        if ov <= 1e-9:
            lo = mid
            best_r = mid
            best_P = Pm
        else:
            hi = mid
    return best_r, best_P


def construct():
    rng = np.random.default_rng(12345)
    seeds = [
        _hex(7),
        _jittered_grid(7, rng),
        _random(rng),
        _lattice(rng),
        _jittered_grid(8, rng),
    ]
    base_r = min(_score(s) for s in seeds)
    best_r = 0.0
    best_P = seeds[0]
    for s in seeds:
        if time.time() > DEADLINE - 20:
            break
        r, P = _solve_for_radius(s.copy(), base_r, rng)
        if r > best_r:
            best_r = r
            best_P = P
    # ensure a valid, in-bounds result
    P = np.clip(np.asarray(best_P, dtype=float), -HALF + best_r, HALF - best_r)
    if not np.all(np.isfinite(P)) or P.shape != (N, 2):
        P = _hex(7)
    return P
