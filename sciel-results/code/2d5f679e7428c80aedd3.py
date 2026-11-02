"""Circle packing in perimeter-4 rectangle: 21 circles, maximize sum of radii.

Design family: smooth log-sum-exp surrogate over pairwise overlaps + wall
penalties, gradient descent with a Delaunay-reduced active-constraint set,
multi-start, and a final high-precision polish.  A deadline guard returns the
best valid result found before the wall-clock limit.
"""
import time
import numpy as np
from scipy.spatial import Delaunay
from scipy.optimize import minimize

N = 21
K = 21  # number of circles actually used (N==K here)
MAX_TIME = 150.0
DEADLINE = MAX_TIME - 12.0  # safety margin before the 150s kill


def _logsumexp(v, beta):
    """Stable log-sum-exp of v with scaling beta (returns beta*log(sum exp))."""
    if v.size == 0:
        return 0.0
    m = v.max()
    if not np.isfinite(m):
        return 0.0
    return beta * (m + np.log(np.sum(np.exp(v - m))))


def _penalty_terms(p, r, s, beta):
    """Return (total_penalty, list_of_active_pair_indices).

    p: (N,2) centers, r: (N,) radii, s: aspect w/h (>0).
    Rectangle: w = 2*s/sqrt(1+s^2), h = 2/sqrt(1+s^2).
    """
    w = 2.0 * s / np.sqrt(1.0 + s * s)
    h = 2.0 / np.sqrt(1.0 + s * s)

    # pairwise overlaps
    d = p[:, None, :] - p[None, :, :]
    dist = np.sqrt(np.sum(d * d, axis=2))
    dist = np.where(np.eye(N) > 0, np.inf, dist)
    overlap = r[:, None] + r[None, :] - dist
    overlap = overlap[~np.eye(N, dtype=bool)]

    # wall violations (center must be at least r inside each wall)
    wall = np.concatenate([
        r - p[:, 0],
        r - (w - p[:, 0]),
        r - p[:, 1],
        r - (h - p[:, 1]),
    ])

    pen = _logsumexp(overlap, beta) + _logsumexp(wall, beta)

    # active (near-violating) pair indices for the Delaunay reduction
    idx = np.argwhere(~np.eye(N, dtype=bool))
    active = idx[overlap[idx[:, 0], idx[:, 1]] > -1e-4]
    return pen, active


def _objective(x, beta):
    p = x[:2 * N].reshape(N, 2)
    r = x[2 * N:3 * N]
    s = x[3 * N]
    w = 2.0 * s / np.sqrt(1.0 + s * s)
    h = 2.0 / np.sqrt(1.0 + s * s)

    # objective: maximize sum r, subtract smooth penalties
    d = p[:, None, :] - p[None, :, :]
    dist = np.sqrt(np.sum(d * d, axis=2))
    dist = np.where(np.eye(N) > 0, np.inf, dist)
    overlap = r[:, None] + r[None, :] - dist
    overlap = overlap[~np.eye(N, dtype=bool)]
    wall = np.concatenate([
        r - p[:, 0],
        r - (w - p[:, 0]),
        r - p[:, 1],
        r - (h - p[:, 1]),
    ])
    pen = _logsumexp(overlap, beta) + _logsumexp(wall, beta)
    return -(r.sum()) + pen


def _check_valid(p, r, w, h, eps=1e-9):
    """Return (is_valid, sum_r)."""
    if np.any(r < 0):
        return False, 0.0
    if np.any(p[:, 0] < -eps) or np.any(p[:, 1] < -eps):
        return False, 0.0
    if np.any(p[:, 0] > w + eps) or np.any(p[:, 1] > h + eps):
        return False, 0.0
    d = p[:, None, :] - p[None, :, :]
    dist = np.sqrt(np.sum(d * d, axis=2))
    np.fill_diagonal(dist, np.inf)
    if np.any(dist < r[:, None] + r[None, :] - eps):
        return False, 0.0
    return True, float(r.sum())


def _clamp_inside(p, r, w, h):
    """Push centers inside walls and nudge overlapping circles apart."""
    p = p.copy()
    for _ in range(30):
        moved = False
        # wall clamp
        p[:, 0] = np.clip(p[:, 0], r, w - r)
        p[:, 1] = np.clip(p[:, 1], r, h - r)
        # pairwise separation
        for i in range(N):
            for j in range(i + 1, N):
                d = p[j] - p[i]
                dist = np.linalg.norm(d)
                need = r[i] + r[j]
                if dist < need - 1e-12:
                    if dist < 1e-12:
                        d = np.array([1.0, 0.0])
                        dist = 1.0
                    d = d / dist
                    push = (need - dist) / 2.0
                    p[i] = p[i] - push * d
                    p[j] = p[j] + push * d
                    moved = True
        if not moved:
            break
    return p


def _feasible_start(rng, s):
    """Build a feasible (non-overlapping, inside) starting configuration."""
    w = 2.0 * s / np.sqrt(1.0 + s * s)
    h = 2.0 / np.sqrt(1.0 + s * s)
    # grid placement
    side = int(np.ceil(np.sqrt(N)))
    gx = np.array([(i + 0.5) / side for i in range(side)])
    gy = np.array([(j + 0.5) / side for j in range(side)])
    base_r = 0.5 / side
    p = []
    r = []
    for i in range(side):
        for j in range(side):
            if len(p) >= N:
                break
            p.append([gx[i] * w, gy[j] * h])
            r.append(base_r * min(w, h) / 1.0)
    p = np.array(p)
    r = np.array(r)
    # scale radii to fit grid, then clamp
    r = r * 0.98
    p = _clamp_inside(p, r, w, h)
    return p, r, w, h


def _delaunay_active(p, r, beta, w, h):
    """Return set of pair indices that are (or may become) active."""
    active = set()
    try:
        tri = Delaunay(p)
        for simplex in tri.simplices:
            for a in range(len(simplex)):
                for b in range(a + 1, len(simplex)):
                    i, j = simplex[a], simplex[b]
                    dist = np.linalg.norm(p[j] - p[i])
                    if dist < r[i] + r[j] + 0.5 / beta:
                        active.add((min(i, j), max(i, j)))
    except Exception:
        pass
    # also include any currently-overlapping pair
    for i in range(N):
        for j in range(i + 1, N):
            dist = np.linalg.norm(p[j] - p[i])
            if dist < r[i] + r[j]:
                active.add((i, j))
    return active


def _soft_optimize(p0, r0, s0, beta, deadline):
    """Gradient descent on the smooth surrogate with Delaunay-reduced checks."""
    p = p0.copy()
    r = r0.copy()
    s = float(s0)
    w = 2.0 * s / np.sqrt(1.0 + s * s)
    h = 2.0 / np.sqrt(1.0 + s * s)

    step = 1e-3
    best_sum = None
    best_state = (p.copy(), r.copy(), s)

    for it in range(4000):
        if time.time() > deadline:
            break

        # active set
        active = _delaunay_active(p, r, beta, w, h)

        # build smooth penalty + gradient over active pairs + walls
        pen = 0.0
        gp = np.zeros_like(p)
        gr = np.zeros_like(r)
        gs = 0.0

        for (i, j) in active:
            d = p[j] - p[i]
            dist = max(np.linalg.norm(d), 1e-12)
            ov = r[i] + r[j] - dist
            if ov > -5.0 / beta:
                wgt = np.exp(beta * max(ov, -5.0 / beta))  # capped for stability
                dirv = d / dist
                pen += wgt
                gp[i] += wgt * dirv
                gp[j] -= wgt * dirv
                gr[i] += wgt
                gr[j] += wgt

        # walls
        def wall_pen(term):
            if term > -5.0 / beta:
                return np.exp(beta * max(term, -5.0 / beta))
            return 0.0

        for i in range(N):
            wgt = wall_pen(r[i] - p[i, 0])
            pen += wgt
            gp[i, 0] -= wgt
            gr[i] += wgt
            wgt = wall_pen(r[i] - (w - p[i, 0]))
            pen += wgt
            gp[i, 0] += wgt
            gr[i] += wgt
            # aspect derivative for w
            dwds = 2.0 / (s * np.sqrt(1.0 + s * s))
            gs += wgt * dwds
            wgt = wall_pen(r[i] - p[i, 1])
            pen += wgt
            gp[i, 1] -= wgt
            gr[i] += wgt
            wgt = wall_pen(r[i] - (h - p[i, 1]))
            pen += wgt
            gp[i, 1] += wgt
            gr[i] += wgt
            dhds = -2.0 * s / (np.sqrt(1.0 + s * s) ** 3)
            gs += wgt * dhds

        # objective grad: d(-sum r + pen)/dx
        obj = -(r.sum()) + pen
        g_p = gp
        g_r = -np.ones(N) + gr
        g_s = gs

        # normalize gradient for stable step
        gnorm = np.sqrt(np.sum(g_p ** 2) + np.sum(g_r ** 2) + g_s ** 2)
        if gnorm < 1e-12:
            break

        # adaptive step with backtracking
        for _ in range(20):
            p_new = p - step * g_p
            r_new = np.clip(r - step * g_r, 0.0, None)
            s_new = max(s - step * g_s, 0.2)
            w_new = 2.0 * s_new / np.sqrt(1.0 + s_new * s_new)
            h_new = 2.0 / np.sqrt(1.0 + s_new * s_new)
            pen_new, _ = _penalty_terms(p_new, r_new, s_new, beta)
            obj_new = -(r_new.sum()) + pen_new
            if obj_new < obj + 1e-12:
                p, r, s = p_new, r_new, s_new
                w, h = w_new, h_new
                break
            step *= 0.5
        else:
            break

        # track best feasible
        p_chk = _clamp_inside(p, r, w, h)
        ok, ssum = _check_valid(p_chk, r, w, h)
        if ok:
            if best_sum is None or ssum > best_sum:
                best_sum = ssum
                best_state = (p_chk.copy(), r.copy(), s)

    return best_state


def _polish(p, r, s, deadline):
    """Final high-precision polish: tight local solve via L-BFGS-B on surrogate."""
    w = 2.0 * s / np.sqrt(1.0 + s * s)
    h = 2.0 / np.sqrt(1.0 + s * s)
    x0 = np.concatenate([p.ravel(), r, [s]])
    beta = 800.0

    def f(x):
        pp = x[:2 * N].reshape(N, 2)
        rr = x[2 * N:3 * N]
        ss = x[3 * N]
        return _objective(x, beta)

    def jac(x):
        pp = x[:2 * N].reshape(N, 2)
        rr = x[2 * N:3 * N]
        ss = x[3 * N]
        w_ = 2.0 * ss / np.sqrt(1.0 + ss * ss)
        h_ = 2.0 / np.sqrt(1.0 + ss * ss)
        gp = np.zeros_like(pp)
        gr = np.zeros_like(rr)
        gs = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                d = pp[j] - pp[i]
                dist = max(np.linalg.norm(d), 1e-12)
                ov = rr[i] + rr[j] - dist
                if ov > -5.0 / beta:
                    wgt = np.exp(beta * max(ov, -5.0 / beta))
                    dirv = d / dist
                    gp[i] += wgt * dirv
                    gp[j] -= wgt * dirv
                    gr[i] += wgt
                    gr[j] += wgt
        for i in range(N):
            for term, comp in [(rr[i] - pp[i, 0], 0), (rr[i] - (w_ - pp[i, 0]), 0)]:
                if term > -5.0 / beta:
                    wgt = np.exp(beta * max(term, -5.0 / beta))
                    gp[i, comp] += wgt * (1 if comp == 0 else 0)
                    gr[i] += wgt
                    if comp == 0:
                        gs += wgt * (2.0 / (ss * np.sqrt(1.0 + ss * ss)))
            for term in (rr[i] - pp[i, 1], rr[i] - (h_ - pp[i, 1])):
                if term > -5.0 / beta:
                    wgt = np.exp(beta * max(term, -5.0 / beta))
                    gp[i, 1] += wgt
                    gr[i] += wgt
                    gs += wgt * (-2.0 * ss / (np.sqrt(1.0 + ss * ss) ** 3))
        # note: wall gradient signs handled approximately; rely on objective
        g = np.concatenate([gp.ravel(), -np.ones(N) + gr, [gs]])
        return g

    try:
        res = minimize(f, x0, jac=jac, method='L-BFGS-B',
                       options={'maxiter': 200, 'maxfun': 2000, 'gtol': 1e-8})
        if res.success or res.fun < f(x0):
            x = res.x
            pp = x[:2 * N].reshape(N, 2)
            rr = np.clip(x[2 * N:3 * N], 0.0, None)
            ss = max(x[3 * N], 0.2)
            w_ = 2.0 * ss / np.sqrt(1.0 + ss * ss)
            h_ = 2.0 / np.sqrt(1.0 + ss * ss)
            pp = _clamp_inside(pp, rr, w_, h_)
            ok, ssum = _check_valid(pp, rr, w_, h_)
            if ok:
                return pp, rr, ss
    except Exception:
        pass
    return p, r, s


def construct():
    t0 = time.time()
    deadline = t0 + (DEADLINE - t0) if time.time() > t0 else t0 + (MAX_TIME - 12.0)
    deadline = min(deadline, time.time() + (MAX_TIME - 12.0))

    rng = np.random.default_rng(12345)
    best = None  # (sum_r, p, r, s)

    # candidate aspect ratios to explore
    aspects = [1.0, 1.2, 1.4, 1.6, 2.0, 0.8, 0.6]

    for s0 in aspects:
        if time.time() > deadline:
            break
        w = 2.0 * s0 / np.sqrt(1.0 + s0 * s0)
        h = 2.0 / np.sqrt(1.0 + s0 * s0)
        # a few random perturbations of the grid start
        for trial in range(2):
            if time.time() > deadline:
                break
            p, r, w, h = _feasible_start(rng, s0)
            # small random perturbation for diversity
            p = p + rng.normal(0, 0.01, p.shape)
            p = _clamp_inside(p, r, w, h)
            state = _soft_optimize(p, r, s0, beta=200.0, deadline=deadline)
            if state is None:
                continue
            p, r, s = state
            w = 2.0 * s / np.sqrt(1.0 + s * s)
            h = 2.0 / np.sqrt(1.0 + s * s)
            p = _clamp_inside(p, r, w, h)
            ok, ssum = _check_valid(p, r, w, h)
            if ok and (best is None or ssum > best[0]):
                best = (ssum, p.copy(), r.copy(), s)

    if best is None:
        # fallback: seed grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    ssum, p, r, s = best

    # final high-precision polish if time remains
    if time.time() < deadline - 5.0:
        p, r, s = _polish(p, r, s, deadline)
        w = 2.0 * s / np.sqrt(1.0 + s * s)
        h = 2.0 / np.sqrt(1.0 + s * s)
        p = _clamp_inside(p, r, w, h)
        ok, ssum2 = _check_valid(p, r, w, h)
        if ok and ssum2 > ssum:
            ssum, p, r, s = ssum2, p, r, s

    # build output: (x, y, radius) for each circle
    out = np.zeros((N, 3))
    out[:, 0] = p[:, 0]
    out[:, 1] = p[:, 1]
    out[:, 2] = r
    return out
