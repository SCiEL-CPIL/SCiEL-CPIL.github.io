"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree

DEADLINE = 116.0  # safety margin before the 150s kill


def _true_score(c):
    """Exact packing radius in double precision."""
    c = np.asarray(c, dtype=float)
    if c.shape != (N, 2) or not np.all(np.isfinite(c)):
        return 0.0
    if np.any(c < -0.5) or np.any(c > 0.5):
        return 0.0
    w = np.min(np.stack([
        0.5 - c[:, 0], c[:, 0] + 0.5,
        0.5 - c[:, 1], c[:, 1] + 0.5,
    ]), axis=0)
    r = float(np.min(w))
    d = np.sqrt(((c[:, None, :] - c[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    r = min(r, float(np.min(d)) * 0.5)
    return r


def _grid_init():
    m = 7
    ticks = (np.arange(m) + 0.5) / m - 0.5
    g = np.array([(x, y) for y in ticks for x in ticks])  # 49 points
    return np.vstack([g, np.array([-2.5 / 7.0, -2.5 / 7.0])])


def _hex_init():
    m = 7
    pts = []
    for i in range(m):
        for j in range(m):
            pts.append(((j + 0.5) / m - 0.5, (i + 0.5) / m - 0.5))
    g = np.array(pts)
    return np.vstack([g, np.array([-2.5 / 7.0, -2.5 / 7.0])])


def _relax(c, r, time_budget=3.0, max_iter=40000, k=6):
    """Force-relax a hard-disk packing at radius r using only the k closest
    neighbours per circle (via a KD-tree) so each step is O(N log N)."""
    c = np.array(c, dtype=float, copy=True)
    c = np.clip(c, -0.5 + r, 0.5 - r)
    start = time.time()
    dt = 0.02
    for _ in range(max_iter):
        if time.time() - start > time_budget:
            break
        # nearest neighbours
        idx = cKDTree(c).query(c, k=k + 1)[1]
        j = idx[:, 1:]                      # (N, k)
        dij = c[np.arange(N)[:, None], :] - c[j]          # (N, k, 2)
        d = np.sqrt((dij ** 2).sum(-1))                   # (N, k)
        d = np.maximum(d, 1e-12)
        overlap = np.maximum(2 * r - d, 0.0)              # (N, k)
        vec = dij / d[..., None]                          # (N, k, 2)
        disp = (overlap[..., None] * vec).sum(axis=1)
        # wall repulsion (linear, strong)
        for axis, sign in ((0, 1), (0, -1), (1, 1), (1, -1)):
            if sign > 0:
                pen = c[:, axis] - (0.5 - r)
            else:
                pen = (-0.5 + r) - c[:, axis]
            pen = np.maximum(pen, 0.0)
            disp[:, axis] += sign * pen / (2 * r)
        c = np.clip(c + dt * disp, -0.5 + r, 0.5 - r)
    return c


def _hill_climb(c, r, time_budget=3.0, max_iter=20000, k=6, step=0.006):
    """Deterministic local search: repeatedly nudge the tightest pair apart
    and accept only if the minimum clearance (min pair dist / 2, wall) rises."""
    c = np.array(c, dtype=float, copy=True)
    c = np.clip(c, -0.5 + r, 0.5 - r)
    start = time.time()
    base = r * 0.999
    for _ in range(max_iter):
        if time.time() - start > time_budget:
            break
        idx = cKDTree(c).query(c, k=k + 1)[1]
        j = idx[:, 1:]
        dij = c[np.arange(N)[:, None], :] - c[j]
        d = np.sqrt((dij ** 2).sum(-1))
        d = np.maximum(d, 1e-12)
        vec = dij / d[..., None]
        i_idx, j_idx = np.unravel_index(np.argmin(d), d.shape)
        v = vec[i_idx, j_idx]
        cand = c.copy()
        cand[i_idx] += step * v
        cand[j_idx] -= step * v
        cand = np.clip(cand, -0.5 + base, 0.5 - base)
        # clearance of candidate
        d2 = np.sqrt(((cand[:, None, :] - cand[None, :, :]) ** 2).sum(-1))
        np.fill_diagonal(d2, np.inf)
        w2 = np.min(np.stack([
            0.5 - cand[:, 0], cand[:, 0] + 0.5,
            0.5 - cand[:, 1], cand[:, 1] + 0.5,
        ]), axis=0)
        sc = min(float(np.min(d2)) * 0.5, float(np.min(w2)))
        if sc > r + 1e-12:
            c = cand
            r = sc
    return c, r


def _grow(c0, time_budget=20.0, relax_budget=3.0, hill_budget=2.5):
    """Grow the radius by repeated relax + hill-climb + binary search."""
    c = np.array(c0, dtype=float, copy=True)
    r = _true_score(c) * 0.99
    r = max(r, 0.05)
    start = time.time()
    improved = True
    while improved and (time.time() - start) < time_budget:
        if time.time() - start > time_budget - 6.0:
            break
        c = _relax(c, r, time_budget=relax_budget)
        c, r2 = _hill_climb(c, r, time_budget=hill_budget)
        r = max(r, r2)
        s = _true_score(c)
        if s >= r - 1e-9:
            r = r * 1.04
            improved = True
        else:
            # binary search the largest feasible radius near current
            lo, hi = s, r
            best_c = c
            for _ in range(5):
                if time.time() - start > time_budget - 4.0:
                    break
                mid = 0.5 * (lo + hi)
                cc = _relax(c, mid, time_budget=relax_budget * 0.6)
                sm = _true_score(cc)
                if sm >= mid - 1e-9:
                    lo, best_c = mid, cc
                else:
                    hi = mid
            r = lo
            c = best_c
            improved = (r > s + 1e-6)
    return c, _true_score(c)


def _softmin_refine(c0, k, time_budget=3.0):
    """Smooth L-BFGS-B polish with a CORRECT soft-min objective and gradient."""
    c0 = np.clip(np.asarray(c0, dtype=float), -0.5, 0.5)

    def obj_and_grad(x):
        pts = x.reshape(-1, 2)
        diff = pts[:, None, :] - pts[None, :, :]
        dists = np.sqrt((diff ** 2).sum(-1))
        np.fill_diagonal(dists, np.inf)
        dw = np.stack([
            0.5 - pts[:, 0], pts[:, 0] + 0.5,
            0.5 - pts[:, 1], pts[:, 1] + 0.5,
        ])
        all_d = np.concatenate([dists.ravel(), dw.ravel()])
        all_d = np.clip(all_d, 1e-8, 1.0)
        weights = np.exp(-k * all_d)
        sw = weights.sum()
        val = -np.log(sw) / k
        # d(-log sw)/d d_m = - w_m / sw
        g = -weights / sw
        grad_pts = np.zeros_like(pts)
        wp = g[:N * N].reshape(N, N)
        mask = ~np.eye(N, dtype=bool)
        safe = np.where(mask, dists, 1.0)
        factors = np.where(mask, wp / safe, 0.0)     # d(-log sw)/d pts_i (pair part)
        wpts = (factors[..., None] * pts[None, :, :]).sum(axis=1)
        s = factors.sum(axis=1)
        grad_pts += s[:, None] * pts - wpts
        ww = g[N * N:].reshape(4, N)
        grad_pts[:, 0] += -ww[0] + ww[1]
        grad_pts[:, 1] += -ww[2] + ww[3]
        return val, -grad_pts.ravel()

    try:
        res = minimize(obj_and_grad, c0.ravel(), jac=True, method='L-BFGS-B',
                       bounds=[(-0.5, 0.5)] * (2 * N),
                       options={'maxiter': 120, 'maxfun': 2500,
                                'ftol': 1e-13, 'gtol': 1e-9, 'iprint': 0})
        c = np.clip(res.x.reshape(N, 2), -0.5, 0.5)
    except Exception:
        c = c0
    return c


def construct():
    start_time = time.time()

    inits = [_grid_init(), _hex_init()]
    rng = np.random.default_rng(12345)
    for _ in range(6):
        inits.append(rng.uniform(-0.42, 0.42, (N, 2)))

    # 1) Grow the radius from several starts; keep the best feasible layout.
    best = inits[0]
    best_r = _true_score(best)
    for c0 in inits:
        if time.time() - start_time > DEADLINE - 18:
            break
        c, s = _grow(c0, time_budget=16.0)
        if s > best_r:
            best_r, best = s, c

    # 2) Soft-min polish from the best layout (and a couple of relaxations).
    candidates = [best, _relax(best, best_r, time_budget=1.5)]
    for k in (60.0, 120.0, 200.0):
        for c0 in candidates:
            if time.time() - start_time > DEADLINE - 6:
                break
            c = _softmin_refine(c0, k, time_budget=2.5)
            s = _true_score(c)
            if s > best_r:
                best_r, best = s, c

    # 3) A final short grow from the polished layout to squeeze more radius.
    if time.time() - start_time < DEADLINE - 4:
        c, s = _grow(best, time_budget=6.0, relax_budget=1.5, hill_budget=1.5)
        if s > best_r:
            best_r, best = s, c

    return best
# EVOLVE-BLOCK-END
