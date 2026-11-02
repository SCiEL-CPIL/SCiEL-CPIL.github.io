"""Circle packing in unit square n=32: DE over radii + center feasibility + L-BFGS polish."""
import time
import numpy as np
from scipy.optimize import minimize

N = 32
EPS = 1e-9
EPSR = 1e-4

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

_N = 32
_EPS = 1e-9
_EPSR = 1e-4
_DEADLINE = 150.0 - 8.0  # safety margin before the 150s kill


def _sum_r(r):
    return float(np.sum(r))


def _feasible(x, y, r):
    """Fast feasibility: is there a placement of centers (fixed radii) with disjoint,
    in-bound circles? Uses a light gradient ascent on the separation objective."""
    x = np.asarray(x, dtype=float).copy()
    y = np.asarray(y, dtype=float).copy()
    r = np.asarray(r, dtype=float)
    n = len(r)

    def sep():
        dx = np.subtract.outer(x, x)
        dy = np.subtract.outer(y, y)
        d = np.sqrt(dx * dx + dy * dy)
        np.fill_diagonal(d, 1e18)
        m = d - r[:, None] - r[None, :]
        return float(np.sum(np.minimum(m, 0.0) ** 2))

    s0 = sep()
    if s0 < 1e-12:
        return True
    for _ in range(300):
        dx = np.subtract.outer(x, x)
        dy = np.subtract.outer(y, y)
        d = np.sqrt(dx * dx + dy * dy)
        np.fill_diagonal(d, 1e-9)
        m = d - r[:, None] - r[None, :]
        neg = np.minimum(m, 0.0)
        w = neg / d
        gx = -np.sum(w, axis=1)
        gy = -np.sum(w, axis=1)
        # boundary
        gx += np.where(x < r, 1.0, 0.0) - np.where(x > 1.0 - r, 1.0, 0.0)
        gy += np.where(y < r, 1.0, 0.0) - np.where(y > 1.0 - r, 1.0, 0.0)
        x = np.clip(x + 0.15 * gx, 0.0, 1.0)
        y = np.clip(y + 0.15 * gy, 0.0, 1.0)
        if sep() < 1e-12:
            return True
    return False


def _de_step(pops, r, best_r, best_val, deadline):
    """One differential-evolution step over radii. Returns (pops, best_r, best_val, spent)."""
    n = len(r)
    F = 0.8
    CR = 0.9
    t0 = time.time()
    new_pops = np.empty_like(pops)
    for i in range(len(pops)):
        a, b, c = np.random.choice(len(pops), 3, replace=False)
        v = pops[a] + F * (pops[b] - pops[c])
        for j in range(n):
            if np.random.rand() < CR:
                r_i = v[j]
            else:
                r_i = pops[i][j]
            r_i = min(max(r_i, 1e-3), 0.5)
            v[j] = r_i
        # feasibility of this radius set
        xs = np.clip(np.random.rand(n), 0.0, 1.0)
        ys = np.clip(np.random.rand(n), 0.0, 1.0)
        if _feasible(xs, ys, v):
            val = _sum_r(v)
            if val > best_val:
                best_r = v.copy()
                best_val = val
            new_pops[i] = v
        else:
            new_pops[i] = pops[i]
    if best_val > _sum_r(pops[i]):
        pass
    # keep best in population
    best_val_pop = max(_sum_r(p) for p in pops)
    if best_val > best_val_pop:
        worst = int(np.argmin([_sum_r(p) for p in pops]))
        new_pops[worst] = best_r
    return new_pops, best_r, best_val, time.time() - t0


def _polish(x, y, r, deadline):
    """L-BFGS-B polish of the full (x,y,r) problem; returns improved (x,y,r)."""
    n = len(r)
    v = np.concatenate([x, y, r])

    def f(vv):
        xx = vv[:n]
        yy = vv[n:2 * n]
        rr = vv[2 * n:]
        dx = np.subtract.outer(xx, xx)
        dy = np.subtract.outer(yy, yy)
        d = np.sqrt(dx * dx + dy * dy)
        np.fill_diagonal(d, 1e9)
        m = d - rr[:, None] - rr[None, :]
        pen = np.sum(np.minimum(m, 0.0) ** 2)
        b = np.sum(np.minimum(rr, xx) ** 2) + np.sum(np.minimum(1.0 - xx, rr) ** 2)
        b += np.sum(np.minimum(rr, yy) ** 2) + np.sum(np.minimum(1.0 - yy, rr) ** 2)
        return pen + b - np.sum(rr)

    def g(vv):
        xx = vv[:n]
        yy = vv[n:2 * n]
        rr = vv[2 * n:]
        gx = np.zeros(n)
        gy = np.zeros(n)
        gr = np.zeros(n)
        dx = np.subtract.outer(xx, xx)
        dy = np.subtract.outer(yy, yy)
        d = np.sqrt(dx * dx + dy * dy)
        np.fill_diagonal(d, 1e-9)
        m = d - rr[:, None] - rr[None, :]
        w = -2.0 * np.minimum(m, 0.0) / d
        np.fill_diagonal(w, 0.0)
        gx += -np.sum(w, axis=1)
        gy += -np.sum(w, axis=1)
        gr += np.sum(w, axis=1)
        # boundary
        gx += 2.0 * np.where(xx < rr, -1.0, 0.0) + 2.0 * np.where(1.0 - xx < rr, 1.0, 0.0)
        gy += 2.0 * np.where(yy < rr, -1.0, 0.0) + 2.0 * np.where(1.0 - yy < rr, 1.0, 0.0)
        gr += 2.0 * np.where(xx < rr, 1.0, 0.0) + 2.0 * np.where(1.0 - xx < rr, 1.0, 0.0)
        gr += 2.0 * np.where(yy < rr, 1.0, 0.0) + 2.0 * np.where(1.0 - yy < rr, 1.0, 0.0)
        gr += -1.0
        return np.concatenate([gx, gy, gr])

    bounds = [(0.0, 1.0)] * n + [(0.0, 1.0)] * n + [(1e-3, 0.5)] * n
    res = minimize(f, v, jac=g, method='L-BFGS-B', bounds=bounds,
                   options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 40})
    return res.x[:n], res.x[n:2 * n], res.x[2 * n:]


def construct():
    start = time.time()
    deadline = start + _DEADLINE
    np.random.seed(12345)
    n = _N

    # initial population of radius vectors
    pop_size = 12
    pops = np.empty((pop_size, n))
    pops[0] = 0.5 / np.sqrt(n) * 0.9
    for i in range(1, pop_size):
        pops[i] = np.clip(np.random.uniform(0.02, 0.25, n), 1e-3, 0.5)

    best_r = pops[0].copy()
    best_val = _sum_r(pops[0])
    best_xy = None

    # evaluate initial population feasibility to seed best
    for i in range(pop_size):
        xs = np.clip(np.random.rand(n), 0.0, 1.0)
        ys = np.clip(np.random.rand(n), 0.0, 1.0)
        if _feasible(xs, ys, pops[i]):
            v = _sum_r(pops[i])
            if v > best_val:
                best_val = v
                best_r = pops[i].copy()
                best_xy = (xs, ys)

    # DE loop
    while time.time() < deadline - 6.0:
        pops, best_r, best_val, spent = _de_step(pops, None, best_r, best_val, deadline)
        if time.time() > deadline - 6.0:
            break

    # get centers for best radii
    if best_xy is None or not _feasible(best_xy[0], best_xy[1], best_r):
        # find centers for best_r via feasibility
        xs = np.clip(np.random.rand(n), 0.0, 1.0)
        ys = np.clip(np.random.rand(n), 0.0, 1.0)
        if _feasible(xs, ys, best_r):
            best_xy = (xs, ys)
        else:
            # fallback: equal radii
            best_r = np.full(n, 0.5 / np.sqrt(n) * 0.9)
            side = int(np.ceil(np.sqrt(n)))
            cells = [((i + 0.5) / side, (j + 0.5) / side) for i in range(side) for j in range(side)]
            xs = np.array([c[0] for c in cells[:n]])
            ys = np.array([c[1] for c in cells[:n]])
            best_xy = (xs, ys)
            best_val = _sum_r(best_r)

    x, y = best_xy
    r = best_r

    # local polish (L-BFGS-B) if time remains
    if time.time() < deadline - 3.0:
        try:
            x2, y2, r2 = _polish(x, y, r, deadline)
            # verify feasibility of polished result
            if _feasible(x2, y2, r2) and _sum_r(r2) > best_val:
                x, y, r = x2, y2, r2
                best_val = _sum_r(r2)
        except Exception:
            pass

    # final safety: ensure disjoint & in-bound; clamp radii if tiny violations
    for _ in range(20):
        dx = np.subtract.outer(x, x)
        dy = np.subtract.outer(y, y)
        d = np.sqrt(dx * dx + dy * dy)
        np.fill_diagonal(d, 1e9)
        m = d - r[:, None] - r[None, :]
        if np.min(m) > -1e-7:
            break
        # shrink radii slightly
        r = r * 0.999
    # clamp inside
    r = np.clip(r, 1e-6, 0.5)
    x = np.clip(x, r, 1.0 - r)
    y = np.clip(y, r, 1.0 - r)

    return np.column_stack([x, y, r])
# EVOLVE-BLOCK-END
