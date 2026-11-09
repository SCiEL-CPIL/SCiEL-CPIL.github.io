"""Circle packing in perimeter-4 rectangle: 21 circles, maximize sum of radii."""
import numpy as np
from scipy.optimize import minimize
import time

N = 21
DEADLINE_MARGIN = 8.0  # safety margin before the 150s wall-clock kill


def _validate(res, tol=1e-5):
    """Return (score, ok). score = sum(r); ok if disjoint and inside a perimeter<=4 box."""
    res = np.asarray(res, dtype=float)
    if res.shape[0] != N:
        return 0.0, False
    c = res[:, 0:2]
    r = res[:, 2]
    if np.any(r <= 0):
        return 0.0, False
    lo = c - r
    hi = c + r
    x0, x1 = lo[:, 0].min(), hi[:, 0].max()
    y0, y1 = lo[:, 1].min(), hi[:, 1].max()
    W, H = x1 - x0, y1 - y0
    if 2 * (W + H) > 4.0 + tol:
        return 0.0, False
    # pairwise disjoint (allow tiny overlap)
    for i in range(N):
        for j in range(i + 1, N):
            d = np.hypot(c[i, 0] - c[j, 0], c[i, 1] - c[j, 1])
            if d < r[i] + r[j] - tol:
                return 0.0, False
    return float(r.sum()), True


def _initial_grid(n):
    """Grid-ish initial placement inside a perimeter-4 box (5x1=1.0, 4x1=1.0)."""
    cols, rows = 5, 4
    pts = []
    for j in range(rows):
        for i in range(cols):
            if len(pts) < n:
                x = 0.5 + i * (1.0 / cols)
                y = 0.5 + j * (1.0 / rows)
                pts.append([x, y])
    pts = np.array(pts[:n], dtype=float)
    r = np.full(n, 0.045)
    return np.column_stack([pts, r])


def _force_relax(res, steps=6000, lr0=0.02, lr1=0.001, k=20.0):
    """Force-directed: expand radii, repel overlaps, keep inside box. Returns (res, score)."""
    res = res.copy()
    c = res[:, 0:2]
    r = res[:, 2].copy()
    best_res = res.copy()
    best_score = 0.0

    for s in range(steps):
        lr = lr0 * (lr1 / lr0) ** (s / max(1, steps - 1))
        # grow radii toward a target (equal-ish share of the box)
        r = np.minimum(r + lr * 0.002, 0.49)

        # forces on centers
        F = np.zeros_like(c)
        for i in range(N):
            for j in range(i + 1, N):
                dx = c[i, 0] - c[j, 0]
                dy = c[i, 1] - c[j, 1]
                d = np.hypot(dx, dy)
                min_d = r[i] + r[j]
                if d < 1e-9:
                    dx = 1e-6
                    dy = 0.0
                    d = 1e-6
                overlap = min_d - d
                if overlap > 0:
                    f = k * overlap
                    ux, uy = dx / d, dy / d
                    F[i, 0] += f * ux
                    F[i, 1] += f * uy
                    F[j, 0] -= f * ux
                    F[j, 1] -= f * uy

        # wall forces: keep each circle inside the box [0,1]x[0,1]
        for i in range(N):
            for axis in range(2):
                lo = r[i]
                hi = 1.0 - r[i]
                if c[i, axis] < lo:
                    F[i, axis] += k * (lo - c[i, axis])
                elif c[i, axis] > hi:
                    F[i, axis] -= k * (c[i, axis] - hi)

        c = c + lr * F
        # clamp centers to feasible region
        for i in range(N):
            for axis in range(2):
                c[i, axis] = np.clip(c[i, axis], r[i], 1.0 - r[i])

        res = np.column_stack([c, r])
        score, ok = _validate(res)
        if ok and score > best_score:
            best_score = score
            best_res = res.copy()
    return best_res, best_score


def _slsqp_polish(res, time_limit=12.0):
    """SLSQP polish: maximize sum(r) subject to box + disjoint constraints."""
    t0 = time.time()
    x0 = res.reshape(-1).copy()

    def neg_obj(x):
        return -x[2::3].sum()

    cons = []
    # box perimeter: 2*((max(x)-min(x)) + (max(y)-min(y))) <= 4
    cons.append({
        'type': 'ineq',
        'fun': lambda x: 4.0 - 2.0 * (((x[0::3].max() - x[0::3].min()) +
                                      (x[1::3].max() - x[1::3].min()))),
    })
    # radii positive
    cons.append({'type': 'ineq', 'fun': lambda x: x[2::3] - 1e-5})
    # pairwise disjoint: dist >= r_i + r_j
    for i in range(N):
        for j in range(i + 1, N):
            ii, jj = i, j
            cons.append({
                'type': 'ineq',
                'fun': lambda x, ii=ii, jj=jj:
                    np.hypot(x[3 * ii] - x[3 * jj], x[3 * ii + 1] - x[3 * jj + 1])
                    - (x[3 * ii + 2] + x[3 * jj + 2]) - 1e-6,
            })

    bounds = [(None, None), (None, None), (1e-5, 0.49)] * N

    best = x0.copy()
    best_score = _validate(x0.reshape(N, 3))[0]
    try:
        res_opt = minimize(neg_obj, x0, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 400, 'ftol': 1e-9,
                                                       'disp': False})
        if res_opt.success or res_opt.fun < neg_obj(x0):
            cand = res_opt.x.reshape(N, 3)
            sc, ok = _validate(cand)
            if ok and sc > best_score:
                best, best_score = cand, sc
    except Exception:
        pass
    return best


def construct():
    start = time.time()
    deadline = 150.0 - DEADLINE_MARGIN

    res = _initial_grid(N)
    best, best_score = _force_relax(res)
    if time.time() > deadline:
        return best

    # polish with SLSQP
    polished = _slsqp_polish(best, time_limit=12.0)
    if time.time() > deadline:
        return best

    sc, ok = _validate(polished)
    if ok and sc > best_score:
        best, best_score = polished, sc

    # a second short force + polish cycle if time remains, to escape local minima
    if time.time() < deadline - 20.0:
        res2 = _initial_grid(N)
        # jitter initial grid slightly for a different basin
        rng = np.random.default_rng(12345)
        res2[:, 0:2] += rng.uniform(-0.03, 0.03, size=(N, 2))
        b2, s2 = _force_relax(res2)
        if time.time() < deadline - 15.0:
            p2 = _slsqp_polish(b2)
            sc2, ok2 = _validate(p2)
            if ok2 and sc2 > best_score:
                best, best_score = p2, sc2

    # final safety: ensure best is valid, else fall back to initial grid
    sc_final, ok_final = _validate(best)
    if not ok_final:
        fb = _initial_grid(N)
        best = fb

    return best
