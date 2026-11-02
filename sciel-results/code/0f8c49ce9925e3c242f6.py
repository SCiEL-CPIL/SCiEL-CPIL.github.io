"""Circle packing in unit square: maximize sum of radii for 26 disjoint circles.

DIFFERENT STRATEGY: a geometric "grow-and-pack" solver instead of force-directed
annealing + penalty L-BFGS. Each circle's radius is set to its EXACT maximal value
given all others (a hard geometric constraint, not a soft penalty), via cyclic
coordinate descent over radii. A swap/anneal neighborhood (randomly exchange two
circle positions and re-grow) escapes local optima. A final SLSQP polish on the
joint (position, radius) variables with explicit non-overlap and wall constraints
recovers any residual slack.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 26
EPS = 1e-9


def _max_radius(pos, r):
    """Exact maximal radius for each circle given all positions and radii."""
    n = len(r)
    mr = np.empty(n)
    for i in range(n):
        dmin = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
        for j in range(n):
            if i != j:
                dmin = min(dmin, np.hypot(pos[i, 0] - pos[j, 0], pos[i, 1] - pos[j, 1]) - r[j])
        mr[i] = max(0.0, dmin)
    return mr


def _grow(pos, r, iters=120):
    """Cyclically grow each radius to its exact max (positions fixed)."""
    r = np.array(r, dtype=float).copy()
    for _ in range(iters):
        for i in range(N):
            dmin = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
            for j in range(N):
                if i != j:
                    dmin = min(dmin, np.hypot(pos[i, 0] - pos[j, 0], pos[i, 1] - pos[j, 1]) - r[j])
            r[i] = max(0.0, dmin)
    return r


def _valid(cands):
    """Return (arr, score) for a valid packing of N circles, else (None, -1)."""
    valid = []
    for x, y, r in cands:
        if r <= 0:
            continue
        if x < r + EPS or x > 1 - r - EPS or y < r + EPS or y > 1 - r - EPS:
            continue
        ok = True
        for x2, y2, r2 in valid:
            if (x - x2) ** 2 + (y - y2) ** 2 < (r + r2) ** 2 - 1e-9:
                ok = False
                break
        if ok:
            valid.append((x, y, r))
    if len(valid) < N:
        return None, -1.0
    arr = np.array(valid[:N])
    return arr, float(arr[:, 2].sum())


def _slsqp(pos, rad, deadline):
    """SLSQP polish on joint (pos, rad) with hard constraints."""
    n = len(rad)
    x0 = np.concatenate([pos.reshape(-1), rad])

    def f(v):
        p = v[:2 * n].reshape(n, 2)
        r = v[2 * n:]
        return -np.sum(r)

    cons = []
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({
                'type': 'ineq',
                'fun': (lambda v, i=i, j=j:
                        np.hypot(v[2 * i] - v[2 * j], v[2 * i + 1] - v[2 * j + 1])
                        - v[2 * n + i] - v[2 * n + j]),
            })
    for i in range(n):
        for ax in range(2):
            cons.append({'type': 'ineq', 'fun': (lambda v, i=i, ax=ax: v[2 * i + ax] - v[2 * n + i])})
            cons.append({'type': 'ineq',
                         'fun': (lambda v, i=i, ax=ax: 1.0 - v[2 * i + ax] - v[2 * n + i])})
    bounds = [(0.0, 1.0)] * (2 * n) + [(1e-5, 0.5)] * n
    try:
        res = minimize(f, x0, method='SLSQP', bounds=bounds, constraints=cons,
                       options={'maxiter': 40, 'ftol': 1e-9})
        if res.success or res.fun < f(x0):
            p = res.x[:2 * n].reshape(n, 2)
            r = res.x[2 * n:]
            # re-exact the radii to remove any numerical slack
            r = _max_radius(p, r)
            return p, r
    except Exception:
        pass
    return pos, rad


def construct():
    t_start = time.time()
    DEADLINE = 140.0  # safety margin before the 150 s kill
    rng = np.random.default_rng(20240517)
    best = None
    best_score = -1.0

    def consider(pos, rad):
        nonlocal best, best_score
        rad = _max_radius(np.asarray(pos, dtype=float), np.asarray(rad, dtype=float))
        arr, s = _valid(list(zip(pos[:, 0], pos[:, 1], rad)))
        if arr is not None and s > best_score:
            best_score = s
            best = arr

    # ---------- structured + random initial positions ----------
    def grid(side):
        cells = [((i + 0.5) / side, (j + 0.5) / side)
                 for i in range(side) for j in range(side)][:N]
        return np.array(cells, dtype=float)

    def hex_layout():
        pts = []
        rows = 5
        for row in range(rows):
            y = (row + 0.5) / rows
            offset = 0.5 / rows if row % 2 else 0.0
            for col in range(rows):
                x = (col + 0.5) / rows + offset
                if 0 <= x <= 1:
                    pts.append((x, y))
            if len(pts) >= N:
                break
        return np.array(pts[:N], dtype=float)

    def rings():
        pts = [(0.5, 0.5)]
        for rad, cnt in [(0.2, 6), (0.35, 12), (0.44, 10)]:
            for k in range(cnt):
                a = 2 * np.pi * k / cnt
                pts.append((0.5 + rad * np.cos(a), 0.5 + rad * np.sin(a)))
            if len(pts) >= N:
                break
        return np.array(pts[:N], dtype=float)

    starts = []
    for side in range(3, 8):
        starts.append(grid(side))
    starts.append(hex_layout())
    starts.append(rings())
    for _ in range(5):
        starts.append(rng.random((N, 2)) * 0.84 + 0.08)

    # ---------- grow-and-pack with swap/anneal neighborhood ----------
    def solve(pos0, seed, n_swap):
        rng_l = np.random.default_rng(seed)
        pos = np.array(pos0, dtype=float).copy()
        r = _grow(pos, np.full(N, 0.5 / np.sqrt(N)), iters=80)
        cur = float(np.sum(r))
        for s in range(n_swap):
            if time.time() > DEADLINE - 8:
                break
            i, j = rng_l.choice(N, size=2, replace=False)
            pos[i], pos[j] = pos[j].copy(), pos[i].copy()
            rn = _grow(pos, r, iters=60)
            sn = float(np.sum(rn))
            # accept if better, or with annealed probability
            if sn > cur or rng_l.random() < np.exp(min(0.0, (sn - cur) / 0.05)):
                r = rn
                cur = sn
        return pos, r

    # ---------- main search loop ----------
    for r in range(len(starts)):
        if time.time() > DEADLINE - 12:
            break
        pos0 = starts[r]
        for trial in range(3):
            if time.time() > DEADLINE - 12:
                break
            seed = 5000 + 11 * r + trial
            n_swap = 25 if trial == 0 else 12
            p, rr = solve(pos0, seed, n_swap)
            consider(p, rr)

    # ---------- final SLSQP polish on the best found ----------
    if best is not None:
        pp, rr = _slsqp(best[:, :2], best[:, 2], DEADLINE)
        consider(pp, rr)
        # one more polish pass
        pp2, rr2 = _slsqp(pp, rr, DEADLINE)
        consider(pp2, rr2)

    if best is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    return best


if __name__ == '__main__':
    out = construct()
    print("sum of radii:", out[:, 2].sum())
    print(out)
