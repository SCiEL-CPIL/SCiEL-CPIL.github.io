"""Circle packing in perimeter-4 rectangle; maximize sum of radii for 21 circles."""
import time
import numpy as np
from scipy.optimize import minimize

N = 21
_DEADLINE = 110.0  # safety margin below 120s wall-clock


def _hex_layout(n):
    """Return (n,2) positions on a hexagonal lattice, roughly centered."""
    pts = []
    ring = 0
    while len(pts) < n:
        if ring == 0:
            pts.append((0.0, 0.0))
        else:
            for k in range(6):
                ang = np.pi / 3.0 * k
                dx = np.cos(ang)
                dy = np.sin(ang)
                for j in range(ring):
                    pts.append((ring * dx + j * (dx - dy), ring * dy + j * (dy - dx)))
        ring += 1
    pts = np.array(pts[:n], dtype=float)
    pts -= pts.mean(axis=0)
    return pts


def _feasible(x, W, H):
    """Return True if all circles are disjoint and inside the rectangle."""
    n = len(x)
    for i in range(n):
        xi, yi, ri = x[i]
        if ri <= 0:
            return False
        if xi - ri < -1e-9 or xi + ri > W + 1e-9:
            return False
        if yi - ri < -1e-9 or yi + ri > H + 1e-9:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(x[i, 0] - x[j, 0], x[i, 1] - x[j, 1])
            if d < x[i, 2] + x[j, 2] - 1e-9:
                return False
    return True


def _value(x, W, H):
    """Sum of radii if feasible, else -inf."""
    if not _feasible(x, W, H):
        return -np.inf
    return float(np.sum(x[:, 2]))


def _optimize_from(x0, W0, H0, time_budget):
    """Run SLSQP from a start; return (x, W, H, val)."""
    n = len(x0)

    def obj(v):
        x = v[:3 * n]
        log_ar = v[3 * n]
        H = 2.0 * np.exp(log_ar)
        W = 4.0 - H
        if W <= 0 or H <= 0:
            return 1e10
        # Objective: minimize negative sum of radii
        val = -np.sum(x[:, 2])
        # Soft penalties to keep SLSQP from running wild
        pen = 0.0
        for i in range(n):
            xi, yi, ri = x[i]
            if ri < 1e-6:
                pen += 1e4 * (1e-6 - ri)
            if xi - ri < 0:
                pen += 1e4 * (0 - (xi - ri))
            if xi + ri > W:
                pen += 1e4 * ((xi + ri) - W)
            if yi - ri < 0:
                pen += 1e4 * (0 - (yi - ri))
            if yi + ri > H:
                pen += 1e4 * ((yi + ri) - H)
        for i in range(n):
            for j in range(i + 1, n):
                d = np.hypot(x[i, 0] - x[j, 0], x[i, 1] - x[j, 1])
                gap = x[i, 2] + x[j, 2] - d
                if gap > 0:
                    pen += 1e4 * gap
        return val + pen

    # Constraints: hard
    def make_bound(i, coord, side_fn):
        def c(v):
            x = v[:3 * n]
            log_ar = v[3 * n]
            H = 2.0 * np.exp(log_ar)
            W = 4.0 - H
            if coord == 0:
                return x[i, 0] - x[i, 2]
            elif coord == 1:
                return x[i, 1] - x[i, 2]
            elif coord == 2:
                return W - (x[i, 0] + x[i, 2])
            else:
                return H - (x[i, 1] + x[i, 2])
        return c

    constraints = []
    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': make_bound(i, 0, None)})
        constraints.append({'type': 'ineq', 'fun': make_bound(i, 1, None)})
        constraints.append({'type': 'ineq', 'fun': make_bound(i, 2, None)})
        constraints.append({'type': 'ineq', 'fun': make_bound(i, 3, None)})

    def make_pair(i, j):
        def c(v):
            x = v[:3 * n]
            d = np.hypot(x[i, 0] - x[j, 0], x[i, 1] - x[j, 1])
            return d - x[i, 2] - x[j, 2]
        return c

    for i in range(n):
        for j in range(i + 1, n):
            constraints.append({'type': 'ineq', 'fun': make_pair(i, j)})

    # Bounds on variables
    lb = np.concatenate([np.zeros(3 * n), [np.log(0.2)]])
    ub = np.concatenate([np.full(3 * n, 2.0), [np.log(3.0)]])

    v0 = np.concatenate([x0.ravel(), [np.log(H0 / W0)]])

    t0 = time.time()
    try:
        res = minimize(
            obj, v0, method='SLSQP',
            bounds=list(zip(lb, ub)),
            constraints=constraints,
            options={'maxiter': 300, 'ftol': 1e-9, 'disp': False},
        )
    except Exception:
        return None

    if time.time() - t0 > time_budget:
        return None

    if not res.success:
        return None

    v = res.x
    x = v[:3 * n].reshape(n, 3)
    log_ar = v[3 * n]
    H = 2.0 * np.exp(log_ar)
    W = 4.0 - H
    if W <= 0 or H <= 0:
        return None
    val = _value(x, W, H)
    if not np.isfinite(val):
        return None
    return (x, W, H, val)


def construct():
    t_start = time.time()
    deadline = t_start + _DEADLINE

    # Initial hexagonal layout
    base = _hex_layout(N)
    # Scale so it fits in a square-ish box initially
    maxc = np.max(np.abs(base))
    if maxc > 0:
        base = base / maxc * 0.9

    # Seed: equal radii sized to fit in a 1x1 square (perimeter 4)
    r_seed = 0.45
    x0 = np.zeros((N, 3))
    x0[:, 0] = base[:, 0] + 0.5
    x0[:, 1] = base[:, 1] + 0.5
    x0[:, 2] = r_seed

    best = None
    best_val = -np.inf

    # First attempt: square 1x1
    result = _optimize_from(x0.copy(), 1.0, 1.0, time_budget=30)
    if result is not None:
        x, W, H, val = result
        if val > best_val:
            best_val = val
            best = (x, W, H)

    if time.time() > deadline - 20:
        if best is not None:
            return best[0]
        return x0

    # Second attempt: slightly different aspect ratio (taller)
    W2, H2 = 0.8, 1.2
    x0b = x0.copy()
    x0b[:, 0] = x0b[:, 0] * (W2 / 1.0)
    x0b[:, 1] = x0b[:, 1] * (H2 / 1.0)
    result = _optimize_from(x0b, W2, H2, time_budget=30)
    if result is not None:
        x, W, H, val = result
        if val > best_val:
            best_val = val
            best = (x, W, H)

    if time.time() > deadline - 20:
        if best is not None:
            return best[0]
        return x0

    # Third attempt: wider rectangle
    W3, H3 = 1.2, 0.8
    x0c = x0.copy()
    x0c[:, 0] = x0c[:, 0] * (W3 / 1.0)
    x0c[:, 1] = x0c[:, 1] * (H3 / 1.0)
    result = _optimize_from(x0c, W3, H3, time_budget=30)
    if result is not None:
        x, W, H, val = result
        if val > best_val:
            best_val = val
            best = (x, W, H)

    if best is not None:
        return best[0]
    return x0
