"""Circle packing in perimeter-4 rectangle: 21 circles, maximize sum of radii.

Structural change from the seed:
  * Differentiable smooth-max (log-sum-exp) penalty formulation optimized with
    L-BFGS-B under simple box bounds (no 63-variable SLSQP constraint set).
  * Multi-start over aspect ratios x diverse initializations (hex, grid, random,
    force-directed) plus simulated-annealing restarts to escape local minima.
  * Progressive smooth-max high-precision polish (eps 1e-3 -> 1e-7, increasing
    penalty weight) to resolve near-tangent contacts and extract extra radius.
  * Hard validity gate + guaranteed valid fallback so an invalid result is never
    returned.
"""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 21
DEADLINE = 140.0          # wall-clock safety margin (evaluator kills at 150 s)
EPS_MIN = 1e-7
EPS_MAX = 1e-3
PEN_MAX = 30.0
R_FLOOR = 1e-5

# --------------------------------------------------------------------------- #
# Objective
# --------------------------------------------------------------------------- #
def _objective_and_grad(x, W, H, lam, eps):
    n = N
    c = x[2:2 + 2 * n].reshape(n, 2)
    r = x[2 + 2 * n:2 + 3 * n]

    # violation terms (positive => constraint broken)
    v = []
    v.append(r[None, :] - c[:, 0])                       # left
    v.append(c[:, 0] + r[None, :] - W)                   # right
    v.append(r[None, :] - c[:, 1])                       # bottom
    v.append(c[:, 1] + r[None, :] - H)                   # top
    for i in range(n):
        for j in range(i + 1, n):
            dx = c[i, 0] - c[j, 0]
            dy = c[i, 1] - c[j, 1]
            d2 = dx * dx + dy * dy
            d = np.sqrt(max(d2, 1e-18))
            v.append(r[i] + r[j] - d)
    v = np.array(v)

    # smooth max = eps * logsumexp(v/eps)  (>= max(v))
    m = v.max()
    if m > 50.0 * eps:
        sm = m + eps * np.log(np.sum(np.exp((v - m) / eps)))
    else:
        sm = eps * np.log(np.sum(np.exp(v / eps)))

    # gradient
    vc = np.clip(v, -1e-3, None) / eps
    w = np.exp(vc)
    wsum = w.sum()
    pw = w / wsum                      # softmax weights over violations
    sm = m + eps * np.log(wsum) if m > 50.0 * eps else eps * np.log(wsum)

    grad = np.zeros_like(x)
    # objective part: -sum(r)  ->  dr = -1
    grad[2 + 2 * n:2 + 3 * n] = -1.0

    # boundary terms
    # term k: v = r_i - c_i0  => dv/dc_i0 = -1, dv/dr_i = +1
    for k in range(n):
        i = k
        grad[2 + 2 * i] -= lam * pw[k]
        grad[2 + 2 * n + i] += lam * pw[k]
    for k in range(n, 2 * n):
        i = k - n
        grad[2 + 2 * i] += lam * pw[k]
        grad[2 + 2 * n + i] += lam * pw[k]
    for k in range(2 * n, 3 * n):
        i = k - 2 * n
        grad[3 + 2 * i] -= lam * pw[k]
        grad[2 + 2 * n + i] += lam * pw[k]
    for k in range(3 * n, 4 * n):
        i = k - 3 * n
        grad[3 + 2 * i] += lam * pw[k]
        grad[2 + 2 * n + i] += lam * pw[k]

    # overlap terms
    base = 4 * n
    for i in range(n):
        for j in range(i + 1, n):
            idx = base + i * n + j
            dx = c[i, 0] - c[j, 0]
            dy = c[i, 1] - c[j, 1]
            d = np.sqrt(max(dx * dx + dy * dy, 1e-18))
            pwij = lam * pw[idx]
            grad[2 + 2 * i] += pwij * dx / d
            grad[3 + 2 * i] += pwij * dy / d
            grad[2 + 2 * j] -= pwij * dx / d
            grad[3 + 2 * j] -= pwij * dy / d
            grad[2 + 2 * n + i] += pwij
            grad[2 + 2 * n + j] += pwij

    obj = -r.sum() + lam * sm
    return obj, grad


# --------------------------------------------------------------------------- #
# Validity / repair
# --------------------------------------------------------------------------- #
def _check_validity(x, W, H, tol=1e-7):
    n = N
    c = x[2:2 + 2 * n].reshape(n, 2)
    r = x[2 + 2 * n:2 + 3 * n]
    if np.any(r < R_FLOOR):
        return False
    if np.any(c[:, 0] - r < -tol) or np.any(c[:, 0] + r - W > tol):
        return False
    if np.any(c[:, 1] - r < -tol) or np.any(c[:, 1] + r - H > tol):
        return False
    for i in range(n):
        for j in range(i + 1, n):
            d2 = (c[i, 0] - c[j, 0]) ** 2 + (c[i, 1] - c[j, 1]) ** 2
            if d2 < (r[i] + r[j]) ** 2 - tol:
                return False
    return True


def _separate_circles(x, W, H, iters=400, tol=1e-7):
    """Push overlapping circles apart (positions only). Returns (x, ok)."""
    n = N
    c = x[2:2 + 2 * n].reshape(n, 2).copy()
    r = x[2 + 2 * n:2 + 3 * n].copy()
    for _ in range(iters):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                dx = c[i, 0] - c[j, 0]
                dy = c[i, 1] - c[j, 1]
                d = np.sqrt(dx * dx + dy * dy)
                req = r[i] + r[j]
                if d < req - 1e-12:
                    if d < 1e-12:
                        dx, dy = 1e-3, 0.0
                        d = np.hypot(dx, dy)
                    push = (req - d) / 2.0 + 1e-9
                    ux, uy = dx / d, dy / d
                    c[i, 0] += push * ux
                    c[i, 1] += push * uy
                    c[j, 0] -= push * ux
                    c[j, 1] -= push * uy
                    moved = True
        # clamp to boundary
        c[:, 0] = np.clip(c[:, 0], r, W - r)
        c[:, 1] = np.clip(c[:, 1], r, H - r)
        if not moved:
            break
    x2 = x.copy()
    x2[2:2 + 2 * n] = c.ravel()
    ok = _check_validity(x2, W, H, tol)
    return x2, ok


# --------------------------------------------------------------------------- #
# Local optimization
# --------------------------------------------------------------------------- #
def _local_optimize(x, W, H, lam, eps, maxiter=300, time_left=None):
    def fun(x):
        o, _ = _objective_and_grad(x, W, H, lam, eps)
        return o

    def gfun(x):
        _, g = _objective_and_grad(x, W, H, lam, eps)
        return g

    n = N
    rmax = min(W, H) * 0.5
    bounds = [(0.0, None)] * 2
    for i in range(n):
        bounds.append((0.0, W))
        bounds.append((0.0, H))
    for i in range(n):
        bounds.append((R_FLOOR, rmax))

    res = minimize(fun, x, jac=gfun, method='L-BFGS-B', bounds=bounds,
                   options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-10})
    return res.x


def _polish(x, W, H, time_left):
    """Progressive smooth-max polish: eps 1e-3 -> 1e-7, increasing penalty."""
    eps_steps = [1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 3e-7, 1e-7]
    best = x
    best_val = -x[2 + 2 * N:].sum()
    for k, eps in enumerate(eps_steps):
        if time_left() < 3.0:
            break
        lam = min(PEN_MAX, 2.0 * (10.0 ** k))
        try:
            x = _local_optimize(x, W, H, lam, eps, maxiter=120, time_left=time_left)
        except Exception:
            break
        val = -x[2 + 2 * N:].sum()
        if val < best_val:
            best_val = val
            best = x.copy()
    return best


# --------------------------------------------------------------------------- #
# Initializations
# --------------------------------------------------------------------------- #
def _hex_init(W, H, rng):
    n = N
    r0 = 0.5 * min(W, H) / np.sqrt(n) * 0.95
    a = r0 * 2.0
    pts = []
    row = 0
    y = r0
    while y <= H - r0 + 1e-9:
        xoff = (a * 0.8660254) if (row % 2) else 0.0
        x = r0 + xoff
        while x <= W - r0 + 1e-9:
            pts.append([x, y])
            x += a * 0.8660254
        y += a * 0.8660254
        row += 1
    pts = np.array(pts)
    if len(pts) == 0:
        pts = np.array([[W / 2, H / 2]])
    # pick n closest to center, then jitter
    cx, cy = W / 2, H / 2
    d = (pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2
    order = np.argsort(d)
    if len(order) >= n:
        sel = pts[order[:n]]
    else:
        sel = np.tile(pts, (int(np.ceil(n / len(pts))) + 1, 1))[:n]
    sel = sel + rng.uniform(-0.05 * r0, 0.05 * r0, sel.shape)
    sel[:, 0] = np.clip(sel[:, 0], r0, W - r0)
    sel[:, 1] = np.clip(sel[:, 1], r0, H - r0)
    r = np.full(n, r0)
    x = np.zeros(2 + 3 * n)
    x[0], x[1] = W, H
    x[2:2 + 2 * n] = sel.ravel()
    x[2 + 2 * n:2 + 3 * n] = r
    return x


def _grid_init(W, H, rng):
    n = N
    cols = int(np.ceil(np.sqrt(n * W / H)))
    rows = int(np.ceil(n / cols))
    cols = max(cols, 1)
    rows = max(rows, 1)
    cw = W / cols
    ch = H / rows
    r0 = 0.5 * min(cw, ch) * 0.98
    pts = []
    for i in range(cols):
        for j in range(rows):
            pts.append([(i + 0.5) * cw, (j + 0.5) * ch])
    pts = np.array(pts[:n])
    if len(pts) < n:
        pts = np.vstack([pts, np.tile(pts, (int(np.ceil(n / len(pts))) + 1, 1))])[:n]
    pts += rng.uniform(-0.02 * r0, 0.02 * r0, pts.shape)
    pts[:, 0] = np.clip(pts[:, 0], r0, W - r0)
    pts[:, 1] = np.clip(pts[:, 1], r0, H - r0)
    x = np.zeros(2 + 3 * n)
    x[0], x[1] = W, H
    x[2:2 + 2 * n] = pts.ravel()
    x[2 + 2 * n:2 + 3 * n] = r0
    return x


def _random_init(W, H, rng):
    n = N
    r0 = 0.5 * min(W, H) / np.sqrt(n) * 0.85
    c = np.column_stack([
        rng.uniform(r0, W - r0, n),
        rng.uniform(r0, H - r0, n),
    ])
    x = np.zeros(2 + 3 * n)
    x[0], x[1] = W, H
    x[2:2 + 2 * n] = c.ravel()
    x[2 + 2 * n:2 + 3 * n] = r0
    return x


def _force_init(W, H, rng, iters=600):
    """Force-directed layout: repel overlaps, pull to walls, then scale to fill."""
    n = N
    r0 = 0.5 * min(W, H) / np.sqrt(n) * 0.9
    c = np.column_stack([
        rng.uniform(r0, W - r0, n),
        rng.uniform(r0, H - r0, n),
    ])
    r = np.full(n, r0)
    for _ in range(iters):
        # repulsion
        for i in range(n):
            for j in range(i + 1, n):
                dx = c[i, 0] - c[j, 0]
                dy = c[i, 1] - c[j, 1]
                d = np.hypot(dx, dy)
                req = r[i] + r[j]
                if d < req:
                    if d < 1e-12:
                        dx, dy = 1e-3, 0.0
                        d = np.hypot(dx, dy)
                    push = 0.5 * (req - d) + 1e-6
                    c[i, 0] += push * dx / d
                    c[i, 1] += push * dy / d
                    c[j, 0] -= push * dx / d
                    c[j, 1] -= push * dy / d
        # pull each circle toward nearest wall (encourage boundary contacts)
        for i in range(n):
            dists = [c[i, 0] - r[i], W - c[i, 0] - r[i],
                     c[i, 1] - r[i], H - c[i, 1] - r[i]]
            m = min(dists)
            if m < 0.3 * r0:
                continue
            if dists[0] == m:
                c[i, 0] -= 0.02 * r0
            elif dists[1] == m:
                c[i, 0] += 0.02 * r0
            elif dists[2] == m:
                c[i, 1] -= 0.02 * r0
            else:
                c[i, 1] += 0.02 * r0
        c[:, 0] = np.clip(c[:, 0], r, W - r)
        c[:, 1] = np.clip(c[:, 1], r, H - r)
    # scale up until a contact is tight (fill the box)
    cx, cy = c[:, 0].mean(), c[:, 1].mean()
    for _ in range(200):
        c2 = cx + (c - cx) * 1.005
        r2 = r * 1.005
        ok = True
        c2[:, 0] = np.clip(c2[:, 0], r2, W - r2)
        c2[:, 1] = np.clip(c2[:, 1], r2, H - r2)
        for i in range(n):
            for j in range(i + 1, n):
                d = np.hypot(c2[i, 0] - c2[j, 0], c2[i, 1] - c2[j, 1])
                if d < r2[i] + r2[j] - 1e-9:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            break
        c, r = c2, r2
    x = np.zeros(2 + 3 * n)
    x[0], x[1] = W, H
    x[2:2 + 2 * n] = c.ravel()
    x[2 + 2 * n:2 + 3 * n] = r
    return x


# --------------------------------------------------------------------------- #
# Fallback (always valid)
# --------------------------------------------------------------------------- #
def _fallback(W, H):
    n = N
    side = int(np.ceil(np.sqrt(n)))
    cw, ch = W / side, H / side
    r = 0.5 * min(cw, ch) * 0.98
    x = np.zeros(2 + 3 * n)
    x[0], x[1] = W, H
    k = 0
    for i in range(side):
        for j in range(side):
            if k >= n:
                break
            x[2 + 2 * k] = (i + 0.5) * cw
            x[3 + 2 * k] = (j + 0.5) * ch
            x[2 + 2 * n + k] = r
            k += 1
    return x


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def construct():
    t0 = time.time()

    def time_left():
        return DEADLINE - (time.time() - t0)

    rng = np.random.default_rng(20240607)

    best = None
    best_val = -np.inf

    def consider(x, W, H):
        nonlocal best, best_val
        if x is None:
            return
        if not _check_validity(x, W, H):
            return
        val = -x[2 + 2 * N:].sum()
        if val > best_val:
            best_val = val
            best = (x.copy(), W, H)

    inits = ['hex', 'grid', 'random', 'force']
    aspect_ratios = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.4, 1.6, 2.0, 2.5]

    # ---- Phase 1: multi-start (annealed) over aspect ratios x inits ---------- #
    for ar in aspect_ratios:
        if time_left() < 6.0:
            break
        W = 2.0 * ar / (1.0 + ar)
        H = 2.0 / (1.0 + ar)
        for init_name in inits:
            if time_left() < 4.0:
                break
            try:
                if init_name == 'hex':
                    x = _hex_init(W, H, rng)
                elif init_name == 'grid':
                    x = _grid_init(W, H, rng)
                elif init_name == 'force':
                    x = _force_init(W, H, rng, iters=400)
                else:
                    x = _random_init(W, H, rng)
            except Exception:
                continue

            # light local optimization to get a feasible-ish start
            try:
                x = _local_optimize(x, W, H, lam=2.0, eps=1e-3, maxiter=120)
            except Exception:
                pass

            # simulated-annealing restarts: perturb, optimize, keep best
            base = x.copy()
            for it in range(6):
                if time_left() < 3.0:
                    break
                try:
                    x = _local_optimize(base, W, H, lam=3.0, eps=3e-4, maxiter=120)
                except Exception:
                    break
                x2, ok = _separate_circles(x, W, H, iters=200)
                if ok:
                    consider(x2, W, H)
                # annealing: scale perturbation by current quality
                scale = 0.05 * min(W, H)
                base = x2 if ok else x
                base = base + rng.normal(0.0, scale, base.shape)
                base[0], base[1] = W, H
                base[2 + 2 * N:2 + 3 * N] = np.clip(base[2 + 2 * N:2 + 3 * N],
                                                    R_FLOOR, min(W, H) * 0.5)

    # ---- Phase 2: high-precision polish of the best candidate --------------- #
    if best is not None:
        xb, Wb, Hb = best
        try:
            xp = _polish(xb, Wb, Hb, time_left)
            xp2, ok = _separate_circles(xp, Wb, Hb, iters=200)
            if ok:
                consider(xp2, Wb, Hb)
        except Exception:
            pass

    # ---- Phase 3: final guaranteed-valid selection -------------------------- #
    if best is None:
        W, H = 1.0, 1.0
        x = _fallback(W, H)
    else:
        x, W, H = best

    x, ok = _separate_circles(x, W, H, iters=300)
    if not ok:
        W, H = 1.0, 1.0
        x = _fallback(W, H)

    # Build (n,3) output: [x, y, r]
    c = x[2:2 + 2 * N].reshape(N, 2)
    r = x[2 + 2 * N:2 + 3 * N]
    out = np.column_stack([c, r])

    # Shift so the bounding box touches the origin and re-verify perimeter.
    out[:, 0] -= out[:, 0].min()
    out[:, 1] -= out[:, 1].min()
    Wf = out[:, 0].max()
    Hf = out[:, 1].max()
    per = 2.0 * (Wf + Hf)
    if per > 4.0:
        s = 4.0 / per
        out *= s

    return out
