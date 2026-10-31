"""Circle packing in perimeter-4 rectangle.

Combines two ideas:
  * Structural (from inspiration): force-based "grow radii + relax positions"
    packing that drives circles into corner/edge pockets, giving a far better
    initial feasible layout than a plain grid.
  * Refinement (from current program): robust scipy SLSQP with a smooth
    penalty objective + analytic gradient, plus a final local-search grow loop
    and a polish pass.  The objective is vectorized for speed.
"""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 21
START = time.time()
DEADLINE = START + 143.0  # safety margin before the 150 s kill

EPS = 1e-9


# --------------------------------------------------------------------------- #
# Fast vectorized penalty objective (minimize -sum(r) + penalties)
# --------------------------------------------------------------------------- #
def _obj_vec(x, W, H, mu=1e4, mu_o=1e3, mu_c=1e2):
    """Return (value, grad).  x is shape (N,3): cx, cy, r."""
    cx = x[:, 0].copy()
    cy = x[:, 1].copy()
    r = x[:, 2].copy()
    n = len(cx)

    cx = np.clip(cx, -1.0, W + 1.0)
    cy = np.clip(cy, -1.0, H + 1.0)
    r = np.clip(r, -1.0, 2.0)

    val = -np.sum(r)
    grad = np.zeros_like(x)
    grad[:, 2] = -1.0

    # --- boundary: r <= min(cx, cy, W-cx, H-cy) --------------------------- #
    lo = np.minimum(np.minimum(cx, cy), np.minimum(W - cx, H - cy))
    viol_b = np.maximum(r - lo, 0.0)
    val += mu * np.sum(viol_b ** 2)
    if np.any(viol_b > 0):
        dl = cx[:, None]
        db = cy[:, None]
        dr = (W - cx)[:, None]
        dt = (H - cy)[:, None]
        lo4 = np.stack([dl, db, dr, dt], axis=1)
        m = lo4.min(axis=1, keepdims=True)
        act = (lo4 <= m + EPS).astype(np.float64)
        cnt = np.maximum(act.sum(axis=1, keepdims=True), 1.0)
        g = (act * np.stack([np.ones_like(dl), np.ones_like(db),
                             -np.ones_like(dr), -np.ones_like(dt)], axis=1)) / cnt
        g_cx = g[:, 0] - g[:, 2]
        g_cy = g[:, 1] - g[:, 3]
        dv = 2.0 * mu * viol_b
        grad[:, 0] += dv * (-g_cx)
        grad[:, 1] += dv * (-g_cy)
        grad[:, 2] += dv * 1.0

    # --- r > 0 ----------------------------------------------------------- #
    viol_r = np.maximum(-r, 0.0)
    val += mu * np.sum(viol_r ** 2)
    if np.any(viol_r > 0):
        grad[:, 2] += 2.0 * mu * viol_r * (-1.0)

    # --- centers inside box (soft) --------------------------------------- #
    vcx_lo = np.maximum(-cx, 0.0)
    vcx_hi = np.maximum(cx - W, 0.0)
    vcy_lo = np.maximum(-cy, 0.0)
    vcy_hi = np.maximum(cy - H, 0.0)
    val += mu_c * (np.sum(vcx_lo**2) + np.sum(vcx_hi**2) +
                   np.sum(vcy_lo**2) + np.sum(vcy_hi**2))
    grad[:, 0] += 2.0 * mu_c * (vcx_hi - vcx_lo)
    grad[:, 1] += 2.0 * mu_c * (vcy_hi - vcy_lo)

    # --- non-overlap (vectorized pairs) ---------------------------------- #
    dx = cx[:, None] - cx[None, :]
    dy = cy[:, None] - cy[None, :]
    d = np.sqrt(dx * dx + dy * dy)
    req = r[:, None] + r[None, :]
    viol = req - d
    iu = np.triu_indices(n, 1)
    v = viol[iu]
    val += mu_o * np.sum(v ** 2)
    if np.any(v > 0):
        dd = np.maximum(d[iu], 1e-12)
        dv = 2.0 * mu_o * v
        gx = dv * (dx[iu] / dd)
        gy = dv * (dy[iu] / dd)
        i = iu[0]
        j = iu[1]
        np.add.at(grad, (i, 0), gx)
        np.add.at(grad, (j, 0), -gx)
        np.add.at(grad, (i, 1), gy)
        np.add.at(grad, (j, 1), -gy)
        np.add.at(grad, (i, 2), dv)
        np.add.at(grad, (j, 2), dv)

    return val, grad


# --------------------------------------------------------------------------- #
# Evaluation (validity + objective)
# --------------------------------------------------------------------------- #
def _evaluate(x, W, H):
    """Return sum of radii if x is a valid packing, else None."""
    cx = np.clip(x[:, 0].copy(), 0.0, W)
    cy = np.clip(x[:, 1].copy(), 0.0, H)
    r = x[:, 2].copy()
    n = len(cx)
    lo = np.minimum(np.minimum(cx, cy), np.minimum(W - cx, H - cy))
    r = np.clip(np.minimum(r, lo), 1e-6, None)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(cx[i] - cx[j], cy[i] - cy[j])
            if d + 1e-8 < r[i] + r[j]:
                return None
    return float(np.sum(r))


# --------------------------------------------------------------------------- #
# Structural initializer: force-based grow + relax
# --------------------------------------------------------------------------- #
def _relax_pack(circles, W, H, max_iter=4000, time_limit=None):
    """Grow radii and relax positions via forces.  Returns list of (x,y,r)."""
    xs = np.array([c[0] for c in circles], dtype=np.float64)
    ys = np.array([c[1] for c in circles], dtype=np.float64)
    rs = np.array([c[2] for c in circles], dtype=np.float64)
    n = len(xs)
    t0 = time.time()
    growth = 0.0006
    lr = 0.12

    for _ in range(max_iter):
        if time_limit is not None and time.time() > time_limit:
            break
        rs_new = rs * (1.0 + growth)

        fx = np.zeros(n)
        fy = np.zeros(n)
        for i in range(n):
            for j in range(i + 1, n):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.hypot(dx, dy)
                min_d = rs_new[i] + rs_new[j]
                if dist < min_d:
                    ov = min_d - dist
                    if dist > 1e-8:
                        nx, ny = dx / dist, dy / dist
                    else:
                        a = random.uniform(0, 2 * np.pi)
                        nx, ny = np.cos(a), np.sin(a)
                    fx[i] += nx * ov
                    fy[i] += ny * ov
                    fx[j] -= nx * ov
                    fy[j] -= ny * ov

        for i in range(n):
            if xs[i] < rs_new[i]:
                fx[i] += (rs_new[i] - xs[i]) * 5.0
            if xs[i] > W - rs_new[i]:
                fx[i] -= (xs[i] - (W - rs_new[i])) * 5.0
            if ys[i] < rs_new[i]:
                fy[i] += (rs_new[i] - ys[i]) * 5.0
            if ys[i] > H - rs_new[i]:
                fy[i] -= (ys[i] - (H - rs_new[i])) * 5.0

        xs_new = np.clip(xs + lr * fx, rs_new, W - rs_new)
        ys_new = np.clip(ys + lr * fy, rs_new, H - rs_new)

        # validity check
        ok = True
        for i in range(n):
            if xs_new[i] < rs_new[i] - 1e-8 or xs_new[i] > W - rs_new[i] + 1e-8:
                ok = False
                break
            if ys_new[i] < rs_new[i] - 1e-8 or ys_new[i] > H - rs_new[i] + 1e-8:
                ok = False
                break
        if ok:
            for i in range(n):
                for j in range(i + 1, n):
                    d2 = (xs_new[i] - xs_new[j]) ** 2 + (ys_new[i] - ys_new[j]) ** 2
                    md = rs_new[i] + rs_new[j]
                    if d2 < md * md - 1e-8:
                        ok = False
                        break
                if not ok:
                    break
        if ok:
            xs, ys, rs = xs_new, ys_new, rs_new
        else:
            growth *= 0.95
            if growth < 1e-6:
                break

    return list(zip(xs, ys, rs))


def _initial_layout(W, H, time_limit=None):
    """Build a diverse set of feasible initial layouts for (W,H)."""
    layouts = []
    Hs = max(H, 1e-3)
    Ws = max(W, 1e-3)

    def add(circles):
        arr = np.array(circles, dtype=np.float64)
        s = _evaluate(arr, W, H)
        if s is not None:
            layouts.append((s, arr))

    # (a) force-relaxed layouts from several seeds
    seeds = []
    # corner / edge biased
    for _ in range(3):
        pts = []
        for _ in range(N):
            cx = random.uniform(0.02 * Ws, 0.98 * Ws)
            cy = random.uniform(0.02 * Hs, 0.98 * Hs)
            pts.append([cx, cy, 0.02 * min(Ws, Hs)])
        seeds.append(pts)
    # uniform
    for _ in range(2):
        pts = []
        for _ in range(N):
            cx = random.uniform(0.05 * Ws, 0.95 * Ws)
            cy = random.uniform(0.05 * Hs, 0.95 * Hs)
            pts.append([cx, cy, 0.03 * min(Ws, Hs)])
        seeds.append(pts)

    for s in seeds:
        if time_limit is not None and time.time() > time_limit:
            break
        packed = _relax_pack(s, W, H, max_iter=2500, time_limit=time_limit)
        add(packed)

    # (b) grid layouts (guaranteed feasible, diverse aspect handling)
    for kx, ky in [(3, 4), (4, 3), (3, 3), (4, 4), (5, 4), (4, 5), (5, 3), (3, 5)]:
        if kx * ky < N:
            continue
        cx = np.linspace(0.0, Ws, kx + 2)[1:-1]
        cy = np.linspace(0.0, Hs, ky + 2)[1:-1]
        pts = []
        for yy in cy:
            for xx in cx:
                r = min(xx, yy, Ws - xx, Hs - yy) * 0.99
                pts.append([xx, yy, max(r, 1e-4)])
        add(pts[:N])

    # (c) pure random
    for _ in range(2):
        pts = []
        for _ in range(N):
            cx = random.uniform(0.05 * Ws, 0.95 * Ws)
            cy = random.uniform(0.05 * Hs, 0.95 * Hs)
            r = min(cx, cy, Ws - cx, Hs - cy) * 0.3
            pts.append([cx, cy, max(r, 1e-4)])
        add(pts)

    return layouts


# --------------------------------------------------------------------------- #
# SLSQP optimizer
# --------------------------------------------------------------------------- #
def _optimize_from(x0, W, H, max_iter=200, time_limit=None):
    best_x = x0.copy()
    best_sum = _evaluate(x0, W, H) or 0.0

    def fun(x):
        return _obj_vec(x, W, H)[0]

    def grad(x):
        return _obj_vec(x, W, H)[1]

    try:
        res = minimize(fun, x0, jac=grad, method='SLSQP',
                       options={'maxiter': max_iter, 'ftol': 1e-10, 'disp': 0})
        if res.x is not None:
            s = _evaluate(res.x, W, H)
            if s is not None and s > best_sum:
                best_sum, best_x = s, res.x
    except Exception:
        pass

    if time_limit is None or time.time() < time_limit:
        try:
            res2 = minimize(fun, best_x, jac=grad, method='SLSQP',
                            options={'maxiter': 120, 'ftol': 1e-12, 'disp': 0})
            if res2.x is not None:
                s = _evaluate(res2.x, W, H)
                if s is not None and s > best_sum:
                    best_sum, best_x = s, res2.x
        except Exception:
            pass

    return best_x, best_sum


# --------------------------------------------------------------------------- #
# Local-search grow loop (inspiration's local_search idea)
# --------------------------------------------------------------------------- #
def _local_grow(arr, W, H, time_limit=None):
    """Try to grow each circle, fixing feasibility by shrinking others / nudging."""
    cx = arr[:, 0].copy()
    cy = arr[:, 1].copy()
    r = arr[:, 2].copy()
    n = len(cx)
    improved = True
    while improved and (time_limit is None or time.time() < time_limit):
        improved = False
        for i in range(n):
            base = r[i]
            lo = min(cx[i], cy[i], W - cx[i], H - cy[i])
            lo = max(lo, 1e-6)
            for _ in range(40):
                trial = min(base * 1.01, lo)
                if trial <= r[i] + 1e-9:
                    break
                cand = r.copy()
                cand[i] = trial
                # quick feasibility
                ok = True
                for j in range(n):
                    if j == i:
                        continue
                    d = np.hypot(cx[i] - cx[j], cy[i] - cy[j])
                    if d + 1e-8 < cand[i] + cand[j]:
                        ok = False
                        break
                if ok:
                    r[i] = trial
                    improved = True
                else:
                    break
    return np.column_stack([cx, cy, r])


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def construct():
    random.seed(42)
    np.random.seed(42)
    best_sum = 0.0
    best_arr = None
    best_W, best_H = 1.0, 1.0

    def time_ok():
        return time.time() < DEADLINE

    Ws = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]

    # Build a global pool of (score, W, H, layout) initial candidates.
    pool = []
    for W in Ws:
        H = 2.0 - W
        if not time_ok():
            break
        layouts = _initial_layout(W, H, time_limit=DEADLINE - 2.0)
        for s, arr in layouts:
            pool.append((s, W, H, arr))

    pool.sort(key=lambda t: -t[0])

    # Optimize the top candidates with SLSQP.
    budget = 14
    for idx in range(min(budget, len(pool))):
        if not time_ok():
            break
        s0, W, H, g0 = pool[idx]
        remaining = DEADLINE - time.time()
        max_iter = 160 if remaining > 70 else 70
        x_opt, s_opt = _optimize_from(g0, W, H, max_iter=max_iter,
                                      time_limit=DEADLINE - 3.0)
        if s_opt > best_sum:
            best_sum, best_arr, best_W, best_H = s_opt, x_opt, W, H

    # Local-search grow on the best found.
    if time_ok() and best_arr is not None:
        grown = _local_grow(best_arr, best_W, best_H, time_limit=DEADLINE - 4.0)
        s = _evaluate(grown, best_W, best_H)
        if s is not None and s > best_sum:
            best_sum, best_arr = s, grown

    # Perturbation restarts from the best.
    if time_ok() and best_arr is not None:
        remaining = DEADLINE - time.time()
        n_extra = min(8, int(remaining // 9))
        for _ in range(n_extra):
            if not time_ok():
                break
            x0 = best_arr + np.random.randn(*best_arr.shape) * 0.02
            x0[:, 0] = np.clip(x0[:, 0], 0.0, best_W)
            x0[:, 1] = np.clip(x0[:, 1], 0.0, best_H)
            x0[:, 2] = np.maximum(x0[:, 2], 1e-4)
            x_opt, s_opt = _optimize_from(x0, best_W, best_H, max_iter=60,
                                          time_limit=DEADLINE - 3.0)
            if s_opt > best_sum:
                best_sum, best_arr = s_opt, x_opt

    # Fallback.
    if best_arr is None:
        W, H = 1.0, 1.0
        cx = np.linspace(0.0, W, 5 + 2)[1:-1]
        cy = np.linspace(0.0, H, 5 + 2)[1:-1]
        pts = []
        for yy in cy:
            for xx in cx:
                r = min(xx, yy, W - xx, H - yy) * 0.99
                pts.append([xx, yy, max(r, 1e-4)])
        best_arr = np.array(pts[:N])
        best_W, best_H = W, H

    # Final clamping to guarantee validity.
    cx = np.clip(best_arr[:, 0], 0.0, best_W)
    cy = np.clip(best_arr[:, 1], 0.0, best_H)
    r = best_arr[:, 2].copy()
    lo = np.minimum(np.minimum(cx, cy), np.minimum(best_W - cx, best_H - cy))
    r = np.clip(np.minimum(r, lo), 1e-6, None)
    out = np.column_stack([cx, cy, r])

    s_final = _evaluate(out, best_W, best_H)
    if s_final is None:
        for s0, W, H, g0 in pool[:5]:
            cx2 = np.clip(g0[:, 0], 0.0, W)
            cy2 = np.clip(g0[:, 1], 0.0, H)
            r2 = g0[:, 2].copy()
            lo2 = np.minimum(np.minimum(cx2, cy2), np.minimum(W - cx2, H - cy2))
            r2 = np.clip(np.minimum(r2, lo2), 1e-6, None)
            out2 = np.column_stack([cx2, cy2, r2])
            if _evaluate(out2, W, H) is not None:
                return out2
        side = 5
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-4)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    return out
