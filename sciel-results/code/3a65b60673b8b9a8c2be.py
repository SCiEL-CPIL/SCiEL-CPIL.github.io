"""Circle packing in perimeter-4 rectangle: unified free-aspect-ratio penalty
optimization + force-directed relaxation + radius-growth polish."""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 21
START = time.time()
DEADLINE = START + 135.0  # safety margin before 150s kill

# ------------------- evaluation -------------------

def _evaluate(x, W, H):
    """Return sum of radii if x is a valid packing, else None.
    x: (n,3) with [cx, cy, r]."""
    cx = np.clip(x[:, 0], 0.0, W)
    cy = np.clip(x[:, 1], 0.0, H)
    r = np.maximum(x[:, 2], 1e-6)
    lo = np.minimum(np.minimum(cx, cy), np.minimum(W - cx, H - cy))
    r = np.minimum(r, lo)
    r = np.maximum(r, 1e-6)
    n = len(cx)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(cx[i] - cx[j], cy[i] - cy[j])
            if d + 1e-7 < r[i] + r[j]:
                return None
    return float(np.sum(r))


# ------------------- penalty objective + gradient -------------------

def _penalty_and_gradient(x, W, H, penalty_weight=100.0):
    """Smooth penalty objective + analytic gradient.
    Objective: -sum(r) + penalty_weight * (sum overlap^2 + sum boundary^2)."""
    cx = np.clip(x[:, 0], 0.0, W)
    cy = np.clip(x[:, 1], 0.0, H)
    r = np.maximum(x[:, 2], 1e-6)
    lo = np.minimum(np.minimum(cx, cy), np.minimum(W - cx, H - cy))
    r = np.minimum(r, lo)
    r = np.maximum(r, 1e-6)
    n = len(cx)

    penalty = 0.0
    grad = np.zeros_like(x)
    grad[:, 2] -= 1.0  # -sum(r)

    # overlaps
    for i in range(n):
        for j in range(i + 1, n):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            d = np.sqrt(dx * dx + dy * dy + 1e-12)
            req = r[i] + r[j]
            if d < req:
                ov = req - d
                penalty += ov * ov
                if d > 1e-10:
                    f = -2.0 * ov
                    grad[i, 0] += penalty_weight * f * (dx / d)
                    grad[i, 1] += penalty_weight * f * (dy / d)
                    grad[j, 0] += penalty_weight * f * (-dx / d)
                    grad[j, 1] += penalty_weight * f * (-dy / d)
                    grad[i, 2] += penalty_weight * (-2.0 * ov)
                    grad[j, 2] += penalty_weight * (-2.0 * ov)

    # boundary
    for i in range(n):
        dl, dr = cx[i], W - cx[i]
        db, dt = cy[i], H - cy[i]
        md = min(dl, dr, db, dt)
        if r[i] > md:
            v = r[i] - md
            penalty += v * v
            g = penalty_weight * 2.0 * v
            grad[i, 2] += g
            if md == dl:
                grad[i, 0] -= g
            elif md == dr:
                grad[i, 0] += g
            elif md == db:
                grad[i, 1] -= g
            else:
                grad[i, 1] += g

    return -float(np.sum(r)) + penalty_weight * penalty, grad


# ------------------- force-directed relaxation -------------------

def _force_relax(x, W, H, iters=80, dt=0.004):
    """Resolve overlaps/boundary violations via pairwise + wall repulsion."""
    x = x.copy().astype(float)
    n = N
    for _ in range(iters):
        if time.time() > DEADLINE - 5:
            break
        cx, cy, r = x[:, 0], x[:, 1], x[:, 2]
        fx = np.zeros(n)
        fy = np.zeros(n)
        for i in range(n):
            if cx[i] - r[i] < 0:
                fx[i] += (cx[i] - r[i]) * 20
            if cx[i] + r[i] > W:
                fx[i] -= (cx[i] + r[i] - W) * 20
            if cy[i] - r[i] < 0:
                fy[i] += (cy[i] - r[i]) * 20
            if cy[i] + r[i] > H:
                fy[i] -= (cy[i] + r[i] - H) * 20
        for i in range(n):
            for j in range(i + 1, n):
                dx = cx[i] - cx[j]
                dy = cy[i] - cy[j]
                d = np.sqrt(dx * dx + dy * dy)
                md = r[i] + r[j]
                if d < md and d > 1e-12:
                    ov = md - d
                    fx[i] += (dx / d) * ov * 10
                    fy[i] += (dy / d) * ov * 10
                    fx[j] -= (dx / d) * ov * 10
                    fy[j] -= (dy / d) * ov * 10
        x[:, 0] = np.clip(x[:, 0] + dt * fx, 0.0, W)
        x[:, 1] = np.clip(x[:, 1] + dt * fy, 0.0, H)
        x[:, 2] = np.maximum(x[:, 2], 1e-6)
        lo = np.minimum(np.minimum(x[:, 0], x[:, 1]), np.minimum(W - x[:, 0], H - x[:, 1]))
        x[:, 2] = np.minimum(x[:, 2], lo)
        x[:, 2] = np.maximum(x[:, 2], 1e-6)
    return x


# ------------------- radius-growth polish -------------------

def _radius_growth_polish(x, W, H, time_budget=8.0):
    """Grow each radius to local slack, re-relax positions, repeat."""
    x = x.copy().astype(float)
    n = N
    start = time.time()
    for outer in range(40):
        if time.time() - start > time_budget:
            break
        cx, cy, r = x[:, 0].copy(), x[:, 1].copy(), x[:, 2].copy()
        grew = False
        for i in range(n):
            slack = min(cx[i] - r[i], W - cx[i] - r[i], cy[i] - r[i], H - cy[i] - r[i])
            for j in range(n):
                if i == j:
                    continue
                d = np.hypot(cx[i] - cx[j], cy[i] - cy[j])
                slack = min(slack, d - r[i] - r[j])
            if slack > 1e-7:
                r[i] += min(0.002, slack * 0.5)
                grew = True
        x[:, 0], x[:, 1], x[:, 2] = cx, cy, r
        x = _force_relax(x, W, H, iters=25, dt=0.001)
        if not grew:
            break
    return x


# ------------------- L-BFGS-B on penalty objective -------------------

def _lbfgsb_optimize(x0, W, H, max_iter=250):
    def og(flat):
        x = flat.reshape(N, 3)
        x[:, 0] = np.clip(x[:, 0], 0.0, W)
        x[:, 1] = np.clip(x[:, 1], 0.0, H)
        x[:, 2] = np.maximum(x[:, 2], 1e-6)
        lo = np.minimum(np.minimum(x[:, 0], x[:, 1]), np.minimum(W - x[:, 0], H - x[:, 1]))
        x[:, 2] = np.minimum(x[:, 2], lo)
        x[:, 2] = np.maximum(x[:, 2], 1e-6)
        obj, grad = _penalty_and_gradient(x, W, H, penalty_weight=400.0)
        return obj, grad.flatten()

    try:
        res = minimize(lambda x: og(x)[0], x0.flatten().copy(),
                       jac=lambda x: og(x)[1], method='L-BFGS-B',
                       options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-6})
        x_opt = res.x.reshape(N, 3)
        x_opt = _force_relax(x_opt, W, H, iters=30, dt=0.002)
        s = _evaluate(x_opt, W, H)
        if s is not None:
            return x_opt, s
    except Exception:
        pass
    return x0, _evaluate(x0, W, H) or 0.0


# ------------------- gradient descent -------------------

def _gradient_descent(x0, W, H, max_iter=400, lr=0.01, penalty_weight=50.0):
    x = x0.copy()
    best_x = x.copy()
    best_sum = _evaluate(x, W, H) or 0.0
    for _ in range(max_iter):
        if time.time() > DEADLINE - 3:
            break
        obj, grad = _penalty_and_gradient(x, W, H, penalty_weight)
        x_new = x - lr * grad
        x_new[:, 0] = np.clip(x_new[:, 0], 0.0, W)
        x_new[:, 1] = np.clip(x_new[:, 1], 0.0, H)
        x_new[:, 2] = np.maximum(x_new[:, 2], 1e-6)
        lo = np.minimum(np.minimum(x_new[:, 0], x_new[:, 1]), np.minimum(W - x_new[:, 0], H - x_new[:, 1]))
        x_new[:, 2] = np.minimum(x_new[:, 2], lo)
        x_new[:, 2] = np.maximum(x_new[:, 2], 1e-6)
        s = _evaluate(x_new, W, H)
        if s is not None and s > best_sum:
            best_sum, best_x = s, x_new.copy()
        obj_new, _ = _penalty_and_gradient(x_new, W, H, penalty_weight)
        lr = min(0.05, max(1e-5, lr * (1.01 if obj_new < obj else 0.9)))
        if obj_new < obj:
            x = x_new
    return best_x, best_sum


# ------------------- simulated annealing -------------------

def _sa_exploration(x0, W, H, steps=1200, T0=0.05, T1=0.001):
    cx, cy, r = x0[:, 0].copy(), x0[:, 1].copy(), x0[:, 2].copy()
    best = _evaluate(np.column_stack([cx, cy, r]), W, H) or 0.0
    bcx, bcy, br = cx.copy(), cy.copy(), r.copy()
    cur = best
    n = N
    for step in range(steps):
        if time.time() > DEADLINE - 3:
            break
        T = T0 * (T1 / T0) ** (step / steps)
        ncx, ncy, nr = cx.copy(), cy.copy(), r.copy()
        for _ in range(random.randint(1, 2)):
            i = random.randint(0, n - 1)
            ncx[i] += random.gauss(0, 0.03)
            ncy[i] += random.gauss(0, 0.03)
            if random.random() < 0.4:
                nr[i] = max(1e-6, nr[i] * random.uniform(0.85, 1.15))
        ncx = np.clip(ncx, 0.0, W)
        ncy = np.clip(ncy, 0.0, H)
        nr = np.maximum(nr, 1e-6)
        lo = np.minimum(np.minimum(ncx, ncy), np.minimum(W - ncx, H - ncy))
        nr = np.minimum(nr, lo)
        nr = np.maximum(nr, 1e-6)
        s = _evaluate(np.column_stack([ncx, ncy, nr]), W, H)
        if s is not None:
            if s > best:
                best, bcx, bcy, br = s, ncx.copy(), ncy.copy(), nr.copy()
            if s >= cur:
                cx, cy, r, cur = ncx, ncy, nr, s
            elif random.random() < np.exp(-(cur - s) / max(T, 1e-8)):
                cx, cy, r, cur = ncx, ncy, nr, s
    return np.column_stack([bcx, bcy, br]), best


# ------------------- initial layouts -------------------

def _grid_layout(kx, ky, W, H, shrink=0.95):
    cx = np.linspace(0.0, W, kx + 2)[1:-1]
    cy = np.linspace(0.0, H, ky + 2)[1:-1]
    pts = []
    for yy in cy:
        for xx in cx:
            r = max(min(xx, yy, W - xx, H - yy) * shrink, 1e-4)
            pts.append([xx, yy, r])
    arr = np.array(pts)
    if len(arr) < N:
        arr = np.vstack([arr, np.tile([0.01, 0.01, 0.005], (N - len(arr), 1))])
    return arr[:N]


def _honeycomb_layout(W, H):
    pts = []
    base_r = min(W, H) / 8.0
    rows = int(H / (base_r * 1.732)) + 2
    cols = int(W / (base_r * 1.5)) + 2
    for row in range(rows):
        y = row * base_r * 1.732
        if y > H:
            break
        off = (base_r * 1.5) if (row % 2 == 1) else 0
        for col in range(cols):
            x = col * base_r * 1.5 + off
            if x > W:
                break
            r = max(min(x, y, W - x, H - y) * 0.9, 1e-4)
            pts.append([x, y, r])
    arr = np.array(pts)
    if len(arr) < N:
        arr = np.vstack([arr, np.tile([0.01, 0.01, 0.005], (N - len(arr), 1))])
    return arr[:N]


def _random_layout(W, H):
    pts = []
    for _ in range(N):
        cx = random.uniform(0.1 * W, 0.9 * W)
        cy = random.uniform(0.1 * H, 0.9 * H)
        r = max(min(cx, cy, W - cx, H - cy) * 0.4, 1e-4)
        pts.append([cx, cy, r])
    return np.array(pts)


def construct():
    best_sum = 0.0
    best_arr = None
    best_W = best_H = 1.0

    def time_ok():
        return time.time() < DEADLINE - 3

    random.seed(42)
    np.random.seed(42)

    candidates = []
    Ws = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
    grid_sizes = [(3, 4), (4, 3), (3, 3), (4, 4), (5, 3), (3, 5)]
    for W in Ws:
        H = 2.0 - W
        if H < 0.3 or W < 0.3:
            continue
        for kx, ky in grid_sizes:
            if kx * ky < N:
                continue
            g = _force_relax(_grid_layout(kx, ky, W, H, 0.95), W, H, iters=40)
            s = _evaluate(g, W, H)
            if s is not None:
                candidates.append((s, W, H, g))
        h = _force_relax(_honeycomb_layout(W, H), W, H, iters=40)
        sh = _evaluate(h, W, H)
        if sh is not None:
            candidates.append((sh, W, H, h))
        for _ in range(2):
            g = _force_relax(_random_layout(W, H), W, H, iters=40)
            s = _evaluate(g, W, H)
            if s is not None:
                candidates.append((s, W, H, g))

    candidates.sort(key=lambda t: -t[0])

    # Phase 1: L-BFGS-B + polish on top candidates
    top_k = min(8, len(candidates))
    for idx in range(top_k):
        if not time_ok():
            break
        s0, W, H, g0 = candidates[idx]
        x_opt, s_opt = _lbfgsb_optimize(g0, W, H, max_iter=200)
        if s_opt > best_sum:
            best_sum, best_arr, best_W, best_H = s_opt, x_opt, W, H
        # radius-growth polish on the current best
        if time_ok() and best_arr is not None:
            xp, sp = _radius_growth_polish(best_arr, best_W, best_H, time_budget=4.0)
            sp = _evaluate(xp, best_W, best_H) or sp
            if sp > best_sum:
                best_sum, best_arr = sp, xp

    # Phase 2: SA + L-BFGS-B cycles
    if time_ok() and best_arr is not None:
        x_sa, s_sa = _sa_exploration(best_arr, best_W, best_H, steps=1200, T0=0.05, T1=0.001)
        if s_sa > best_sum:
            best_sum, best_arr = s_sa, x_sa

    # Phase 3: extra SA + L-BFGS-B + polish cycles
    if time_ok() and best_arr is not None:
        remaining = DEADLINE - time.time()
        n_extra = min(4, int(remaining // 15))
        for _ in range(n_extra):
            if not time_ok():
                break
            x0 = best_arr + np.random.randn(*best_arr.shape) * 0.02
            x0[:, 0] = np.clip(x0[:, 0], 0.0, best_W)
            x0[:, 1] = np.clip(x0[:, 1], 0.0, best_H)
            x0[:, 2] = np.maximum(x0[:, 2], 1e-4)
            x_sa, s_sa = _sa_exploration(x0, best_W, best_H, steps=700, T0=0.04, T1=0.001)
            if s_sa > best_sum:
                best_sum, best_arr = s_sa, x_sa
            if time_ok():
                x_opt, s_opt = _lbfgsb_optimize(best_arr, best_W, best_H, max_iter=100)
                if s_opt > best_sum:
                    best_sum, best_arr = s_opt, x_opt
            if time_ok():
                xp, sp = _radius_growth_polish(best_arr, best_W, best_H, time_budget=3.0)
                sp = _evaluate(xp, best_W, best_H) or sp
                if sp > best_sum:
                    best_sum, best_arr = sp, xp

    # Final polish
    if time_ok() and best_arr is not None:
        xp, sp = _radius_growth_polish(best_arr, best_W, best_H, time_budget=5.0)
        sp = _evaluate(xp, best_W, best_H) or sp
        if sp > best_sum:
            best_sum, best_arr = sp, xp

    if best_arr is None:
        W, H = 1.0, 1.0
        best_arr = _grid_layout(5, 5, W, H, 0.95)
        best_W, best_H = W, H

    # Final clamping + validation
    cx = np.clip(best_arr[:, 0], 0.0, best_W)
    cy = np.clip(best_arr[:, 1], 0.0, best_H)
    r = np.maximum(best_arr[:, 2], 1e-6)
    lo = np.minimum(np.minimum(cx, cy), np.minimum(best_W - cx, best_H - cy))
    r = np.minimum(r, lo)
    r = np.maximum(r, 1e-6)
    out = np.column_stack([cx, cy, r])
    if _evaluate(out, best_W, best_H) is None:
        side = 5
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-4)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    return out
