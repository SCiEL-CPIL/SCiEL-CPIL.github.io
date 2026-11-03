"""Circle packing in perimeter-4 rectangle: differentiable joint optimization."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21
DEADLINE = 140.0  # safety margin before 150s kill


def _soft_min(x, beta):
    """Smooth minimum via log-sum-exp."""
    x = np.asarray(x, dtype=np.float64)
    x = np.clip(x, -500, 500)
    m = np.min(x)
    if m < -400:
        return m
    s = np.sum(np.exp(-beta * (x - m)))
    return m - np.log(s) / beta


def _soft_max(x, beta):
    return -_soft_min(-np.asarray(x, dtype=np.float64), beta)


def _check_valid(circles, W, H, tol=1e-6):
    """Return (is_valid, min_margin). circles: (N,3) [x,y,r]."""
    circles = np.asarray(circles, dtype=np.float64)
    n = len(circles)
    if n != N:
        return False, -1.0
    if W <= 0 or H <= 0:
        return False, -1.0
    # perimeter
    perim = 2.0 * (W + H)
    if perim > 4.0 + 1e-5:
        return False, -1.0
    # bounds
    for i in range(n):
        x, y, r = circles[i]
        if r < -1e-9:
            return False, -1.0
        if x - r < -tol or x + r > W + tol:
            return False, -1.0
        if y - r < -tol or y + r > H + tol:
            return False, -1.0
    # pairwise
    for i in range(n):
        for j in range(i + 1, n):
            xi, yi, ri = circles[i]
            xj, yj, rj = circles[j]
            d = np.hypot(xi - xj, yi - yj)
            if d < ri + rj - tol:
                return False, -1.0
    # compute actual min margin for diagnostics
    min_margin = 1e18
    for i in range(n):
        x, y, r = circles[i]
        min_margin = min(min_margin, x - r, W - (x + r), y - r, H - (y + r))
    for i in range(n):
        for j in range(i + 1, N):
            xi, yi, ri = circles[i]
            xj, yj, rj = circles[j]
            d = np.hypot(xi - xj, yi - yj)
            min_margin = min(min_margin, d - (ri + rj))
    return True, min_margin


def _init_hex(W, H):
    """Hexagonal close-packed initialization for N circles in W x H rectangle."""
    # target radius
    r0 = min(W, H) / 6.0
    # hex grid
    dx = 2.0 * r0
    dy = np.sqrt(3.0) * r0
    rows = int(np.ceil(H / dy)) + 2
    cols = int(np.ceil(W / dx)) + 2
    centers = []
    for row in range(rows):
        y = dy * (row + 0.5)
        offset = (dx / 2.0) if (row % 2) else 0.0
        for col in range(cols):
            x = dx * (col + 0.5) + offset
            if 0 <= x <= W and 0 <= y <= H:
                centers.append((x, y))
    # sort by distance to center, take N
    cx, cy = W / 2.0, H / 2.0
    centers.sort(key=lambda p: (p[0] - cx) ** 2 + (p[1] - cy) ** 2)
    centers = centers[:N]
    # initial radius: min distance to boundary / 2, and min pairwise / 2
    rs = []
    for (x, y) in centers:
        d_bound = min(x, W - x, y, H - y)
        rs.append(d_bound * 0.4)
    rs = np.array(rs)
    # reduce to ensure pairwise
    for _ in range(50):
        ok = True
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(centers[i][0] - centers[j][0], centers[i][1] - centers[j][1])
                need = (rs[i] + rs[j]) * 1.05
                if d < need:
                    ok = False
                    scale = d / need * 0.95
                    rs[i] *= scale
                    rs[j] *= scale
        if ok:
            break
    circles = np.array([[cx_, cy_, r] for (cx_, cy_), r in zip(centers, rs)])
    return circles


def _objective_and_grad(z, W, H, beta, mu):
    """z = [x1,y1,r1, ..., xN,yN,rN]. Minimize -sum(r) with smooth constraints."""
    n = N
    z = np.asarray(z, dtype=np.float64)
    xs = z[0::3]
    ys = z[1::3]
    rs = z[2::3]

    # smooth objective: -sum(r)
    f = -np.sum(rs)

    # gradient of objective
    g = np.zeros_like(z)
    g[2::3] = -1.0

    # constraint: bounds (soft)
    # x - r >= 0, W - x - r >= 0, y - r >= 0, H - y - r >= 0
    c1 = xs - rs
    c2 = W - xs - rs
    c3 = ys - rs
    c4 = H - ys - rs
    # soft penalty: -mu * sum(soft_min(c, beta))
    for c, idx_x, idx_r, sign_x, sign_r in [
        (c1, 0, 2, 1, -1),
        (c2, 0, 2, -1, -1),
        (c3, 1, 2, 1, -1),
        (c4, 1, 2, -1, -1),
    ]:
        c_clip = np.clip(c, -500, 500)
        m = np.min(c_clip)
        if m > -400:
            exps = np.exp(-beta * (c_clip - m))
            s = np.sum(exps)
            f += -mu * (m - np.log(s) / beta)
            # gradient of soft_min: -exps/s
            grad_sm = -exps / s
            g[idx_x] += mu * grad_sm * sign_x
            g[idx_r] += mu * grad_sm * sign_r

    # pairwise: dist - (ri + rj) >= 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            d = np.hypot(dx, dy)
            if d < 1e-12:
                d = 1e-12
            c = d - (rs[i] + rs[j])
            c_clip = np.clip(c, -500, 500)
            m = c_clip
            if m > -400:
                exps = np.exp(-beta * (c_clip - m))
                s = np.sum(exps)
                if s < 1e-300:
                    s = 1e-300
                f += -mu * (m - np.log(s) / beta)
                grad_sm = -exps / s
                # d(c)/dx_i = dx/d, d(c)/dx_j = -dx/d
                g[3 * i] += mu * grad_sm * (dx / d)
                g[3 * i + 1] += mu * grad_sm * (dy / d)
                g[3 * j] += mu * grad_sm * (-dx / d)
                g[3 * j + 1] += mu * grad_sm * (-dy / d)
                g[3 * i + 2] += mu * grad_sm * (-1.0)
                g[3 * j + 2] += mu * grad_sm * (-1.0)

    return f, g


def _solve_for_aspect(W, H, max_iter, beta_start, beta_end, mu_start, mu_end, time_budget):
    """Optimize circle positions/radii for fixed W, H. Returns (circles, value)."""
    t0 = time.time()
    circles = _init_hex(W, H)
    z = circles.reshape(-1)
    best_z = z.copy()
    best_val = np.sum(z[2::3])
    best_valid, _ = _check_valid(best_z.reshape(-1, 3), W, H)
    if not best_valid:
        # shrink to make valid
        for _ in range(100):
            valid, _ = _check_valid(best_z.reshape(-1, 3), W, H)
            if valid:
                break
            best_z[2::3] *= 0.9
        best_val = np.sum(best_z[2::3])

    n_iter = 0
    while n_iter < max_iter:
        if time.time() - t0 > time_budget:
            break
        # anneal beta and mu
        frac = n_iter / max(1, max_iter - 1)
        beta = beta_start * (beta_end / beta_start) ** frac
        mu = mu_start * (mu_end / mu_start) ** frac
        try:
            res = minimize(
                _objective_and_grad,
                z,
                args=(W, H, beta, mu),
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 30, 'gtol': 1e-8, 'ftol': 1e-12},
            )
            z = res.x
        except Exception:
            pass
        # project: clip to bounds
        xs = z[0::3]; ys = z[1::3]; rs = z[2::3]
        rs = np.clip(rs, 1e-6, None)
        xs = np.clip(xs, rs, W - rs)
        ys = np.clip(ys, rs, H - rs)
        z = np.empty_like(z)
        z[0::3] = xs; z[1::3] = ys; z[2::3] = rs
        n_iter += 1
        val = np.sum(z[2::3])
        valid, _ = _check_valid(z.reshape(-1, 3), W, H)
        if valid and val > best_val:
            best_val = val
            best_z = z.copy()
            best_valid = True
    return best_z.reshape(-1, 3), best_val


def _polish(circles, W, H, iters=100, tol=1e-10):
    """High-precision local polish: alternate radius update and position Nesterov-like."""
    circles = circles.copy()
    for _ in range(iters):
        # update radii: r_i = min over constraints
        new_r = np.empty(N)
        for i in range(N):
            x, y, r = circles[i]
            d_bound = min(x, W - x, y, H - y)
            d_pair = np.inf
            for j in range(N):
                if i == j:
                    continue
                xj, yj, rj = circles[j]
                d = np.hypot(x - xj, y - yj)
                d_pair = min(d_pair, d - rj)
            new_r[i] = max(1e-9, min(d_bound, d_pair))
        # move positions toward free space (gradient ascent of min margin)
        for i in range(N):
            x, y, r = circles[i]
            # gradient of min-margin w.r.t. x_i, y_i
            # use all constraints
            gx = 0.0; gy = 0.0
            # boundary
            if x - r == min(x, W - x, y, H - y):
                gx -= 1.0
            if W - x - r == min(x, W - x, y, H - y):
                gx += 1.0
            if y - r == min(x, W - x, y, H - y):
                gy -= 1.0
            if H - y - r == min(x, W - x, y, H - y):
                gy += 1.0
            # pairwise
            for j in range(N):
                if i == j:
                    continue
                xj, yj, rj = circles[j]
                d = np.hypot(x - xj, y - yj)
                if d < 1e-12:
                    d = 1e-12
                if d - (r + rj) == min([x - r, W - x - r, y - r, H - y - r,
                                         min([np.hypot(x - circles[k][0], y - circles[k][1]) - (r + circles[k][2]) for k in range(N) if k != i])]):
                    gx += (x - xj) / d
                    gy += (y - yj) / d
            step = 1e-4
            x_new = np.clip(x + step * gx, r, W - r)
            y_new = np.clip(y + step * gy, r, H - r)
            circles[i] = [x_new, y_new, r]
        # recompute radii
        for i in range(N):
            x, y, r = circles[i]
            d_bound = min(x, W - x, y, H - y)
            d_pair = np.inf
            for j in range(N):
                if i == j:
                    continue
                xj, yj, rj = circles[j]
                d = np.hypot(x - xj, y - yj)
                d_pair = min(d_pair, d - rj)
            circles[i][2] = max(1e-9, min(d_bound, d_pair))
    return circles


def construct():
    t_start = time.time()
    deadline = t_start + DEADLINE

    best_circles = None
    best_val = 0.0
    best_W = best_H = 1.0

    # initial grid solution as fallback
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
    init_circles = np.array(cells[:N])
    # scale to perimeter 4 square: side=1
    init_circles = init_circles * 1.0
    valid, _ = _check_valid(init_circles, 1.0, 1.0)
    if valid:
        best_circles = init_circles.copy()
        best_val = np.sum(init_circles[:, 2])
        best_W, best_H = 1.0, 1.0

    # aspect ratio sweep: W from 0.5 to 1.5 (H = 2 - W for perimeter 4)
    # sample aspects
    aspects = np.linspace(0.4, 1.6, 25)
    # sort by expected quality (near square first)
    aspects = sorted(aspects, key=lambda a: abs(a - 1.0))

    for W in aspects:
        if time.time() > deadline - 20:
            break
        H = 2.0 - W
        if H <= 0.1:
            continue
        time_budget = min(15.0, deadline - time.time() - 10)
        if time_budget < 2:
            break
        try:
            circles, val = _solve_for_aspect(
                W, H,
                max_iter=40,
                beta_start=50.0,
                beta_end=5000.0,
                mu_start=1.0,
                mu_end=1e4,
                time_budget=time_budget,
            )
            valid, _ = _check_valid(circles, W, H)
            if valid and val > best_val:
                best_val = val
                best_circles = circles.copy()
                best_W, best_H = W, H
        except Exception:
            continue

    # polish the best
    if best_circles is not None and time.time() < deadline - 5:
        try:
            polished = _polish(best_circles, best_W, best_H, iters=200, tol=1e-10)
            valid, _ = _check_valid(polished, best_W, best_H)
            if valid and np.sum(polished[:, 2]) > best_val:
                best_val = np.sum(polished[:, 2])
                best_circles = polished
        except Exception:
            pass

    # final validation; if invalid, fall back to grid
    if best_circles is None:
        best_circles = init_circles
        best_W, best_H = 1.0, 1.0
    else:
        valid, _ = _check_valid(best_circles, best_W, best_H)
        if not valid:
            # shrink
            for _ in range(200):
                valid, _ = _check_valid(best_circles, best_W, best_H)
                if valid:
                    break
                best_circles[:, 2] *= 0.9
            if not valid:
                best_circles = init_circles
                best_W, best_H = 1.0, 1.0

    return best_circles


if __name__ == '__main__':
    res = construct()
    print(res)
