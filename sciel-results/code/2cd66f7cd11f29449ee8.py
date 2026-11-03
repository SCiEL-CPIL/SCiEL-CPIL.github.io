import numpy as np
import time
import random
from scipy.optimize import minimize, linprog

N = 21
TIME_LIMIT = 140.0
START = time.time()

def _elapsed():
    return time.time() - START

def _time_ok(margin=4.0):
    return _elapsed() < TIME_LIMIT - margin

def _softplus(x, beta=100.0):
    xb = beta * x
    return np.where(xb > 0.0,
                    x + np.log1p(np.exp(-xb)) / beta,
                    np.log1p(np.exp(xb)) / beta)

def _softplus_grad(x, beta=100.0):
    xb = beta * x
    return np.where(xb > 0.0,
                    1.0 / (1.0 + np.exp(-xb)),
                    np.exp(xb) / (1.0 + np.exp(xb)))

def _objective_joint(state, beta=100.0):
    t = np.clip(state[0], 0.15, 0.85)
    w = 2.0 * t
    h = 2.0 * (1.0 - t)

    xs = state[1::3]
    ys = state[2::3]
    rs = state[3::3]

    obj = -np.sum(rs)
    grad = np.zeros_like(state)

    v1 = rs - xs
    obj += np.sum(_softplus(v1, beta))
    g1 = _softplus_grad(v1, beta)
    grad[1::3] -= g1
    grad[3::3] += g1

    v2 = xs + rs - w
    obj += np.sum(_softplus(v2, beta))
    g2 = _softplus_grad(v2, beta)
    grad[1::3] += g2
    grad[3::3] += g2

    v3 = rs - ys
    obj += np.sum(_softplus(v3, beta))
    g3 = _softplus_grad(v3, beta)
    grad[2::3] -= g3
    grad[3::3] += g3

    v4 = ys + rs - h
    obj += np.sum(_softplus(v4, beta))
    g4 = _softplus_grad(v4, beta)
    grad[2::3] += g4
    grad[3::3] += g4

    v5 = -rs
    obj += np.sum(_softplus(v5, beta))
    g5 = _softplus_grad(v5, beta)
    grad[3::3] -= g5

    dx = xs[:, np.newaxis] - xs[np.newaxis, :]
    dy = ys[:, np.newaxis] - ys[np.newaxis, :]
    d = np.sqrt(dx * dx + dy * dy + 1e-12)
    r_sum = rs[:, np.newaxis] + rs[np.newaxis, :]
    v = r_sum - d
    iu = np.triu_indices(N, k=1)
    v_upper = v[iu]
    d_upper = d[iu]
    dx_upper = dx[iu]
    dy_upper = dy[iu]

    obj += np.sum(_softplus(v_upper, beta))
    g = _softplus_grad(v_upper, beta)

    idx_i = iu[0]
    idx_j = iu[1]
    for k in range(len(idx_i)):
        i = idx_i[k]
        j = idx_j[k]
        gk = g[k]
        dk = d_upper[k]
        dxk = dx_upper[k]
        dyk = dy_upper[k]

        grad[3 * i + 1] += -gk * dxk / dk
        grad[3 * i + 2] += -gk * dyk / dk
        grad[3 * i + 3] += gk

        grad[3 * j + 1] += gk * dxk / dk
        grad[3 * j + 2] += gk * dyk / dk
        grad[3 * j + 3] += gk

    grad[3::3] += -1.0

    grad[0] += -2.0 * np.sum(g2) + 2.0 * np.sum(g4)

    return obj, grad

def _strictly_valid(circles, w, h, tol=1e-7):
    xs = circles[:, 0]
    ys = circles[:, 1]
    rs = circles[:, 2]
    if np.any(rs < 1e-9):
        return False
    if np.any(xs - rs < -tol) or np.any(xs + rs > w + tol):
        return False
    if np.any(ys - rs < -tol) or np.any(ys + rs > h + tol):
        return False
    dx = xs[:, np.newaxis] - xs[np.newaxis, :]
    dy = ys[:, np.newaxis] - ys[np.newaxis, :]
    d2 = dx * dx + dy * dy
    r_sum = rs[:, np.newaxis] + rs[np.newaxis, :]
    iu = np.triu_indices(N, k=1)
    if np.any(d2[iu] < (r_sum[iu] - tol) ** 2):
        return False
    return True

def _solve_lp_radii(xs, ys, w, h):
    c_obj = -np.ones(N)
    A_ub = []
    b_ub = []
    for i in range(N):
        for j in range(i + 1, N):
            d = np.sqrt((xs[i] - xs[j]) ** 2 + (ys[i] - ys[j]) ** 2)
            row = np.zeros(N)
            row[i] = 1.0
            row[j] = 1.0
            A_ub.append(row)
            b_ub.append(d)
        row = np.zeros(N)
        row[i] = 1.0
        A_ub.append(row)
        b_ub.append(xs[i])
        A_ub.append(row.copy())
        b_ub.append(w - xs[i])
        A_ub.append(row.copy())
        b_ub.append(ys[i])
        A_ub.append(row.copy())
        b_ub.append(h - ys[i])
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    bounds = [(0, None) for _ in range(N)]
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds,
                      method='highs', options={'presolve': True})
        if res.success and res.x is not None:
            return res.x
    except Exception:
        pass
    return None

def _lp_polish(w, h, xs, ys):
    rs = _solve_lp_radii(xs, ys, w, h)
    if rs is None:
        return None
    if np.any(rs < 1e-9):
        return None
    rs = rs * (1.0 - 1e-7)
    circles = np.column_stack([xs, ys, rs])
    if _strictly_valid(circles, w, h, tol=1e-7):
        return circles
    return None

def _physics_simulate(x0, y0, w0, h0, r0, max_steps=4000, dt=0.002,
                      cooling_rate=0.9995):
    if not _time_ok():
        return w0, h0, r0, x0.copy(), y0.copy()

    sigma = 2.0 * r0
    eps = 1.0
    boundary_strength = 50.0

    x = x0.copy().astype(float)
    y = y0.copy().astype(float)
    vx = np.zeros(N)
    vy = np.zeros(N)
    temperature = 0.01
    damping = 0.99

    for step in range(max_steps):
        if not _time_ok():
            break

        fx = np.zeros(N)
        fy = np.zeros(N)

        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx * dx + dy * dy
                dist = np.sqrt(dist_sq)
                if dist < 1e-10:
                    dx = random.uniform(-1e-6, 1e-6)
                    dy = random.uniform(-1e-6, 1e-6)
                    dist_sq = dx * dx + dy * dy
                    dist = np.sqrt(dist_sq)
                sr6 = (sigma / dist) ** 6
                sr12 = sr6 * sr6
                force_mag = 24 * eps * (2 * sr12 - sr6) / dist
                fx[i] += force_mag * dx
                fy[i] += force_mag * dy
                fx[j] -= force_mag * dx
                fy[j] -= force_mag * dy

        for i in range(N):
            if x[i] < r0:
                fx[i] += boundary_strength * (r0 - x[i])
            if x[i] > w0 - r0:
                fx[i] -= boundary_strength * (x[i] - (w0 - r0))
            if y[i] < r0:
                fy[i] += boundary_strength * (r0 - y[i])
            if y[i] > h0 - r0:
                fy[i] -= boundary_strength * (y[i] - (h0 - r0))

        vx = vx * damping + fx * dt
        vy = vy * damping + fy * dt
        x += vx * dt
        y += vy * dt
        temperature *= cooling_rate
        if step % 100 == 0 and temperature > 1e-4:
            x += np.random.normal(0, temperature * 0.01, N)
            y += np.random.normal(0, temperature * 0.01, N)
        x = np.clip(x, -0.5, w0 + 0.5)
        y = np.clip(y, -0.5, h0 + 0.5)

    xs_min = x.min() - r0
    xs_max = x.max() + r0
    ys_min = y.min() - r0
    ys_max = y.max() + r0
    w_actual = xs_max - xs_min
    h_actual = ys_max - ys_min
    if w_actual + h_actual > 0:
        scale = 2.0 / (w_actual + h_actual)
        w_final = scale * w_actual
        h_final = scale * h_actual
        r_final = scale * r0
        x_final = (x - xs_min) * scale
        y_final = (y - ys_min) * scale
    else:
        w_final, h_final, r_final = w0, h0, r0
        x_final, y_final = x.copy(), y.copy()
    return w_final, h_final, r_final, x_final, y_final

def _joint_optimize(w, h, xs, ys, r, max_iter=400):
    """Returns (w, h, circles) or None. Box is exactly w=2t,h=2(1-t)."""
    if not _time_ok():
        return None
    t0 = w / (w + h)
    state = np.zeros(1 + 3 * N)
    state[0] = t0
    state[1::3] = xs
    state[2::3] = ys
    state[3::3] = r
    bounds = [(0.15, 0.85)] + [(0.0, 2.0)] * (2 * N) + [(1e-4, 1.0)] * N
    try:
        res = minimize(_objective_joint, state, jac=True, method='L-BFGS-B',
                       bounds=bounds, options={'maxiter': max_iter,
                                               'ftol': 1e-14, 'gtol': 1e-10,
                                               'maxfun': 6000, 'disp': False})
    except Exception:
        return None
    s = res.x
    t = np.clip(s[0], 0.15, 0.85)
    wv = 2.0 * t
    hv = 2.0 * (1.0 - t)
    xs2 = s[1::3]
    ys2 = s[2::3]
    rs2 = s[3::3]
    circles = np.column_stack([xs2, ys2, rs2])
    if not _strictly_valid(circles, wv, hv, tol=1e-7):
        return None
    return wv, hv, circles

def construct():
    global START
    START = time.time()

    def pack_hexagonal(rows):
        pts = []
        for k, n_circles in enumerate(rows):
            y_k = k * np.sqrt(3)
            offset = 1.0 if k % 2 == 1 else 0.0
            for i in range(n_circles):
                pts.append((i * 2.0 + offset, y_k))
        pts = np.array(pts)
        pts -= pts.mean(axis=0)
        xs = pts[:, 0]
        ys = pts[:, 1]
        w_total = (xs.max() - xs.min()) + 2.0
        h_total = (ys.max() - ys.min()) + 2.0
        scale = 2.0 / (w_total + h_total)
        r = scale * 1.0
        w = scale * w_total
        h = scale * h_total
        xs_f = pts[:, 0] * scale + w / 2.0
        ys_f = pts[:, 1] * scale + h / 2.0
        return w, h, r, xs_f, ys_f

    def pack_grid(rows, cols):
        pts = []
        for i in range(cols):
            for j in range(rows):
                if i * rows + j >= N:
                    break
                pts.append((i * 2.0, j * 2.0))
        pts = np.array(pts[:N])
        pts -= pts.mean(axis=0)
        xs = pts[:, 0]
        ys = pts[:, 1]
        w_total = (xs.max() - xs.min()) + 2.0
        h_total = (ys.max() - ys.min()) + 2.0
        scale = 2.0 / (w_total + h_total)
        r = scale * 1.0
        w = scale * w_total
        h = scale * h_total
        xs_f = pts[:, 0] * scale + w / 2.0
        ys_f = pts[:, 1] * scale + h / 2.0
        return w, h, r, xs_f, ys_f

    configs = []
    configs.append(pack_hexagonal([7, 7, 7]))
    configs.append(pack_hexagonal([4, 5, 4, 4, 4]))
    configs.append(pack_hexagonal([4, 4, 4, 4, 4, 1]))
    configs.append(pack_grid(5, 5))
    configs.append(pack_grid(3, 7))
    configs.append(pack_grid(4, 6))
    for _ in range(6):
        if not _time_ok():
            break
        w, h, r = 1.0, 1.0, 0.05
        xs = np.random.uniform(r, w - r, N)
        ys = np.random.uniform(r, h - r, N)
        configs.append((w, h, r, xs, ys))

    best = None
    best_sum = -1.0

    def consider(circles):
        nonlocal best, best_sum
        if circles is None:
            return
        s = np.sum(circles[:, 2])
        if s > best_sum:
            best = circles.copy()
            best_sum = s

    for cfg in configs:
        if not _time_ok():
            break
        w, h, r, xs, ys = cfg
        w_p, h_p, r_p, xs_p, ys_p = _physics_simulate(xs, ys, w, h, r)

        # variable-radius joint optimization from physics result
        res1 = _joint_optimize(w_p, h_p, xs_p, ys_p, r_p)
        if res1 is not None:
            wv, hv, circ = res1
            lp = _lp_polish(wv, hv, circ[:, 0], circ[:, 1])
            consider(lp if lp is not None else circ)

        # joint directly from original config
        if _time_ok():
            res2 = _joint_optimize(w, h, xs, ys, r, max_iter=200)
            if res2 is not None:
                wv, hv, circ = res2
                lp = _lp_polish(wv, hv, circ[:, 0], circ[:, 1])
                consider(lp if lp is not None else circ)

    # Final high-precision polish on the best configuration
    if best is not None and _time_ok():
        xs_b, ys_b, rs_b = best[:, 0], best[:, 1], best[:, 2]
        wv = xs_b.max() + rs_b.max()
        hv = ys_b.max() + rs_b.max()
        # ensure box has w+h ~ 2 for joint; normalize
        s = 2.0 / (wv + hv)
        wv, hv = s * wv, s * hv
        res = _joint_optimize(wv, hv, xs_b * s, ys_b * s, rs_b * s, max_iter=800)
        if res is not None:
            wv2, hv2, circ = res
            lp = _lp_polish(wv2, hv2, circ[:, 0], circ[:, 1])
            consider(lp if lp is not None else circ)
        lp = _lp_polish(wv, hv, xs_b * s, ys_b * s)
        if lp is not None:
            consider(lp)

    if best is None:
        w, h, r, xs, ys = configs[0]
        return np.column_stack([xs, ys, np.full(N, r)])

    # Final safety: verify against a valid box and shrink if needed
    wv = best[:, 0].max() + best[:, 2].max()
    hv = best[:, 1].max() + best[:, 2].max()
    s = 2.0 / (wv + hv)
    wv, hv = s * wv, s * hv
    if not _strictly_valid(best, wv, hv, tol=1e-7):
        best = best.copy()
        best[:, 2] *= (1.0 - 1e-5)
    return best
