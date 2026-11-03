"""Circle packing in perimeter-4 rectangle: pack 21 circles, maximize sum of radii.
Combines hexagonal structured initialization with a softplus-gradient optimizer that
co-optimizes the rectangle aspect ratio (perimeter-4 constraint) plus a high-precision
polish pass.
"""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 21
TIME_LIMIT = 146.0  # Safety margin before 150s wall-clock kill


def _softplus(x, beta=100.0):
    """Stable softplus function."""
    xb = beta * x
    return np.where(xb > 0,
                    x + np.log1p(np.exp(-xb)) / beta,
                    np.log1p(np.exp(xb)) / beta)


def _softplus_grad(x, beta=100.0):
    """Gradient of softplus (scaled sigmoid)."""
    xb = beta * x
    return np.where(xb > 0,
                    1.0 / (1.0 + np.exp(-xb)),
                    np.exp(xb) / (1.0 + np.exp(xb)))


def _rect_dims(t):
    """Perimeter-4 rectangle from aspect parameter t in (0,1): w=2t, h=2(1-t), w+h=2."""
    t = min(max(t, 0.05), 0.95)
    w = 2.0 * t
    h = 2.0 * (1.0 - t)
    return w, h


def _objective_and_grad(x, beta=100.0):
    """
    Objective: -sum(r) + softplus penalties for boundary/overlap/positivity.
    x: shape (1 + 3N,) = [t, x1, y1, r1, x2, y2, r2, ...]
    """
    t = x[0]
    w, h = _rect_dims(t)
    c = x[1:].reshape(-1, 3)
    xs = c[:, 0]
    ys = c[:, 1]
    rs = c[:, 2]

    obj = -np.sum(rs)
    g = np.zeros_like(x)

    # Boundary penalties (with respect to x, y, r)
    v1 = rs - xs          # left
    obj += np.sum(_softplus(v1, beta))
    g1 = _softplus_grad(v1, beta)
    g[1::3] -= g1
    g[3::3] += g1

    v2 = xs + rs - w      # right
    obj += np.sum(_softplus(v2, beta))
    g2 = _softplus_grad(v2, beta)
    g[1::3] += g2
    g[3::3] += g2

    v3 = rs - ys          # bottom
    obj += np.sum(_softplus(v3, beta))
    g3 = _softplus_grad(v3, beta)
    g[2::3] -= g3
    g[3::3] += g3

    v4 = ys + rs - h      # top
    obj += np.sum(_softplus(v4, beta))
    g4 = _softplus_grad(v4, beta)
    g[2::3] += g4
    g[3::3] += g4

    # Radius positivity (weak)
    v5 = -rs
    obj += np.sum(_softplus(v5, beta))
    g5 = _softplus_grad(v5, beta)
    g[3::3] -= g5

    # Pairwise overlap penalties (vectorized inner loop)
    for i in range(N):
        dx = xs[i] - xs[i + 1:]
        dy = ys[i] - ys[i + 1:]
        d = np.sqrt(dx * dx + dy * dy + 1e-12)
        v = rs[i] + rs[i + 1:] - d
        sp = _softplus(v, beta)
        gg = _softplus_grad(v, beta)
        obj += np.sum(sp)

        g[1 + 3 * i] += -np.sum(gg * dx / d)
        g[2 + 3 * i] += -np.sum(gg * dy / d)
        g[3 + 3 * i] += np.sum(gg)

        idx = np.arange(i + 1, N)
        g[1 + 3 * idx] += gg * dx / d
        g[2 + 3 * idx] += gg * dy / d
        g[3 + 3 * idx] += gg

    # Gradient for maximizing radii (minimizing -sum(r))
    g[3::3] += -1.0

    # Gradient of the right/top softplus penalties w.r.t. t (aspect ratio)
    # d(w)/dt = 2, d(h)/dt = -2
    gw = 2.0
    gh = -2.0
    g[0] += gw * np.sum(g2) + gh * np.sum(g4)

    return obj, g


def _strictly_valid(circles, w, h, tol=1e-8):
    """Check circles within bounds and pairwise disjoint."""
    n = circles.shape[0]
    xs = circles[:, 0]
    ys = circles[:, 1]
    rs = circles[:, 2]
    if np.any(rs < 1e-9):
        return False
    if np.any(xs - rs < -tol) or np.any(xs + rs > w + tol):
        return False
    if np.any(ys - rs < -tol) or np.any(ys + rs > h + tol):
        return False
    for i in range(n):
        dx = xs[i] - xs[i + 1:]
        dy = ys[i] - ys[i + 1:]
        d2 = dx * dx + dy * dy
        need = rs[i] + rs[i + 1:] - tol
        if np.any(d2 < need * need):
            return False
    return True


def _hex_init(w, h):
    """Hex-packed initial circles for a given rectangle; picks best valid layout."""
    row_configs = [
        [3, 4, 4, 4, 3, 3], [4, 4, 4, 4, 3, 2], [2, 4, 4, 4, 4, 3],
        [3, 4, 4, 4, 4, 2], [4, 3, 4, 4, 4, 2], [2, 3, 4, 4, 4, 4],
        [3, 3, 4, 4, 4, 3], [4, 4, 3, 4, 4, 2], [2, 4, 3, 4, 4, 4],
        [3, 4, 3, 4, 4, 3], [5, 4, 4, 4, 4], [4, 5, 4, 4, 4],
        [3, 5, 4, 4, 5], [4, 4, 5, 4, 4], [2, 5, 4, 5, 5],
        [5, 5, 5, 5, 1], [1, 5, 5, 5, 5], [3, 4, 5, 4, 5],
        [5, 4, 5, 4, 3], [4, 4, 4, 4, 4, 1], [1, 4, 4, 4, 4, 4],
        [4, 4, 4, 4, 5], [5, 4, 4, 4, 4], [3, 4, 4, 5, 5],
        [5, 5, 4, 4, 3], [4, 4, 4, 5, 4],
    ]

    best = None
    best_val = -np.inf

    for row_counts in row_configs:
        if sum(row_counts) != N:
            continue
        n_rows = len(row_counts)
        max_cols = max(row_counts)
        est_r_h = w / (max_cols + 0.5)
        est_r_v = h / (n_rows * np.sqrt(3) / 2 + 1.0)
        r_est = min(est_r_h, est_r_v) * 0.95
        if r_est < 0.01:
            continue

        circles = []
        for row_idx, n_cols in enumerate(row_counts):
            y = (row_idx + 0.5) * r_est * np.sqrt(3)
            x_offset = r_est if row_idx % 2 == 1 else 0.0
            row_width = n_cols * 2 * r_est
            center_x = w / 2
            for col_idx in range(n_cols):
                x = center_x - row_width / 2 + x_offset + (col_idx + 0.5) * 2 * r_est
                circles.append([x, y, r_est])
        circles = np.array(circles)
        if _strictly_valid(circles, w, h):
            val = float(np.sum(circles[:, 2]))
            if val > best_val:
                best_val = val
                best = circles.copy()

    if best is None:
        best = _grid_init(w, h)
    return best


def _grid_init(w, h):
    """Simple grid initialization."""
    a = int(np.ceil(np.sqrt(N * w / h)))
    b = int(np.ceil(N / a))
    while a * b < N:
        b += 1
    circles = []
    for i in range(N):
        col = i % a
        row = i // a
        r = min(w / (a + 0.5), h / (b + 0.5)) * 0.9
        x = (col + 0.5) * w / a
        y = (row + 0.5) * h / b
        circles.append([x, y, r])
    return np.array(circles)


def _random_init(w, h, seed):
    """Jittered random initialization with non-overlap attempts."""
    rng = np.random.RandomState(seed)
    r0 = min(w, h) / 2.0 * 0.2
    circles = np.zeros((N, 3))
    circles[:, 0] = rng.uniform(r0, w - r0, N)
    circles[:, 1] = rng.uniform(r0, h - r0, N)
    circles[:, 2] = r0 * rng.uniform(0.8, 1.2, N)
    return circles


def _scale_to_perim(circles, target_perim=4.0):
    """Scale circles so their bounding box has perimeter <= target."""
    xs = circles[:, 0]
    ys = circles[:, 1]
    rs = circles[:, 2]
    min_x = np.min(xs - rs)
    max_x = np.max(xs + rs)
    min_y = np.min(ys - rs)
    max_y = np.max(ys + rs)
    width = max_x - min_x
    height = max_y - min_y
    perim = 2.0 * (width + height)
    if perim <= target_perim + 1e-9:
        return circles
    scale = target_perim / perim
    cx = (min_x + max_x) / 2.0
    cy = (min_y + max_y) / 2.0
    xs_new = cx + (xs - cx) * scale
    ys_new = cy + (ys - cy) * scale
    rs_new = rs * scale
    return np.column_stack([xs_new, ys_new, rs_new])


def _polish(x0, deadline, beta_start=100.0, beta_end=400.0):
    """High-precision polish: raise beta and tighten tolerance for a final solve."""
    res = None
    try:
        res = minimize(
            lambda p: _objective_and_grad(p, beta=beta_start)[0],
            x0,
            jac=lambda p: _objective_and_grad(p, beta=beta_start)[1],
            method='SLSQP',
            bounds=[(0.05, 0.95)] + [(0, 2.0)] * (2 * N) + [(1e-6, 1.0)] * N,
            options={'maxiter': 300, 'ftol': 1e-14}
        )
    except Exception:
        return x0
    if res is not None and res.fun < 1e6:
        return res.x
    return x0


def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT

    best_value = -np.inf
    best_result = None
    best_state = None

    # Aspect-ratio starting points (t = w/2, so t in (0,1))
    t_starts = [0.5, 0.42, 0.58, 0.38, 0.62, 0.35, 0.65, 0.30, 0.70, 0.45, 0.55, 0.28, 0.72]

    for t0 in t_starts:
        if time.time() > deadline - 8:
            break
        w, h = _rect_dims(t0)

        # Build candidate initializations (state = [t, x, y, r, ...])
        inits = []
        for circ in [_hex_init(w, h), _grid_init(w, h),
                     _random_init(w, h, 1), _random_init(w, h, 2)]:
            state = np.concatenate([[t0], circ.flatten()])
            inits.append(state)

        for x0 in inits:
            if time.time() > deadline - 3:
                break
            try:
                res = minimize(
                    lambda p: _objective_and_grad(p)[0],
                    x0,
                    jac=lambda p: _objective_and_grad(p)[1],
                    method='SLSQP',
                    bounds=[(0.05, 0.95)] + [(0, 2.0)] * (2 * N) + [(1e-6, 1.0)] * N,
                    options={'maxiter': 250, 'ftol': 1e-12}
                )
                if res.success or res.fun < 1e6:
                    x_opt = res.x
                    t = x_opt[0]
                    w, h = _rect_dims(t)
                    circles_opt = x_opt[1:].reshape(-1, 3)
                    if _strictly_valid(circles_opt, w, h):
                        val = np.sum(circles_opt[:, 2])
                        if val > best_value:
                            best_value = val
                            best_result = circles_opt.copy()
                            best_state = x_opt.copy()
            except Exception:
                pass

    # Final high-precision polish on the best state (co-optimizes t as well)
    if best_state is not None and time.time() < deadline - 2:
        polished = _polish(best_state, deadline)
        t = polished[0]
        w, h = _rect_dims(t)
        circles_p = polished[1:].reshape(-1, 3)
        if _strictly_valid(circles_p, w, h):
            val = np.sum(circles_p[:, 2])
            if val > best_value:
                best_value = val
                best_result = circles_p.copy()

    # Final scaling to guarantee perimeter <= 4
    if best_result is not None:
        best_result = _scale_to_perim(best_result, 4.0)
    else:
        best_result = _grid_init(1.0, 1.0)
        best_result = _scale_to_perim(best_result, 4.0)

    return best_result
