"""Circle packing in perimeter-4 rectangle: pack 21 circles, maximize sum of radii.

Model-and-optimize approach:
 - Parameterize the answer as (w, r_i, x_i, y_i) for i in 0..20 with h = 2 - w.
 - Directly maximize sum(r_i) subject to boundary and non-overlap constraints.
 - Use a smooth barrier/penalty formulation with L-BFGS-B for speed.
 - Multiple random restarts with hexagonal/grid seeds for diversity.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 21


def _validity_margin(result, tol=1e-4):
    """Return minimum clearance (positive = valid). result is (N,3): x, y, r."""
    xs = result[:, 0]
    ys = result[:, 1]
    rs = result[:, 2]
    if np.any(rs < 1e-6):
        return -1.0
    # boundary: circles must be inside [0, w] x [0, h] where w + h = 2
    # But we don't know w, h from result directly. The bounding box defines them.
    # Actually the contract says "bounding rectangle has perimeter at most 4".
    # So we need: 2*(bw + bh) <= 4, i.e., bw + bh <= 2.
    min_x = np.min(xs - rs)
    max_x = np.max(xs + rs)
    min_y = np.min(ys - rs)
    max_y = np.max(ys + rs)
    bw = max_x - min_x
    bh = max_y - min_y
    if bw + bh > 2.0 + tol:
        return -(bw + bh - 2.0)
    # circles must not overlap
    for i in range(N):
        for j in range(i + 1, N):
            d = np.sqrt((xs[i] - xs[j]) ** 2 + (ys[i] - ys[j]) ** 2)
            m = d - (rs[i] + rs[j])
            if m < -tol:
                return m
    return min(bw + bh - 2.0, 0.0)  # should be <= 0


def _objective(v, lam=50.0):
    """
    Objective: maximize sum(r) minus barrier violations.
    v = [w, r_0..r_20, x_0..x_20, y_0..y_20]
    h = 2 - w
    """
    w = v[0]
    h = 2.0 - w
    rs = v[1:1 + N]
    xs = v[1 + N:1 + 2 * N]
    ys = v[1 + 2 * N:1 + 3 * N]

    if w <= 0.1 or h <= 0.1:
        return 1e10

    # Boundary constraints: each circle inside [0,w] x [0,h]
    # x_i - r_i >= 0, w - x_i - r_i >= 0, y_i - r_i >= 0, h - y_i - r_i >= 0
    pen = 0.0
    for i in range(N):
        pen += max(0.0, -(xs[i] - rs[i])) ** 2
        pen += max(0.0, -(w - xs[i] - rs[i])) ** 2
        pen += max(0.0, -(ys[i] - rs[i])) ** 2
        pen += max(0.0, -(h - ys[i] - rs[i])) ** 2

    # Non-overlap: dist(i,j) >= r_i + r_j
    for i in range(N):
        for j in range(i + 1, N):
            d = np.sqrt((xs[i] - xs[j]) ** 2 + (ys[i] - ys[j]) ** 2)
            viol = rs[i] + rs[j] - d
            if viol > 0:
                pen += viol ** 2

    sum_r = np.sum(rs)
    return -sum_r + lam * pen


def _objective_grad(v, lam=50.0):
    """Gradient of _objective for L-BFGS-B."""
    w = v[0]
    h = 2.0 - w
    rs = v[1:1 + N]
    xs = v[1 + N:1 + 2 * N]
    ys = v[1 + 2 * N:1 + 3 * N]

    if w <= 0.1 or h <= 0.1:
        return np.array([0.0] + [0.0] * (3 * N))

    grad = np.zeros(1 + 3 * N)
    # d(-sum_r)/dr_i = -1
    grad[1:1 + N] = -1.0

    # Boundary penalty gradients
    for i in range(N):
        # x_i - r_i >= 0
        viol1 = max(0.0, -(xs[i] - rs[i]))
        if viol1 > 0:
            grad[1 + N + i] += 2 * lam * viol1  # d/dx_i
            grad[1 + i] += -2 * lam * viol1     # d/dr_i
        # w - x_i - r_i >= 0
        viol2 = max(0.0, -(w - xs[i] - rs[i]))
        if viol2 > 0:
            grad[0] += 2 * lam * viol2           # d/dw
            grad[1 + N + i] += -2 * lam * viol2  # d/dx_i
            grad[1 + i] += -2 * lam * viol2      # d/dr_i
        # y_i - r_i >= 0
        viol3 = max(0.0, -(ys[i] - rs[i]))
        if viol3 > 0:
            grad[1 + 2 * N + i] += 2 * lam * viol3
            grad[1 + i] += -2 * lam * viol3
        # h - y_i - r_i >= 0
        viol4 = max(0.0, -(h - ys[i] - rs[i]))
        if viol4 > 0:
            grad[0] += -2 * lam * viol4          # d/dh = -d/dw
            grad[1 + 2 * N + i] += -2 * lam * viol4
            grad[1 + i] += -2 * lam * viol4

    # Non-overlap penalty gradients
    for i in range(N):
        for j in range(i + 1, N):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            d = np.sqrt(dx * dx + dy * dy)
            if d < 1e-12:
                continue
            viol = rs[i] + rs[j] - d
            if viol > 0:
                # d(pen)/dr_i = 2*viol, d(pen)/dr_j = 2*viol
                grad[1 + i] += 2 * lam * viol
                grad[1 + j] += 2 * lam * viol
                # d(pen)/dx_i = 2*viol * dx/d, similarly for others
                grad[1 + N + i] += 2 * lam * viol * dx / d
                grad[1 + N + j] += -2 * lam * viol * dx / d
                grad[1 + 2 * N + i] += 2 * lam * viol * dy / d
                grad[1 + 2 * N + j] += -2 * lam * viol * dy / d

    return grad


def _make_result(v):
    """Convert parameter vector to (N,3) result array."""
    w = v[0]
    h = 2.0 - w
    rs = v[1:1 + N]
    xs = v[1 + N:1 + 2 * N]
    ys = v[1 + 2 * N:1 + 3 * N]
    result = np.zeros((N, 3))
    result[:, 0] = xs
    result[:, 1] = ys
    result[:, 2] = np.maximum(rs, 0.0)
    return result


def _scale_to_perimeter(result):
    """Scale result about its bounding-box center so perimeter <= 4."""
    xs = result[:, 0]
    ys = result[:, 1]
    rs = result[:, 2]
    min_x = np.min(xs - rs)
    max_x = np.max(xs + rs)
    min_y = np.min(ys - rs)
    max_y = np.max(ys + rs)
    bw = max_x - min_x
    bh = max_y - min_y
    perim = 2.0 * (bw + bh)
    if perim > 4.0 + 1e-9 and perim > 0:
        scale = 4.0 / perim
        cx = 0.5 * (min_x + max_x)
        cy = 0.5 * (min_y + max_y)
        for i in range(N):
            result[i, 0] = cx + (result[i, 0] - cx) * scale
            result[i, 1] = cy + (result[i, 1] - cy) * scale
            result[i, 2] = result[i, 2] * scale
    return result


def _hex_initial(w, h, seed=0):
    """Create hexagonal arrangement of 21 circles in rectangle w x h."""
    rng = np.random.RandomState(seed)
    # Try different row configurations
    configs = [[7, 7, 7], [4, 5, 4, 4, 4], [3, 4, 4, 4, 4, 2], [5, 5, 5, 3, 3]]
    cfg = configs[seed % len(configs)]
    while sum(cfg) < N:
        cfg = [7, 7, 7]
        break

    pts = []
    for k, n in enumerate(cfg):
        y = k * np.sqrt(3.0)
        off = 1.0 if k % 2 == 1 else 0.0
        for i in range(n):
            if len(pts) >= N:
                break
            pts.append((i * 2.0 + off, y))
    pts = np.array(pts[:N])
    xs = pts[:, 0]
    ys = pts[:, 1]
    cx = 0.5 * (xs.max() + xs.min())
    cy = 0.5 * (ys.max() + ys.min())
    xs = xs - cx
    ys = ys - cy
    wc = xs.max() - xs.min()
    hc = ys.max() - ys.min()
    if wc > 0 and hc > 0:
        # Scale to fit in w x h with margin
        s = min((w - 0.5) / (wc + 1.0), (h - 0.5) / (hc + 1.0))
        s = max(s, 0.01)
        xs_f = xs * s + 0.5 * w
        ys_f = ys * s + 0.5 * h
    else:
        xs_f = np.linspace(0.1, w - 0.1, N)
        ys_f = np.linspace(0.1, h - 0.1, N)
    # Equal initial radius
    r0 = min(w, h) * 0.15
    return w, xs_f, ys_f, np.full(N, r0)


def _grid_initial(w, h, seed=0):
    """Create grid arrangement of 21 circles."""
    rows = 5
    cols = 5
    xs, ys = [], []
    for r in range(rows):
        for c in range(cols):
            if len(xs) >= N:
                break
            xs.append((c + 0.5) / cols * w)
            ys.append((r + 0.5) / rows * h)
    xs = np.array(xs[:N])
    ys = np.array(ys[:N])
    r0 = min(w, h) * 0.15
    return w, xs, ys, np.full(N, r0)


def _random_initial(w, h, seed=0):
    """Create random arrangement."""
    rng = np.random.RandomState(seed)
    xs = rng.uniform(0.1, w - 0.1, N)
    ys = rng.uniform(0.1, h - 0.1, N)
    r0 = min(w, h) * 0.12
    return w, xs, ys, np.full(N, r0)


def _equal_radius_slsqp(xs, ys, r, w, h, max_iter=300):
    """
    Jointly optimize aspect ratio (w) and positions (x_i, y_i) for N circles
    with a shared radius r, maximizing r subject to:
      - w + h = 2 (so h = 2 - w)
      - circles inside [0,w] x [0,h]
      - non-overlap: dist(i,j) >= 2*r
    Variables: z = [w, r, x_0..x_{N-1}, y_0..y_{N-1}]
    """
    n = len(xs)
    z0 = np.zeros(1 + 1 + 2 * n)
    z0[0] = w
    z0[1] = r
    z0[2:2 + n] = xs
    z0[2 + n:2 + 2 * n] = ys

    def obj(z):
        # maximize r -> minimize -r
        return -z[1]

    def jac(z):
        g = np.zeros_like(z)
        g[1] = -1.0
        return g

    # Inequality constraints: g_i(z) >= 0
    def make_boundary_constraints():
        cons = []
        for i in range(n):
            # x_i - r >= 0
            cons.append({'type': 'ineq',
                         'fun': lambda z, i=i: z[2 + i] - z[1],
                         'jac': lambda z, i=i: np.concatenate([[0, -1],
                                                               np.zeros(2 * n)])})
            # w - x_i - r >= 0
            cons.append({'type': 'ineq',
                         'fun': lambda z, i=i: z[0] - z[2 + i] - z[1],
                         'jac': lambda z, i=i: np.concatenate([[1, -1],
                                                               np.zeros(2 * n)])})
            # y_i - r >= 0
            cons.append({'type': 'ineq',
                         'fun': lambda z, i=i: z[2 + n + i] - z[1],
                         'jac': lambda z, i=i: np.concatenate([[0, -1],
                                                               np.zeros(2 * n)])})
            # h - y_i - r >= 0  =>  (2-w) - y_i - r >= 0
            cons.append({'type': 'ineq',
                         'fun': lambda z, i=i: 2.0 - z[0] - z[2 + n + i] - z[1],
                         'jac': lambda z, i=i: np.concatenate([[-1, -1],
                                                               np.zeros(2 * n)])})
        return cons

    def make_overlap_constraints():
        cons = []
        for i in range(n):
            for j in range(i + 1, n):
                def fun(z, i=i, j=j):
                    dx = z[2 + i] - z[2 + j]
                    dy = z[2 + n + i] - z[2 + n + j]
                    d = np.sqrt(dx * dx + dy * dy)
                    return d - 2.0 * z[1]
                def jac_fn(z, i=i, j=j):
                    dx = z[2 + i] - z[2 + j]
                    dy = z[2 + n + i] - z[2 + n + j]
                    d = np.sqrt(dx * dx + dy * dy)
                    if d < 1e-12:
                        d = 1e-12
                    g = np.zeros(2 + 2 * n)
                    g[1] = -2.0
                    g[2 + i] = dx / d
                    g[2 + j] = -dx / d
                    g[2 + n + i] = dy / d
                    g[2 + n + j] = -dy / d
                    return g
                cons.append({'type': 'ineq', 'fun': fun, 'jac': jac_fn})
        return cons

    constraints = make_boundary_constraints() + make_overlap_constraints()
    bounds = [(0.3, 1.7), (0.01, 1.0)] + [(0.01, 1.99)] * (2 * n)

    try:
        res = minimize(obj, z0, jac=jac, method='SLSQP',
                       bounds=bounds, constraints=constraints,
                       options={'maxiter': max_iter, 'ftol': 1e-12})
        if res.success or (res.fun < -0.01 and res.x[1] > 0):
            w_opt = res.x[0]
            r_opt = res.x[1]
            xs_opt = res.x[2:2 + n]
            ys_opt = res.x[2 + n:2 + 2 * n]
            return w_opt, r_opt, xs_opt, ys_opt
    except Exception:
        pass
    return None


def _polish_equal_radius(result, max_iter=500):
    """
    High-precision polish: take a valid packing, extract its bounding box,
    then run SLSQP equal-radius optimization starting from that configuration.
    Returns improved result or None.
    """
    xs = result[:, 0].copy()
    ys = result[:, 1].copy()
    rs = result[:, 2].copy()

    # Use the median radius as initial shared radius
    r0 = float(np.median(rs))
    min_x = np.min(xs - rs)
    max_x = np.max(xs + rs)
    min_y = np.min(ys - rs)
    max_y = np.max(ys + rs)
    bw = max_x - min_x
    bh = max_y - min_y
    w0 = bw
    h0 = bh

    # Normalize positions to [0, w0] x [0, h0]
    xs_norm = xs - min_x
    ys_norm = ys - min_y

    out = _equal_radius_slsqp(xs_norm, ys_norm, r0, w0, h0, max_iter=max_iter)
    if out is None:
        return None

    w_opt, r_opt, xs_opt, ys_opt = out
    if r_opt <= 0 or w_opt <= 0.1 or w_opt >= 1.9:
        return None

    # Build result
    res = np.zeros((N, 3))
    res[:, 0] = xs_opt
    res[:, 1] = ys_opt
    res[:, 2] = r_opt

    # Verify validity
    if _validity_margin(res) < -1e-6:
        # Try scaling down slightly
        _scale_to_perimeter(res)
        if _validity_margin(res) < -1e-4:
            return None

    return res


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    best_result = None
    best_value = -np.inf

    def consider(result):
        nonlocal best_result, best_value
        if result is None:
            return
        if _validity_margin(result) < -1e-4:
            return
        val = float(np.sum(result[:, 2]))
        if val > best_value:
            best_value = val
            best_result = result.copy()

    # Try different aspect ratios
    aspect_ratios = [1.0, 0.8, 1.2, 0.7, 1.4, 0.6, 1.6, 0.9, 1.1]
    init_types = ['hex', 'grid', 'random']

    seed_counter = 0
    for ar in aspect_ratios:
        if time.time() > deadline - 30.0:
            break
        w = 2.0 * ar / (1.0 + ar)
        h = 2.0 - w

        for init_type in init_types:
            if time.time() > deadline - 25.0:
                break

            if init_type == 'hex':
                w_init, xs, ys, rs = _hex_initial(w, h, seed_counter)
            elif init_type == 'grid':
                w_init, xs, ys, rs = _grid_initial(w, h, seed_counter)
            else:
                w_init, xs, ys, rs = _random_initial(w, h, seed_counter)
            seed_counter += 1

            # Build initial parameter vector
            v0 = np.zeros(1 + 3 * N)
            v0[0] = w_init
            v0[1:1 + N] = rs
            v0[1 + N:1 + 2 * N] = xs
            v0[1 + 2 * N:1 + 3 * N] = ys

            bounds = [(0.2, 1.8)] + [(0.01, 1.0)] * N + [(0.0, 2.0)] * (2 * N)

            # Use L-BFGS-B with gradient
            for lam in [100.0, 50.0]:
                if time.time() > deadline - 20.0:
                    break
                try:
                    res = minimize(_objective, v0, jac=_objective_grad,
                                   method='L-BFGS-B', bounds=bounds,
                                   args=(lam,),
                                   options={'maxiter': 200, 'ftol': 1e-10,
                                             'gtol': 1e-8, 'disp': False})
                except Exception:
                    continue

                if res.success or res.fun < 1e6:
                    result = _make_result(res.x)
                    _scale_to_perimeter(result)
                    consider(result)

                    # If result is good, try to refine with smaller penalty
                    if _validity_margin(result) > -1e-5:
                        v1 = res.x.copy()
                        try:
                            res2 = minimize(_objective, v1, jac=_objective_grad,
                                            method='L-BFGS-B', bounds=bounds,
                                            args=(5.0,),
                                            options={'maxiter': 100, 'ftol': 1e-10,
                                                      'gtol': 1e-8, 'disp': False})
                        except Exception:
                            pass
                        if res2.success or res2.fun < 1e6:
                            result2 = _make_result(res2.x)
                            _scale_to_perimeter(result2)
                            consider(result2)

    # Phase 2: Joint aspect-ratio + equal-radius SLSQP refinement
    # For each promising initial configuration, let the optimizer find the
    # optimal aspect ratio and shared radius simultaneously.
    if best_result is not None:
        # Polish the best result found so far
        if time.time() < deadline - 10.0:
            polished = _polish_equal_radius(best_result, max_iter=400)
            if polished is not None:
                consider(polished)

    # Phase 3: Direct equal-radius SLSQP from diverse initial configs
    if time.time() < deadline - 15.0:
        for ar in [1.0, 0.85, 1.15, 0.75, 1.3]:
            if time.time() > deadline - 10.0:
                break
            w = 2.0 * ar / (1.0 + ar)
            h = 2.0 - w
            # Use hex initial positions
            w_init, xs, ys, rs = _hex_initial(w, h, 42)
            r0 = float(np.min(rs))
            out = _equal_radius_slsqp(xs, ys, r0, w, h, max_iter=300)
            if out is not None:
                w_opt, r_opt, xs_opt, ys_opt = out
                if r_opt > 0:
                    res_arr = np.zeros((N, 3))
                    res_arr[:, 0] = xs_opt
                    res_arr[:, 1] = ys_opt
                    res_arr[:, 2] = r_opt
                    if _validity_margin(res_arr) >= -1e-5:
                        consider(res_arr)
                    else:
                        _scale_to_perimeter(res_arr)
                        consider(res_arr)

    # Final fallback: simple grid
    if best_result is None or best_value <= 0:
        w = 1.0
        h = 1.0
        result = np.zeros((N, 3))
        for i in range(N):
            row = i // 5
            col = i % 5
            result[i, 0] = (col + 0.5) / 5.0 * w
            result[i, 1] = (row + 0.5) / 5.0 * h
            result[i, 2] = 0.48 / 5.0
        _scale_to_perimeter(result)
        return result

    return best_result
