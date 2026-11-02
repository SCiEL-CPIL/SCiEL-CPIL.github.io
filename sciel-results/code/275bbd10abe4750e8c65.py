"""Circle packing in perimeter-4 rectangle: pack 21 circles, maximize sum of radii.

Combines:
 - Delaunay-based LP radii solver (exact feasibility, decoupled structure)
 - equal-radius SLSQP refinement jointly optimizing aspect ratio w and radius r
 - hexagonal / grid / random initial configurations
 - Smooth gradient-based polish (L-BFGS-B with softplus penalties) for per-circle radii
 - Rigid-body assembly: optimize centers via gradient descent on smooth min-pairwise-distance,
   then solve LP for optimal radius assignment at each step
 - Vectorized validity checks for speed
 - Ultra-high-precision final polish with high penalty weights
"""
import numpy as np
import time
import random
from scipy.optimize import minimize, linprog
from scipy.spatial import Delaunay

N = 21


def _delaunay_edges(xs, ys):
    """Return list of (i, j) edges from Delaunay triangulation of centers."""
    points = np.column_stack((xs, ys))
    try:
        tri = Delaunay(points)
    except Exception:
        return None
    edges = set()
    for simplex in tri.simplices:
        edges.add(tuple(sorted((simplex[0], simplex[1]))))
        edges.add(tuple(sorted((simplex[1], simplex[2]))))
        edges.add(tuple(sorted((simplex[0], simplex[2]))))
    return list(edges)


def _solve_lp_radii(xs, ys, w, h):
    """
    Solve LP for per-circle radii given fixed centers and rectangle.
    Maximize sum(r) s.t.  r_i + r_j <= d_ij  (Delaunay edges)
                          r_i <= min(x_i, w-x_i, y_i, h-y_i)
                          r_i >= 0
    Returns array of radii (zeros on failure).
    """
    limits = np.minimum(np.minimum(xs, w - xs), np.minimum(ys, h - ys))
    if np.any(limits < 0):
        return np.zeros(N)

    A_list = []
    b_list = []
    edges = _delaunay_edges(xs, ys)
    if edges is not None:
        for i, j in edges:
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx * dx + dy * dy)
            row = np.zeros(N)
            row[i] = 1
            row[j] = 1
            A_list.append(row)
            b_list.append(dist)

    A_bound = np.eye(N)
    if A_list:
        A_ub = np.vstack([A_list, A_bound])
        b_ub = np.concatenate([b_list, limits])
    else:
        A_ub = A_bound
        b_ub = limits

    c = -np.ones(N)
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=(0, None), method='highs')
        if res.success:
            return np.maximum(0.0, res.x)
    except Exception:
        pass
    return np.maximum(0.0, limits * 0.9)


def _max_equal_radius(xs, ys, w, h):
    """
    Largest r such that a single-radius packing is feasible.
    """
    limits = np.minimum(np.minimum(xs, w - xs), np.minimum(ys, h - ys))
    r = float(np.min(limits))
    if r < 0:
        return 0.0
    edges = _delaunay_edges(xs, ys)
    if edges is not None:
        for i, j in edges:
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            r = min(r, 0.5 * np.sqrt(dx * dx + dy * dy))
    return max(0.0, r)


def _make_hex(rows, w, h):
    """Hexagonal layout of given per-row counts, scaled to fit w x h."""
    pts = []
    for k, n in enumerate(rows):
        y = k * np.sqrt(3.0)
        off = 1.0 if k % 2 == 1 else 0.0
        for i in range(n):
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
    wt = wc + 2.0
    ht = hc + 2.0
    s = 2.0 / (wt + ht)
    xs_f = xs * s + 0.5 * (s * wt)
    ys_f = ys * s + 0.5 * (s * ht)
    return xs_f, ys_f, s * wt, s * ht


def _make_grid(rows, cols, w, h):
    pts = []
    for j in range(rows):
        for i in range(cols):
            if len(pts) >= N:
                break
            pts.append((i * 2.0, j * 2.0))
    pts = np.array(pts[:N])
    xs = pts[:, 0]
    ys = pts[:, 1]
    cx = 0.5 * (xs.max() + xs.min())
    cy = 0.5 * (ys.max() + ys.min())
    xs = xs - cx
    ys = ys - cy
    wt = (xs.max() - xs.min()) + 2.0
    ht = (ys.max() - ys.min()) + 2.0
    s = 2.0 / (wt + ht)
    xs_f = xs * s + 0.5 * (s * wt)
    ys_f = ys * s + 0.5 * (s * ht)
    return xs_f, ys_f, s * wt, s * ht


def _make_random(w, h, seed):
    rng = np.random.RandomState(seed)
    xs = rng.uniform(0.1, 0.9, N) * w
    ys = rng.uniform(0.1, 0.9, N) * h
    return xs, ys, w, h


def _rigid_body_assembly(w, h, xs, ys, deadline, max_iter=300, lr_init=0.01):
    """
    Rigid-body assembly: optimize centers via gradient descent on a smooth
    objective that encourages separation (minimizes overlap) while staying
    within bounds. Then solve LP for optimal radii at each step.
    
    The objective is: -sum(LP_radii) + penalty for out-of-bounds.
    We use a smooth surrogate: maximize min pairwise distance via
    a sum-of-softplus approach, then LP gives the actual radii.
    
    Returns (w, h, xs, ys, rs) on success.
    """
    if time.time() > deadline - 8.0:
        return w, h, xs, ys, np.zeros(N)

    idx_i, idx_j = np.triu_indices(N, k=1)
    n_pairs = len(idx_i)

    x = xs.copy().astype(np.float64)
    y = ys.copy().astype(np.float64)

    best_sum = -np.inf
    best_x, best_y = x.copy(), y.copy()
    best_rs = np.zeros(N)

    lr = lr_init
    for it in range(max_iter):
        if time.time() > deadline - 8.0:
            break

        # Compute LP radii for current positions
        rs = _solve_lp_radii(x, y, w, h)
        sum_r = float(np.sum(rs))

        if sum_r > best_sum:
            best_sum = sum_r
            best_x = x.copy()
            best_y = y.copy()
            best_rs = rs.copy()

        # Gradient: we want to move centers to increase LP radii.
        # Approximate gradient: for each pair, if they are too close, push apart.
        # Use the LP dual information implicitly: if r_i + r_j is binding, push apart.
        dx = x[idx_i] - x[idx_j]
        dy = y[idx_i] - y[idx_j]
        dist_sq = dx * dx + dy * dy
        dist = np.sqrt(dist_sq + 1e-12)
        min_dist = rs[idx_i] + rs[idx_j]
        gap = dist - min_dist

        # Smooth penalty for negative gap (overlap in LP sense)
        # Gradient of sum of softplus(-gap) w.r.t. positions
        beta = 50.0
        sig = 1.0 / (1.0 + np.exp(np.clip(gap * beta, -20, 20)))

        gx = np.zeros(N)
        gy = np.zeros(N)

        # Pairwise repulsion
        coeff = beta * sig / dist
        np.add.at(gx, idx_i, coeff * dx)
        np.add.at(gx, idx_j, -coeff * dx)
        np.add.at(gy, idx_i, coeff * dy)
        np.add.at(gy, idx_j, -coeff * dy)

        # Boundary repulsion: push away from walls
        # Left: if x_i < r_i, push right
        bl = np.maximum(0, rs - x)
        br = np.maximum(0, x + rs - w)
        bb = np.maximum(0, rs - y)
        bt = np.maximum(0, y + rs - h)

        gx += bl - br
        gy += bb - bt

        # Adaptive learning rate with momentum-like decay
        x = np.clip(x + lr * gx, 1e-4, w - 1e-4)
        y = np.clip(y + lr * gy, 1e-4, h - 1e-4)

        # Decay learning rate
        lr *= 0.995
        if lr < 1e-5:
            lr = 1e-5

    # Final LP solve
    rs_final = _solve_lp_radii(best_x, best_y, w, h)
    if np.sum(rs_final) > best_sum:
        return w, h, best_x, best_y, rs_final
    return w, h, best_x, best_y, best_rs


def _refine_slsqp(w, h, xs, ys, r, deadline):
    """
    Equal-radius SLSQP refinement.
    Variables: [w, r, x0..x20, y0..y20].  h = 2 - w.
    Maximize r  s.t.  boundary + pairwise non-overlap.
    """
    if time.time() > deadline - 3.0:
        return w, h, r, xs, ys

    v0 = np.zeros(2 + 2 * N)
    v0[0] = w
    v0[1] = r
    v0[2:2 + N] = xs
    v0[2 + N:2 + 2 * N] = ys

    def objective(v):
        if v[0] <= 0.2 or (2.0 - v[0]) <= 0.2 or v[1] <= 0:
            return 1e6
        return -v[1]

    def make_cons():
        cons = [
            {'type': 'ineq', 'fun': lambda v: v[0] - 0.2},
            {'type': 'ineq', 'fun': lambda v: 2.0 - v[0] - 0.2},
        ]
        for i in range(N):
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[2 + i] - v[1]})
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[0] - v[2 + i] - v[1]})
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[2 + N + i] - v[1]})
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: (2.0 - v[0]) - v[2 + N + i] - v[1]})
        for i in range(N):
            for j in range(i + 1, N):
                cons.append({
                    'type': 'ineq',
                    'fun': lambda v, i=i, j=j: np.sqrt((v[2 + i] - v[2 + j]) ** 2
                                                       + (v[2 + N + i] - v[2 + N + j]) ** 2) - 2.0 * v[1]
                })
        return cons

    bounds = [(0.2, 1.8), (0.001, 1.0)] + [(0.0, 2.0)] * (2 * N)

    try:
        res = minimize(objective, v0, method='SLSQP', bounds=bounds,
                       constraints=make_cons(),
                       options={'maxiter': 120, 'ftol': 1e-9, 'disp': False})
    except Exception:
        return w, h, r, xs, ys

    if res.success:
        w_r = res.x[0]
        r_r = res.x[1]
        h_r = 2.0 - w_r
        xs_r = res.x[2:2 + N]
        ys_r = res.x[2 + N:2 + 2 * N]
        if w_r > 0 and h_r > 0 and r_r > 0:
            return w_r, h_r, r_r, xs_r, ys_r
    return w, h, r, xs, ys


def _polish_gradient(w, h, xs, ys, rs, deadline, max_iter=500, penalty_weight=500.0):
    """
    Smooth gradient-based polish using L-BFGS-B with softplus penalties.
    Variables: [w, x0..x20, y0..y20, r0..r20]. h = 2 - w.
    Maximizes sum(r_i) subject to boundary and non-overlap via softplus penalties.
    Returns (w, h, xs, ys, rs) or input on failure.
    """
    if time.time() > deadline - 5.0:
        return w, h, xs, ys, rs

    v0 = np.zeros(1 + 3 * N)
    v0[0] = w
    v0[1:1 + N] = xs
    v0[1 + N:1 + 2 * N] = ys
    v0[1 + 2 * N:1 + 3 * N] = rs

    idx_i, idx_j = np.triu_indices(N, k=1)

    def objective_and_grad(v):
        w_val = v[0]
        h_val = 2.0 - w_val
        x = v[1:1 + N]
        y = v[1 + N:1 + 2 * N]
        r = v[1 + 2 * N:1 + 3 * N]

        if w_val < 0.05 or h_val < 0.05:
            return 1e6, np.zeros_like(v)

        obj = -np.sum(r)
        grad = np.zeros_like(v)
        grad[1 + 2 * N:1 + 3 * N] = -1.0

        eps_b = 1e-3
        eps_o = 1e-3

        # Boundary violations
        v_l = r - x
        sp_l = eps_b * np.log1p(np.exp(np.clip(v_l / eps_b, -20, 20)))
        obj += penalty_weight * np.sum(sp_l)
        sig_l = 1.0 / (1.0 + np.exp(-np.clip(v_l / eps_b, -20, 20)))
        grad[1:1 + N] += -penalty_weight * sig_l
        grad[1 + 2 * N:1 + 3 * N] += penalty_weight * sig_l

        v_r = x + r - w_val
        sp_r = eps_b * np.log1p(np.exp(np.clip(v_r / eps_b, -20, 20)))
        obj += penalty_weight * np.sum(sp_r)
        sig_r = 1.0 / (1.0 + np.exp(-np.clip(v_r / eps_b, -20, 20)))
        grad[0] += penalty_weight * np.sum(-sig_r)
        grad[1:1 + N] += penalty_weight * sig_r
        grad[1 + 2 * N:1 + 3 * N] += penalty_weight * sig_r

        v_b = r - y
        sp_b = eps_b * np.log1p(np.exp(np.clip(v_b / eps_b, -20, 20)))
        obj += penalty_weight * np.sum(sp_b)
        sig_b = 1.0 / (1.0 + np.exp(-np.clip(v_b / eps_b, -20, 20)))
        grad[1 + N:1 + 2 * N] += -penalty_weight * sig_b
        grad[1 + 2 * N:1 + 3 * N] += penalty_weight * sig_b

        v_t = y + r - h_val
        sp_t = eps_b * np.log1p(np.exp(np.clip(v_t / eps_b, -20, 20)))
        obj += penalty_weight * np.sum(sp_t)
        sig_t = 1.0 / (1.0 + np.exp(-np.clip(v_t / eps_b, -20, 20)))
        grad[0] += penalty_weight * np.sum(sig_t)
        grad[1 + N:1 + 2 * N] += penalty_weight * sig_t
        grad[1 + 2 * N:1 + 3 * N] += penalty_weight * sig_t

        # Overlap penalty
        dx = x[idx_i] - x[idx_j]
        dy = y[idx_i] - y[idx_j]
        dist_sq = dx * dx + dy * dy
        min_dist = r[idx_i] + r[idx_j]
        dist_smooth = np.sqrt(dist_sq + 1e-10)
        g = min_dist - dist_smooth
        sp_o = eps_o * np.log1p(np.exp(np.clip(g / eps_o, -20, 20)))
        obj += 4.0 * penalty_weight * np.sum(sp_o)
        sig_o = 1.0 / (1.0 + np.exp(-np.clip(g / eps_o, -20, 20)))

        coeff = 4.0 * penalty_weight * sig_o / dist_smooth
        np.add.at(grad, 1 + idx_i, coeff * dx)
        np.add.at(grad, 1 + idx_j, -coeff * dx)
        np.add.at(grad, 1 + N + idx_i, coeff * dy)
        np.add.at(grad, 1 + N + idx_j, -coeff * dy)
        np.add.at(grad, 1 + 2 * N + idx_i, 4.0 * penalty_weight * sig_o)
        np.add.at(grad, 1 + 2 * N + idx_j, 4.0 * penalty_weight * sig_o)

        return obj, grad

    def objective_only(v):
        return objective_and_grad(v)[0]

    bounds = [(0.05, 1.95)] + [(0.0, 2.0)] * (3 * N)

    try:
        res = minimize(objective_only, v0, jac=objective_and_grad, method='L-BFGS-B',
                       bounds=bounds, options={'maxiter': max_iter, 'ftol': 1e-14, 'disp': False})
    except Exception:
        return w, h, xs, ys, rs

    if res.success or res.fun < 1e3:
        w_r = res.x[0]
        h_r = 2.0 - w_r
        xs_r = res.x[1:1 + N]
        ys_r = res.x[1 + N:1 + 2 * N]
        rs_r = np.maximum(0.0, res.x[1 + 2 * N:1 + 3 * N])
        if w_r > 0.05 and h_r > 0.05:
            return w_r, h_r, xs_r, ys_r, rs_r
    return w, h, xs, ys, rs


def _make_result_per_circle(w, h, xs, ys, rs):
    result = np.zeros((N, 3))
    result[:, 0] = xs
    result[:, 1] = ys
    result[:, 2] = np.maximum(rs, 0.0)
    _scale_to_perimeter(result)
    return result


def _make_result_equal(w, h, xs, ys, r):
    result = np.zeros((N, 3))
    result[:, 0] = xs
    result[:, 1] = ys
    result[:, 2] = r
    _scale_to_perimeter(result)
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


def _validity_margin(result, tol=1e-5):
    """Return minimum clearance (positive = valid). Vectorized."""
    xs = result[:, 0]
    ys = result[:, 1]
    rs = result[:, 2]
    if np.any(rs < 1e-6):
        return -1.0
    min_x = np.min(xs - rs)
    max_x = np.max(xs + rs)
    min_y = np.min(ys - rs)
    max_y = np.max(ys + rs)
    bw = max_x - min_x
    bh = max_y - min_y
    perim = 2.0 * (bw + bh)
    if perim > 4.0 + tol:
        return -1.0
    idx_i, idx_j = np.triu_indices(N, k=1)
    dx = xs[idx_i] - xs[idx_j]
    dy = ys[idx_i] - ys[idx_j]
    dist = np.sqrt(dx * dx + dy * dy)
    min_dist = rs[idx_i] + rs[idx_j]
    m = np.min(dist - min_dist)
    return float(m)


def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    random.seed(42)
    np.random.seed(42)

    best_result = None
    best_value = -np.inf

    def consider(result):
        nonlocal best_result, best_value
        if result is None:
            return
        if _validity_margin(result) < -1e-5:
            return
        val = float(np.sum(result[:, 2]))
        if val > best_value:
            best_value = val
            best_result = result.copy()

    # ---------- initial configurations ----------
    inits = []
    for rows in [[7, 7, 7], [4, 5, 4, 4, 4], [4, 4, 4, 4, 4, 1],
                 [3, 4, 4, 4, 4, 2], [5, 5, 5, 3, 3], [6, 7, 8]]:
        if sum(rows) < N:
            continue
        xs, ys, w, h = _make_hex(rows, 1.0, 1.0)
        inits.append((w, h, xs, ys))
    for rows, cols in [(5, 5), (3, 7), (4, 6), (7, 3), (6, 4)]:
        xs, ys, w, h = _make_grid(rows, cols, 1.0, 1.0)
        inits.append((w, h, xs, ys))
    for seed in range(8):
        w = 1.0 + 0.15 * (seed - 3.5) / 7.0
        h = 2.0 - w
        xs, ys, w, h = _make_random(w, h, seed + 100)
        inits.append((w, h, xs, ys))

    # ---------- process each init ----------
    for idx, (w, h, xs, ys) in enumerate(inits):
        if time.time() > deadline - 20.0:
            break

        # 1) LP-based per-circle radii result
        rs_lp = _solve_lp_radii(xs, ys, w, h)
        res_lp = _make_result_per_circle(w, h, xs, ys, rs_lp)
        consider(res_lp)

        # 2) Equal-radius seed from LP max feasible r, then SLSQP refine
        r0 = _max_equal_radius(xs, ys, w, h)
        if r0 > 0:
            res_eq0 = _make_result_equal(w, h, xs, ys, r0)
            consider(res_eq0)
        if time.time() < deadline - 12.0:
            w2, h2, r2, xs2, ys2 = _refine_slsqp(w, h, xs, ys, max(r0, 0.02), deadline)
            r2 = _max_equal_radius(xs2, ys2, w2, h2)
            if r2 > 0:
                res_eq1 = _make_result_equal(w2, h2, xs2, ys2, r2)
                consider(res_eq1)

        # 3) Rigid-body assembly: optimize centers, LP for radii
        if time.time() < deadline - 15.0:
            w3, h3, xs3, ys3, rs3 = _rigid_body_assembly(w, h, xs, ys, deadline, max_iter=200)
            if np.any(rs3 > 0):
                res_rba = _make_result_per_circle(w3, h3, xs3, ys3, rs3)
                consider(res_rba)

        # 4) Gradient polish on per-circle radii from LP
        if time.time() < deadline - 10.0:
            w4, h4, xs4, ys4, rs4 = _polish_gradient(w, h, xs, ys, rs_lp, deadline, max_iter=300)
            if np.any(rs4 > 0):
                res_poly = _make_result_per_circle(w4, h4, xs4, ys4, rs4)
                consider(res_poly)

            # Second polish pass
            if time.time() < deadline - 8.0:
                w5, h5, xs5, ys5, rs5 = _polish_gradient(w4, h4, xs4, ys4, rs4, deadline, max_iter=300)
                if np.any(rs5 > 0):
                    res_poly2 = _make_result_per_circle(w5, h5, xs5, ys5, rs5)
                    consider(res_poly2)

    # ---------- Final high-precision polish on best result ----------
    if best_result is not None and time.time() < deadline - 12.0:
        xs_b = best_result[:, 0].copy()
        ys_b = best_result[:, 1].copy()
        rs_b = best_result[:, 2].copy()
        min_x = np.min(xs_b - rs_b)
        max_x = np.max(xs_b + rs_b)
        min_y = np.min(ys_b - rs_b)
        max_y = np.max(ys_b + rs_b)
        w_b = max_x - min_x
        h_b = max_y - min_y
        xs_b = xs_b - min_x
        ys_b = ys_b - min_y

        # Ultra-high-precision polish with high penalty weight
        w_f, h_f, xs_f, ys_f, rs_f = _polish_gradient(w_b, h_b, xs_b, ys_b, rs_b, deadline,
                                                       max_iter=500, penalty_weight=500.0)
        if np.any(rs_f > 0):
            res_final = _make_result_per_circle(w_f, h_f, xs_f, ys_f, rs_f)
            consider(res_final)

        # Even higher precision second pass
        if time.time() < deadline - 6.0:
            w_f2, h_f2, xs_f2, ys_f2, rs_f2 = _polish_gradient(w_f, h_f, xs_f, ys_f, rs_f, deadline,
                                                                max_iter=500, penalty_weight=1000.0)
            if np.any(rs_f2 > 0):
                res_final2 = _make_result_per_circle(w_f2, h_f2, xs_f2, ys_f2, rs_f2)
                consider(res_final2)

        # Rigid-body assembly on best result for further improvement
        if time.time() < deadline - 8.0:
            w_f3, h_f3, xs_f3, ys_f3, rs_f3 = _rigid_body_assembly(w_f, h_f, xs_f, ys_f, deadline,
                                                                    max_iter=200)
            if np.any(rs_f3 > 0):
                res_final3 = _make_result_per_circle(w_f3, h_f3, xs_f3, ys_f3, rs_f3)
                consider(res_final3)

    # ---------- final fallback ----------
    if best_result is None or best_value <= 0:
        w = 1.0
        h = 1.0
        side = 5
        result = np.zeros((N, 3))
        for i in range(N):
            row = i // side
            col = i % side
            result[i, 0] = (col + 0.5) / side * w
            result[i, 1] = (row + 0.5) / side * h
            result[i, 2] = 0.48 / side
        _scale_to_perimeter(result)
        return result

    return best_result
