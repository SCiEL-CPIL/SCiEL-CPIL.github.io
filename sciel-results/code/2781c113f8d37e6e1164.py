"""Circle packing in perimeter-4 rectangle: pack 21 circles, maximize sum of radii.

Combines:
 - Delaunay-based LP radii solver (exact feasibility, decoupled structure)
 - equal-radius SLSQP refinement jointly optimizing aspect ratio w and radius r
 - Smooth gradient-based polish (L-BFGS-B with softplus penalties) for per-circle radii
 - Vectorized validity checks for speed
 - Hexagonal / grid / random initial configurations
"""
import numpy as np
import time
import random
from scipy.optimize import minimize, linprog
from scipy.spatial import Delaunay

N = 21


def _softplus(x):
    return np.where(x > 20, x, np.log1p(np.exp(x)))


def _sigmoid(x):
    return np.where(x > 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


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
    Largest r such that a single-radius packing is feasible:
    r <= min_i (x_i, w-x_i, y_i, h-y_i)
    r <= d_ij / 2 for all Delaunay edges (i, j).
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


def _refine_slsqp(w, h, xs, ys, r, deadline):
    """
    Equal-radius SLSQP refinement.
    Variables: [w, r, x0..x20, y0..y20].  h = 2 - w.
    Maximize r  s.t.  boundary + pairwise non-overlap.
    Returns (w, h, r, xs, ys) or the input on failure.
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


def _polish_gradient(w, h, xs, ys, rs, deadline, max_iter=500):
    """
    Smooth gradient-based polish using L-BFGS-B with softplus penalty.
    Variables: [w, x0..x20, y0..y20, r0..r20]. h = 2 - w.
    Maximizes sum(r_i) subject to boundary and non-overlap via softplus penalties.
    Returns (w, h, xs, ys, rs) or input on failure.
    """
    if time.time() > deadline - 5.0:
        return w, h, xs, ys, rs

    # State vector: [w, x0..x20, y0..y20, r0..r20]
    v = np.zeros(1 + 3 * N)
    v[0] = w
    v[1:1 + N] = xs
    v[1 + N:1 + 2 * N] = ys
    v[1 + 2 * N:1 + 3 * N] = rs

    def _unpack(v):
        w_ = v[0]
        h_ = 2.0 - w_
        xs_ = v[1:1 + N]
        ys_ = v[1 + N:1 + 2 * N]
        rs_ = v[1 + 2 * N:1 + 3 * N]
        return w_, h_, xs_, ys_, rs_

    def _objective_and_grad(v, beta):
        w_, h_, xs_, ys_, rs_ = _unpack(v)
        if w_ <= 0.05 or h_ <= 0.05:
            return 1e6, np.zeros_like(v)

        obj = -np.sum(rs_)
        grad = np.zeros_like(v)
        grad[1 + 2 * N:1 + 3 * N] = -1.0  # d(-sum(r))/dr

        # Boundary penalties
        c1 = xs_ - rs_
        c2 = w_ - xs_ - rs_
        c3 = ys_ - rs_
        c4 = h_ - ys_ - rs_

        s1 = _sigmoid(-beta * c1)
        s2 = _sigmoid(-beta * c2)
        s3 = _sigmoid(-beta * c3)
        s4 = _sigmoid(-beta * c4)

        obj += np.sum(_softplus(-beta * c1)) + np.sum(_softplus(-beta * c2))
        obj += np.sum(_softplus(-beta * c3)) + np.sum(_softplus(-beta * c4))

        # d/dx_i
        grad[1:1 + N] += -beta * s1 + beta * s2
        # d/dy_i
        grad[1 + N:1 + 2 * N] += -beta * s3 + beta * s4
        # d/dr_i
        grad[1 + 2 * N:1 + 3 * N] += beta * s1 - beta * s2 + beta * s3 - beta * s4
        # d/dw
        grad[0] += np.sum(-beta * s2 + beta * s4)

        # Pairwise penalties (vectorized)
        dx = xs_[:, None] - xs_[None, :]
        dy = ys_[:, None] - ys_[None, :]
        d = np.sqrt(dx * dx + dy * dy)
        np.fill_diagonal(d, 1e10)
        c = d - rs_[:, None] - rs_[None, :]
        np.fill_diagonal(c, 1e10)
        s = _sigmoid(-beta * c)
        obj += 0.5 * np.sum(_softplus(-beta * c))

        # Gradients for pairwise terms
        grad_pair_x = np.zeros((N, N))
        grad_pair_y = np.zeros((N, N))
        
        mask = ~np.eye(N, dtype=bool)
        grad_pair_x[mask] = -beta * s[mask] * dx[mask] / d[mask]
        grad_pair_y[mask] = -beta * s[mask] * dy[mask] / d[mask]
        
        grad[1:1 + N] += np.sum(grad_pair_x, axis=1)
        grad[1 + N:1 + 2 * N] += np.sum(grad_pair_y, axis=1)
        grad[1 + 2 * N:1 + 3 * N] += np.sum(beta * s, axis=1)

        return obj, grad

    bounds = [(0.05, 1.95)] + [(0.0, 2.0)] * (2 * N) + [(0.0, 1.0)] * N

    # Annealing schedule for beta
    betas = [10.0, 30.0, 100.0, 300.0]
    
    best_obj = _objective_and_grad(v, betas[0])[0]
    best_v = v.copy()

    for beta in betas:
        if time.time() > deadline - 2.0:
            break
        try:
            # Use previous best as start
            res = minimize(lambda v_, b=beta: _objective_and_grad(v_, b), best_v, jac=True, method='L-BFGS-B',
                           bounds=bounds, options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-10, 'disp': False})
            if res.fun < best_obj:
                best_obj = res.fun
                best_v = res.x.copy()
        except Exception:
            pass

    w_r = best_v[0]
    h_r = 2.0 - w_r
    xs_r = best_v[1:1 + N]
    ys_r = best_v[1 + N:1 + 2 * N]
    rs_r = np.maximum(0.0, best_v[1 + 2 * N:1 + 3 * N])
    
    if w_r > 0.05 and h_r > 0.05:
        return w_r, h_r, xs_r, ys_r, rs_r
    return w, h, xs, ys, rs


def _make_result_from_lp(w, h, xs, ys):
    """Build a valid (N,3) result using per-circle LP radii, then scale to perimeter <= 4."""
    rs = _solve_lp_radii(xs, ys, w, h)
    result = np.zeros((N, 3))
    result[:, 0] = xs
    result[:, 1] = ys
    result[:, 2] = rs
    result[:, 2] = np.maximum(result[:, 2], 0.0)
    _scale_to_perimeter(result)
    return result


def _make_result_equal(w, h, xs, ys, r):
    result = np.zeros((N, 3))
    result[:, 0] = xs
    result[:, 1] = ys
    result[:, 2] = r
    _scale_to_perimeter(result)
    return result


def _make_result_variable(w, h, xs, ys, rs):
    result = np.zeros((N, 3))
    result[:, 0] = xs
    result[:, 1] = ys
    result[:, 2] = np.maximum(0.0, rs)
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


def _validity_margin(result, tol=1e-6):
    """Return minimum clearance (positive = valid)."""
    xs = result[:, 0]
    ys = result[:, 1]
    rs = result[:, 2]
    if np.any(rs < 1e-7):
        return -1.0
    # Check pairwise non-overlap (vectorized)
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    d = np.sqrt(dx * dx + dy * dy)
    np.fill_diagonal(d, 1e10)
    min_clearance = np.min(d - rs[:, None] - rs[None, :])
    if min_clearance < -tol:
        return min_clearance
    # Perimeter check
    min_x = np.min(xs - rs)
    max_x = np.max(xs + rs)
    min_y = np.min(ys - rs)
    max_y = np.max(ys + rs)
    bw = max_x - min_x
    bh = max_y - min_y
    perim = 2.0 * (bw + bh)
    if perim > 4.0 + tol:
        return -1.0
    return min(min_clearance, 4.0 - perim) + tol


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

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
                 [3, 4, 4, 4, 4, 2], [5, 5, 5, 3, 3], [6, 7, 8],
                 [3, 3, 3, 3, 3, 3, 3], [2, 3, 4, 4, 4, 4], [4, 3, 4, 3, 4]]:
        if sum(rows) < N:
            continue
        xs, ys, w, h = _make_hex(rows, 1.0, 1.0)
        inits.append((w, h, xs, ys))
    for rows, cols in [(5, 5), (3, 7), (4, 6), (7, 3), (6, 4), (2, 11), (11, 2)]:
        xs, ys, w, h = _make_grid(rows, cols, 1.0, 1.0)
        inits.append((w, h, xs, ys))
    for seed in range(12):
        w = 1.0 + 0.3 * (seed - 5.5) / 11.0
        h = 2.0 - w
        xs, ys, w, h = _make_random(w, h, seed + 100)
        inits.append((w, h, xs, ys))

    # ---------- process each init ----------
    for idx, (w, h, xs, ys) in enumerate(inits):
        if time.time() > deadline - 12.0:
            break

        # 1) LP-based per-circle radii result (guaranteed-feasible structure)
        res_lp = _make_result_from_lp(w, h, xs, ys)
        consider(res_lp)

        # 2) equal-radius seed from LP max feasible r, then SLSQP refine
        r0 = _max_equal_radius(xs, ys, w, h)
        if r0 > 0:
            res_eq0 = _make_result_equal(w, h, xs, ys, r0)
            consider(res_eq0)

        # 3) Gradient polish on LP radii
        if time.time() < deadline - 10.0:
            rs0 = _solve_lp_radii(xs, ys, w, h)
            w2, h2, xs2, ys2, rs2 = _polish_gradient(w, h, xs, ys, rs0, deadline, max_iter=800)
            res_var = _make_result_variable(w2, h2, xs2, ys2, rs2)
            consider(res_var)

        # 4) SLSQP equal-radius refinement
        if time.time() < deadline - 6.0:
            w3, h3, r3, xs3, ys3 = _refine_slsqp(w, h, xs, ys, max(r0, 0.02), deadline)
            r3 = _max_equal_radius(xs3, ys3, w3, h3)
            if r3 > 0:
                res_eq1 = _make_result_equal(w3, h3, xs3, ys3, r3)
                consider(res_eq1)

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
