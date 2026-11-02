"""Circle packing in perimeter-4 rectangle: pack 21 circles, maximize sum of radii.

Combines:
 - Delaunay-based LP radii solver (exact feasibility, decoupled structure)
 - equal-radius SLSQP refinement jointly optimizing aspect ratio w and radius r
 - hexagonal / grid / random initial configurations
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
    if np.any(limits < -1e-9):
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


def _is_valid(result):
    """Check if result is a valid packing: disjoint interiors, within bounding box of perim <= 4."""
    if result.shape != (N, 3):
        return False
    xs = result[:, 0]
    ys = result[:, 1]
    rs = result[:, 2]
    if np.any(rs < 0):
        return False
    min_x = np.min(xs - rs)
    max_x = np.max(xs + rs)
    min_y = np.min(ys - rs)
    max_y = np.max(ys + rs)
    bw = max_x - min_x
    bh = max_y - min_y
    perim = 2.0 * (bw + bh)
    if perim > 4.0 + 1e-5:
        return False
    for i in range(N):
        for j in range(i + 1, N):
            dist = np.sqrt((xs[i] - xs[j])**2 + (ys[i] - ys[j])**2)
            if dist < rs[i] + rs[j] - 1e-7:
                return False
    return True


def _value(result):
    """Return sum of radii if valid, else -inf."""
    if not _is_valid(result):
        return -np.inf
    return float(np.sum(result[:, 2]))


def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    best_result = None
    best_value = -np.inf

    def _try_result(result):
        nonlocal best_result, best_value
        val = _value(result)
        if val > best_value:
            best_value = val
            best_result = result.copy()

    # Generate candidate configurations
    candidates = []

    # Hexagonal layouts with various row counts
    for rows in [1, 2, 3, 4, 5, 6, 7]:
        if rows > N:
            break
        # Distribute N circles into 'rows' rows as evenly as possible
        base = N // rows
        extra = N % rows
        row_counts = [base + 1] * extra + [base] * (rows - extra)
        for w_try in [0.9, 1.0, 1.1, 0.95, 1.05, 0.85, 1.15]:
            h_try = 2.0 - w_try
            xs, ys, w_act, h_act = _make_hex(row_counts, w_try, h_try)
            r_eq = _max_equal_radius(xs, ys, w_act, h_act)
            if r_eq > 0:
                candidates.append(('equal', w_act, h_act, xs, ys, r_eq))
            # LP radii
            result_lp = _make_result_from_lp(w_act, h_act, xs, ys)
            if _is_valid(result_lp):
                candidates.append(('lp', w_act, h_act, xs, ys, None, result_lp))

    # Grid layouts
    for rows in [3, 4, 5, 6]:
        cols = (N + rows - 1) // rows
        for w_try in [0.9, 1.0, 1.1, 0.95, 1.05]:
            h_try = 2.0 - w_try
            xs, ys, w_act, h_act = _make_grid(rows, cols, w_try, h_try)
            r_eq = _max_equal_radius(xs, ys, w_act, h_act)
            if r_eq > 0:
                candidates.append(('equal', w_act, h_act, xs, ys, r_eq))
            result_lp = _make_result_from_lp(w_act, h_act, xs, ys)
            if _is_valid(result_lp):
                candidates.append(('lp', w_act, h_act, xs, ys, None, result_lp))

    # Random initializations
    for seed in range(20):
        w_try = random.uniform(0.85, 1.15)
        h_try = 2.0 - w_try
        xs, ys, w_act, h_act = _make_random(w_try, h_try, seed)
        r_eq = _max_equal_radius(xs, ys, w_act, h_act)
        if r_eq > 0:
            candidates.append(('equal', w_act, h_act, xs, ys, r_eq))

    # Evaluate candidates
    for cand in candidates:
        if time.time() > deadline - 15:
            break

        if cand[0] == 'equal':
            _, w, h, xs, ys, r = cand
            result = _make_result_equal(w, h, xs, ys, r)
            _try_result(result)

            # Try SLSQP refinement
            if time.time() < deadline - 20:
                w_r, h_r, r_r, xs_r, ys_r = _refine_slsqp(w, h, xs, ys, r, deadline)
                result_ref = _make_result_equal(w_r, h_r, xs_r, ys_r, r_r)
                _try_result(result_ref)

                # After SLSQP, try LP radii on the refined centers
                if time.time() < deadline - 15:
                    result_lp2 = _make_result_from_lp(w_r, h_r, xs_r, ys_r)
                    _try_result(result_lp2)

        elif cand[0] == 'lp':
            _, w, h, xs, ys, _, result = cand
            _try_result(result)

    # If we have a good result, try to polish it further
    if best_result is not None and time.time() < deadline - 10:
        # Extract centers and try LP again with tighter tolerance
        xs = best_result[:, 0]
        ys = best_result[:, 1]
        # Estimate w, h from bounding box
        min_x = np.min(xs - best_result[:, 2])
        max_x = np.max(xs + best_result[:, 2])
        min_y = np.min(ys - best_result[:, 2])
        max_y = np.max(ys + best_result[:, 2])
        w_est = max_x - min_x
        h_est = max_y - min_y
        # Shift to origin
        xs2 = xs - min_x
        ys2 = ys - min_y
        rs_lp = _solve_lp_radii(xs2, ys2, w_est, h_est)
        if np.all(rs_lp > 0):
            result_polish = np.column_stack([xs2, ys2, rs_lp])
            _scale_to_perimeter(result_polish)
            _try_result(result_polish)

    # Fallback: guaranteed valid 5x5 grid in 1x1 square
    if best_result is None or best_value < 0:
        w = 1.0
        h = 1.0
        side = 5
        result = np.zeros((N, 3))
        for i in range(N):
            row = i // side
            col = i % side
            result[i, 0] = (col + 0.5) / side
            result[i, 1] = (row + 0.5) / side
            result[i, 2] = 0.48 / side
        return result

    return best_result
