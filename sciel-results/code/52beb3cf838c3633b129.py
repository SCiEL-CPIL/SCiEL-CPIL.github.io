"""Circle packing in perimeter-4 rectangle: pack 21 circles, maximize sum of radii.

Key structural insight: the bounding box of the circles IS the rectangle, so we
optimize the bounding-box width W and height H (with W + H <= 2 for perimeter <= 4)
directly, jointly with all circle centers and radii.

Pipeline:
  1. Many diverse initial configurations (hex, grid, random) at varied aspect ratios.
  2. Per-circle LP radii solver (Delaunay edges) for a strong feasible start.
  3. High-precision gradient-based polish (L-BFGS-B with softplus penalties) that
     jointly optimizes W, H, centers, and radii.
  4. Global search over aspect ratio using the polished result as a warm start.
  5. Vectorized validity checks for speed.
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
    """Largest r such that a single-radius packing is feasible."""
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


def _make_hex(rows):
    """Hexagonal layout of given per-row counts, centered at origin, unit spacing."""
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
    return xs, ys, wt, ht


def _make_grid(rows, cols):
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
    return xs, ys, wt, ht


def _fit_to_aspect(xs, ys, wt, ht, aspect):
    """
    Given centered points with natural width wt and height ht (including radius
    padding of 1 on each side), scale so that the bounding box has the given aspect
    ratio (W/H) and W + H = 2 (perimeter 4).
    Returns (xs, ys, W, H) where points are in [0,W]x[0,H] with margin.
    """
    # We want W/H = aspect and W + H = 2.
    H = 2.0 / (1.0 + aspect)
    W = 2.0 - H
    # Scale points so that their natural extent (wt, ht) fits into (W, H) with
    # the constraint that each circle has room. We scale by the limiting factor.
    # The natural width/height already include a +2 padding (radius 1 on each side).
    # We want the scaled padding to be consistent. Simplest: scale so that
    # wt * s = W and ht * s = H would require s = W/wt = H/ht; use min to fit both,
    # then center.
    s = min(W / wt, H / ht)
    xs_f = xs * s + 0.5 * W
    ys_f = ys * s + 0.5 * H
    return xs_f, ys_f, W, H


def _make_random(W, H, seed):
    rng = np.random.RandomState(seed)
    xs = rng.uniform(0.1, 0.9, N) * W
    ys = rng.uniform(0.1, 0.9, N) * H
    return xs, ys, W, H


def _polish_gradient(W, H, xs, ys, rs, deadline, max_iter=600, beta=80.0):
    """
    High-precision gradient polish using L-BFGS-B with softplus penalties.
    Variables: [W, H, x0..x20, y0..y20, r0..r20].
    Constraints handled by penalties:
      - W + H <= 2 (perimeter)
      - each circle inside [0,W]x[0,H]
      - pairwise non-overlap
    Maximizes sum(r_i).
    Returns (W, H, xs, ys, rs) or input on failure.
    """
    if time.time() > deadline - 6.0:
        return W, H, xs, ys, rs

    # State vector: [W, H, x0..x20, y0..y20, r0..r20]
    nvar = 2 + 3 * N
    v0 = np.zeros(nvar)
    v0[0] = W
    v0[1] = H
    v0[2:2 + N] = xs
    v0[2 + N:2 + 2 * N] = ys
    v0[2 + 2 * N:2 + 3 * N] = rs

    def _unpack(v):
        W_ = v[0]
        H_ = v[1]
        xs_ = v[2:2 + N]
        ys_ = v[2 + N:2 + 2 * N]
        rs_ = v[2 + 2 * N:2 + 3 * N]
        return W_, H_, xs_, ys_, rs_

    def _objective_and_grad(v):
        W_, H_, xs_, ys_, rs_ = _unpack(v)
        if W_ <= 0.05 or H_ <= 0.05:
            return 1e6, np.zeros(nvar)

        obj = -np.sum(rs_)
        grad = np.zeros(nvar)
        grad[2 + 2 * N:2 + 3 * N] = -1.0  # d(-sum(r))/dr

        # Perimeter penalty: W + H <= 2
        c_perim = 2.0 - (W_ + H_)
        sp_perim = _softplus(-beta * c_perim)
        obj += sp_perim
        s_perim = _sigmoid(-beta * c_perim)
        grad[0] += -beta * s_perim
        grad[1] += -beta * s_perim

        # Boundary penalties
        c1 = xs_ - rs_              # x_i - r_i >= 0
        c2 = W_ - xs_ - rs_         # W - x_i - r_i >= 0
        c3 = ys_ - rs_              # y_i - r_i >= 0
        c4 = H_ - ys_ - rs_         # H - y_i - r_i >= 0

        s1 = _sigmoid(-beta * c1)
        s2 = _sigmoid(-beta * c2)
        s3 = _sigmoid(-beta * c3)
        s4 = _sigmoid(-beta * c4)

        obj += np.sum(_softplus(-beta * c1)) + np.sum(_softplus(-beta * c2))
        obj += np.sum(_softplus(-beta * c3)) + np.sum(_softplus(-beta * c4))

        # d/dx_i
        grad[2:2 + N] += -beta * s1 + beta * s2
        # d/dy_i
        grad[2 + N:2 + 2 * N] += -beta * s3 + beta * s4
        # d/dr_i
        grad[2 + 2 * N:2 + 3 * N] += beta * s1 - beta * s2 + beta * s3 - beta * s4
        # d/dW
        grad[0] += np.sum(beta * s2)
        # d/dH
        grad[1] += np.sum(beta * s4)

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
        mask = ~np.eye(N, dtype=bool)
        gx = np.zeros((N, N))
        gy = np.zeros((N, N))
        gx[mask] = -beta * s[mask] * dx[mask] / d[mask]
        gy[mask] = -beta * s[mask] * dy[mask] / d[mask]

        grad[2:2 + N] += np.sum(gx, axis=1)
        grad[2 + N:2 + 2 * N] += np.sum(gy, axis=1)
        grad[2 + 2 * N:2 + 3 * N] += np.sum(beta * s, axis=1)

        return obj, grad

    bounds = [(0.05, 1.95), (0.05, 1.95)] + [(0.0, 2.0)] * (2 * N) + [(0.0, 1.0)] * N

    best_x = v0.copy()
    best_fun = _objective_and_grad(v0)[0]
    try:
        res = minimize(_objective_and_grad, v0, jac=True, method='L-BFGS-B',
                       bounds=bounds, options={'maxiter': max_iter, 'ftol': 1e-13,
                                                'gtol': 1e-11, 'disp': False})
        if res.fun < best_fun:
            best_fun = res.fun
            best_x = res.x
    except Exception:
        pass

    # Second polish pass from the best point
    if time.time() < deadline - 3.0:
        try:
            res2 = minimize(_objective_and_grad, best_x, jac=True, method='L-BFGS-B',
                            bounds=bounds, options={'maxiter': 300, 'ftol': 1e-14,
                                                     'gtol': 1e-12, 'disp': False})
            if res2.fun < best_fun:
                best_fun = res2.fun
                best_x = res2.x
        except Exception:
            pass

    W_r = best_x[0]
    H_r = best_x[1]
    xs_r = best_x[2:2 + N]
    ys_r = best_x[2 + N:2 + 2 * N]
    rs_r = np.maximum(0.0, best_x[2 + 2 * N:2 + 3 * N])
    if W_r > 0.05 and H_r > 0.05:
        return W_r, H_r, xs_r, ys_r, rs_r
    return W, H, xs, ys, rs


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


def _make_result_variable(W, H, xs, ys, rs):
    result = np.zeros((N, 3))
    result[:, 0] = xs
    result[:, 1] = ys
    result[:, 2] = np.maximum(0.0, rs)
    _scale_to_perimeter(result)
    return result


def _make_result_from_lp(W, H, xs, ys):
    rs = _solve_lp_radii(xs, ys, W, H)
    result = np.zeros((N, 3))
    result[:, 0] = xs
    result[:, 1] = ys
    result[:, 2] = np.maximum(0.0, rs)
    _scale_to_perimeter(result)
    return result


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

    # ---------- Build diverse initial configurations at varied aspect ratios ----------
    # aspect = W/H, with W + H = 2.
    aspects = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 1.7, 2.0]

    inits = []
    # Hexagonal layouts
    hex_rows_list = [
        [7, 7, 7], [4, 5, 4, 4, 4], [4, 4, 4, 4, 4, 1],
        [3, 4, 4, 4, 4, 2], [5, 5, 5, 3, 3], [6, 7, 8],
        [3, 3, 3, 3, 3, 3, 3], [2, 3, 4, 4, 4, 4], [4, 3, 4, 3, 4],
        [7, 7, 7], [5, 6, 5, 5], [3, 5, 5, 5, 3], [6, 6, 6, 3],
    ]
    for rows in hex_rows_list:
        if sum(rows) < N:
            continue
        xs, ys, wt, ht = _make_hex(rows)
        # Use the natural aspect of this layout plus a couple of varied ones
        nat_aspect = wt / ht
        for aspect in [nat_aspect, max(0.5, nat_aspect * 0.8), min(2.0, nat_aspect * 1.25)]:
            xf, yf, W, H = _fit_to_aspect(xs, ys, wt, ht, aspect)
            inits.append((W, H, xf, yf))

    # Grid layouts
    for rows, cols in [(5, 5), (3, 7), (4, 6), (7, 3), (6, 4), (2, 11), (11, 2), (7, 3), (4, 5)]:
        xs, ys, wt, ht = _make_grid(rows, cols)
        nat_aspect = wt / ht
        for aspect in [nat_aspect, max(0.5, nat_aspect * 0.8), min(2.0, nat_aspect * 1.25)]:
            xf, yf, W, H = _fit_to_aspect(xs, ys, wt, ht, aspect)
            inits.append((W, H, xf, yf))

    # Random layouts at varied aspects
    for seed in range(16):
        aspect = aspects[seed % len(aspects)]
        H = 2.0 / (1.0 + aspect)
        W = 2.0 - H
        xs, ys, W, H = _make_random(W, H, seed + 100)
        inits.append((W, H, xs, ys))

    # ---------- Process each init ----------
    for idx, (W, H, xs, ys) in enumerate(inits):
        if time.time() > deadline - 12.0:
            break

        # 1) LP-based per-circle radii result (guaranteed-feasible structure)
        res_lp = _make_result_from_lp(W, H, xs, ys)
        consider(res_lp)

        # 2) Gradient polish on LP radii (joint W, H, centers, radii)
        if time.time() < deadline - 10.0:
            rs0 = _solve_lp_radii(xs, ys, W, H)
            W2, H2, xs2, ys2, rs2 = _polish_gradient(W, H, xs, ys, rs0, deadline, max_iter=500)
            res_var = _make_result_variable(W2, H2, xs2, ys2, rs2)
            consider(res_var)

    # ---------- Global search over aspect ratio using best as warm start ----------
    # If we have a good result, try re-optimizing at different aspect ratios by
    # taking its centers (rescaled to fit) and re-polishing.
    if best_result is not None and time.time() < deadline - 20.0:
        base = best_result.copy()
        xs_b = base[:, 0]
        ys_b = base[:, 1]
        rs_b = base[:, 2]
        # Re-center base to origin
        cx_b = 0.5 * (np.max(xs_b + rs_b) + np.min(xs_b - rs_b))
        cy_b = 0.5 * (np.max(ys_b + rs_b) + np.min(ys_b - rs_b))
        xs_c = xs_b - cx_b
        ys_c = ys_b - cy_b
        # natural extent
        wt_b = np.max(xs_b + rs_b) - np.min(xs_b - rs_b)
        ht_b = np.max(ys_b + rs_b) - np.min(ys_b - rs_b)
        for aspect in [0.7, 0.85, 1.0, 1.15, 1.4]:
            if time.time() > deadline - 10.0:
                break
            xf, yf, W, H = _fit_to_aspect(xs_c, ys_c, wt_b, ht_b, aspect)
            rs0 = _solve_lp_radii(xf, yf, W, H)
            if np.sum(rs0) < 1e-6:
                rs0 = rs_b * (min(W / wt_b, H / ht_b))
            W2, H2, xs2, ys2, rs2 = _polish_gradient(W, H, xf, yf, rs0, deadline, max_iter=400)
            res_var = _make_result_variable(W2, H2, xs2, ys2, rs2)
            consider(res_var)

    # ---------- final fallback ----------
    if best_result is None or best_value <= 0:
        W = 1.0
        H = 1.0
        side = 5
        result = np.zeros((N, 3))
        for i in range(N):
            row = i // side
            col = i % side
            result[i, 0] = (col + 0.5) / side * W
            result[i, 1] = (row + 0.5) / side * H
            result[i, 2] = 0.48 / side
        _scale_to_perimeter(result)
        return result

    return best_result
