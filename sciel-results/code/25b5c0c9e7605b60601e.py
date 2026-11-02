"""Circle packing in perimeter-4 rectangle: pack 21 circles, maximize sum of radii.

Combines:
 - Delaunay-based LP radii solver (exact feasibility, decoupled structure)
 - equal-radius SLSQP refinement jointly optimizing aspect ratio w and radius r
 - hexagonal / grid / random initial configurations
 - physics-based (Lennard-Jones + soft boundary) settling as an additional initializer
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
    if len(pts) < N:
        # Pad with duplicates slightly offset
        while len(pts) < N:
            pts.append(pts[-1] + (0.1, 0.1))
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


def _physics_settle(w, h, seed, max_steps=5000, dt=0.001, epsilon=1.0, sigma=0.3, k_wall=50.0):
    """
    Physics-based settling using Lennard-Jones-like repulsion and soft wall confinement.
    Returns (xs, ys) after settling.
    """
    rng = np.random.RandomState(seed)
    xs = rng.uniform(0.15, 0.85, N) * w
    ys = rng.uniform(0.15, 0.85, N) * h
    vs = np.zeros((N, 2))
    
    # Precompute pair indices
    pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    for step in range(max_steps):
        fx = np.zeros(N)
        fy = np.zeros(N)
        
        # Pairwise LJ-like repulsion
        for i, j in pairs:
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            r2 = dx * dx + dy * dy
            r = np.sqrt(r2) if r2 > 1e-12 else 1e-6
            if r < 2.0 * sigma:
                # Repulsive part of LJ: 4*eps*(sigma/r)^12 - 4*eps*(sigma/r)^6
                # Simplified: strong repulsion when r < sigma
                inv_r = 1.0 / r
                inv_r6 = (sigma * inv_r) ** 6
                f_mag = 48.0 * epsilon * inv_r6 * (inv_r6 - 0.5)
                fx[i] += f_mag * dx * inv_r
                fy[i] += f_mag * dy * inv_r
                fx[j] -= f_mag * dx * inv_r
                fy[j] -= f_mag * dy * inv_r
        
        # Wall forces (soft)
        for i in range(N):
            # Left wall
            if xs[i] < sigma:
                fx[i] += k_wall * (sigma - xs[i])
            # Right wall
            if xs[i] > w - sigma:
                fx[i] -= k_wall * (xs[i] - (w - sigma))
            # Bottom wall
            if ys[i] < sigma:
                fy[i] += k_wall * (sigma - ys[i])
            # Top wall
            if ys[i] > h - sigma:
                fy[i] -= k_wall * (ys[i] - (h - sigma))
        
        # Velocity Verlet update with damping
        vs[:, 0] = 0.95 * (vs[:, 0] + fx * dt)
        vs[:, 1] = 0.95 * (vs[:, 1] + fy * dt)
        xs += vs[:, 0] * dt
        ys += vs[:, 1] * dt
        
        # Hard clamp to stay in bounds
        xs = np.clip(xs, 0.01, w - 0.01)
        ys = np.clip(ys, 0.01, h - 0.01)
    
    return xs, ys


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


def _make_result_from_lp(w, h, xs, ys):
    """Build a valid (N,3) result using per-circle LP radii, then scale to perimeter <= 4."""
    rs = _solve_lp_radii(xs, ys, w, h)
    result = np.zeros((N, 3))
    result[:, 0] = xs
    result[:, 1] = ys
    result[:, 2] = rs
    # ensure non-negative
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


def _is_valid(result, tol=1e-6):
    """Check if result is a valid packing."""
    xs = result[:, 0]
    ys = result[:, 1]
    rs = result[:, 2]
    
    if np.any(rs < 0):
        return False
    
    min_x = np.min(xs - rs)
    max_x = np.max(xs + rs)
    min_y = np.min(ys - rs)
    max_y = np.max(ys + rs)
    
    perim = 2.0 * ((max_x - min_x) + (max_y - min_y))
    if perim > 4.0 + tol:
        return False
    
    # Check overlaps
    for i in range(N):
        for j in range(i + 1, N):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < rs[i] + rs[j] - tol:
                return False
    return True


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill
    
    best_value = -np.inf
    best_result = None
    
    random.seed(42)
    np.random.seed(42)
    
    # Generate initial configurations
    inits = []
    
    # Hexagonal configurations with different row structures
    hex_configs = [
        [5, 5, 5, 4, 2],   # 21 circles
        [4, 5, 5, 5, 2],
        [3, 5, 5, 5, 3],
        [5, 4, 5, 4, 3],
        [2, 5, 5, 5, 4],
    ]
    for rows in hex_configs:
        if sum(rows) >= N:
            w, h = 1.0, 1.0
            xs, ys, w_new, h_new = _make_hex(rows, w, h)
            inits.append(('hex', w_new, h_new, xs, ys))
    
    # Grid configurations
    grid_configs = [
        (4, 6),   # 4x6 = 24, take 21
        (5, 5),   # 5x5 = 25, take 21
        (3, 7),   # 3x7 = 21
        (7, 3),
        (6, 4),
    ]
    for rows, cols in grid_configs:
        w, h = 1.0, 1.0
        xs, ys, w_new, h_new = _make_grid(rows, cols, w, h)
        inits.append(('grid', w_new, h_new, xs, ys))
    
    # Random initializations
    for seed in [1, 2, 3, 4, 5]:
        w = 1.0
        h = 1.0
        xs, ys, w_new, h_new = _make_random(w, h, seed)
        inits.append(('random', w_new, h_new, xs, ys))
    
    # Physics-based initializations
    for seed in [10, 20, 30]:
        w = 1.0
        h = 1.0
        xs, ys = _physics_settle(w, h, seed, max_steps=2000)
        inits.append(('physics', w, h, xs, ys))
    
    # Try each initialization
    for init_type, w, h, xs, ys in inits:
        if time.time() > deadline - 10:
            break
        
        # Method 1: LP radii solver
        result_lp = _make_result_from_lp(w, h, xs, ys)
        if _is_valid(result_lp):
            val = np.sum(result_lp[:, 2])
            if val > best_value:
                best_value = val
                best_result = result_lp.copy()
        
        # Method 2: Equal radius SLSQP refinement
        if time.time() > deadline - 15:
            continue
        
        r_max = _max_equal_radius(xs, ys, w, h)
        if r_max > 0.01:
            w_r, h_r, r_r, xs_r, ys_r = _refine_slsqp(w, h, xs, ys, r_max, deadline)
            if time.time() > deadline - 5:
                continue
            
            result_eq = _make_result_equal(w_r, h_r, xs_r, ys_r, r_r)
            if _is_valid(result_eq):
                val = np.sum(result_eq[:, 2])
                if val > best_value:
                    best_value = val
                    best_result = result_eq.copy()
    
    # If we have a valid best, try to refine with more iterations
    if best_result is not None and time.time() < deadline - 20:
        # Try to improve the best result by perturbing and re-solving
        for trial in range(3):
            if time.time() > deadline - 10:
                break
            
            # Extract current best centers
            xs = best_result[:, 0].copy()
            ys = best_result[:, 1].copy()
            rs = best_result[:, 2].copy()
            
            # Compute bounding box
            min_x = np.min(xs - rs)
            max_x = np.max(xs + rs)
            min_y = np.min(ys - rs)
            max_y = np.max(ys + rs)
            w = max_x - min_x
            h = max_y - min_y
            
            if w <= 0 or h <= 0:
                continue
            
            # Normalize to [0, w] x [0, h]
            xs = xs - min_x
            ys = ys - min_y
            
            # Try LP again with slight perturbation
            rng = np.random.RandomState(42 + trial)
            xs_pert = xs + rng.randn(N) * 0.01 * w
            ys_pert = ys + rng.randn(N) * 0.01 * h
            
            result_pert = _make_result_from_lp(w, h, xs_pert, ys_pert)
            if _is_valid(result_pert):
                val = np.sum(result_pert[:, 2])
                if val > best_value:
                    best_value = val
                    best_result = result_pert.copy()
    
    # Fallback if no valid solution found
    if best_result is None:
        w = 1.0
        h = 1.0
        result = np.zeros((N, 3))
        # Simple 5x5 grid, take first 21
        side = 5
        for i in range(N):
            row = i // side
            col = i % side
            result[i, 0] = (col + 0.5) / side
            result[i, 1] = (row + 0.5) / side
            result[i, 2] = 0.48 / side
        _scale_to_perimeter(result)
        best_result = result
    
    return best_result
