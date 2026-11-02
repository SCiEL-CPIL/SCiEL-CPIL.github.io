"""Heilbronn convex regions n=14 – Feasibility-based L-BFGS optimization."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 14
M = N // 2  # 7 independent points for central symmetry

# ─── fast area helpers ────────────────────────────────────────────────────────

def _min_triangle_area_vec(points):
    """Vectorized minimum triangle area (not doubled) for N points."""
    n = len(points)
    min_area = float('inf')
    x = points[:, 0]
    y = points[:, 1]
    for i in range(n):
        for j in range(i + 1, n):
            if j + 1 >= n:
                continue
            dx_ji = x[j] - x[i]
            dy_ji = y[j] - y[i]
            dx_ki = x[j+1:] - x[i]
            dy_ki = y[j+1:] - y[i]
            cross = dx_ji * dy_ki - dx_ki * dy_ji
            areas = 0.5 * np.abs(cross)
            local_min = np.min(areas)
            if local_min < min_area:
                min_area = local_min
    return min_area

def _hull_area_fast(points):
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    hull_area = _hull_area_fast(points)
    if hull_area < 1e-10:
        return 0.0
    min_area = _min_triangle_area_vec(points)
    return min_area / hull_area

def _all_triangle_areas(points):
    n = len(points)
    x = points[:, 0]
    y = points[:, 1]
    areas_list = []
    for i in range(n):
        for j in range(i + 1, n):
            if j + 1 >= n:
                continue
            dx_ji = x[j] - x[i]
            dy_ji = y[j] - y[i]
            dx_ki = x[j+1:] - x[i]
            dy_ki = y[j+1:] - y[i]
            cross = dx_ji * dy_ki - dx_ki * dy_ji
            tri_areas = 0.5 * np.abs(cross)
            if len(tri_areas) > 0:
                areas_list.append(tri_areas)
    if not areas_list:
        return np.array([0.0])
    return np.concatenate(areas_list)

# ─── central-symmetric parameterization ───────────────────────────────────────

def _decode_central_symmetric(params):
    """
    params: shape (M, 2) – M independent points.
    Returns: shape (N, 2) – full centrally-symmetric set, sorted by angle from origin.
    Central symmetry: if p is in the set, -p is also in the set.
    """
    pts = np.vstack([params, -params])  # (N, 2)
    # Sort by angle from origin for consistent convex ordering
    angles = np.arctan2(pts[:, 1], pts[:, 0])
    order = np.argsort(angles)
    return pts[order]

# ─── New Feasibility-Based Objective ─────────────────────────────────────────

def _objective_feasibility_symmetric(params_flat, penalty_weight=100.0):
    """
    Maximize the minimum triangle area, penalizing deviation from unit hull area.
    This is equivalent to maximizing min_area / hull_area but more stable for gradients.
    We minimize: -min_area + penalty_weight * (hull_area - 1)^2
    """
    params = params_flat.reshape(M, 2)
    if np.any(np.isnan(params)) or np.any(np.isinf(params)):
        return 1e10
    
    points = _decode_central_symmetric(params)
    min_area = _min_triangle_area_vec(points)
    hull_area = _hull_area_fast(points)
    
    if hull_area < 1e-10 or min_area < 1e-12:
        return 1e10
    
    # We want to maximize min_area, so we minimize -min_area.
    # We want hull_area to be 1, so we add penalty (hull_area - 1)^2.
    # Note: The score is min_area / hull_area.
    # If we fix hull_area=1, maximizing min_area is equivalent.
    # The penalty guides the optimizer to keep hull_area near 1.
    return -min_area + penalty_weight * (hull_area - 1.0)**2

def _objective_feasibility_cartesian(points_flat, penalty_weight=100.0):
    """
    Same as above but for full Cartesian coordinates.
    """
    points = points_flat.reshape(N, 2)
    if np.any(np.isnan(points)) or np.any(np.isinf(points)):
        return 1e10
    
    min_area = _min_triangle_area_vec(points)
    hull_area = _hull_area_fast(points)
    
    if hull_area < 1e-10 or min_area < 1e-12:
        return 1e10
        
    return -min_area + penalty_weight * (hull_area - 1.0)**2

# ─── normalization ────────────────────────────────────────────────────────────

def _normalize_unit_area(points):
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area > 1e-10:
            scale = np.sqrt(1.0 / hull_area)
            return points * scale
    except Exception:
        pass
    xs = points[:, 0]
    ys = points[:, 1]
    w = xs.max() - xs.min()
    h = ys.max() - ys.min()
    if w > 1e-10 and h > 1e-10:
        scale = 1.0 / np.sqrt(w * h)
        return points * scale
    return points

# ─── initializations in symmetric space ───────────────────────────────────────

def _generate_sym_inits():
    """Generate diverse initializations in the M×2 symmetric parameter space."""
    inits = []
    rng = np.random.default_rng(42)

    # Regular M-gon (first half of regular 14-gon)
    angles_m = np.linspace(0, np.pi, M, endpoint=False)
    reg = np.column_stack([np.cos(angles_m), np.sin(angles_m)])
    inits.append(reg.flatten())

    # Rotated regular M-gons
    for rot in [0.1, 0.2, 0.4, 0.7, 1.0, 1.3, 1.6]:
        a = angles_m + rot
        pts = np.column_stack([np.cos(a), np.sin(a)])
        inits.append(pts.flatten())

    # Jittered regular M-gon
    for jitter in [0.05, 0.1, 0.15, 0.2, 0.3, 0.5]:
        jittered = reg + rng.normal(0, jitter, size=(M, 2))
        inits.append(jittered.flatten())

    # Two-ring: inner + outer (symmetric pairs)
    for r_inner in [0.3, 0.4, 0.5, 0.6, 0.7]:
        for n_in in [3, 4, 5]:
            n_out = M - n_in
            a_in = np.linspace(0, np.pi, n_in, endpoint=False)
            a_out = np.linspace(0, np.pi, n_out, endpoint=False) + np.pi / max(n_out, 1)
            pts_in = r_inner * np.column_stack([np.cos(a_in), np.sin(a_in)])
            pts_out = np.column_stack([np.cos(a_out), np.sin(a_out)])
            pts = np.vstack([pts_in, pts_out])
            inits.append(pts.flatten())

    # Elliptical
    for ax, ay in [(1.0, 0.5), (1.0, 0.7), (1.2, 0.8), (1.3, 0.9), (1.5, 0.7)]:
        a_e = np.linspace(0, np.pi, M, endpoint=False)
        pts = np.column_stack([ax * np.cos(a_e), ay * np.sin(a_e)])
        inits.append(pts.flatten())

    # Alternating radii
    for alt in [0.1, 0.2, 0.3, 0.4]:
        a_alt = np.linspace(0, np.pi, M, endpoint=False)
        radii = np.ones(M)
        radii[::2] = 1.0 + alt
        radii[1::2] = 1.0 - alt
        pts = np.column_stack([radii * np.cos(a_alt), radii * np.sin(a_alt)])
        inits.append(pts.flatten())

    # Random convex-like (sort by angle, keep in upper half for symmetry)
    for seed in range(10):
        rng_s = np.random.default_rng(seed * 1000 + 7)
        pts_raw = rng_s.uniform(-1, 1, size=(M, 2))
        for i in range(M):
            if pts_raw[i, 1] < 0:
                pts_raw[i, 1] = -pts_raw[i, 1]
        angles_c = np.arctan2(pts_raw[:, 1], pts_raw[:, 0])
        sorted_idx = np.argsort(angles_c)
        inits.append(pts_raw[sorted_idx].flatten())

    # Fibonacci-like
    golden_angle = 2.399963229728653
    for r_scale in [1.0, 0.8, 1.2]:
        pts_fib = np.zeros((M, 2))
        for i in range(M):
            angle = i * golden_angle % np.pi
            r = r_scale * np.sqrt((i + 0.5) / M)
            pts_fib[i, 0] = r * np.cos(angle)
            pts_fib[i, 1] = r * np.sin(angle)
        inits.append(pts_fib.flatten())

    # Pure random
    for seed in range(12):
        rng_s = np.random.default_rng(seed * 777 + 13)
        pts_rand = rng_s.uniform(-1, 1, size=(M, 2))
        angles_r = np.arctan2(pts_rand[:, 1], pts_rand[:, 0])
        sorted_idx = np.argsort(angles_r)
        inits.append(pts_rand[sorted_idx].flatten())

    return inits

# ─── main construct ───────────────────────────────────────────────────────────

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin

    best_points = None
    best_score = 0.0

    def _update(points):
        nonlocal best_points, best_score
        s = _score(points)
        if s > best_score:
            best_score = s
            best_points = points.copy()

    inits = _generate_sym_inits()

    # Phase 1: L-BFGS from diverse symmetric initializations
    # This is faster and more precise than DE or NM for this smooth-ish objective.
    for x0 in inits:
        remaining = deadline - time.time()
        if remaining < 5:
            break
        try:
            res = minimize(
                _objective_feasibility_symmetric,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-11}
            )
            pts = _decode_central_symmetric(res.x.reshape(M, 2))
            _update(pts)
        except Exception:
            pass

    # Phase 2: Polish best symmetric solution with higher precision
    if best_points is not None:
        bp = best_points.copy()
        angles_bp = np.arctan2(bp[:, 1], bp[:, 0])
        order_bp = np.argsort(angles_bp)
        bp_sorted = bp[order_bp]
        sym_params_best = bp_sorted[:M]

        remaining = deadline - time.time()
        if remaining > 5:
            try:
                res_polish = minimize(
                    _objective_feasibility_symmetric,
                    sym_params_best.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-12}
                )
                pts_polish = _decode_central_symmetric(res_polish.x.reshape(M, 2))
                _update(pts_polish)
            except Exception:
                pass

    # Phase 3: Break symmetry – full 28D Cartesian polish
    # Sometimes breaking symmetry allows for a better packing.
    if best_points is not None:
        cart_flat = best_points.flatten()
        
        remaining = deadline - time.time()
        if remaining > 10:
            try:
                res_cart = minimize(
                    _objective_feasibility_cartesian,
                    cart_flat,
                    method='L-BFGS-B',
                    options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-11}
                )
                _update(res_cart.x.reshape(N, 2))
            except Exception:
                pass

        # Another round of polishing in Cartesian space
        remaining = deadline - time.time()
        if remaining > 5:
            try:
                res_cart2 = minimize(
                    _objective_feasibility_cartesian,
                    best_points.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-12}
                )
                _update(res_cart2.x.reshape(N, 2))
            except Exception:
                pass

    # Fallback
    if best_points is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])

    # Normalize to unit hull area
    final_points = _normalize_unit_area(best_points)

    # Safety checks
    try:
        hull = ConvexHull(final_points)
        if hull.volume < 1e-10:
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
            final_points = _normalize_unit_area(np.column_stack([np.cos(angles), np.sin(angles)]))
    except Exception:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        final_points = _normalize_unit_area(np.column_stack([np.cos(angles), np.sin(angles)]))

    if np.any(np.isnan(final_points)) or np.any(np.isinf(final_points)):
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        final_points = _normalize_unit_area(np.column_stack([np.cos(angles), np.sin(angles)]))

    return final_points
