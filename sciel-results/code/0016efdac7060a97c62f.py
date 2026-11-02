import numpy as np
import time
import scipy.optimize as opt
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13
TRIPLES = [(i, j, k) for i in range(N) for j in range(i + 1, N) for k in range(j + 1, N)]

# Precompute indices for fast triangle area computation
TRI_I = np.array([t[0] for t in TRIPLES])
TRI_J = np.array([t[1] for t in TRIPLES])
TRI_K = np.array([t[2] for t in TRIPLES])


def _convex_hull_area(points):
    """Compute convex hull area using scipy.spatial.ConvexHull."""
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume is area
    except Exception:
        # Fallback: sort by angle and use shoelace
        cx, cy = points[:, 0].mean(), points[:, 1].mean()
        angles = np.arctan2(points[:, 1] - cy, points[:, 0] - cx)
        sorted_idx = np.argsort(angles)
        p = points[sorted_idx]
        x, y = p[:, 0], p[:, 1]
        xn, yn = np.roll(x, -1), np.roll(y, -1)
        return 0.5 * abs(np.sum(x * yn - xn * y))


def _min_triangle_area(points):
    """Compute minimum triangle area among all triples."""
    pi = points[TRI_I]
    pj = points[TRI_J]
    pk = points[TRI_K]
    v1x = pj[:, 0] - pi[:, 0]
    v1y = pj[:, 1] - pi[:, 1]
    v2x = pk[:, 0] - pi[:, 0]
    v2y = pk[:, 1] - pi[:, 1]
    cross = np.abs(v1x * v2y - v1y * v2x)
    return 0.5 * np.min(cross)


def _score(points):
    """Compute min triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    min_area = _min_triangle_area(points)
    return min_area / hull_area


def _neg_score(x_flat):
    """Objective for minimization: negative score."""
    points = x_flat.reshape(N, 2)
    return -_score(points)


def _make_symmetric_config(n_outer, n_inner, r_inner, phase_outer, phase_inner):
    """Create points on two concentric regular polygons."""
    pts = []
    if n_outer > 0:
        theta_o = np.linspace(phase_outer, 2 * np.pi + phase_outer, n_outer, endpoint=False)
        pts.extend(zip(np.cos(theta_o), np.sin(theta_o)))
    if n_inner > 0:
        theta_i = np.linspace(phase_inner, 2 * np.pi + phase_inner, n_inner, endpoint=False)
        pts.extend(zip(r_inner * np.cos(theta_i), r_inner * np.sin(theta_i)))
    return np.array(pts)


def _make_hex_symmetric(n_on_hex, n_on_axis, hex_radius, axis_scale, phase_hex, phase_axis):
    """
    Points on a regular hexagon (n_on_hex vertices) plus points on 3 symmetry axes
    (n_on_axis points distributed along the 3 axes).
    """
    pts = []
    if n_on_hex > 0:
        theta_h = np.linspace(phase_hex, 2 * np.pi + phase_hex, n_on_hex, endpoint=False)
        pts.extend(zip(hex_radius * np.cos(theta_h), hex_radius * np.sin(theta_h)))
    if n_on_axis > 0:
        # 3 axes at 0, 60, 120 degrees (or 30, 90, 150 for the other set)
        axis_angles = np.array([0, 2 * np.pi / 3, 4 * np.pi / 3]) + phase_axis
        # Distribute n_on_axis points along the 3 axes
        per_axis = [n_on_axis // 3 + (1 if i < n_on_axis % 3 else 0) for i in range(3)]
        for ax in range(3):
            n_ax = per_axis[ax]
            if n_ax == 0:
                continue
            # Place points at radial distances along the axis
            radii = np.linspace(axis_scale * 0.2, axis_scale, n_ax)
            for r in radii:
                pts.append((r * np.cos(axis_angles[ax]), r * np.sin(axis_angles[ax])))
    return np.array(pts)


def _make_heptagon_symmetric(n_on_hept, n_on_axis, hept_radius, axis_scale, phase_hept, phase_axis):
    """Points on a regular heptagon plus points on symmetry axes."""
    pts = []
    if n_on_hept > 0:
        theta_h = np.linspace(phase_hept, 2 * np.pi + phase_hept, n_on_hept, endpoint=False)
        pts.extend(zip(hept_radius * np.cos(theta_h), hept_radius * np.sin(theta_h)))
    if n_on_axis > 0:
        # Heptagon has 7-fold symmetry; use 7 axes
        axis_angles = np.linspace(0, 2 * np.pi, 7, endpoint=False) + phase_axis
        per_axis = [n_on_axis // 7 + (1 if i < n_on_axis % 7 else 0) for i in range(7)]
        for ax in range(7):
            n_ax = per_axis[ax]
            if n_ax == 0:
                continue
            radii = np.linspace(axis_scale * 0.3, axis_scale, n_ax)
            for r in radii:
                pts.append((r * np.cos(axis_angles[ax]), r * np.sin(axis_angles[ax])))
    return np.array(pts)


def _optimize_from_start(x0, deadline, max_iter=300):
    """Run L-BFGS-B optimization from x0, respecting deadline."""
    if time.time() > deadline:
        return None, -np.inf
    
    stop_flag = [False]
    
    def callback(xk):
        if time.time() > deadline:
            stop_flag[0] = True
            return True
        return False
    
    try:
        res = opt.minimize(_neg_score, x0, method='L-BFGS-B',
                          bounds=[(-2.0, 2.0)] * (2 * N),
                          options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10},
                          callback=callback)
        if not stop_flag[0]:
            pts = res.x.reshape(N, 2)
            s = _score(pts)
            return pts, s
    except Exception:
        pass
    return None, -np.inf


def _nelder_mead_polish(x0, deadline, max_iter=200):
    """Polish with Nelder-Mead."""
    if time.time() > deadline:
        return None, -np.inf
    
    stop_flag = [False]
    
    def callback(xk):
        if time.time() > deadline:
            stop_flag[0] = True
            return True
        return False
    
    try:
        res = opt.minimize(_neg_score, x0, method='Nelder-Mead',
                          options={'maxiter': max_iter, 'adaptive': True, 'xatol': 1e-12, 'fatol': 1e-12},
                          callback=callback)
        if not stop_flag[0]:
            pts = res.x.reshape(N, 2)
            s = _score(pts)
            return pts, s
    except Exception:
        pass
    return None, -np.inf


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin
    
    best_score = -np.inf
    best_points = None
    
    # Generate diverse initial configurations
    initial_configs = []
    
    # Regular 13-gon
    theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
    initial_configs.append(np.column_stack((np.cos(theta), np.sin(theta))))
    
    # Concentric polygon configs
    for n_o, n_i, r_i, ph_o, ph_i in [
        (7, 6, 0.5, 0, 0),
        (7, 6, 0.5, 0, np.pi / 7),
        (7, 6, 0.4, 0, np.pi / 6),
        (7, 6, 0.6, 0, np.pi / 7),
        (6, 7, 0.5, 0, 0),
        (6, 7, 0.5, 0, np.pi / 6),
        (8, 5, 0.5, 0, 0),
        (8, 5, 0.5, 0, np.pi / 8),
        (5, 8, 0.5, 0, 0),
        (5, 8, 0.5, 0, np.pi / 5),
        (9, 4, 0.5, 0, 0),
        (4, 9, 0.5, 0, 0),
    ]:
        initial_configs.append(_make_symmetric_config(n_o, n_i, r_i, ph_o, ph_i))
    
    # Hexagon-symmetric configs
    for n_h, n_ax, r_h, s_ax, ph_h, ph_ax in [
        (6, 7, 1.0, 0.5, 0, 0),
        (6, 7, 1.0, 0.5, 0, np.pi / 6),
        (6, 7, 1.0, 0.7, 0, np.pi / 6),
        (6, 7, 0.8, 0.5, 0, np.pi / 6),
        (5, 8, 1.0, 0.5, 0, 0),
        (5, 8, 1.0, 0.5, np.pi / 5, np.pi / 6),
        (7, 6, 1.0, 0.4, 0, 0),
        (7, 6, 1.0, 0.4, np.pi / 7, np.pi / 6),
    ]:
        initial_configs.append(_make_hex_symmetric(n_h, n_ax, r_h, s_ax, ph_h, ph_ax))
    
    # Heptagon-symmetric configs
    for n_h, n_ax, r_h, s_ax, ph_h, ph_ax in [
        (7, 6, 1.0, 0.5, 0, 0),
        (7, 6, 1.0, 0.5, 0, np.pi / 7),
        (7, 6, 1.0, 0.7, 0, np.pi / 7),
        (6, 7, 1.0, 0.5, 0, 0),
        (6, 7, 1.0, 0.5, np.pi / 6, np.pi / 7),
    ]:
        initial_configs.append(_make_heptagon_symmetric(n_h, n_ax, r_h, s_ax, ph_h, ph_ax))
    
    # Random perturbations of regular polygon
    np.random.seed(42)
    for _ in range(5):
        theta_p = np.linspace(0, 2 * np.pi, N, endpoint=False) + np.random.uniform(-0.1, 0.1, N)
        r_p = np.ones(N) + np.random.uniform(-0.1, 0.1, N)
        initial_configs.append(np.column_stack((r_p * np.cos(theta_p), r_p * np.sin(theta_p))))
    
    # Evaluate all initial configs
    config_scores = []
    for idx, config in enumerate(initial_configs):
        if time.time() > deadline - 30:
            break
        s = _score(config)
        config_scores.append((s, idx))
        if s > best_score:
            best_score = s
            best_points = config.copy()
    
    config_scores.sort(reverse=True)
    
    # Optimize from top configurations
    n_tried = 0
    for s_init, idx in config_scores:
        if time.time() > deadline - 20:
            break
        if n_tried >= 8:
            break
        x0 = initial_configs[idx].flatten()
        pts, s = _optimize_from_start(x0, deadline, max_iter=200)
        n_tried += 1
        if pts is not None and s > best_score:
            best_score = s
            best_points = pts
    
    # Polish with Nelder-Mead if time remains
    if best_points is not None and time.time() < deadline - 10:
        pts, s = _nelder_mead_polish(best_points.flatten(), deadline, max_iter=150)
        if pts is not None and s > best_score:
            best_score = s
            best_points = pts
    
    # Final safety: if no valid points, use regular polygon
    if best_points is None:
        theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack((np.cos(theta), np.sin(theta)))
    
    return best_points
