"""
Optimization approach: Convex position parameterization using sorted angles on unit circle.
Reduces search space from 2N to N dimensions, automatically enforcing convex hull constraint.
Uses Nelder-Mead optimization on angular parameters to maximize minimum triangle area.
"""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 14

def _angles_to_points(angles):
    """
    Convert sorted angles to points on unit circle.
    Angles must be in [0, 2*pi) and sorted.
    """
    angles = np.sort(angles % (2 * np.pi))
    return np.column_stack([np.cos(angles), np.sin(angles)])

def _points_to_angles(points):
    """
    Convert points to their angles (assuming points are on or near unit circle).
    """
    angles = np.arctan2(points[:, 1], points[:, 0])
    return angles % (2 * np.pi)

def _min_triangle_area(points):
    """
    Calculate the minimum area of any triangle formed by triples of points.
    Vectorized for speed.
    """
    n = len(points)
    if n < 3:
        return 0.0
    x = points[:, 0]
    y = points[:, 1]
    min_area = float('inf')
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            dx_ji = x[j] - x[i]
            dy_ji = y[j] - y[i]
            dx_ki = x[j+1:] - x[i]
            dy_ki = y[j+1:] - y[i]
            cross = dx_ji * dy_ki - dy_ji * dx_ki
            areas = 0.5 * np.abs(cross)
            local_min = np.min(areas)
            if local_min < min_area:
                min_area = local_min
    return min_area

def _hull_area(points):
    """
    Calculate the area of the convex hull of the points.
    """
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """
    Calculate the score: min_triangle_area / hull_area.
    """
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_area = _min_triangle_area(points)
    return min_area / hull_area

def _objective_angles(angle_params):
    """
    Objective function for minimization on angular parameters.
    We want to maximize the score, so we minimize the inverse.
    angle_params: array of N angles (not necessarily sorted)
    """
    if not np.all(np.isfinite(angle_params)):
        return 1e10
    
    # Convert angles to points
    pts = _angles_to_points(angle_params)
    
    # Check for degenerate configurations
    hull_area = _hull_area(pts)
    if hull_area < 1e-10:
        return 1e10
        
    min_area = _min_triangle_area(pts)
    if min_area < 1e-12:
        return 1e10
        
    # Minimize hull_area / min_area is equivalent to maximizing min_area / hull_area
    return hull_area / min_area

def _generate_initial_angle_configs():
    """
    Generate a diverse set of initial angle configurations.
    """
    configs = []
    rng = np.random.default_rng(42)
    
    # 1. Regular 14-gon (equally spaced angles)
    angles_reg = np.linspace(0, 2 * np.pi, N, endpoint=False)
    configs.append(angles_reg.copy())
    
    # 2. Jittered regular polygon
    for std in [0.05, 0.1, 0.15, 0.2, 0.3]:
        jitter = rng.normal(0, std, size=N)
        configs.append(angles_reg + jitter)
    
    # 3. Random configurations
    for _ in range(15):
        angles = rng.uniform(0, 2 * np.pi, N)
        configs.append(angles)
    
    # 4. Clustered configurations (some points close together)
    for _ in range(5):
        base = rng.uniform(0, 2 * np.pi, N // 2)
        jitter = rng.normal(0, 0.1, size=N // 2)
        clustered = np.sort(np.concatenate([base, base + jitter]))
        configs.append(clustered % (2 * np.pi))
    
    # 5. Two-group configurations
    for gap in [0.5, 1.0, 1.5, 2.0]:
        angles1 = np.linspace(0, gap, N // 2, endpoint=False)
        angles2 = np.linspace(gap + 0.1, 2 * np.pi - 0.1, N - N // 2, endpoint=False)
        configs.append(np.sort(np.concatenate([angles1, angles2])) % (2 * np.pi))
    
    return configs

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    best_score = 0.0
    best_pts = None
    best_obj = float('inf')
    best_angles = None
    
    def update_best(pts, angles=None):
        nonlocal best_score, best_pts, best_obj, best_angles
        if pts is None:
            return
        pts = np.asarray(pts, dtype=float)
        if pts.shape != (N, 2) or not np.all(np.isfinite(pts)):
            return
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts.copy()
            best_obj = _objective_angles(angles) if angles is not None else _objective_angles(_points_to_angles(pts))
            if angles is not None:
                best_angles = angles.copy()

    # Phase 1: Multi-start Nelder-Mead on angular parameters
    inits = _generate_initial_angle_configs()
    
    for i, angles0 in enumerate(inits):
        if time.time() > deadline - 40:
            break
        try:
            res = minimize(_objective_angles, angles0, method='Nelder-Mead',
                           options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
            if res.fun < best_obj:
                # Convert optimized angles back to points
                pts = _angles_to_points(res.x)
                update_best(pts, res.x)
        except Exception:
            pass

    # Phase 2: Additional local refinement with different optimizers
    if best_angles is not None and time.time() < deadline - 20:
        try:
            # Powell's method for refinement
            res_powell = minimize(_objective_angles, best_angles, method='Powell',
                                  options={'maxiter': 2000, 'xtol': 1e-10, 'ftol': 1e-12})
            if res_powell.fun < best_obj:
                pts = _angles_to_points(res_powell.x)
                update_best(pts, res_powell.x)
        except Exception:
            pass

    # Phase 3: Final high-precision polish
    if best_angles is not None and time.time() < deadline - 5:
        try:
            res_final = minimize(_objective_angles, best_angles, method='Nelder-Mead',
                                 options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})
            if res_final.fun < best_obj:
                pts = _angles_to_points(res_final.x)
                update_best(pts, res_final.x)
        except Exception:
            pass

    # Fallback to regular polygon if nothing worked
    if best_pts is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_pts = _angles_to_points(angles)

    # Normalize to unit area
    # The score is invariant to scaling, but the task says "normalized to unit area".
    # We should return points such that their convex hull has area 1.
    try:
        hull = ConvexHull(best_pts)
        area = hull.volume
        if area > 1e-10:
            scale = np.sqrt(1.0 / area)
            best_pts = best_pts * scale
    except Exception:
        pass

    return best_pts
