"""Rewritten program: concentric-ring parametric packing with local refinement."""
import time
import numpy as np
from scipy.optimize import minimize

N = 50

# Deadline guard: 150s wall-clock limit, keep safety margin
_DEADLINE = 125.0  # seconds


def _compute_radius(points):
    """Compute the feasible radius for a set of points."""
    n = len(points)
    if n < 2:
        return 0.0
    # Container constraint
    dists = np.sqrt(np.sum(points ** 2, axis=1))
    r_container = 1.0 - np.max(dists)
    if r_container < 0:
        return 0.0
    # Pairwise constraint
    # Compute all pairwise distances
    dx = points[:, 0:1] - points[0:1, 0]
    dy = points[:, 1:2] - points[0:1, 1]
    # Only need upper triangle
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    d = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d)
    r_pair = min_pair / 2.0
    return min(r_container, r_pair)


def _build_initial_rings():
    """Build initial concentric ring configuration: 1+6+12+18+13 = 50."""
    rings = [1, 6, 12, 18, 13]
    # Initial radii (approximate)
    radii = [0.0, 0.35, 0.65, 0.88, 1.0]
    # For each ring, angles are evenly spaced with an offset
    points = []
    for ring_idx, count in enumerate(rings):
        r = radii[ring_idx]
        if count == 1:
            points.append([0.0, 0.0])
        else:
            # Offset angle to stagger rings
            offset = np.pi / count if ring_idx % 2 == 0 else 0.0
            angles = np.linspace(0, 2 * np.pi, count, endpoint=False) + offset
            for a in angles:
                points.append([r * np.cos(a), r * np.sin(a)])
    return np.array(points[:N])


def _params_to_points(params):
    """Convert parameter vector to point coordinates.
    
    Parameters: [r1, r2, ..., r5, theta1_1, theta1_2, ..., theta5_13]
    Ring 1: 1 point (center) - no angle needed
    Ring 2: 6 points - 6 angles
    Ring 3: 12 points - 12 angles
    Ring 4: 18 points - 18 angles
    Ring 5: 13 points - 13 angles
    Total: 5 radii + 49 angles = 54 params
    """
    rings = [1, 6, 12, 18, 13]
    points = []
    param_idx = 0
    for ring_idx, count in enumerate(rings):
        r = params[param_idx]
        param_idx += 1
        # Clamp radius to [0, 1]
        r = max(0.0, min(1.0, r))
        if count == 1:
            points.append([0.0, 0.0])
        else:
            for j in range(count):
                theta = params[param_idx]
                param_idx += 1
                points.append([r * np.cos(theta), r * np.sin(theta)])
    return np.array(points[:N])


def _objective(params):
    """Objective: negative of the feasible radius (minimize)."""
    points = _params_to_points(params)
    r = _compute_radius(points)
    return -r


def _refine_local(points, time_budget=30.0):
    """Local refinement using direct optimization on point coordinates."""
    start = time.time()
    best_points = points.copy()
    best_r = _compute_radius(points)
    
    # Use L-BFGS-B to maximize the minimum distance
    def neg_radius(pts_flat):
        pts = pts_flat.reshape(-1, 2)
        return -_compute_radius(pts)
    
    x0 = points.flatten()
    
    try:
        result = minimize(
            neg_radius, x0, method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        if result.fun < -best_r:
            best_points = result.x.reshape(-1, 2)
            best_r = -result.fun
    except Exception:
        pass
    
    elapsed = time.time() - start
    if elapsed > time_budget:
        return best_points
    
    # Try a second refinement pass with tighter tolerance
    start2 = time.time()
    try:
        result2 = minimize(
            neg_radius, best_points.flatten(), method='L-BFGS-B',
            options={'maxiter': 100, 'ftol': 1e-16, 'gtol': 1e-12}
        )
        if result2.fun < -best_r:
            best_points = result2.x.reshape(-1, 2)
            best_r = -result2.fun
    except Exception:
        pass
    
    return best_points


def construct():
    """Construct 50 circle centers in unit circle to maximize common radius."""
    t_start = time.time()
    
    # Build initial ring configuration
    points = _build_initial_rings()
    current_r = _compute_radius(points)
    
    # Optimize ring radii and angles using L-BFGS-B
    # Parameters: 5 radii + 49 angles = 54
    rings = [1, 6, 12, 18, 13]
    n_angles = sum(c - 1 for c in rings)  # 49
    n_params = 5 + n_angles
    
    # Initial params from the ring configuration
    initial_radii = [0.0, 0.35, 0.65, 0.88, 1.0]
    params = np.zeros(n_params)
    params[0:5] = initial_radii
    param_idx = 5
    for ring_idx, count in enumerate(rings):
        if count > 1:
            r = initial_radii[ring_idx]
            offset = np.pi / count if ring_idx % 2 == 0 else 0.0
            for j in range(count):
                theta = 2 * np.pi * j / count + offset
                params[param_idx] = theta
                param_idx += 1
    
    # Set bounds
    bounds = []
    for i in range(5):
        bounds.append((0.0, 1.0))
    for i in range(n_angles):
        bounds.append((0.0, 2 * np.pi))
    
    # First optimization pass
    elapsed = time.time() - t_start
    if elapsed < _DEADLINE - 40:
        try:
            result = minimize(
                _objective, params, method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if result.fun < -current_r:
                points = _params_to_points(result.x)
                current_r = -result.fun
        except Exception:
            pass
    
    # Second optimization pass from improved point
    elapsed = time.time() - t_start
    if elapsed < _DEADLINE - 30:
        # Re-derive params from current points
        # Simply use current points for local refinement
        points = _refine_local(points, time_budget=min(25.0, _DEADLINE - (time.time() - t_start) - 5))
        current_r = _compute_radius(points)
    
    # Final high-precision polish
    elapsed = time.time() - t_start
    if elapsed < _DEADLINE - 10:
        def neg_radius_final(pts_flat):
            pts = pts_flat.reshape(-1, 2)
            return -_compute_radius(pts)
        
        try:
            result_final = minimize(
                neg_radius_final, points.flatten(), method='L-BFGS-B',
                options={'maxiter': 50, 'ftol': 1e-16, 'gtol': 1e-12}
            )
            if result_final.fun < -current_r:
                points = result_final.x.reshape(-1, 2)
                current_r = -result_final.fun
        except Exception:
            pass
    
    # Validate: ensure all points are inside the unit circle
    dists = np.sqrt(np.sum(points ** 2, axis=1))
    # If any point is outside, scale down
    max_dist = np.max(dists)
    if max_dist > 0.999:
        scale = 0.999 / max_dist
        points = points * scale
    
    # Ensure finite values
    if not np.all(np.isfinite(points)):
        points = _build_initial_rings()
    
    return points
