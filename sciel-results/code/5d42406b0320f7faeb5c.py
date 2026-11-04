"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time
from itertools import combinations

def _triangle_areas(points):
    """Compute areas of all triangles formed by triples of points."""
    n = len(points)
    areas = np.zeros((n, n, n))
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
                area = 0.5 * abs(cross)
                areas[i, j, k] = area
                areas[j, i, k] = area
                areas[j, k, i] = area
                areas[k, j, i] = area
                areas[k, i, j] = area
                areas[i, k, j] = area
    return areas

def _min_triangle_area(points):
    """Compute the minimum area of any triangle formed by 3 points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
                area = 0.5 * abs(cross)
                if area < min_area:
                    min_area = area
    return min_area

def _convex_hull_area(points):
    """Compute the area of the convex hull of the given points."""
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume is area
    except Exception:
        # Fallback: use shoelace formula on sorted points
        # This is a rough fallback
        return 1.0

def _score(points):
    """Compute the score: min triangle area / convex hull area."""
    min_area = _min_triangle_area(points)
    hull_area = _convex_hull_area(points)
    if hull_area <= 0:
        return 0.0
    return min_area / hull_area

def _objective(params, n=N):
    """Objective function to minimize: negative of the score (min triangle area / hull area)."""
    points = params.reshape(n, 2)
    score = _score(points)
    return -score

def _log_sum_exp_surrogate(params, n=N, beta=50.0):
    """
    Smooth surrogate using log-sum-exp to approximate min triangle area.
    We want to maximize the minimum area, so we minimize -log(sum(exp(-beta * area_i))).
    This is smooth and differentiable.
    """
    points = params.reshape(n, 2)
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
                area = 0.5 * abs(cross)
                areas.append(area)
    areas = np.array(areas)
    # Log-sum-exp trick for numerical stability
    max_area = np.max(areas)
    log_sum = np.log(np.sum(np.exp(-beta * (areas - max_area)))) + (-beta * max_area)
    # We want to maximize min area, so minimize -log_sum_exp
    # But we also need to account for hull area
    hull_area = _convex_hull_area(points)
    if hull_area <= 0:
        return 1e10
    # The surrogate for min area / hull_area
    # min_area ≈ -1/beta * log_sum_exp(-beta * areas)
    # So score ≈ min_area / hull_area
    # We want to maximize this, so minimize -score
    approx_min_area = -log_sum / beta
    return -approx_min_area / hull_area

def _objective_radial(params, n=N):
    """
    Objective for radial parameterization.
    params: array of shape (n, 2) where first n/2 are angles and rest are radii,
    or we use a different encoding.
    Actually, let's use: params = [angles (n,), radii (n,)]
    But we need to ensure convexity. A simpler approach:
    Use angles sorted, and radii > 0.
    """
    n = len(params) // 2
    angles = params[:n]
    radii = params[n:]
    # Sort angles to ensure convex ordering
    angles = np.sort(angles)
    # Compute points
    points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    return _log_sum_exp_surrogate(points.flatten(), n=n)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_points = None
    best_score = -float('inf')
    
    # Initial best: regular polygon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    regular_points = np.column_stack([np.cos(angles), np.sin(angles)])
    best_points = regular_points.copy()
    best_score = _score(best_points)
    
    # Multiple random starts with L-BFGS-B
    np.random.seed(42)
    
    n_starts = 30
    for s in range(n_starts):
        if time.time() > deadline - 5.0:
            break
        
        # Random initial points
        if s == 0:
            # Start from regular polygon with slight perturbation
            angles = np.linspace(0, 2*np.pi, N, endpoint=False) + np.random.normal(0, 0.1, N)
            radii = 1.0 + np.random.normal(0, 0.1, N)
            params = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)]).flatten()
        else:
            # Random points in unit disk
            angles = np.sort(np.random.uniform(0, 2*np.pi, N))
            radii = np.random.uniform(0.5, 1.5, N)
            params = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)]).flatten()
        
        try:
            result = minimize(
                _objective,
                params,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-12}
            )
            
            points = result.x.reshape(N, 2)
            score = _score(points)
            
            if score > best_score:
                best_score = score
                best_points = points.copy()
                
        except Exception as e:
            pass
    
    # Polish with direct optimization on the best found
    if time.time() < deadline - 10.0:
        try:
            result = minimize(
                _objective,
                best_points.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-14}
            )
            points = result.x.reshape(N, 2)
            score = _score(points)
            if score > best_score:
                best_score = score
                best_points = points.copy()
        except Exception:
            pass
    
    # Try radial parameterization for additional improvement
    if time.time() < deadline - 15.0:
        try:
            # Extract angles and radii from best_points
            cx = np.mean(best_points[:, 0])
            cy = np.mean(best_points[:, 1])
            rel_points = best_points - np.array([cx, cy])
            radii = np.sqrt(np.sum(rel_points**2, axis=1))
            angles = np.arctan2(rel_points[:, 1], rel_points[:, 0])
            
            # Sort by angle
            sort_idx = np.argsort(angles)
            angles = angles[sort_idx]
            radii = radii[sort_idx]
            
            # Ensure angles are in [0, 2*pi)
            angles = angles % (2*np.pi)
            angles = np.sort(angles)
            
            radial_params = np.concatenate([angles, radii])
            
            result = minimize(
                _objective_radial,
                radial_params,
                method='L-BFGS-B',
                bounds=[(0, 2*np.pi)] * N + [(0.01, None)] * N,
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-12}
            )
            
            # Convert back to points
            angles_opt = np.sort(result.x[:N]) % (2*np.pi)
            radii_opt = np.maximum(result.x[N:], 0.01)
            points_opt = np.column_stack([radii_opt * np.cos(angles_opt), radii_opt * np.sin(angles_opt)])
            
            score = _score(points_opt)
            if score > best_score:
                best_score = score
                best_points = points_opt.copy()
        except Exception:
            pass
    
    # Final polish
    if time.time() < deadline - 5.0:
        try:
            result = minimize(
                _objective,
                best_points.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-14}
            )
            points = result.x.reshape(N, 2)
            score = _score(points)
            if score > best_score:
                best_points = points.copy()
        except Exception:
            pass
    
    return best_points

# EVOLVE-BLOCK-END
