"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull
from itertools import combinations


def _min_triangle_area(points):
    """Compute the minimum area over all triples of points."""
    n = len(points)
    min_area = float('inf')
    # Use vectorized computation for speed
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                v1 = points[j] - points[i]
                v2 = points[k] - points[i]
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                if area < min_area:
                    min_area = area
    return min_area


def _convex_hull_area(points):
    """Compute the area of the convex hull of the points."""
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume is area
    except Exception:
        return 0.0


def _score(points):
    """Score: minimum triangle area divided by convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    min_area = _min_triangle_area(points)
    return min_area / hull_area


def _regular_n_gon(n, radius=1.0, center=(0.0, 0.0)):
    """Generate a regular n-gon."""
    angles = 2 * np.pi * np.arange(n) / n
    xs = center[0] + radius * np.cos(angles)
    ys = center[1] + radius * np.sin(angles)
    return np.column_stack([xs, ys])


def _polar_to_cartesian(params):
    """Convert polar parameters (r, theta) pairs to Cartesian coordinates."""
    n = len(params) // 2
    radii = params[:n]
    thetas = params[n:]
    xs = radii * np.cos(thetas)
    ys = radii * np.sin(thetas)
    return np.column_stack([xs, ys])


def _cartesian_to_polar(points):
    """Convert Cartesian points to polar parameters (r, theta)."""
    n = len(points)
    xs = points[:, 0]
    ys = points[:, 1]
    radii = np.sqrt(xs**2 + ys**2)
    thetas = np.arctan2(ys, xs)
    return np.concatenate([radii, thetas])


def _objective_polar(params):
    """Objective function for optimization: negative score (we minimize)."""
    n = N
    points = _polar_to_cartesian(params)
    score = _score(points)
    return -score


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 140s deadline, 10s safety margin
    
    # Start with regular 14-gon
    initial_points = _regular_n_gon(N, radius=1.0)
    best_points = initial_points.copy()
    best_score = _score(best_points)
    
    # Convert to polar for optimization
    initial_polar = _cartesian_to_polar(initial_points)
    
    # Define bounds for polar parameters
    # Radii: between 0.3 and 1.5 (allow some variation from unit circle)
    # Angles: between 0 and 2*pi
    n = N
    bounds = [(0.3, 1.5)] * n + [(0.0, 2 * np.pi)] * n
    
    # Try differential evolution first
    try:
        def _de_objective(params):
            points = _polar_to_cartesian(params)
            return -_score(points)
        
        result = differential_evolution(
            _de_objective,
            bounds=bounds,
            seed=42,
            maxiter=50,
            tol=1e-6,
            polish=False,
            update_interval=1
        )
        
        elapsed = time.time() - start_time
        if result.fun < -best_score:
            best_polar = result.x
            best_points = _polar_to_cartesian(best_polar)
            best_score = -result.fun
        
        # Refine with Nelder-Mead
        if time.time() < deadline - 20:
            try:
                result2 = minimize(
                    _objective_polar,
                    best_polar if 'best_polar' in dir() else initial_polar,
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
                )
                if result2.fun < -best_score:
                    best_points = _polar_to_cartesian(result2.x)
                    best_score = -result2.fun
            except Exception:
                pass
    except Exception:
        pass
    
    # If time remains, try a few random restarts
    if time.time() < deadline - 30:
        for seed in [1, 2, 3, 4, 5]:
            if time.time() > deadline - 15:
                break
            try:
                rng = np.random.RandomState(seed)
                # Perturb the regular polygon
                angles = 2 * np.pi * np.arange(N) / N
                angle_perturb = rng.normal(0, 0.05, N)
                radii = 1.0 + rng.normal(0, 0.1, N)
                radii = np.clip(radii, 0.5, 1.5)
                
                xs = radii * np.cos(angles + angle_perturb)
                ys = radii * np.sin(angles + angle_perturb)
                points = np.column_stack([xs, ys])
                
                polar = _cartesian_to_polar(points)
                
                result = minimize(
                    _objective_polar,
                    polar,
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-7, 'fatol': 1e-9}
                )
                
                if result.fun < -best_score:
                    best_points = _polar_to_cartesian(result.x)
                    best_score = -result.fun
            except Exception:
                pass
    
    # Final check: ensure we have valid points
    if _convex_hull_area(best_points) < 1e-10:
        best_points = _regular_n_gon(N, radius=1.0)
    
    return best_points
# EVOLVE-BLOCK-END
