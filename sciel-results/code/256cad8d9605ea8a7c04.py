"""Heilbronn convex regions n=14: maximize min triangle area / convex hull area."""
import numpy as np
import time
from scipy.optimize import minimize

N = 14

def _min_triangle_area(pts):
    """Compute the minimum area over all triples of points."""
    n = len(pts)
    min_area = np.inf
    # Vectorized computation of all triangle areas
    for i in range(n):
        for j in range(i + 1, n):
            v1 = pts[j] - pts[i]
            for k in range(j + 1, n):
                v2 = pts[k] - pts[i]
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                if area < min_area:
                    min_area = area
    return min_area

def _convex_hull_area(pts):
    """Compute the area of the convex hull of the point set."""
    # Use scipy's ConvexHull
    from scipy.spatial import ConvexHull
    hull = ConvexHull(pts)
    return hull.volume  # In 2D, volume is area

def _score(pts):
    """Score = min triangle area / convex hull area."""
    try:
        hull_area = _convex_hull_area(pts)
        if hull_area < 1e-12:
            return 0.0
        min_tri = _min_triangle_area(pts)
        return min_tri / hull_area
    except Exception:
        return 0.0

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    # Initialize with regular 14-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    best_points = points.copy()
    best_score = _score(points)

    # Gradient-free optimization using Nelder-Mead
    # We'll try multiple restarts with different initial perturbations
    def objective(params):
        """Objective to minimize: negative of the score."""
        pts = np.column_stack([
            params[0:N] * np.cos(params[N:2*N]),
            params[0:N] * np.sin(params[N:2*N])
        ])
        return -_score(pts)

    # Initial parameters: radii and angles
    initial_params = np.concatenate([np.ones(N), angles])

    # Run Nelder-Mead optimization
    try:
        result = minimize(
            objective,
            initial_params,
            method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
        )
        
        if result.success or result.fun < -best_score:
            optimal_params = result.x
            optimal_points = np.column_stack([
                optimal_params[0:N] * np.cos(optimal_params[N:2*N]),
                optimal_params[0:N] * np.sin(optimal_params[N:2*N])
            ])
            optimal_score = -result.fun
            if optimal_score > best_score:
                best_points = optimal_points
                best_score = optimal_score
    except Exception:
        pass

    # Check time and potentially do a second round if time permits
    if time.time() < deadline - 20:
        # Try a different initialization: slightly perturbed regular polygon
        rng = np.random.RandomState(42)
        for trial in range(3):
            if time.time() > deadline - 10:
                break
            perturbation = rng.normal(0, 0.1, N)
            perturbed_angles = angles + perturbation
            initial_params2 = np.concatenate([np.ones(N), perturbed_angles])
            
            try:
                result2 = minimize(
                    objective,
                    initial_params2,
                    method='Nelder-Mead',
                    options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
                )
                
                if result2.success or result2.fun < -best_score:
                    optimal_params = result2.x
                    optimal_points = np.column_stack([
                        optimal_params[0:N] * np.cos(optimal_params[N:2*N]),
                        optimal_params[0:N] * np.sin(optimal_params[N:2*N])
                    ])
                    optimal_score = -result2.fun
                    if optimal_score > best_score:
                        best_points = optimal_points
                        best_score = optimal_score
            except Exception:
                pass

    # Ensure we return the best points found
    return best_points
