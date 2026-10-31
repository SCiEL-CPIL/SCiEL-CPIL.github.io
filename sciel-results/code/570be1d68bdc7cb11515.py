"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull
from itertools import combinations
import time

def _min_triangle_area(points):
    """Compute the minimum triangle area over all triples of points."""
    n = len(points)
    min_area = np.inf
    for i, j, k in combinations(range(n), 3):
        v1 = points[j] - points[i]
        v2 = points[k] - points[i]
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        area = 0.5 * abs(cross)
        if area < min_area:
            min_area = area
    return min_area

def _convex_hull_area(points):
    """Compute the area of the convex hull of the points."""
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume is area
    except Exception:
        # Fallback: compute area using polygon formula if hull fails
        # Sort points by angle from centroid
        cx, cy = np.mean(points, axis=0)
        angles = np.arctan2(points[:, 1] - cy, points[:, 0] - cx)
        sorted_points = points[np.argsort(angles)]
        # Shoelace formula
        x = sorted_points[:, 0]
        y = sorted_points[:, 1]
        area = 0.5 * abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
        return area

def _score(points):
    """Score = min triangle area / convex hull area."""
    min_ta = _min_triangle_area(points)
    hull_area = _convex_hull_area(points)
    if hull_area <= 0:
        return 0.0
    return min_ta / hull_area

def _objective(params):
    """Objective function for minimization (we maximize score, so minimize negative score)."""
    points = params.reshape(-1, 2)
    # Check for degenerate cases
    if np.any(np.isnan(points)) or np.any(np.isinf(points)):
        return 1e10
    score = _score(points)
    return -score

def construct():
    """Construct 13 points to maximize minimum triangle area / convex hull area."""
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    best_score = 0.0
    best_points = None
    
    # Initial guess: regular 13-gon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    init_points = np.column_stack([np.cos(angles), np.sin(angles)])
    init_params = init_points.flatten()
    
    # Differential evolution bounds: points in [-2, 2]^2
    bounds = [(-2, 2)] * (2 * N)
    
    # Run differential evolution with time limit
    try:
        result = differential_evolution(
            _objective, bounds,
            maxiter=200,
            popsize=15,
            tol=1e-8,
            mutation=(0.3, 1.0),
            recombination=0.8,
            seed=42,
            polish=True,
            callback=None
        )
        
        if result.success or result.fun < 0:
            best_params = result.x
            best_points = best_params.reshape(-1, 2)
            best_score = -result.fun
    except Exception as e:
        pass
    
    # Check time
    if time.time() > deadline - 30:
        if best_points is not None:
            return best_points
        return init_points
    
    # Local refinement with Nelder-Mead
    if best_points is not None:
        try:
            local_result = minimize(
                _objective,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            if local_result.fun < -best_score:
                best_params = local_result.x
                best_points = best_params.reshape(-1, 2)
                best_score = -local_result.fun
        except Exception:
            pass
    
    # Try a few more local refinements from different starting points
    if time.time() < deadline - 20:
        for seed in [1, 2, 3]:
            if time.time() > deadline - 20:
                break
            try:
                # Perturb the best solution
                noise = np.random.RandomState(seed).normal(0, 0.1, size=2*N)
                start_params = best_params + noise
                local_result = minimize(
                    _objective,
                    start_params,
                    method='Nelder-Mead',
                    options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12}
                )
                if local_result.fun < -best_score:
                    best_params = local_result.x
                    best_points = best_params.reshape(-1, 2)
                    best_score = -local_result.fun
            except Exception:
                pass
    
    if best_points is None:
        best_points = init_points
    
    return best_points
# EVOLVE-BLOCK-END
