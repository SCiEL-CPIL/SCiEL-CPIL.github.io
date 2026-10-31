"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import combinations

N = 13

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull
from itertools import combinations
import time


def _all_triangle_areas(points):
    """Compute areas of all triangles formed by triples of points."""
    n = len(points)
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                v1 = points[j] - points[i]
                v2 = points[k] - points[i]
                cross = v1[0] * v2[1] - v1[1] * v2[0]
                areas.append(0.5 * abs(cross))
    return np.array(areas)


def _convex_hull_area(points):
    """Compute the area of the convex hull of the points."""
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume is area
    except Exception:
        return 0.0


def _score(points):
    """Compute the score: min triangle area / convex hull area."""
    if len(points) < 3:
        return 0.0
    areas = _all_triangle_areas(points)
    min_area = np.min(areas)
    hull_area = _convex_hull_area(points)
    if hull_area <= 0:
        return 0.0
    return min_area / hull_area


def _objective(x):
    """Objective function for optimization (minimize negative score)."""
    points = x.reshape((-1, 2))
    # Check for degenerate cases
    if np.any(np.isinf(points)) or np.any(np.isnan(points)):
        return 0.0
    # Compute score
    areas = _all_triangle_areas(points)
    min_area = np.min(areas)
    hull_area = _convex_hull_area(points)
    if hull_area <= 0:
        return 0.0
    return -min_area / hull_area


def _initial_guess():
    """Generate an initial guess based on a regular polygon with some perturbation."""
    # Start with points on a circle
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    return points.flatten()


def construct():
    """Construct 13 points to maximize the minimum triangle area / convex hull area."""
    start_time = time.time()
    deadline = start_time + 145  # 5 second safety margin
    
    # Start with a reasonable initial guess
    x0 = _initial_guess()
    
    # First, try differential evolution for global optimization
    # Bounds: points in [0, 1] x [0, 1]
    bounds = [(0, 1)] * (2 * N)
    
    best_points = x0.reshape((-1, 2))
    best_score = _score(best_points)
    
    # Try differential evolution
    try:
        result = differential_evolution(
            _objective, 
            bounds,
            seed=42,
            maxiter=100,
            tol=1e-6,
            polish=False,
            init='sobol',
            update_interval=10
        )
        if time.time() > deadline:
            return best_points
        
        if result.fun < 0:  # Negative means positive score
            candidate_points = result.x.reshape((-1, 2))
            candidate_score = _score(candidate_points)
            if candidate_score > best_score:
                best_points = candidate_points
                best_score = candidate_score
    except Exception:
        pass
    
    # Now polish with local optimization
    if time.time() < deadline:
        try:
            result2 = minimize(
                _objective,
                best_points.flatten(),
                method='L-BFGS-B',
                bounds=[(0, 1)] * (2 * N),
                options={'maxiter': 500, 'ftol': 1e-12}
            )
            if time.time() > deadline:
                return best_points
            
            if result2.fun < 0:
                candidate_points = result2.x.reshape((-1, 2))
                candidate_score = _score(candidate_points)
                if candidate_score > best_score:
                    best_points = candidate_points
                    best_score = candidate_score
        except Exception:
            pass
    
    # Try a few more local optimizations from different starting points
    for _ in range(5):
        if time.time() > deadline:
            break
        
        # Generate random starting point
        rng = np.random.RandomState(np.random.randint(0, 2**31))
        x0 = rng.uniform(0, 1, size=2 * N)
        
        try:
            result3 = minimize(
                _objective,
                x0,
                method='L-BFGS-B',
                bounds=[(0, 1)] * (2 * N),
                options={'maxiter': 200, 'ftol': 1e-12}
            )
            if time.time() > deadline:
                break
            
            if result3.fun < 0:
                candidate_points = result3.x.reshape((-1, 2))
                candidate_score = _score(candidate_points)
                if candidate_score > best_score:
                    best_points = candidate_points
                    best_score = candidate_score
        except Exception:
            continue
    
    return best_points
# EVOLVE-BLOCK-END
