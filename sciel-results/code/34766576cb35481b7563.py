"""Heilbronn convex regions - place 13 points to maximize minimum triangle area."""
import numpy as np
from scipy.optimize import differential_evolution
import time

N = 13

def _min_triangle_area(points):
    """Compute minimum triangle area among all triples."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                cross = (points[j][0] - points[i][0]) * (points[k][1] - points[i][1]) - \
                        (points[j][1] - points[i][1]) * (points[k][0] - points[i][0])
                area = 0.5 * abs(cross)
                if area < min_area:
                    min_area = area
    return min_area

def _convex_hull_area(points):
    """Compute convex hull area."""
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(points)
        return hull.volume
    except:
        return 1e-10

def _score(points):
    """Score = min triangle area / convex hull area."""
    min_area = _min_triangle_area(points)
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    return min_area / hull_area

def _objective(params, radii):
    """Objective function for differential evolution: minimize negative score."""
    angles = params
    points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    return -_score(points)

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # Strategy: points on a circle with slightly perturbed radii
    # Start with equally spaced angles on unit circle
    base_angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    base_radii = np.ones(N)
    
    # Initial score with regular polygon
    initial_points = np.column_stack([base_radii * np.cos(base_angles), 
                                       base_radii * np.sin(base_angles)])
    best_score = _score(initial_points)
    best_points = initial_points.copy()
    
    # Try multiple radius configurations
    radius_configs = [
        np.ones(N),
        np.ones(N) * 0.95,
        np.ones(N) * 1.05,
    ]
    
    # Add some perturbed radius configs
    rng = np.random.RandomState(42)
    for _ in range(5):
        radii = 1.0 + rng.uniform(-0.1, 0.1, N)
        radius_configs.append(radii)
    
    # Optimize angles for each radius config
    for radii in radius_configs:
        if time.time() > deadline:
            break
        
        # Differential evolution to optimize angles
        bounds = [(0, 2 * np.pi)] * N
        
        try:
            result = differential_evolution(
                _objective,
                bounds,
                args=(radii,),
                seed=42,
                maxiter=50,
                popsize=15,
                tol=1e-6,
                polish=True
            )
            
            if result.fun < 0:  # Negative means positive score
                points = np.column_stack([radii * np.cos(result.x), 
                                           radii * np.sin(result.x)])
                score = _score(points)
                if score > best_score:
                    best_score = score
                    best_points = points.copy()
        except Exception:
            continue
    
    # Additional local refinement using scipy.optimize.minimize
    if time.time() < deadline:
        from scipy.optimize import minimize
        
        # Try a few random starting points for local optimization
        for attempt in range(3):
            if time.time() > deadline:
                break
            
            start_angles = base_angles + rng.uniform(-0.2, 0.2, N)
            
            def local_obj(angles):
                points = np.column_stack([base_radii * np.cos(angles), 
                                           base_radii * np.sin(angles)])
                return -_score(points)
            
            try:
                res = minimize(local_obj, start_angles, method='Nelder-Mead',
                              options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8})
                if res.fun < 0:
                    points = np.column_stack([base_radii * np.cos(res.x), 
                                               base_radii * np.sin(res.x)])
                    score = _score(points)
                    if score > best_score:
                        best_score = score
                        best_points = points.copy()
            except Exception:
                continue
    
    # Normalize: center at origin and scale to unit area
    centroid = best_points.mean(axis=0)
    best_points = best_points - centroid
    
    # Scale to unit area
    hull_area = _convex_hull_area(best_points)
    if hull_area > 1e-10:
        scale = 1.0 / np.sqrt(hull_area)
        best_points = best_points * scale
    
    return best_points
