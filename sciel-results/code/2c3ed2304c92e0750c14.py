"""Optimized 13-point configuration to maximize minimum triangle area / hull area."""
import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

N = 13

def _score(points):
    """Compute min triangle area / convex hull area."""
    n = points.shape[0]
    # Compute convex hull area
    # Use a simple hull algorithm or scipy
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        # Fallback: use polygon area if all points are on hull
        hull_area = _polygon_area(points)
    
    if hull_area < 1e-12:
        return 0.0
    
    # Compute minimum triangle area
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
                if area < min_area:
                    min_area = area
    
    return min_area / hull_area

def _polygon_area(points):
    """Shoelace formula for polygon area (assumes points in order)."""
    n = len(points)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += points[i][0] * points[j][1] - points[j][0] * points[i][1]
    return abs(area) * 0.5

def _points_from_params(params):
    """Convert 26 parameters (13 radii, 13 angles) to points."""
    radii = params[:N]
    angles = params[N:]
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return np.column_stack([x, y])

def _objective(params):
    """Negative score for minimization."""
    points = _points_from_params(params)
    s = _score(points)
    return -s

def construct():
    start_time = time.time()
    deadline = 135.0  # Leave 15s safety margin
    
    # Initial guess: regular 13-gon
    angles0 = np.linspace(0, 2 * np.pi, N, endpoint=False)
    radii0 = np.ones(N)
    params0 = np.concatenate([radii0, angles0])
    
    # Bounds for differential evolution
    # Radii: [0.1, 2.0], Angles: [0, 2*pi]
    bounds = [(0.1, 2.0)] * N + [(0.0, 2 * np.pi)] * N
    
    best_params = params0.copy()
    best_score = _score(_points_from_params(params0))
    
    # Try differential evolution first
    try:
        result = differential_evolution(
            _objective,
            bounds,
            seed=42,
            maxiter=100,
            popsize=15,
            tol=1e-4,
            polish=False,
            callback=lambda x, convergence: None
        )
        
        if result.fun < -best_score:
            best_score = -result.fun
            best_params = result.x
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 10:
        return _points_from_params(best_params)
    
    # Refine with L-BFGS-B
    try:
        result2 = minimize(
            _objective,
            best_params,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-8}
        )
        
        if result2.fun < -best_score:
            best_score = -result2.fun
            best_params = result2.x
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 5:
        return _points_from_params(best_params)
    
    # Try a few local restarts if time permits
    rng = np.random.default_rng(42)
    for restart in range(5):
        if time.time() - start_time > deadline - 5:
            break
        
        # Perturb from best
        perturbation = rng.normal(0, 0.1, size=2 * N)
        start_params = best_params + perturbation
        # Wrap angles
        start_params[N:] = start_params[N:] % (2 * np.pi)
        # Clip radii
        start_params[:N] = np.clip(start_params[:N], 0.1, 2.0)
        
        try:
            result3 = minimize(
                _objective,
                start_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-8}
            )
            
            if result3.fun < -best_score:
                best_score = -result3.fun
                best_params = result3.x
        except Exception:
            pass
    
    return _points_from_params(best_params)
