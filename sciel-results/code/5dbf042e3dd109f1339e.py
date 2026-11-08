"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import differential_evolution

def _score(points):
    """
    Compute the minimum triangle area divided by the convex hull area.
    Points: array of shape (N, 2)
    """
    if len(points) < 3:
        return 0.0
    
    # Compute convex hull area
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(points)
        hull_area = hull.volume  # In 2D, volume is area
        if hull_area <= 1e-12:
            return 0.0
    except Exception:
        return 0.0
    
    # Compute minimum triangle area
    min_area = float('inf')
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                # Area of triangle = 0.5 * |cross product|
                area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
                if area < min_area:
                    min_area = area
    
    if min_area <= 1e-12:
        return 0.0
    
    return min_area / hull_area

def _evaluate_params(params):
    """
    Convert polar parameters to Cartesian and compute score.
    params: array of shape (2*N,) where [0::2] are radii, [1::2] are angles.
    """
    radii = params[0::2]
    angles = params[1::2]
    
    # Ensure radii are positive and angles are in [0, 2*pi)
    radii = np.clip(radii, 0.01, 1.0)
    angles = np.mod(angles, 2 * np.pi)
    
    # Sort by angle to ensure convex position (points on convex hull)
    sort_idx = np.argsort(angles)
    radii = radii[sort_idx]
    angles = angles[sort_idx]
    
    points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    
    return _score(points)

def _neg_score(params):
    """Negative score for minimization."""
    return -_evaluate_params(params)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    # Initial best
    best_score = 0.0
    best_points = None
    
    # Try a few random initial configurations
    rng = np.random.RandomState(N)
    
    # Differential evolution in polar coordinates
    # Bounds: radii in [0.1, 1.0], angles in [0, 2*pi)
    bounds = []
    for i in range(N):
        bounds.append((0.1, 1.0))  # radius
        bounds.append((0, 2 * np.pi))  # angle
    
    # Run differential evolution
    try:
        result = differential_evolution(
            _neg_score,
            bounds,
            seed=N,
            maxiter=50,
            popsize=15,
            tol=1e-6,
            mutation=(0.5, 1.0),
            recombination=0.7,
            polish=True
        )
        
        if result.success or result.fun < 0:
            best_params = result.x
            score = -result.fun
            if score > best_score:
                best_score = score
                
                # Reconstruct points
                radii = best_params[0::2]
                angles = best_params[1::2]
                sort_idx = np.argsort(angles)
                radii = radii[sort_idx]
                angles = angles[sort_idx]
                best_points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        
    except Exception:
        pass
    
    # If DE didn't find a good solution, try some structured initial guesses
    if best_points is None or best_score < 0.01:
        # Regular polygon on unit circle
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        radii = np.ones(N)
        points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        score = _score(points)
        if score > best_score:
            best_score = score
            best_points = points
    
    # Check time and return best
    if best_points is None:
        # Fallback: random points
        best_points = rng.rand(N, 2)
    
    return best_points
# EVOLVE-BLOCK-END
