import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import itertools

N = 14

def _score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    Returns 0 if invalid.
    """
    if len(points) < 3:
        return 0.0
    
    # Center points to improve numerical stability
    # Actually, the score is invariant to translation and scaling, 
    # but centering helps the optimizer.
    
    # Check for degeneracy: if points are too close or collinear
    # ConvexHull requires at least 3 non-collinear points.
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area < 1e-10:
            return 0.0
    except Exception:
        return 0.0

    # Calculate min triangle area
    min_area = float('inf')
    n = len(points)
    # Vectorized calculation for speed, though N=14 is small
    # Using itertools for clarity and low overhead for small N
    for i, j, k in itertools.combinations(range(n), 3):
        p1 = points[i]
        p2 = points[j]
        p3 = points[k]
        # Area = 0.5 * |cross(p2-p1, p3-p1)|
        cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
        area = 0.5 * abs(cross)
        if area < min_area:
            min_area = area
            
    if min_area == 0.0 or min_area == float('inf'):
        return 0.0

    return min_area / hull_area

def _objective(x):
    """
    Objective function for minimization.
    x is a flat array of shape (2*N,).
    """
    points = x.reshape(-1, 2)
    # Optional: Center the points to remove translation invariance
    # This can help the optimizer.
    points = points - np.mean(points, axis=0)
    
    s = _score(points)
    # We want to maximize s, so minimize -s
    # Add a small penalty if points are too far apart to keep scale reasonable
    # Although score is scale invariant, large coordinates can cause numerical issues
    # Let's check the magnitude
    if np.any(np.abs(points) > 100):
        return 1e6 # Large penalty
    
    return -s

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s safety margin
    
    # Initial guess: Regular N-gon
    angles = 2 * np.pi * np.arange(N) / N
    init_points = np.column_stack((np.cos(angles), np.sin(angles)))
    x0 = init_points.flatten()
    
    best_x = x0.copy()
    best_score = _score(init_points)
    
    # Powell method
    # maxiter and maxfev control the runtime
    # Powell can be slow, so we limit iterations
    # We can also try Nelder-Mead as a fallback or alternative
    
    # Try Powell first
    try:
        res = minimize(_objective, x0, method='Powell', 
                       options={'maxiter': 500, 'maxfev': 5000, 'xtol': 1e-5, 'ftol': 1e-5})
        if res.x is not None:
            cand_x = res.x
            cand_points = cand_x.reshape(-1, 2)
            cand_points = cand_points - np.mean(cand_points, axis=0) # Re-center
            cand_score = _score(cand_points)
            if cand_score > best_score:
                best_score = cand_score
                best_x = cand_x
    except Exception as e:
        pass # Ignore optimization errors, keep best found so far
    
    elapsed = time.time() - start_time
    if elapsed > deadline:
        return best_x.reshape(-1, 2)
    
    # Try a few random perturbations of the regular polygon to escape local minima
    # The regular polygon is a strong candidate, but maybe slightly irregular is better?
    # For convex regions, regular polygons are often optimal for "max min distance" type problems,
    # but "max min triangle area" might favor slightly different configurations.
    
    rng = np.random.RandomState(42)
    
    for i in range(10): # Max 10 extra tries
        if time.time() > deadline:
            break
            
        # Perturb the best found or the regular polygon
        # Let's perturb the regular polygon
        angles = 2 * np.pi * np.arange(N) / N
        radii = np.ones(N)
        
        # Add small random noise to angles and radii
        noise_theta = rng.normal(0, 0.1, N)
        noise_r = rng.normal(0, 0.1, N)
        
        new_angles = angles + noise_theta
        new_radii = 1.0 + noise_r
        
        new_points = np.column_stack((new_radii * np.cos(new_angles), 
                                       new_radii * np.sin(new_angles)))
        
        # Center it
        new_points = new_points - np.mean(new_points, axis=0)
        
        x_start = new_points.flatten()
        
        try:
            res = minimize(_objective, x_start, method='Powell', 
                           options={'maxiter': 200, 'maxfev': 2000, 'xtol': 1e-5, 'ftol': 1e-5})
            if res.x is not None:
                cand_x = res.x
                cand_points = cand_x.reshape(-1, 2)
                cand_points = cand_points - np.mean(cand_points, axis=0)
                cand_score = _score(cand_points)
                if cand_score > best_score:
                    best_score = cand_score
                    best_x = cand_x
        except Exception:
            pass
            
    return best_x.reshape(-1, 2)
