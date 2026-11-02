"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution, minimize
import time
import itertools

def _calc_score(pts):
    """
    Calculate the minimum triangle area divided by the convex hull area.
    pts: numpy array of shape (N, 2)
    """
    n = len(pts)
    if n < 3:
        return 0.0
    
    # Compute convex hull area
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
        if hull_area < 1e-12:
            return 0.0
    except Exception:
        return 0.0
    
    # Compute minimum triangle area
    min_area = float('inf')
    # Precompute indices for efficiency? N=13 is small, direct loop is fine.
    for i in range(n):
        xi, yi = pts[i]
        for j in range(i + 1, n):
            xj, yj = pts[j]
            dx1 = xj - xi
            dy1 = yj - yi
            for k in range(j + 1, n):
                xk, yk = pts[k]
                dx2 = xk - xi
                dy2 = yk - yi
                # Twice the area is the cross product magnitude
                area2 = abs(dx1 * dy2 - dy1 * dx2)
                if area2 < min_area:
                    min_area = area2
                    if min_area < 1e-15:
                        return 0.0
    
    # min_area here is 2 * Area. hull_area is Area.
    # Ratio = (min_area / 2) / hull_area
    return (min_area / 2.0) / hull_area

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    def objective(x):
        pts = x.reshape(N, 2)
        # To prevent numerical issues with very large/small numbers,
        # we can normalize here? No, DE explores the space.
        # But if points fly to 1e100, cross product might overflow?
        # 1e100 * 1e100 = 1e200, within double range (1e308).
        # So it's safe.
        s = _calc_score(pts)
        return -s

    bounds = [(-10, 10)] * (2 * N)
    
    best_score = -float('inf')
    best_params = None
    
    # Try multiple restarts with different seeds
    seeds = [42, 123, 999]
    
    for seed in seeds:
        if time.time() > deadline - 30:
            break
            
        try:
            res = differential_evolution(
                objective, 
                bounds, 
                maxiter=50, 
                popsize=15, 
                tol=1e-8, 
                seed=seed,
                polish=False
            )
            
            current_score = -res.fun
            if current_score > best_score:
                best_score = current_score
                best_params = res.x.copy()
                
        except Exception:
            continue
        
        # If we have a good candidate, polish it
        if best_params is not None and time.time() < deadline - 10:
            try:
                res_nm = minimize(
                    objective, 
                    best_params, 
                    method='Nelder-Mead', 
                    options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10}
                )
                nm_score = -res_nm.fun
                if nm_score > best_score:
                    best_score = nm_score
                    best_params = res_nm.x.copy()
            except Exception:
                pass

    if best_params is None:
        # Fallback: Random points
        rng = np.random.RandomState(0)
        best_params = rng.uniform(-1, 1, (N, 2)).flatten()
        
    # Reshape to (N, 2)
    result = best_params.reshape(N, 2)
    
    # Optional: Normalize to [0, 1] square for cleaner output
    # This does not change the score
    min_x, min_y = np.min(result, axis=0)
    max_x, max_y = np.max(result, axis=0)
    width = max_x - min_x
    height = max_y - min_y
    max_dim = max(width, height)
    if max_dim > 1e-10:
        result = (result - min_x) / max_dim
        # Center it? Not necessary, but nice.
        # Actually, just scaling is enough.
        
    return result
# EVOLVE-BLOCK-END
