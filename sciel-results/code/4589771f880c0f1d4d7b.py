import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 13

def _score_points(pts):
    """Compute min triangle area / convex hull area."""
    n = len(pts)
    if n < 3:
        return 0.0
    
    # Compute convex hull area
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
        if hull_area <= 1e-9:
            return 0.0
    except Exception:
        return 0.0
        
    # Compute all triangle areas
    min_area = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                # Cross product magnitude / 2
                v1 = pts[j] - pts[i]
                v2 = pts[k] - pts[i]
                cross = v1[0]*v2[1] - v1[1]*v2[0]
                area = 0.5 * abs(cross)
                if area < min_area:
                    min_area = area
                    
    return min_area / hull_area

def _soft_min_surrogate(pts, beta):
    """
    Negative of the soft-min area divided by hull area.
    We minimize this to maximize the score.
    """
    n = len(pts)
    # Compute hull area
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
        if hull_area <= 1e-9:
            return 1e10 # Large penalty
    except Exception:
        return 1e10
        
    # Compute log-sum-exp of -beta * areas
    # To avoid overflow/underflow, we can compute areas first
    # M = C(13,3) = 286 triangles. This is small enough to compute explicitly.
    areas = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                v1 = pts[j] - pts[i]
                v2 = pts[k] - pts[i]
                cross = v1[0]*v2[1] - v1[1]*v2[0]
                area = 0.5 * abs(cross)
                areas.append(area)
                
    areas = np.array(areas)
    # Soft min: -1/beta * log(sum(exp(-beta * areas)))
    # Using log-sum-exp trick for stability
    max_val = np.max(-beta * areas)
    log_sum_exp = max_val + np.log(np.sum(np.exp(-beta * areas - max_val)))
    soft_min_area = -log_sum_exp / beta
    
    score = soft_min_area / hull_area
    return -score

def construct():
    start_time = time.time()
    deadline = 145.0 # 5s safety margin
    
    # Initial symmetric guess: Center + 12-gon
    # Or Center + 2 Hexagons
    # Let's try Center + 12-gon first
    angles = np.linspace(0, 2*np.pi, 12, endpoint=False)
    r = 1.0
    pts_init = np.vstack([
        np.array([[0.0, 0.0]]),
        np.column_stack([r * np.cos(angles), r * np.sin(angles)])
    ])
    
    best_pts = pts_init.copy()
    best_score = _score_points(best_pts)
    
    # List of betas for annealing
    betas = [5.0, 20.0, 50.0, 100.0, 200.0]
    
    current_pts = best_pts.copy()
    
    for beta in betas:
        if time.time() > deadline:
            break
            
        # Local optimization
        try:
            res = minimize(
                _soft_min_surrogate, 
                current_pts.flatten(), 
                args=(beta,), 
                method='L-BFGS-B', 
                options={'maxiter': 50, 'ftol': 1e-12}
            )
            new_pts = res.x.reshape(-1, 2)
            
            # Check if valid and improved
            new_score = _score_points(new_pts)
            if new_score > best_score:
                best_score = new_score
                best_pts = new_pts.copy()
                current_pts = new_pts.copy()
        except Exception:
            pass
            
        # Add a small random perturbation to escape local minima?
        # Maybe just rely on the beta schedule.
        
    # Final polish with exact min (high beta or just check)
    # The soft min with high beta is a good approximation.
    # We can do one more step with a very high beta if time permits.
    if time.time() < deadline - 5:
        try:
            res = minimize(
                _soft_min_surrogate, 
                current_pts.flatten(), 
                args=(500.0,), 
                method='L-BFGS-B', 
                options={'maxiter': 100, 'ftol': 1e-14}
            )
            new_pts = res.x.reshape(-1, 2)
            new_score = _score_points(new_pts)
            if new_score > best_score:
                best_score = new_score
                best_pts = new_pts.copy()
        except Exception:
            pass

    return best_pts
