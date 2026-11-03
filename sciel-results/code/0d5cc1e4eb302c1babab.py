import numpy as np
from scipy.optimize import minimize
import time
from itertools import combinations

N = 13

def _triangle_areas(pts):
    """Compute all triangle areas from n points."""
    n = len(pts)
    areas = np.zeros((n, n, n)) # Or just list
    min_area = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                a = pts[i]
                b = pts[j]
                c = pts[k]
                area = 0.5 * abs((b[0]-a[0])*(c[1]-a[1]) - (c[0]-a[0])*(b[1]-a[1]))
                if area < min_area:
                    min_area = area
    return min_area

def _convex_hull_area(pts):
    """Compute area of convex hull of points."""
    # Assuming points are in convex position, we can just use shoelace
    # But to be safe, use scipy
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _score(pts):
    """Calculate the score: min triangle area / hull area."""
    hull_area = _convex_hull_area(pts)
    if hull_area < 1e-10:
        return 0.0
    min_area = _triangle_areas(pts)
    return min_area / hull_area

def _is_convex(pts):
    """Check if points form a convex polygon (in the given order)."""
    n = len(pts)
    sign = 0
    for i in range(n):
        p0 = pts[i]
        p1 = pts[(i+1)%n]
        p2 = pts[(i+2)%n]
        cross = (p1[0]-p0[0])*(p2[1]-p0[1]) - (p1[1]-p0[1])*(p2[0]-p0[0])
        if cross != 0:
            if sign == 0:
                sign = 1 if cross > 0 else -1
            else:
                if (cross > 0 and sign < 0) or (cross < 0 and sign > 0):
                    return False
    return True

def _penalty(pts):
    """Penalty for non-convexity."""
    # Sort by angle around centroid to ensure consistent order for convexity check
    centroid = np.mean(pts, axis=0)
    angles = np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])
    sorted_idx = np.argsort(angles)
    sorted_pts = pts[sorted_idx]
    
    if not _is_convex(sorted_pts):
        return 1000.0
    return 0.0

def _objective(x):
    """Objective function for minimization."""
    pts = x.reshape(N, 2)
    
    # Check for degenerate points (too close)
    for i in range(N):
        for j in range(i+1, N):
            if np.linalg.norm(pts[i] - pts[j]) < 1e-6:
                return 1e6
    
    # Penalty for non-convexity
    pen = _penalty(pts)
    if pen > 0:
        return pen
    
    # Calculate score
    s = _score(pts)
    
    # We want to maximize s, so minimize -s
    return -s

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # Initial guess: Regular 13-gon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    x0 = np.column_stack((np.cos(angles), np.sin(angles))).flatten()
    
    best_score = _score(x0.reshape(N, 2))
    best_pts = x0.reshape(N, 2)
    
    # Try to optimize
    # Use SLSQP or L-BFGS-B
    # Since we have a penalty, L-BFGS-B might struggle with the non-smooth penalty.
    # SLSQP is better for small problems with penalties.
    
    try:
        res = minimize(_objective, x0, method='SLSQP', options={'maxiter': 200, 'ftol': 1e-12})
        
        if res.success:
            pts_opt = res.x.reshape(N, 2)
            score_opt = _score(pts_opt)
            if score_opt > best_score:
                best_score = score_opt
                best_pts = pts_opt
    except Exception:
        pass
    
    # If time allows, try a few random perturbations
    while time.time() < deadline - 10.0:
        # Perturb the best points
        perturb = np.random.normal(0, 0.1, size=(N, 2))
        x_init = (best_pts + perturb).flatten()
        
        try:
            res = minimize(_objective, x_init, method='SLSQP', options={'maxiter': 100, 'ftol': 1e-12})
            if res.success:
                pts_opt = res.x.reshape(N, 2)
                score_opt = _score(pts_opt)
                if score_opt > best_score:
                    best_score = score_opt
                    best_pts = pts_opt
        except Exception:
            pass
        
        if time.time() > deadline - 5.0:
            break
            
    return best_pts

# EVOLVE-BLOCK-START
import random
import numpy as np

def construct():
    # Call the optimized construct from above? 
    # No, the function must be named construct.
    # I will put the logic inside construct directly or call the helper.
    
    # Re-define or use the logic from above.
    # To keep it clean, I'll put the logic in the EVOLVE block.
    
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Initial guess: Regular 13-gon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    x0 = np.column_stack((np.cos(angles), np.sin(angles))).flatten()
    
    best_score = _score(x0.reshape(N, 2))
    best_pts = x0.reshape(N, 2)
    
    try:
        res = minimize(_objective, x0, method='SLSQP', options={'maxiter': 200, 'ftol': 1e-12})
        if res.success:
            pts_opt = res.x.reshape(N, 2)
            score_opt = _score(pts_opt)
            if score_opt > best_score:
                best_score = score_opt
                best_pts = pts_opt
    except Exception:
        pass
    
    while time.time() < deadline - 10.0:
        perturb = np.random.normal(0, 0.1, size=(N, 2))
        x_init = (best_pts + perturb).flatten()
        
        try:
            res = minimize(_objective, x_init, method='SLSQP', options={'maxiter': 100, 'ftol': 1e-12})
            if res.success:
                pts_opt = res.x.reshape(N, 2)
                score_opt = _score(pts_opt)
                if score_opt > best_score:
                    best_score = score_opt
                    best_pts = pts_opt
        except Exception:
            pass
        
        if time.time() > deadline - 5.0:
            break
            
    return best_pts
# EVOLVE-BLOCK-END
