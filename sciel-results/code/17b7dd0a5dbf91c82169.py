"""Rewritten program: Heilbronn convex regions, 13 points, maximize min triangle area / hull area."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 13

# EVOLVE-BLOCK-START

def _min_triangle_area(pts):
    """Compute the minimum area of any triangle formed by triples of points."""
    n = len(pts)
    min_area = np.inf
    # Vectorized computation of all triangle areas
    # For speed, we can use a more efficient approach for n=13
    # Number of triples: C(13,3) = 286, which is small
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                # Area = 0.5 * |cross(pj - pi, pk - pi)|
                dx1 = pts[j, 0] - pts[i, 0]
                dy1 = pts[j, 1] - pts[i, 1]
                dx2 = pts[k, 0] - pts[i, 0]
                dy2 = pts[k, 1] - pts[i, 1]
                area = 0.5 * abs(dx1 * dy2 - dy1 * dx2)
                if area < min_area:
                    min_area = area
    return min_area

def _convex_hull_area(pts):
    """Compute the area of the convex hull of the given points."""
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(pts)
        return hull.volume  # For 2D, volume is area
    except Exception:
        # Fallback: if points are collinear or degenerate
        return 1e-10

def _score(pts):
    """Score: minimum triangle area divided by convex hull area."""
    min_tri = _min_triangle_area(pts)
    hull_area = _convex_hull_area(pts)
    if hull_area < 1e-12:
        return 0.0
    return min_tri / hull_area

def _neg_score(pts):
    """Negative score for minimization."""
    return -_score(pts)

def _normalize(pts):
    """Normalize points to unit area convex hull."""
    hull_area = _convex_hull_area(pts)
    if hull_area < 1e-12:
        return pts
    scale = 1.0 / np.sqrt(hull_area)
    return pts * scale

def _initial_symmetric_config():
    """
    Create an initial configuration exploiting 13-fold rotational symmetry.
    Place points on concentric regular 13-gons with varying radii.
    """
    n = N
    # Start with points on a circle
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    radii = np.ones(n)
    pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    return pts

def _refine_with_powell(pts, time_limit=30.0):
    """Refine points using Powell's method with time limit."""
    start_time = time.time()
    x0 = pts.flatten()
    
    def objective(x):
        p = x.reshape(-1, 2)
        return _neg_score(p)
    
    try:
        result = minimize(objective, x0, method='Powell', 
                          options={'maxiter': 200, 'disp': False})
        return result.x.reshape(-1, 2)
    except Exception:
        return pts

def _refine_with_bfgs(pts, time_limit=20.0):
    """Refine points using BFGS for high-precision polish."""
    x0 = pts.flatten()
    
    def objective(x):
        p = x.reshape(-1, 2)
        return _neg_score(p)
    
    def gradient(x):
        p = x.reshape(-1, 2)
        # Numerical gradient
        eps = 1e-7
        grad = np.zeros_like(x)
        for i in range(len(x)):
            xp = x.copy()
            xm = x.copy()
            xp[i] += eps
            xm[i] -= eps
            grad[i] = (objective(xp.reshape(-1, 2)) - objective(xm.reshape(-1, 2))) / (2 * eps)
        return grad
    
    try:
        result = minimize(objective, x0, jac=gradient, method='BFGS',
                          options={'maxiter': 100, 'disp': False})
        return result.x.reshape(-1, 2)
    except Exception:
        return pts

def construct():
    """Construct 13 points to maximize minimum triangle area / convex hull area."""
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Start with symmetric configuration
    best_pts = _initial_symmetric_config()
    best_score = _score(best_pts)
    
    # Powell refinement
    if time.time() < deadline - 40:
        refined = _refine_with_powell(best_pts)
        refined_score = _score(refined)
        if refined_score > best_score:
            best_pts = refined
            best_score = refined_score
    
    # BFGS polish
    if time.time() < deadline - 10:
        polished = _refine_with_bfgs(best_pts)
        polished_score = _score(polished)
        if polished_score > best_score:
            best_pts = polished
            best_score = polished_score
    
    # Additional restarts if time permits
    attempts = 0
    while time.time() < deadline - 20 and attempts < 5:
        # Random perturbation of the best solution
        perturbation = np.random.randn(N, 2) * 0.1
        trial_pts = best_pts + perturbation
        trial_score = _score(trial_pts)
        
        if trial_score > best_score:
            best_pts = trial_pts
            best_score = trial_score
        
        # Refine the trial
        if time.time() < deadline - 15:
            refined_trial = _refine_with_powell(trial_pts)
            refined_trial_score = _score(refined_trial)
            if refined_trial_score > best_score:
                best_pts = refined_trial
                best_score = refined_trial_score
        
        attempts += 1
    
    # Final BFGS polish
    if time.time() < deadline - 5:
        final_polished = _refine_with_bfgs(best_pts)
        final_score = _score(final_polished)
        if final_score > best_score:
            best_pts = final_polished
            best_score = final_score
    
    # Normalize to unit area
    best_pts = _normalize(best_pts)
    
    return best_pts

# EVOLVE-BLOCK-END
