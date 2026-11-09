import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull
import time

N = 14

def _compute_score(points):
    """
    Compute the score: min triangle area / convex hull area.
    Points are normalized to be centered at origin and have unit RMS radius.
    """
    if len(points) < 3:
        return 0.0
    
    # Normalize
    pts = points.copy()
    # Center
    pts -= np.mean(pts, axis=0)
    # Scale
    rms = np.sqrt(np.sum(pts**2) / N)
    if rms < 1e-9:
        return 0.0
    pts /= rms
    
    # Calculate all triangle areas
    n = len(pts)
    min_area = np.inf
    
    # Vectorized calculation of areas for all triples is memory intensive for large N,
    # but N=14 is small (364 triangles).
    # We can loop or use combinations.
    # For speed, we can compute cross products.
    
    # Let's use a loop for clarity, it's fast enough for N=14
    # Or vectorize:
    # i, j, k
    areas = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                # Area = 0.5 * |cross(p_j - p_i, p_k - p_i)|
                v1 = pts[j] - pts[i]
                v2 = pts[k] - pts[i]
                cross = v1[0]*v2[1] - v1[1]*v2[0]
                area = 0.5 * abs(cross)
                if area < min_area:
                    min_area = area
    
    if min_area == 0.0 or np.isinf(min_area):
        return 0.0

    # Calculate Hull Area
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
    except Exception:
        return 0.0
    
    if hull_area < 1e-9:
        return 0.0
        
    return min_area / hull_area

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Leave 10s margin
    
    # Initial guess: Regular 14-gon
    angles = 2 * np.pi * np.arange(N) / N
    init_pts = np.column_stack((np.cos(angles), np.sin(angles)))
    
    # Flatten for optimizer
    init_flat = init_pts.flatten()
    
    best_score = _compute_score(init_pts)
    best_pts = init_pts
    
    def objective(flat_pts):
        nonlocal best_score, best_pts
        pts = flat_pts.reshape(N, 2)
        score = _compute_score(pts)
        if score > best_score:
            best_score = score
            best_pts = pts.copy()
        # We want to maximize, so minimize negative
        return -score

    # Use Nelder-Mead
    # Maxiter needs to be high enough, but we check time.
    # We can use a callback to check time.
    
    def callback(xk):
        return time.time() > deadline

    try:
        res = minimize(objective, init_flat, method='Nelder-Mead', 
                       options={'maxiter': 5000, 'xatol': 1e-5, 'fatol': 1e-5, 'disp': False},
                       callback=callback)
        
        if time.time() < deadline:
            # Try a second pass with a different starting point or tighter tolerance?
            # Or just return best found.
            pass
            
    except Exception:
        pass

    return best_pts
