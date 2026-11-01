import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time
import itertools

N = 14

def _min_triangle_area(pts):
    """Compute the minimum triangle area over all triples of points."""
    n = len(pts)
    min_area = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                a = pts[i]
                b = pts[j]
                c = pts[k]
                area = 0.5 * abs((b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0]))
                if area < min_area:
                    min_area = area
    return min_area

def _convex_hull_area(pts):
    """Compute the area of the convex hull of the points."""
    try:
        hull = ConvexHull(pts)
        return hull.volume  # In 2D, volume is area
    except Exception:
        # Fallback: if all points are collinear or degenerate
        return 0.0

def _score(pts):
    """Score = min triangle area / convex hull area."""
    hull_area = _convex_hull_area(pts)
    if hull_area <= 1e-12:
        return 0.0
    min_tri = _min_triangle_area(pts)
    return min_tri / hull_area

def _regular_polygon(n, radius=1.0):
    """Generate vertices of a regular n-gon."""
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    return np.column_stack((radius * np.cos(angles), radius * np.sin(angles)))

def _optimize_from_start(start_pts, max_iter=200, timeout=10):
    """Try to improve a starting configuration using Nelder-Mead."""
    start_time = time.time()
    
    def objective(x):
        if time.time() - start_time > timeout:
            return 0.0
        pts = x.reshape(-1, 2)
        s = _score(pts)
        return -s  # Minimize negative score
    
    x0 = start_pts.flatten()
    try:
        res = minimize(objective, x0, method='Nelder-Mead', 
                      options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-10})
        best_pts = res.x.reshape(-1, 2)
        best_score = _score(best_pts)
        return best_pts, best_score
    except Exception:
        return start_pts, _score(start_pts)

def construct():
    deadline = 120.0  # Safety margin before 150s limit
    start_time = time.time()
    
    # Start with regular 14-gon
    best_pts = _regular_polygon(N)
    best_score = _score(best_pts)
    
    # Try local optimization from regular polygon
    if time.time() - start_time < deadline - 30:
        opt_pts, opt_score = _optimize_from_start(best_pts, max_iter=100, timeout=15)
        if opt_score > best_score:
            best_pts = opt_pts
            best_score = opt_score
    
    # Try perturbations around the best configuration
    rng = np.random.default_rng(42)
    attempts = 0
    max_attempts = 20
    
    while time.time() - start_time < deadline - 15 and attempts < max_attempts:
        attempts += 1
        # Create perturbed version
        noise_scale = 0.05 / (1 + attempts * 0.1)  # Decreasing noise
        perturbed = best_pts + rng.normal(0, noise_scale, size=(N, 2))
        
        # Try to optimize from perturbed start
        if time.time() - start_time < deadline - 20:
            opt_pts, opt_score = _optimize_from_start(perturbed, max_iter=50, timeout=5)
            if opt_score > best_score:
                best_pts = opt_pts
                best_score = opt_score
    
    # Final attempt: try a few more random restarts if time permits
    if time.time() - start_time < deadline - 10:
        for _ in range(3):
            if time.time() - start_time >= deadline - 5:
                break
            # Random points in unit disk
            angles = rng.uniform(0, 2*np.pi, N)
            radii = np.sqrt(rng.uniform(0, 1, N))
            rand_pts = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
            s = _score(rand_pts)
            if s > best_score:
                best_pts = rand_pts
                best_score = s
    
    return best_pts
