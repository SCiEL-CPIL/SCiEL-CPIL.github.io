import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull

def heilbronn_convex13() -> np.ndarray:
    n = 13
    
    # Start with a regular 13-gon as initial guess
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x0 = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Normalize so convex hull area = 1
    try:
        hull = ConvexHull(x0)
        hull_area = hull.volume
        x0 = x0 / np.sqrt(hull_area)
    except Exception:
        pass
    
    # Use a grid-based search with local refinement
    # Try multiple initial configurations and optimize each
    
    def compute_min_area(pts):
        """Compute minimum triangle area among all C(13,3)=286 triangles"""
        areas = np.zeros((n, n, n))
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    x1, y1 = pts[i]
                    x2, y2 = pts[j]
                    x3, y3 = pts[k]
                    a = 0.5 * abs(x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))
                    areas[i, j, k] = a
        # Only consider valid triangles (i < j < k)
        min_area = min(areas[i, j, k] for i in range(n) for j in range(i+1, n) for k in range(j+1, n))
        return min_area
    
    def convex_hull_area(pts):
        """Compute convex hull area"""
        try:
            hull = ConvexHull(pts)
            return hull.volume
        except Exception:
            return 0.0
    
    def score(pts):
        """Compute the metric: A_min / A_hull"""
        hull_area = convex_hull_area(pts)
        if hull_area < 1e-10:
            return 0.0
        min_area = compute_min_area(pts)
        return min_area / hull_area
    
    # Try multiple optimization strategies
    best_points = x0.copy()
    best_score = score(x0)
    
    # Strategy 1: Local optimization from regular polygon
    from scipy.optimize import minimize
    
    def objective(p):
        pts = p.reshape(n, 2)
        # Penalty if points go outside unit circle
        max_r = np.max(np.sqrt(p[0::2]**2 + p[1::2]**2))
        penalty = 100.0 * max(0, max_r - 1.0)**2
        min_a = compute_min_area(pts)
        return -min_a + penalty
    
    # Try Nelder-Mead with better settings
    p0 = x0.flatten()
    result = minimize(objective, p0, method='Nelder-Mead', 
                      options={'maxiter': 20000, 'xatol': 1e-10, 'fatol': 1e-10,
                               'initial_simplex': None})
    
    pts = result.x.reshape(n, 2)
    s = score(pts)
    if s > best_score:
        best_score = s
        best_points = pts
    
    # Strategy 2: Try different initial configurations
    for trial in range(5):
        # Perturb the initial guess
        np.random.seed(42 + trial)
        noise = np.random.randn(n, 2) * 0.1
        p0_trial = (x0 + noise).flatten()
        
        result = minimize(objective, p0_trial, method='Nelder-Mead',
                         options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-10})
        
        pts = result.x.reshape(n, 2)
        s = score(pts)
        if s > best_score:
            best_score = s
            best_points = pts
    
    # Final normalization and validation
    if not np.all(np.isfinite(best_points)):
        return x0
    
    try:
        hull = ConvexHull(best_points)
        if hull.volume < 1e-6:
            return x0
    except Exception:
        return x0
    
    return best_points
