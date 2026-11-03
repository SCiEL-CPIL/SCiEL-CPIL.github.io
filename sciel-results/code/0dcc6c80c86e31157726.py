import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 14

# EVOLVE-BLOCK-START
def _triangle_area(p1, p2, p3):
    v1 = p2 - p1
    v2 = p3 - p1
    return 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])

def _score(points):
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area < 1e-9:
            return -1e6
    except Exception:
        return -1e6
        
    min_area = float('inf')
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                area = _triangle_area(points[i], points[j], points[k])
                if area < min_area:
                    min_area = area
                    
    if min_area < 1e-9:
        return -1e6
        
    return min_area / hull_area

def _optimize_from_start(start_points):
    x0 = start_points.flatten()
    
    def objective(x_flat):
        pts = x_flat.reshape(N, 2)
        center = np.mean(pts, axis=0)
        pts = pts - center
        norms = np.linalg.norm(pts, axis=1)
        mean_norm = np.mean(norms)
        if mean_norm < 1e-9:
            return 1e6
        pts = pts / mean_norm
        
        s = _score(pts)
        if s < 0:
            return 1e6
        return -s

    try:
        result = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-7, 'fatol': 1e-7})
        best_pts = result.x.reshape(N, 2)
        center = np.mean(best_pts, axis=0)
        best_pts = best_pts - center
        norms = np.linalg.norm(best_pts, axis=1)
        mean_norm = np.mean(norms)
        if mean_norm > 1e-9:
            best_pts = best_pts / mean_norm
        return best_pts, -result.fun
    except Exception:
        return start_points, -1e6

def construct():
    best_points = None
    best_score = -1e6
    
    # Strategy 1: Regular Polygon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    reg_pts = np.column_stack((np.cos(angles), np.sin(angles)))
    pts, score = _optimize_from_start(reg_pts)
    if score > best_score:
        best_score = score
        best_points = pts
        
    # Strategy 2: Perturbed Regular Polygon (Break symmetry)
    rng = np.random.RandomState(42)
    for i in range(5):
        noise = rng.normal(0, 0.1, size=(N, 2))
        perturbed = reg_pts + noise
        pts, score = _optimize_from_start(perturbed)
        if score > best_score:
            best_score = score
            best_points = pts
            
    # Strategy 3: Two concentric circles? Or Grid?
    # A grid might be bad for convex hull area if not on boundary.
    # Let's try a "squashed" polygon or just rely on the polygon optimization.
    
    return best_points
# EVOLVE-BLOCK-END
