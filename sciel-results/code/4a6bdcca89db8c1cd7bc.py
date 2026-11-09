import numpy as np
from scipy.optimize import differential_evolution
import time

def _vertex_score(polygon, i):
    v = polygon[i]
    d = np.linalg.norm(polygon - v, axis=1)
    d[i] = np.inf
    d = np.sort(d)
    # d has 19 elements
    # Check d_0 < 0.05
    if d[0] < 0.05:
        return 0.0
    
    # Find k: first index minimizing d_{k+1} - d_k
    gaps = np.diff(d)
    k = np.argmin(gaps)
    
    # D = (d_k + d_{k+1})/2
    D = (d[k] + d[k+1]) / 2.0
    if D < 0.1:
        return 0.0
    
    # Select 4 distances
    if k == 0:
        dists = d[0:4]
    elif k == 17: # Note: d has 19 elements, indices 0..18. k max is 18?
        # gaps has 18 elements. k in 0..17.
        # If k=17, d_{17}, d_{18} are the pair.
        # "d_15..d_18" means d[15], d[16], d[17], d[18]
        dists = d[15:19]
    else:
        # d_{k-1}..d_{k+2}
        dists = d[k-1:k+3]
        
    # raw = mean of max(d/D, D/d)
    ratios = np.maximum(dists / D, D / dists)
    raw = np.mean(ratios)
    
    return min(1.0, 1.0 / raw)

def _min_vertex_score(polygon):
    scores = [_vertex_score(polygon, i) for i in range(len(polygon))]
    return min(scores)

def _is_valid(polygon):
    # Check convexity
    n = len(polygon)
    if n < 3: return False
    cross_products = []
    for i in range(n):
        p0 = polygon[i]
        p1 = polygon[(i+1)%n]
        p2 = polygon[(i+2)%n]
        v1 = p1 - p0
        v2 = p2 - p1
        cross = v1[0]*v2[1] - v1[1]*v2[0]
        cross_products.append(cross)
        
    # All must be same sign and non-zero
    if any(c == 0 for c in cross_products): return False
    if np.any(np.array(cross_products) > 0) and np.any(np.array(cross_products) < 0): return False
    
    # Check no repeated points
    for i in range(n):
        for j in range(i+1, n):
            if np.allclose(polygon[i], polygon[j]):
                return False
    return True

def optimize_5gon():
    # Initial guess: Regular pentagon
    angles = np.linspace(0, 2*np.pi, 5, endpoint=False)
    r = 1.0
    x = r * np.cos(angles)
    y = r * np.sin(angles)
    init = np.column_stack((x, y)).flatten()
    
    bounds = [(-2, 2)] * 10 # x, y for 5 points
    
    def objective(params):
        poly = params.reshape(5, 2)
        # Normalize to avoid scale issues during optimization?
        # The score is scale invariant.
        # But we need to ensure it's not degenerate.
        if not _is_valid(poly):
            return 100.0 # High penalty
        
        score = _min_vertex_score(poly)
        return -score
        
    result = differential_evolution(objective, bounds, seed=42, maxiter=1000, tol=1e-4)
    return result.x.reshape(5, 2), -result.fun

def equidistant_points_polygon20() -> np.ndarray:
    start_time = time.time()
    deadline = 600.0
    margin = 10.0
    
    # Step 1: Get a good 5-gon
    # If we have time, optimize it. Otherwise use regular.
    # Optimization might take some time. Let's do a quick local search first.
    
    # For now, let's assume we can optimize quickly.
    # If not, fallback to regular 20-gon.
    
    try:
        # Quick optimization
        poly5, score5 = optimize_5gon()
        
        # Check if it's better than regular
        # Regular 20-gon score is ~0.67
        if score5 > 0.67:
            # Construct 20-gon from 4 copies of poly5
            # Place them on a circle of radius R=10
            R = 10.0
            base_angles = np.linspace(0, 2*np.pi, 4, endpoint=False)
            
            polygon = np.zeros((20, 2))
            for j in range(4):
                center = np.array([R * np.cos(base_angles[j]), R * np.sin(base_angles[j])])
                # Rotate the 5-gon to align with radial direction?
                # Or just place it.
                # To ensure convexity, the "outward" direction of the 5-gon should align with the radial vector.
                # We can rotate poly5 so that its "center" (centroid) is at center, and it faces out.
                centroid = np.mean(poly5, axis=0)
                poly5_centered = poly5 - centroid
                
                # Determine "outward" normal of the 5-gon
                # Simplest: align the vertex with max projection on radial vector?
                # Or just rotate by base_angles[j].
                angle_rot = base_angles[j]
                cos_r, sin_r = np.cos(angle_rot), np.sin(angle_rot)
                rot_mat = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
                
                poly5_rot = poly5_centered @ rot_mat.T
                poly5_trans = poly5_rot + center
                
                polygon[j*5:(j+1)*5] = poly5_trans
            
            # Check validity of the 20-gon
            if _is_valid(polygon):
                # Normalize
                centroid20 = np.mean(polygon, axis=0)
                polygon_norm = polygon - centroid20
                dists = np.linalg.norm(polygon_norm, axis=1)
                mean_dist = np.mean(dists)
                if mean_dist > 0.01:
                    polygon_norm = polygon_norm / mean_dist
                    return polygon_norm
            
    except Exception as e:
        pass
        
    # Fallback: Regular 20-gon
    n = 20
    radius = 1.0
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    polygon = np.stack([x, y], axis=1)
    return polygon
