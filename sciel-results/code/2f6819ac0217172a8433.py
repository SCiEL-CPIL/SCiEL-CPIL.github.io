import numpy as np
from scipy.optimize import minimize

def _score_polygon(points):
    """
    Calculate the minimum vertex score for a given polygon.
    """
    n = len(points)
    # Normalize
    centroid = np.mean(points, axis=0)
    points_centered = points - centroid
    dists_to_centroid = np.linalg.norm(points_centered, axis=1)
    mean_dist = np.mean(dists_to_centroid)
    
    if mean_dist <= 0.01:
        return 0.0
        
    points_norm = points_centered / mean_dist
    
    min_score = 1.0
    
    for i in range(n):
        # Distances from i to all others
        diffs = points_norm - points_norm[i]
        d = np.linalg.norm(diffs, axis=1)
        d[i] = np.inf # Exclude self
        
        # Sort distances
        d_sorted = np.sort(d)
        
        # Check for repeated points (distance < 0.05)
        if d_sorted[0] < 0.05:
            min_score = 0.0
            break
            
        # Find k: first index minimizing d_{k+1} - d_k
        gaps = np.diff(d_sorted)
        k = np.argmin(gaps)
        
        D = (d_sorted[k] + d_sorted[k+1]) / 2.0
        
        if D < 0.1:
            min_score = 0.0
            break
            
        # Select 4 distances
        if k == 0:
            d_sel = d_sorted[0:4]
        elif k == n - 2: # k=17 for n=20
            d_sel = d_sorted[15:19]
        else:
            d_sel = d_sorted[k-1:k+3]
            
        # Calculate raw score
        ratios = np.maximum(d_sel / D, D / d_sel)
        raw = np.mean(ratios)
        
        score_i = min(1.0, 1.0 / raw)
        
        if score_i < min_score:
            min_score = score_i
            
    return min_score

def _generate_polygon(r, theta):
    n = 20
    k = np.arange(1, n + 1)
    z = r * np.exp(1j * theta)
    points_complex = z ** k
    points = np.column_stack((points_complex.real, points_complex.imag))
    return points

def _objective(params):
    r, theta = params
    # Constraints: r > 0, theta > 0
    if r <= 0 or theta <= 0:
        return 1000.0
        
    points = _generate_polygon(r, theta)
    
    # Check convexity
    # Calculate cross products of consecutive edges
    # If any cross product has different sign, not convex
    n = len(points)
    signs = []
    for i in range(n):
        p0 = points[i]
        p1 = points[(i + 1) % n]
        p2 = points[(i + 2) % n]
        v1 = p1 - p0
        v2 = p2 - p1
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        if abs(cross) < 1e-9:
            return 1000.0 # Collinear
        signs.append(np.sign(cross))
        
    if len(set(signs)) > 1:
        return 1000.0 # Not strictly convex
        
    score = _score_polygon(points)
    return -score # Minimize negative score

def equidistant_points_polygon20() -> np.ndarray:
    # Initial guess: Regular polygon
    theta0 = 2.0 * np.pi / 20.0
    r0 = 1.0
    
    # Optimization
    # Bounds for r and theta
    bounds = [(0.8, 1.2), (0.1 * np.pi, 0.2 * np.pi)]
    
    # Use Nelder-Mead
    res = minimize(_objective, [r0, theta0], method='Nelder-Mead', bounds=bounds, options={'xatol': 1e-5, 'fatol': 1e-5, 'maxiter': 1000})
    
    r_opt, theta_opt = res.x
    
    # Generate final points
    points = _generate_polygon(r_opt, theta_opt)
    
    return points
