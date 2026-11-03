import numpy as np
from scipy.optimize import minimize

N = 14

def min_triangle_area(points):
    # points: (N, 2)
    min_area = np.inf
    for i in range(N):
        for j in range(i+1, N):
            for k in range(j+1, N):
                p1, p2, p3 = points[i], points[j], points[k]
                # Area = 0.5 * |cross(p2-p1, p3-p1)|
                v1 = p2 - p1
                v2 = p3 - p1
                area = 0.5 * abs(v1[0]*v2[1] - v1[1]*v2[0])
                if area < min_area:
                    min_area = area
    return min_area

def hull_area(points):
    # Convex hull area
    from scipy.spatial import ConvexHull
    hull = ConvexHull(points)
    return hull.volume # In 2D, volume is area

def score(points):
    a_min = min_triangle_area(points)
    a_hull = hull_area(points)
    if a_hull == 0: return 0
    return a_min / a_hull

def construct():
    # Initial guess: Regular 14-gon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Optimization
    # We want to maximize score.
    # Since score is scale invariant, we can fix the scale during optimization to avoid drift?
    # Or just optimize the raw points and normalize at the end.
    # Minimizing -score is non-smooth due to min().
    # Instead, we can maximize the min triangle area subject to hull area = 1.
    
    def objective(x):
        pts = x.reshape(N, 2)
        # Normalize to unit hull area to keep search space bounded
        h_area = hull_area(pts)
        if h_area < 1e-6:
            return 0
        scale = np.sqrt(1.0 / h_area)
        pts_scaled = pts * scale
        return -min_triangle_area(pts_scaled)
        
    # Use Nelder-Mead for robustness against non-smoothness
    result = minimize(objective, points.flatten(), method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-4, 'fatol': 1e-6})
    
    best_points = result.x.reshape(N, 2)
    
    # Final normalization to unit hull area
    h_area = hull_area(best_points)
    if h_area > 0:
        best_points = best_points * np.sqrt(1.0 / h_area)
        
    return best_points
