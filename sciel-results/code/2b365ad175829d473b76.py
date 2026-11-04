import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 13

def construct():
    # Initialize with a regular N-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    points = np.column_stack((np.cos(angles), np.sin(angles)))
    
    # Define the objective function to minimize (negative of the score)
    def objective(x):
        pts = x.reshape(-1, 2)
        try:
            # Center the points
            centroid = np.mean(pts, axis=0)
            pts = pts - centroid
            
            # Calculate convex hull area
            hull = ConvexHull(pts)
            hull_area = hull.volume
            
            if hull_area < 1e-10:
                return 0.0
            
            # Normalize to unit area
            scale = np.sqrt(hull_area)
            pts = pts / scale
            # Now hull area is 1 (approximately, due to scaling)
            # Recalculate hull area to be precise? 
            # Scaling by 1/sqrt(A) makes area 1.
            
            # Calculate minimum triangle area
            min_area = float('inf')
            n = len(pts)
            for i in range(n):
                for j in range(i + 1, n):
                    for k in range(j + 1, n):
                        p1 = pts[i]
                        p2 = pts[j]
                        p3 = pts[k]
                        area = 0.5 * abs(p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1]))
                        if area < min_area:
                            min_area = area
            
            if min_area == float('inf'):
                return 0.0
            
            # Score is min_area (since hull area is 1)
            # We want to maximize score, so minimize -score
            return -min_area

        except Exception:
            return 0.0

    # Optimize the shape
    # Use Nelder-Mead as it doesn't require gradients and is robust
    res = minimize(objective, points.flatten(), method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6})
    
    optimized_points = res.x.reshape(-1, 2)
    
    return optimized_points
