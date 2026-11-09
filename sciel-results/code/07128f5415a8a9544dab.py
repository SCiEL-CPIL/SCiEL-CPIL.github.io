import numpy as np
import random

N = 13

def construct():
    # Initial guess: Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Simple Local Search
    best_score = 0
    best_points = points.copy()
    
    def score(pts):
        # Calculate min triangle area
        min_area = float('inf')
        n = len(pts)
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    # Area = 0.5 * |cross(pj-pi, pk-pi)|
                    v1 = pts[j] - pts[i]
                    v2 = pts[k] - pts[i]
                    area = 0.5 * abs(v1[0]*v2[1] - v1[1]*v2[0])
                    if area < min_area:
                        min_area = area
        if min_area == 0:
            return 0
        
        # Calculate hull area
        # Use scipy if available, else simple convex hull
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(pts)
            hull_area = hull.volume # In 2D, volume is area
        except:
            # Fallback: Graham Scan or just approximate
            # For simplicity, assume we can import scipy
            import scipy.spatial
            hull = scipy.spatial.ConvexHull(pts)
            hull_area = hull.volume
            
        return min_area / hull_area

    best_score = score(points)
    
    rng = np.random.default_rng(42)
    for _ in range(200): # 200 iterations
        # Perturb all points
        new_points = points + rng.normal(0, 0.05, points.shape)
        s = score(new_points)
        if s > best_score:
            best_score = s
            points = new_points
    
    return points
