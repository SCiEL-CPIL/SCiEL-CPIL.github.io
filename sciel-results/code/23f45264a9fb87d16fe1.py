import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 13

def get_score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    """
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area < 1e-10:
            return 0.0
    except:
        return 0.0

    min_area = float('inf')
    n = len(points)
    # Vectorized calculation for speed, though N=13 is small
    # 13C3 = 286 triangles
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                # Area = 0.5 * | (x2-x1)(y3-y1) - (y2-y1)(x3-x1) |
                area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]))
                if area < min_area:
                    min_area = area
    
    if min_area == float('inf'):
        return 0.0
        
    return min_area / hull_area

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Initial guess: Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    radius = 1.0
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    initial_points = np.column_stack((x, y))
    
    best_score = get_score(initial_points)
    best_points = initial_points.copy()
    
    def objective(flat_points):
        pts = flat_points.reshape(N, 2)
        return -get_score(pts)
        
    # Try to optimize from the regular polygon
    if time.time() < deadline:
        try:
            res = minimize(
                objective, 
                initial_points.flatten(), 
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            if res.fun < -best_score:
                best_points = res.x.reshape(N, 2)
                best_score = -res.fun
        except Exception as e:
            pass
            
    # If time allows, try a few random perturbations/restarts
    # to ensure we didn't get stuck in a bad local minimum, though regular N-gon is very robust
    if time.time() < deadline - 5.0:
        rng = np.random.default_rng(42)
        for _ in range(3):
            if time.time() >= deadline:
                break
            # Perturb the regular polygon
            noise = rng.normal(0, 0.1, (N, 2))
            start_pts = initial_points + noise
            try:
                res = minimize(
                    objective, 
                    start_pts.flatten(), 
                    method='Nelder-Mead',
                    options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
                )
                if res.fun < -best_score:
                    best_points = res.x.reshape(N, 2)
                    best_score = -res.fun
            except Exception as e:
                pass

    return best_points
