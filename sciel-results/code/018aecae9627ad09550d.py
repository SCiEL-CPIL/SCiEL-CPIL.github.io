import numpy as np
import scipy.spatial
import time

N = 14

def construct():
    # Time limit check
    start_time = time.time()
    time_limit = 110.0 # Leave 10s buffer
    
    # 1. Initialize with Regular N-gon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    points = np.vstack([np.cos(angles), np.sin(angles)]).T
    
    # 2. Score Function
    def get_score(pts):
        # Calculate all triangle areas
        # Use vectorized operations for speed
        # i, j, k indices
        areas = []
        n = len(pts)
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    # Area = 0.5 * |cross(p_j - p_i, p_k - p_i)|
                    v1 = pts[j] - pts[i]
                    v2 = pts[k] - pts[i]
                    area = 0.5 * abs(v1[0]*v2[1] - v1[1]*v2[0])
                    areas.append(area)
        min_area = np.min(areas)
        
        # Calculate Hull Area
        try:
            hull = scipy.spatial.ConvexHull(pts)
            hull_area = hull.volume # In 2D, .volume is area
        except Exception:
            # If points are collinear or less than 3 non-collinear, area is 0
            return 0.0
            
        if hull_area == 0:
            return 0.0
            
        return min_area / hull_area

    current_score = get_score(points)
    best_points = points.copy()
    best_score = current_score
    
    # 3. Local Search
    step_size = 0.1
    iterations = 0
    max_iterations = 20000
    
    while iterations < max_iterations:
        if time.time() - start_time > time_limit:
            break
            
        # Pick random point
        idx = np.random.randint(N)
        # Perturb
        perturbation = np.random.normal(0, step_size, 2)
        new_points = points.copy()
        new_points[idx] += perturbation
        
        new_score = get_score(new_points)
        
        if new_score > best_score:
            best_score = new_score
            best_points = new_points.copy()
            # Accept
            points = new_points
            # Maybe increase step size slightly if we found a good one? Or keep same.
            # Let's keep step size constant or decrease slowly
        else:
            # Rejection: do nothing to points, but maybe decrease step size
            pass
            
        iterations += 1
        
        # Cool down
        if iterations % 100 == 0:
            step_size *= 0.95
            
    return best_points
