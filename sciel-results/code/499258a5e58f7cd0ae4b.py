import numpy as np
import time
from scipy.optimize import minimize
import random

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2.0])
AREA = np.sqrt(3) / 4.0

def _min_triangle_area(points):
    """Compute the minimum area of any triangle formed by 3 of the points."""
    n = len(points)
    min_area = float('inf')
    # Vectorized for speed
    # points shape (n, 2)
    # We need area for all triples i, j, k
    # Area = 0.5 * | (pj - pi) x (pk - pi) |
    # We can loop over i, and vectorize j, k
    for i in range(n):
        pi = points[i]
        # Vectors from pi to all other points
        v = points - pi
        # We need cross products v[j] x v[k] for j > i, k > j
        # Actually, simpler to just do triple loop for n=11, it's 165 iterations.
        # 165 * (few flops) is very fast.
        for j in range(i + 1, n):
            v1 = points[j] - pi
            for k in range(j + 1, n):
                v2 = points[k] - pi
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                if area < min_area:
                    min_area = area
    return min_area / AREA

def _is_inside_triangle(p):
    x, y = p
    if y < -1e-9: return False
    if x < -1e-9 or x > 1.0 + 1e-9: return False
    max_y = np.sqrt(3) * x if x <= 0.5 else np.sqrt(3) * (1.0 - x)
    return y <= max_y + 1e-9

def _params_to_points(params):
    """
    Convert symmetric parameters to 11 points.
    params: [y_c, x_1, y_1, x_2, y_2, ..., x_5, y_5]
    Point 0: Center (0.5, y_c)
    Points 1,2: (0.5-x_1, y_1), (0.5+x_1, y_1)
    ...
    Points 9,10: (0.5-x_5, y_5), (0.5+x_5, y_5)
    """
    points = []
    y_c = params[0]
    points.append(np.array([0.5, y_c]))
    
    idx = 1
    for i in range(5):
        x_i = params[idx]
        y_i = params[idx+1]
        idx += 2
        points.append(np.array([0.5 - x_i, y_i]))
        points.append(np.array([0.5 + x_i, y_i]))
        
    return np.array(points)

def _objective(params):
    """
    Objective function for minimization.
    Returns negative of min area (to maximize).
    Also adds penalty if points are outside triangle.
    """
    points = _params_to_points(params)
    
    # Check bounds
    for p in points:
        if not _is_inside_triangle(p):
            return 1000.0  # Large penalty
    
    min_area = _min_triangle_area(points)
    return -min_area

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_score = 0.0
    best_params = None
    best_points = None
    
    rng = random.Random(42)
    
    # Strategy: Multiple random restarts with Nelder-Mead optimization
    # The space is 11-dimensional.
    
    # Define a helper to generate random initial params
    def random_params():
        # y_c should be in [0, h]
        # x_i should be small enough to stay in triangle
        # y_i should be in [0, h]
        # Rough constraints:
        # For point (0.5 +/- x, y) to be in triangle:
        # y <= sqrt(3) * (0.5 - x)  (if x>0, left side constraint is y <= sqrt(3)*(0.5-x) ?? No)
        # Left side line: y = sqrt(3) x. Point (0.5-x, y).
        # y <= sqrt(3) * (0.5 - x) is NOT correct.
        # Line from (0,0) to (0.5, h): y = sqrt(3) x.
        # Point (x_p, y_p) is inside if y_p <= sqrt(3) x_p (for x_p < 0.5)
        # Here x_p = 0.5 - x_i.
        # So y_i <= sqrt(3) * (0.5 - x_i)
        # Similarly for right side: y_i <= sqrt(3) * (1 - (0.5 + x_i)) = sqrt(3) * (0.5 - x_i)
        # So constraint is y_i <= sqrt(3) * (0.5 - x_i)
        # And y_i >= 0, x_i >= 0
        
        params = []
        # Center point
        y_c = rng.uniform(0.1, 0.8)
        params.append(y_c)
        
        for i in range(5):
            # Choose x_i such that there is room for y_i
            # Max y_i is sqrt(3)*(0.5 - x_i)
            # Let's pick x_i in [0.01, 0.4]
            x_i = rng.uniform(0.01, 0.4)
            max_y = np.sqrt(3) * (0.5 - x_i)
            if max_y < 0:
                max_y = 0.01
            y_i = rng.uniform(0.01, max_y)
            params.append(x_i)
            params.append(y_i)
            
        return np.array(params)

    # Run several optimizations
    n_restarts = 50
    for r in range(n_restarts):
        if time.time() > deadline:
            break
            
        p0 = random_params()
        
        # Use Nelder-Mead
        # maxiter needs to be high enough, but we have time limits
        try:
            res = minimize(_objective, p0, method='Nelder-Mead', 
                           options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 200, 'adaptive': True})
            
            if res.fun < -best_score: # res.fun is -area, so lower is better
                best_score = -res.fun
                best_params = res.x
                best_points = _params_to_points(res.x)
                
        except Exception as e:
            pass

    # If Nelder-Mead failed or didn't find anything good, try a simple hill climb on the best found
    if best_params is not None and time.time() < deadline:
        # Local refinement
        p_best = best_params.copy()
        current_val = _objective(p_best)
        step_size = 0.05
        for it in range(100):
            if time.time() > deadline:
                break
            improved = False
            for dim in range(len(p_best)):
                for direction in [-1, 1]:
                    p_new = p_best.copy()
                    p_new[dim] += direction * step_size
                    val_new = _objective(p_new)
                    if val_new < current_val:
                        p_best = p_new
                        current_val = val_new
                        improved = True
            if not improved:
                step_size *= 0.5
                if step_size < 1e-5:
                    break
        
        final_val = _objective(p_best)
        if final_val < -best_score:
            best_score = -final_val
            best_points = _params_to_points(p_best)

    # Fallback if no valid points found
    if best_points is None:
        # Return a default valid configuration
        # Just return random points inside
        pts = []
        for i in range(11):
            while True:
                x = rng.random()
                y = rng.random() * np.sqrt(3) * 0.5
                if _is_inside_triangle(np.array([x,y])):
                    pts.append(np.array([x,y]))
                    break
        best_points = np.array(pts)
        
    return best_points
