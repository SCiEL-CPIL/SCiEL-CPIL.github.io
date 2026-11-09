import numpy as np
import time
from scipy.optimize import minimize

N = 11
H = np.sqrt(3) / 2

def construct():
    # Vertices of the equilateral triangle
    # v0 = (0,0), v1 = (1,0), v2 = (0.5, sqrt(3)/2)
    
    def is_inside(points):
        # points: (N, 2)
        px = points[:, 0]
        py = points[:, 1]
        cond1 = py >= -1e-9
        cond2 = py <= np.sqrt(3) * px + 1e-9
        cond3 = py <= np.sqrt(3) * (1 - px) + 1e-9
        return np.all(cond1 & cond2 & cond3)
        
    def min_area(points):
        # points: (N, 2)
        # Calculate areas of all triangles formed by 3 distinct points
        # Indices i < j < k
        idx = np.indices((N, N, N))
        mask = (idx[0] < idx[1]) & (idx[1] < idx[2])
        i, j, k = idx[:, :, :, mask].T
        
        p_i = points[i]
        p_j = points[j]
        p_k = points[k]
        
        # Vectorized area calculation: 0.5 * | (j-i) x (k-i) |
        v1 = p_j - p_i
        v2 = p_k - p_i
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        areas = 0.5 * np.abs(cross)
        
        return np.min(areas)
    
    def objective(points):
        if not is_inside(points):
            return 1.0 # Penalty
        return -min_area(points) # Minimize -area
        
    # Initialize random points
    def random_points():
        pts = []
        while len(pts) < N:
            x = np.random.rand()
            y = np.random.rand() * H
            if y <= np.sqrt(3)*x + 1e-9 and y <= np.sqrt(3)*(1-x) + 1e-9:
                pts.append([x, y])
        return np.array(pts)
        
    # Simulated Annealing
    current_pts = random_points()
    current_obj = objective(current_pts)
    best_pts = current_pts.copy()
    best_obj = current_obj
    
    T = 0.02
    T_min = 1e-6
    alpha = 0.995
    step_size = 0.05
    
    start_time = time.time()
    time_limit = 100 # Leave 20s buffer
    
    while time.time() - start_time < time_limit and T > T_min:
        # Pick a random point to move
        idx = np.random.randint(N)
        # Generate new position
        new_pts = current_pts.copy()
        noise = np.random.normal(0, step_size, 2)
        new_pts[idx] += noise
        
        # Check if inside
        if not is_inside(new_pts):
            continue
            
        new_obj = objective(new_pts)
        
        # Acceptance criterion
        if new_obj < current_obj:
            current_pts = new_pts
            current_obj = new_obj
            if current_obj < best_obj:
                best_pts = current_pts.copy()
                best_obj = current_obj
        else:
            delta = new_obj - current_obj
            if delta < 1e-12:
                continue
            if np.random.rand() < np.exp(-delta / T):
                current_pts = new_pts
                current_obj = new_obj
                
        T *= alpha
        
    # Final polish with Nelder-Mead on the best found
    def obj_func(x):
        pts = x.reshape(-1, 2)
        if not is_inside(pts):
            return 1.0
        return -min_area(pts)
        
    # Run a short Nelder-Mead to polish
    try:
        result = minimize(obj_func, best_pts.flatten(), method='Nelder-Mead', 
                          options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10})
        
        final_pts = result.x.reshape(-1, 2)
        
        # Ensure validity
        if is_inside(final_pts):
            final_obj = objective(final_pts)
            if final_obj < best_obj:
                best_pts = final_pts
                best_obj = final_obj
    except Exception:
        pass
        
    return best_pts
