import numpy as np
from scipy.optimize import minimize
import time

def moving_sofa_2d() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 620  # safety margin
    
    # Sofa grid
    xs = np.arange(-3, 1.5, 0.01)
    ys = np.arange(-1.5, 1.5, 0.01)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    sofa_points = np.column_stack([X.ravel(), Y.ravel()])
    n_points = len(sofa_points)
    
    # Base corridor in local coords
    # Part 1: 0<=u<=1, -3<=v<=1
    # Part 2: -3<=u<=1, 0<=v<=1
    
    def in_base(u, v):
        """Check if point (u,v) is in the base corridor (local coords)."""
        p1 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
        p2 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
        return p1 | p2
    
    def compute_sofa_area(path):
        """path: (n,3) array of (x,y,a_deg). Compute area of intersection."""
        # Include start and end
        full_path = np.vstack([np.array([[0.0, 0.0, 0.0]]), path, np.array([[-2.0, 0.0, 90.0]])])
        
        # Densify: between consecutive rows, add linear interpolations
        densified = [full_path[0]]
        for i in range(len(full_path)-1):
            x1, y1, a1 = full_path[i]
            x2, y2, a2 = full_path[i+1]
            m = max(1, int(np.ceil(abs(a2 - a1))))
            for k in range(1, m+1):
                t = k / m
                x = x1 + t*(x2-x1)
                y = y1 + t*(y2-y1)
                a = a1 + t*(a2-a1)
                densified.append([x, y, a])
        densified = np.array(densified)
        
        # For each placement, compute which sofa points are inside
        # A point is in the sofa if it's inside ALL placements
        mask = np.ones(n_points, dtype=bool)
        
        for x, y, a_deg in densified:
            a_rad = np.deg2rad(a_deg)
            cos_a = np.cos(a_rad)
            sin_a = np.sin(a_rad)
            
            # R(-a)(w - (x,y)): rotate by -a
            # R(-a) = [[cos(a), sin(a)], [-sin(a), cos(a)]]
            dx = sofa_points[:, 0] - x
            dy = sofa_points[:, 1] - y
            
            u = cos_a * dx + sin_a * dy
            v = -sin_a * dx + cos_a * dy
            
            inside = in_base(u, v)
            mask &= inside
        
        area = np.sum(mask) * 0.0001
        return area
    
    # Initial guess: use the baseline path
    best_path = np.array([
        [-0.1, 0.1, 15.0],
        [-0.2, 0.2, 20.0],
        [-0.3, 0.3, 34.0],
        [-0.3, 0.3, 40.0],
        [-0.4, 0.3, 50.0],
        [-0.5, 0.2, 55.0],
        [-0.6, 0.2, 75.0],
        [-0.7, 0.3, 86.0],
    ], dtype=np.float64)
    
    best_area = compute_sofa_area(best_path)
    
    # Try to improve with local optimization
    # Parameterize by the 8*3=24 values
    def objective(params):
        if time.time() > deadline - 5:
            return -best_area  # return negative of best to stop
        path = params.reshape(-1, 3)
        # Ensure angles are in reasonable range
        path[:, 2] = np.clip(path[:, 2], -180, 180)
        try:
            area = compute_sofa_area(path)
            return -area  # minimize negative area
        except:
            return -best_area
    
    # Nelder-Mead optimization
    x0 = best_path.ravel()
    result = minimize(objective, x0, method='Nelder-Mead', 
                      options={'maxiter': 500, 'xatol': 0.01, 'fatol': 0.0001})
    
    if result.fun < -best_area:
        best_path = result.x.reshape(-1, 3)
        best_area = -result.fun
    
    # Try a few random restarts if time permits
    n_restarts = 5
    for _ in range(n_restarts):
        if time.time() > deadline - 60:
            break
        
        # Perturb the best path
        perturbation = np.random.randn(*best_path.shape) * 0.1
        new_path = best_path + perturbation
        new_path[:, 2] = np.clip(new_path[:, 2], 5, 85)  # keep angles reasonable
        
        result = minimize(objective, new_path.ravel(), method='Nelder-Mead',
                         options={'maxiter': 200, 'xatol': 0.01, 'fatol': 0.0001})
        
        if result.fun < -best_area:
            best_path = result.x.reshape(-1, 3)
            best_area = -result.fun
    
    # Final check
    final_area = compute_sofa_area(best_path)
    if final_area > best_area:
        best_area = final_area
    
    return best_path
