import numpy as np
from scipy.optimize import minimize
import time

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    
    # Define hex lattice basis
    sqrt3 = np.sqrt(3)
    u = np.array([sqrt3, 0.0])
    v = np.array([sqrt3 / 2.0, 1.5])
    
    # Generate candidate points
    points = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            p = i * u + j * v
            dist_sq = np.dot(p, p)
            points.append((dist_sq, p))
            
    # Sort by distance from origin
    points.sort(key=lambda x: x[0])
    
    # Take the 12 closest points
    cluster_points = [p[1] for p in points[:N]]
    
    # Objective function
    def calc_L(params):
        theta, x_c, y_c = params
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        
        max_L = 0.0
        for p in cluster_points:
            dx = p[0] - x_c
            dy = p[1] - y_c
            xr = dx * cos_t - dy * sin_t
            yr = dx * sin_t + dy * cos_t
            
            # Constraints for outer hex (angle 0, centered at 0,0)
            # L >= (sqrt(3) x + y) / sqrt(3)
            # L >= (-sqrt(3) x + y) / sqrt(3)
            # L >= y / (sqrt(3)/2)
            # L >= (-sqrt(3) x - y) / sqrt(3)
            # L >= -y / (sqrt(3)/2)
            # L >= (sqrt(3) x - y) / sqrt(3)
            
            L1 = (sqrt3 * xr + yr) / sqrt3
            L2 = (-sqrt3 * xr + yr) / sqrt3
            L3 = yr / (sqrt3 / 2)
            L4 = (-sqrt3 * xr - yr) / sqrt3
            L5 = -yr / (sqrt3 / 2)
            L6 = (sqrt3 * xr - yr) / sqrt3
            
            L = max(L1, L2, L3, L4, L5, L6)
            if L > max_L:
                max_L = L
                
        return max_L

    # Initial guess
    x0 = [0.0, 0.0, 0.0]
    
    # Optimize
    try:
        res = minimize(calc_L, x0, method='Nelder-Mead', options={'xatol': 1e-7, 'fatol': 1e-7, 'maxiter': 2000})
        theta_opt, x_c_opt, y_c_opt = res.x
        L_opt = res.fun
    except Exception:
        # Fallback to grid search if optimization fails
        L_opt = float('inf')
        best_params = (0.0, 0.0, 0.0)
        for theta in np.linspace(-np.pi/6, np.pi/6, 12):
            for xc in np.linspace(-2, 2, 21):
                for yc in np.linspace(-2, 2, 21):
                    L = calc_L([theta, xc, yc])
                    if L < L_opt:
                        L_opt = L
                        best_params = (theta, xc, yc)
        theta_opt, x_c_opt, y_c_opt = best_params

    # Construct inner hexes
    inner = []
    for p in cluster_points:
        x_rot = p[0] * np.cos(theta_opt) - p[1] * np.sin(theta_opt)
        y_rot = p[0] * np.sin(theta_opt) + p[1] * np.cos(theta_opt)
        x_final = x_rot + x_c_opt
        y_final = y_rot + y_c_opt
        inner.append([x_final, y_final, np.degrees(theta_opt)])
        
    return inner, (0.0, 0.0), L_opt, 0.0
# EVOLVE-BLOCK-END
