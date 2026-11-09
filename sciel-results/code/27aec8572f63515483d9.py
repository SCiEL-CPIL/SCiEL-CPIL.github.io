import numpy as np
from scipy.optimize import minimize

def squares_in_square10() -> tuple:
    n = 10
    # Initial configuration: Cross Pattern (4 corners, 4 edges, 2 center)
    # s_c = 0.25, s_e = 0.5, s_m = 0 (degenerate)
    # Let's use the optimal LP solution: s_c=0.375, s_e=0.25, s_m=0.25
    # Or s_c=0.25, s_e=0.5, s_m=0.
    # Let's start with s_c=0.3, s_e=0.4, s_m=0.1 (sum = 1.2+1.6+0.2=3.0)
    
    s_c = 0.3
    s_e = 0.4
    s_m = 0.1
    
    centers = np.zeros((n, 2))
    sides = np.zeros(n)
    
    # Corners (0-3)
    centers[0] = [s_c/2, s_c/2]
    centers[1] = [1-s_c/2, s_c/2]
    centers[2] = [s_c/2, 1-s_c/2]
    centers[3] = [1-s_c/2, 1-s_c/2]
    sides[0:4] = s_c
    
    # Edges (4-7)
    centers[4] = [0.5, s_e/2]      # Bottom
    centers[5] = [0.5, 1-s_e/2]    # Top
    centers[6] = [s_e/2, 0.5]      # Left
    centers[7] = [1-s_e/2, 0.5]    # Right
    sides[4:8] = s_e
    
    # Center (8-9)
    # Stack vertically in the middle
    # Available height 1 - 2*s_e = 0.2.
    # s_m = 0.1. Two squares height 0.2.
    centers[8] = [0.5, 0.5 - s_m/2 - s_m/2] # 0.5 - 0.1 = 0.4? 
    # Let's place them at y = 0.5 +/- s_m/2 + offset?
    # Gap is [s_e, 1-s_e] = [0.4, 0.6].
    # Square 8: [0.4, 0.5] -> center 0.45
    # Square 9: [0.5, 0.6] -> center 0.55
    centers[8] = [0.5, 0.45]
    centers[9] = [0.5, 0.55]
    sides[8:10] = s_m
    
    # Optimization
    # Variables: cx1..cx10, cy1..cy10, s1..s10
    x0 = np.concatenate([centers.flatten(), sides])
    
    def objective(x):
        c = x[:2*n].reshape(n, 2)
        s = x[2*n:]
        # Ensure positive sides
        s = np.maximum(s, 1e-6)
        
        # Boundary penalty
        bound_pen = 0
        for i in range(n):
            # x bounds
            if c[i, 0] - s[i]/2 < 0: bound_pen += (0 - (c[i, 0] - s[i]/2))**2
            if c[i, 0] + s[i]/2 > 1: bound_pen += ((c[i, 0] + s[i]/2) - 1)**2
            # y bounds
            if c[i, 1] - s[i]/2 < 0: bound_pen += (0 - (c[i, 1] - s[i]/2))**2
            if c[i, 1] + s[i]/2 > 1: bound_pen += ((c[i, 1] + s[i]/2) - 1)**2
            
        # Overlap penalty
        overlap_pen = 0
        for i in range(n):
            for j in range(i+1, n):
                dx = abs(c[i, 0] - c[j, 0]) - (s[i] + s[j])/2
                dy = abs(c[i, 1] - c[j, 1]) - (s[i] + s[j])/2
                # If dx < 0 and dy < 0, they overlap
                # Penalty is area of intersection
                ox = max(0, -dx)
                oy = max(0, -dy)
                overlap_pen += ox * oy
                
        # Maximize sum of sides -> Minimize -sum(sides) + penalty
        return -np.sum(s) + 1000 * (bound_pen + overlap_pen)

    # Use L-BFGS-B
    # Bounds: centers [0,1], sides [1e-6, 1]
    bounds = []
    for i in range(n):
        bounds.append((0, 1)) # cx
        bounds.append((0, 1)) # cy
    for i in range(n):
        bounds.append((1e-6, 1)) # s
        
    try:
        res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-10})
        x_opt = res.x
    except Exception:
        x_opt = x0
        
    c_opt = x_opt[:2*n].reshape(n, 2)
    s_opt = x_opt[2*n:]
    
    # Post-processing to ensure strict validity
    # Check and shrink if necessary
    tol = 1e-7
    def is_valid(c, s):
        for i in range(n):
            if not (np.isfinite(c[i,0]) and np.isfinite(c[i,1]) and np.isfinite(s[i])):
                return False
            if s[i] <= 0:
                return False
            if c[i,0] - s[i]/2 < -tol or c[i,0] + s[i]/2 > 1 + tol:
                return False
            if c[i,1] - s[i]/2 < -tol or c[i,1] + s[i]/2 > 1 + tol:
                return False
        for i in range(n):
            for j in range(i+1, n):
                t = (s[i] + s[j])/2 - tol
                if abs(c[i,0] - c[j,0]) < t and abs(c[i,1] - c[j,1]) < t:
                    return False
        return True

    if not is_valid(c_opt, s_opt):
        # Shrink sides until valid
        scale = 1.0
        for _ in range(100):
            s_test = s_opt * scale
            if is_valid(c_opt, s_test):
                s_opt = s_test
                break
            scale *= 0.99
            
    return c_opt, s_opt
