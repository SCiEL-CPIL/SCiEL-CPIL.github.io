import numpy as np
import time
from scipy.optimize import minimize

def squares_in_square10() -> tuple:
    n = 10
    start_time = time.time()
    deadline = start_time + 590  # 10s margin
    
    # Variables: x0, y0, s0, x1, y1, s1, ...
    # Shape: (30,)
    
    def unpack(z):
        # z is (30,)
        # returns centers (10,2), sides (10,)
        x = z[0::3]
        y = z[1::3]
        s = z[2::3]
        centers = np.stack([x, y], axis=1)
        return centers, s

    def penalty(z):
        centers, s = unpack(z)
        x = centers[:, 0]
        y = centers[:, 1]
        
        # Boundary penalties
        # x - s/2 >= 0  => s/2 - x <= 0
        # x + s/2 <= 1  => x + s/2 - 1 <= 0
        pen_x_left = np.maximum(0, s/2 - x)
        pen_x_right = np.maximum(0, x + s/2 - 1)
        pen_y_left = np.maximum(0, s/2 - y)
        pen_y_right = np.maximum(0, y + s/2 - 1)
        
        boundary_pen = np.sum(pen_x_left**2 + pen_x_right**2 + pen_y_left**2 + pen_y_right**2)
        
        # Overlap penalties
        overlap_pen = 0.0
        for i in range(n):
            for j in range(i+1, n):
                dx = abs(x[i] - x[j])
                dy = abs(y[i] - y[j])
                req_x = (s[i] + s[j]) / 2
                req_y = (s[i] + s[j]) / 2
                
                # If dx < req_x and dy < req_y, they overlap
                # Penalty is (req_x - dx)^2 + (req_y - dy)^2 if positive
                ox = max(0, req_x - dx)
                oy = max(0, req_y - dy)
                overlap_pen += ox**2 + oy**2
                
        return 100 * (boundary_pen + overlap_pen)

    def objective(z):
        centers, s = unpack(z)
        return -np.sum(s) + penalty(z)

    # Initial guesses
    def get_grid_init():
        k = 4 # 4x4 grid for 10 squares
        side = 1.0 / k
        idx = np.arange(n)
        cols = idx % k
        rows = idx // k
        cx = (cols + 0.5) * side
        cy = (rows + 0.5) * side
        centers = np.stack([cx, cy], axis=1)
        sides = np.full(n, side)
        # Pack into z
        z = np.zeros(3*n)
        z[0::3] = centers[:, 0]
        z[1::3] = centers[:, 1]
        z[2::3] = sides
        return z

    def get_random_init():
        # Random centers in [0.2, 0.8] to avoid boundaries initially
        # Random sides in [0.1, 0.4]
        cx = np.random.uniform(0.2, 0.8, n)
        cy = np.random.uniform(0.2, 0.8, n)
        s = np.random.uniform(0.1, 0.4, n)
        z = np.zeros(3*n)
        z[0::3] = cx
        z[1::3] = cy
        z[2::3] = s
        return z

    bounds = [(0, 1) for _ in range(n)] + [(0, 1) for _ in range(n)] + [(1e-6, 1.0) for _ in range(n)]
    # Actually, x,y can be 0 to 1. s can be small to 1.
    # But if x=0 and s=1, it violates boundary. The penalty handles it.
    # L-BFGS-B bounds:
    # x: [0, 1]
    # y: [0, 1]
    # s: [1e-6, 1.0]
    
    best_z = None
    best_score = -np.inf
    
    # Run several optimizations
    num_restarts = 20
    
    for i in range(num_restarts):
        if time.time() > deadline:
            break
            
        if i == 0:
            z0 = get_grid_init()
        else:
            z0 = get_random_init()
            
        # Add small noise to grid init to break symmetry
        if i == 0:
            z0 += np.random.normal(0, 0.01, 3*n)
            
        try:
            res = minimize(objective, z0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8})
            if res.success or res.fun < best_score: # Check if better
                 # Note: res.fun is -sum(s) + penalty. Lower is better.
                 # We want to maximize sum(s).
                 # So we track the best res.fun (lowest cost).
                 if res.fun < best_score:
                     best_score = res.fun
                     best_z = res.x
        except Exception:
            pass

    if best_z is None:
        # Fallback to grid
        best_z = get_grid_init()

    centers, sides = unpack(best_z)
    
    # Post-processing: Ensure strict validity by scaling if necessary
    # Calculate max scale factor k
    k = 1.0
    
    # Boundary checks
    for i in range(n):
        xi, yi, si = centers[i, 0], centers[i, 1], sides[i]
        if si > 0:
            k = min(k, 2*xi/si, 2*(1-xi)/si, 2*yi/si, 2*(1-yi)/si)
            
    # Overlap checks
    for i in range(n):
        for j in range(i+1, n):
            xi, yi, si = centers[i, 0], centers[i, 1], sides[i]
            xj, yj, sj = centers[j, 0], centers[j, 1], sides[j]
            dx = abs(xi - xj)
            dy = abs(yi - yj)
            denom = si + sj
            if denom > 0:
                k = min(k, 2*dx/denom, 2*dy/denom)
                
    # Ensure k is positive and <= 1
    k = max(1e-6, min(1.0, k))
    
    sides = sides * k
    # Centers remain the same
    
    # Final check and return
    # If still invalid (due to floating point), shrink more
    # But k calculation should handle it.
    
    return centers, sides
