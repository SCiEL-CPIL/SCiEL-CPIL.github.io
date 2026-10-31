import numpy as np
import scipy.optimize as opt
import time

N = 26

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_score = 0.0
    best_result = None
    
    # Initial guess: Hexagonal lattice
    # Fit as many as possible in a hex grid, then trim/pad to N
    # For N=26, a 5x5 grid is 25, close enough. Or 6x5=30.
    # Let's use a grid that roughly fits.
    # Side length approx sqrt(N). sqrt(26) ~ 5.1.
    # Let's try a few grid configurations for initialization.
    
    def get_initial_guess(seed):
        rng = np.random.RandomState(seed)
        # Hexagonal lattice parameters
        # Row height = r * sqrt(3)
        # Col width = 2 * r
        # We want to fit 26 circles.
        # Let's estimate radius. Area = 26 * pi * r^2 <= 1. r <= 0.11.
        # But packing density ~ 0.9. 26 * pi * r^2 * 0.9 <= 1 => r ~ 0.1.
        
        # Generate points on a hex grid
        # We will generate a larger grid and pick the best N or just use a grid that fits.
        # A 5x5 grid has 25 points. Add 1? Or use 6x5 and remove 4?
        # Let's use a 5x5 grid centered, and add one in the center or corner?
        # Actually, just use a 5x5 grid of 25 points, and the 26th is a duplicate with jitter?
        # No, better to use a 6x5 grid (30 points) and randomly remove 4, or optimize positions freely.
        
        # Simplest robust init: Random jitter on a grid.
        # Grid 5x5 = 25 points.
        # We need 26.
        # Let's use a 6x5 grid, remove 4 random points, keep 26.
        
        rows = 5
        cols = 6
        points = []
        for i in range(rows):
            for j in range(cols):
                x = (j + 0.5) / cols
                y = (i + 0.5) / rows
                # Hex offset for odd rows
                if i % 2 == 1:
                    x += 0.5 / cols
                # Ensure within [0,1] roughly
                if 0 <= x <= 1 and 0 <= y <= 1:
                    points.append([x, y])
        
        # We have ~30 points. Shuffle and take 26.
        rng.shuffle(points)
        points = points[:N]
        
        # Jitter
        jitter = rng.uniform(-0.02, 0.02, size=(N, 2))
        points = np.array(points) + jitter
        
        # Initial radii
        # Estimate max radius based on min distance
        min_dist = np.inf
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(points[i] - points[j])
                min_dist = min(min_dist, d)
        # Also consider boundary
        for i in range(N):
            d_b = min(points[i][0], 1-points[i][0], points[i][1], 1-points[i][1])
            min_dist = min(min_dist, d_b)
            
        r_init = 0.5 * min_dist * 0.9
        r_init = max(r_init, 0.05) # Safety
        
        radii = np.full(N, r_init)
        
        # Flatten to (x1, y1, r1, x2, y2, r2, ...)
        z = np.zeros(3 * N)
        for i in range(N):
            z[3*i] = points[i][0]
            z[3*i+1] = points[i][1]
            z[3*i+2] = radii[i]
            
        return z

    def unpack(z):
        x = z[0::3]
        y = z[1::3]
        r = z[2::3]
        return x, y, r

    def objective(z):
        _, _, r = unpack(z)
        return -np.sum(r)

    def constraints(z):
        x, y, r = unpack(z)
        cons = []
        # Boundary constraints
        # x_i - r_i >= 0
        cons.append({'type': 'ineq', 'fun': lambda z, x=x, r=r: x - r})
        # 1 - x_i - r_i >= 0
        cons.append({'type': 'ineq', 'fun': lambda z, x=x, r=r: 1 - x - r})
        # y_i - r_i >= 0
        cons.append({'type': 'ineq', 'fun': lambda z, y=y, r=r: y - r})
        # 1 - y_i - r_i >= 0
        cons.append({'type': 'ineq', 'fun': lambda z, y=y, r=r: 1 - y - r})
        
        # Overlap constraints
        # (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
        def overlap_fun(z, i, j):
            xi, yi, ri = z[3*i], z[3*i+1], z[3*i+2]
            xj, yj, rj = z[3*j], z[3*j+1], z[3*j+2]
            return (xi - xj)**2 + (yi - yj)**2 - (ri + rj)**2
        
        for i in range(N):
            for j in range(i+1, N):
                cons.append({'type': 'ineq', 'fun': overlap_fun, 'args': (i, j)})
                
        return cons

    def bounds():
        b = []
        for i in range(N):
            b.append((0, 1)) # x
            b.append((0, 1)) # y
            b.append((0, 0.5)) # r (max possible is 0.5)
        return b

    num_restarts = 0
    max_restarts = 20
    
    while time.time() < deadline and num_restarts < max_restarts:
        seed = num_restarts
        z0 = get_initial_guess(seed)
        
        # Check if initial is valid, if so record
        x, y, r = unpack(z0)
        if is_valid(x, y, r):
            score = np.sum(r)
            if score > best_score:
                best_score = score
                best_result = np.column_stack((x, y, r))
        
        try:
            res = opt.minimize(objective, z0, method='SLSQP', bounds=bounds(), 
                                constraints=constraints(z0), 
                                options={'maxiter': 200, 'ftol': 1e-8})
            
            if res.success and res.fun is not None:
                z_opt = res.x
                x_opt, y_opt, r_opt = unpack(z_opt)
                
                if is_valid(x_opt, y_opt, r_opt):
                    score = np.sum(r_opt)
                    if score > best_score:
                        best_score = score
                        best_result = np.column_stack((x_opt, y_opt, r_opt))
        except Exception as e:
            pass
        
        num_restarts += 1

    if best_result is None:
        # Fallback to initial guess if optimization failed
        z0 = get_initial_guess(0)
        x, y, r = unpack(z0)
        best_result = np.column_stack((x, y, r))

    return best_result

def is_valid(x, y, r):
    # Check boundaries
    if np.any(x < r) or np.any(x > 1 - r):
        return False
    if np.any(y < r) or np.any(y > 1 - r):
        return False
    if np.any(r <= 0):
        return False
        
    # Check overlaps
    for i in range(N):
        for j in range(i+1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist_sq = dx*dx + dy*dy
            r_sum = r[i] + r[j]
            if dist_sq < r_sum * r_sum - 1e-9:
                return False
    return True
