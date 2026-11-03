import numpy as np
from scipy.optimize import minimize
import time

N = 21

def construct():
    start_time = time.time()
    deadline = 110.0  # Leave margin for overhead
    
    # 1. Initialization: Hexagonal pattern
    # Generate a hexagonal grid of points
    # We want to initialize 21 circles in a 1x1 square (W=1, H=1)
    # Let's try to pack them densely.
    
    def generate_hex_init(n_circles, box_size=1.0):
        # Generate points in a hexagonal lattice
        # Spacing s
        # We estimate s based on area
        # Area per circle ~ pi * r^2. Hex density ~ 0.9.
        # Box area = 1. N=21.
        # 21 * pi * r^2 * 0.9 ~ 1 => r ~ 0.116
        # Lattice spacing s ~ 2*r ~ 0.23
        # Let's try a few spacings and pick the one that fits best?
        # Or just use a standard grid and let optimizer fix it.
        
        # Simple robust init: Random uniform in [0.1, 0.9]
        # But hex is better.
        
        # Generate a large hex grid
        s = 0.25
        points = []
        for i in range(-5, 10):
            for j in range(-5, 10):
                x = i * s + (j % 2) * s * 0.5
                y = j * s * np.sqrt(3) / 2
                # Center the grid around (0.5, 0.5)
                # We will normalize later or just pick points inside [0,1]
                points.append((x, y))
        
        points = np.array(points)
        # Shift to center
        center = np.array([0.5, 0.5])
        # Scale to fit in 1x1?
        # Let's just pick points closest to center from a grid that covers the square
        # Better: Create a grid that explicitly covers [0,1]x[0,1]
        
        # Let's try a simpler init: 5x5 grid with corners removed (21 circles)
        # This is basically the seed but maybe slightly better arranged?
        # No, let's stick to the hex idea.
        
        # Generate hex points in [0, 1.5] x [0, 1.5]
        s = 0.2
        pts = []
        for i in range(10):
            for j in range(10):
                x = i * s + (j % 2) * s * 0.5
                y = j * s * np.sqrt(3) / 2
                if 0 <= x <= 1 and 0 <= y <= 1:
                    pts.append([x, y])
        
        pts = np.array(pts)
        if len(pts) < N:
            # Fallback to random
            np.random.seed(42)
            pts = np.random.uniform(0.1, 0.9, (N, 2))
        else:
            # Pick N points closest to center
            dists = np.sum((pts - 0.5)**2, axis=1)
            idx = np.argsort(dists)[:N]
            pts = pts[idx]
            
        # Initial radii: small
        r_init = 0.05 * np.ones(N)
        W_init = 1.0
        
        return np.concatenate([pts, r_init.reshape(-1, 1), [[W_init]]]).flatten()

    x0 = generate_hex_init(N)
    
    def objective(params):
        # params: x1, y1, r1, ..., xN, yN, rN, W
        W = params[-1]
        H = 2.0 - W
        if W <= 0 or H <= 0:
            return 1e10
        
        coords = params[:-1].reshape(N, 3)
        x = coords[:, 0]
        y = coords[:, 1]
        r = coords[:, 2]
        
        # Sum of radii
        sum_r = np.sum(r)
        
        # Penalties
        penalty = 0.0
        
        # Wall constraints
        # x >= r, x <= W - r
        # y >= r, y <= H - r
        # Using squared violations
        wall_viol = np.maximum(0, r - x) + np.maximum(0, x - (W - r))
        wall_viol += np.maximum(0, r - y) + np.maximum(0, y - (H - r))
        penalty += np.sum(wall_viol**2)
        
        # Circle-circle constraints
        # dist >= ri + rj
        # dist^2 >= (ri+rj)^2
        # We use dist - (ri+rj) to avoid sqrt in gradient if possible, but penalty is non-smooth.
        # Let's use squared distance difference?
        # max(0, (ri+rj)^2 - dist^2)
        
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx*dx + dy*dy
                min_dist_sq = (r[i] + r[j])**2
                viol = min_dist_sq - dist_sq
                if viol > 0:
                    penalty += viol**2
        
        # Aspect ratio penalty? No, W is optimized.
        # Keep W in [0.5, 1.5]
        if W < 0.5 or W > 1.5:
            penalty += (W - 1.0)**2 * 1000
        
        return -sum_r + 10.0 * penalty

    # Optimize
    best_x = x0
    best_val = objective(x0)
    
    # Use L-BFGS-B
    # Bounds
    # x, y in [0, 2], r in [0, 1], W in [0.5, 1.5]
    bounds = []
    for i in range(N):
        bounds.append((0.0, 2.0)) # x
        bounds.append((0.0, 2.0)) # y
        bounds.append((0.0, 1.0)) # r
    bounds.append((0.5, 1.5)) # W
    
    try:
        res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
        
        if res.fun < best_val:
            best_val = res.fun
            best_x = res.x
            
        # Polish with higher precision if time permits
        if time.time() < deadline - 30:
            # Run again with tighter tolerance
            res2 = minimize(objective, best_x, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10})
            if res2.fun < best_val:
                best_val = res2.fun
                best_x = res2.x
                
    except Exception as e:
        pass # Keep best_x from init or first run

    # Extract result
    W = best_x[-1]
    H = 2.0 - W
    coords = best_x[:-1].reshape(N, 3)
    
    # Ensure validity: clip to box, ensure no overlap?
    # The penalty method might leave small violations.
    # We must return a VALID configuration.
    # If there are violations, we need to shrink radii or move circles.
    # Simple fix: shrink all radii by a small factor to ensure disjointness?
    # Or check and adjust.
    
    # Let's verify and adjust
    x = coords[:, 0]
    y = coords[:, 1]
    r = coords[:, 2]
    
    # Check wall constraints
    for i in range(N):
        if x[i] < r[i]:
            x[i] = r[i]
        if x[i] > W - r[i]:
            x[i] = W - r[i]
        if y[i] < r[i]:
            y[i] = r[i]
        if y[i] > H - r[i]:
            y[i] = H - r[i]
            
    # Check circle-circle constraints
    # If overlapping, shrink the smaller radius?
    # This is tricky. A robust way is to reduce all radii by a small epsilon if any overlap exists.
    # Or, we can do a final "repair" pass.
    
    valid = True
    # Quick check
    for i in range(N):
        if x[i] < r[i] - 1e-9 or x[i] > W - r[i] + 1e-9 or y[i] < r[i] - 1e-9 or y[i] > H - r[i] + 1e-9:
            valid = False
            break
    if valid:
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                if dx*dx + dy*dy < (r[i] + r[j])**2:
                    valid = False
                    break
            if not valid: break
            
    if not valid:
        # Fallback: Shrink radii
        # Find max overlap
        max_overlap = 0
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                req = r[i] + r[j]
                if dist < req:
                    overlap = (req - dist) / 2.0
                    max_overlap = max(max_overlap, overlap)
        # Shrink all radii by max_overlap + small margin
        shrink = max_overlap + 1e-4
        r = r - shrink
        # Also check walls
        for i in range(N):
            if r[i] > x[i]: r[i] = x[i] * 0.9
            if r[i] > W - x[i]: r[i] = (W - x[i]) * 0.9
            if r[i] > y[i]: r[i] = y[i] * 0.9
            if r[i] > H - y[i]: r[i] = (H - y[i]) * 0.9
            if r[i] < 0: r[i] = 0.01
            
    result = np.column_stack((x, y, r))
    return result

# EVOLVE-BLOCK-END
