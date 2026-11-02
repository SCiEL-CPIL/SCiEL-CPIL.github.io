import numpy as np
import time
from scipy.optimize import minimize

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin for 150s limit
    
    best_val = -np.inf
    best_x = None
    
    # Initial guess strategy: Hexagonal packing
    # We try a few different aspect ratios for initialization
    aspect_ratios = [1.0, 1.2, 0.8, 1.5, 0.66]
    
    for ar in aspect_ratios:
        if time.time() > deadline:
            break
            
        # Generate hexagonal grid points
        # Approximate packing in a rectangle
        # Let's just use a distorted grid or simple hex logic
        # For N=21, a 5x4 grid with one removed or 6x4 with 3 removed works.
        # Let's try a 5x4 grid (20 circles) + 1 in the middle? Or 6x4 (24) remove 3.
        # Simpler: Use a grid that fits N.
        # Side ~ sqrt(21) ~ 4.58. So 5x5 grid is 25. Remove 4.
        # Or 6x4 = 24. Remove 3.
        
        # Let's use a 6x4 grid, removing 3 corners to make it 21?
        # Or just use a 5x5 grid and remove 4.
        # Let's try to generate a hex-like layout manually or via a simple loop.
        
        # Actually, a good generic init is a grid.
        # Let's create a 5x5 grid, remove 4 corners.
        rows, cols = 5, 5
        points = []
        for i in range(rows):
            for j in range(cols):
                # Skip 4 corners
                if (i==0 and j==0) or (i==0 and j==cols-1) or (i==rows-1 and j==0) or (i==rows-1 and j==cols-1):
                    continue
                points.append((j, i))
        
        # If we have more/less, adjust. 25-4=21. Perfect.
        
        # Scale to fit aspect ratio 'ar'
        # Let width = ar, height = 1 (relative). Then scale to perimeter 4.
        # w_rel = ar, h_rel = 1.
        # w + h = 2.
        # w = 2 * ar / (ar + 1)
        # h = 2 / (ar + 1)
        
        w_init = 2.0 * ar / (ar + 1.0)
        h_init = 2.0 / (ar + 1.0)
        
        # Map grid indices (0..4, 0..4) to [0, w_init] x [0, h_init]
        # Add margin
        margin = 0.1
        x_coords = []
        y_coords = []
        r_init = min(w_init, h_init) / max(rows, cols) * 0.4 # rough radius
        
        for (j, i) in points:
            # Center of cell
            x = (j + 0.5) * (w_init / cols)
            y = (i + 0.5) * (h_init / rows)
            x_coords.append(x)
            y_coords.append(y)
            
        x_coords = np.array(x_coords)
        y_coords = np.array(y_coords)
        
        # Initial radii
        # Estimate max radius based on grid spacing
        dx = w_init / cols
        dy = h_init / rows
        r_init = min(dx, dy) / 2.0 * 0.9
        
        # Variable vector: x1, y1, r1, ..., x21, y21, r21, w
        x0 = np.zeros(N * 3 + 1)
        for k in range(N):
            x0[3*k] = x_coords[k]
            x0[3*k+1] = y_coords[k]
            x0[3*k+2] = r_init
        x0[-1] = w_init
        
        # Define objective and constraints for SLSQP
        def objective(v):
            # v: [x1, y1, r1, ..., xN, yN, rN, w]
            rs = v[2:3*N:3]
            return -np.sum(rs)
            
        def constraints(v):
            cons = []
            xs = v[0:3*N:3]
            ys = v[1:3*N:3]
            rs = v[2:3*N:3]
            w = v[-1]
            h = 2.0 - w
            
            # Boundary constraints:
            # x_i >= r_i, x_i <= w - r_i
            # y_i >= r_i, y_i <= h - r_i
            # r_i >= 0 (handled by bounds)
            # w > 0, h > 0 (handled by bounds on w)
            
            # Distance constraints:
            # (xi - xj)^2 + (yi - yj)^2 >= (ri + rj)^2
            for i in range(N):
                for j in range(i+1, N):
                    dx = xs[i] - xs[j]
                    dy = ys[i] - ys[j]
                    dist_sq = dx*dx + dy*dy
                    rad_sum = rs[i] + rs[j]
                    # dist_sq - rad_sum^2 >= 0
                    cons.append({'type': 'ineq', 'fun': lambda v, i=i, j=j: (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 - (v[3*i+2] + v[3*j+2])**2})
            
            # Box constraints for x, y relative to r and w, h
            # x_i - r_i >= 0
            # w - x_i - r_i >= 0
            # y_i - r_i >= 0
            # h - y_i - r_i >= 0
            
            for i in range(N):
                # x >= r
                cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*i] - v[3*i+2]})
                # x <= w - r  =>  w - x - r >= 0
                cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[-1] - v[3*i] - v[3*i+2]})
                # y >= r
                cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*i+1] - v[3*i+2]})
                # y <= h - r  =>  (2-w) - y - r >= 0
                cons.append({'type': 'ineq', 'fun': lambda v, i=i: 2.0 - v[-1] - v[3*i+1] - v[3*i+2]})
                
            return cons

        bounds = []
        for k in range(N):
            bounds.append((0.0, 2.0)) # x
            bounds.append((0.0, 2.0)) # y
            bounds.append((1e-6, 1.0)) # r
        bounds.append((0.1, 1.9)) # w (must be > 0 and < 2)
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints, 
                           options={'maxiter': 500, 'ftol': 1e-9, 'disp': False})
            
            if res.success:
                val = -res.fun
                if val > best_val:
                    best_val = val
                    best_x = res.x
                    
        except Exception:
            pass
            
        if time.time() > deadline:
            break
            
    # If no result found, fallback to a very simple grid
    if best_x is None:
        side = 5
        w = 1.0
        h = 1.0
        cells = []
        for i in range(side):
            for j in range(side):
                if (i==0 and j==0) or (i==0 and j==4) or (i==4 and j==0) or (i==4 and j==4):
                    continue
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 0.01
                cells.append([x, y, r])
        return np.array(cells)
        
    # Extract result
    xs = best_x[0:3*N:3]
    ys = best_x[1:3*N:3]
    rs = best_x[2:3*N:3]
    
    # Clip to ensure validity
    w = best_x[-1]
    h = 2.0 - w
    
    # Ensure radii are small enough to fit
    # If any constraint is violated due to numerical error, scale down radii
    min_dist = np.inf
    for i in range(N):
        for j in range(i+1, N):
            d = np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)
            min_dist = min(min_dist, d)
            
    # Check bounds
    max_r_allowed = min(w, h) / 2.0
    for i in range(N):
        # Check if circle fits in box
        max_r_box = min(xs[i], w-xs[i], ys[i], h-ys[i])
        if rs[i] > max_r_box:
            rs[i] = max_r_box * 0.99
            
    # Check pairwise
    for i in range(N):
        for j in range(i+1, N):
            d = np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)
            max_r_pair = d / 2.0
            # We need ri + rj <= d
            # If ri + rj > d, scale both down? Or just scale the larger one?
            # Simple fix: scale all radii by a factor if necessary
            pass
            
    # Robust fix: if any overlap, scale all radii down slightly
    # Calculate max violation
    max_viol = 0
    for i in range(N):
        for j in range(i+1, N):
            d = np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)
            s = rs[i] + rs[j]
            if s > d:
                viol = (s - d) / d
                max_viol = max(max_viol, viol)
                
    if max_viol > 0:
        scale = 1.0 / (1.0 + max_viol)
        rs *= scale
        
    # Final check and construction
    result = np.zeros((N, 3))
    for i in range(N):
        result[i] = [xs[i], ys[i], rs[i]]
        
    return result
