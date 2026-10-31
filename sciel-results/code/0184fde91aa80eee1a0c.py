"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, LinearConstraint

def construct():
    t0 = time.time()
    deadline = t0 + 145.0  # safety margin before 150s kill

    # We'll search over rectangle dimensions. Perimeter = 2*(W+H) = 4 => W+H = 2.
    # Let W = w, H = 2 - w, with w in (0, 2).
    
    best_score = -1
    best_result = None
    best_w = 1.0

    # Strategy: For a given aspect ratio, place circles in a grid-like pattern
    # and then optimize positions and radii to maximize sum of radii subject to
    # non-overlap and boundary constraints.
    
    # We'll try several aspect ratios and grid configurations.
    # For 21 circles, possible grid layouts: 5x5 (25 slots, use 21), 4x6 (24 slots, use 21), etc.
    
    # Let's define a function to evaluate a configuration given w (width), and initial positions.
    
    def evaluate_configuration(w, H, x0, r0):
        """
        Optimize circle positions and radii to maximize sum(r) subject to:
        - Circles stay within rectangle [0,w] x [0,H]
        - No overlap: dist(i,j) >= r_i + r_j
        - r_i > 0
        """
        n = len(x0)
        # Variables: x_1, y_1, r_1, x_2, y_2, r_2, ...
        # Total 3n variables.
        
        def objective(v):
            # v is 3n vector: [x_0, y_0, r_0, x_1, y_1, r_1, ...]
            # Minimize -sum(r_i)
            r = v[2::3]
            return -np.sum(r)
        
        def constraints(v):
            cons = []
            x = v[0::3]
            y = v[1::3]
            r = v[2::3]
            
            # Boundary constraints: r_i <= x_i, r_i <= w - x_i, r_i <= y_i, r_i <= H - y_i
            # And r_i > 0
            for i in range(n):
                # x_i - r_i >= 0
                cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*i] - v[3*i+2]})
                # w - x_i - r_i >= 0
                cons.append({'type': 'ineq', 'fun': lambda v, i=i: w - v[3*i] - v[3*i+2]})
                # y_i - r_i >= 0
                cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*i+1] - v[3*i+2]})
                # H - y_i - r_i >= 0
                cons.append({'type': 'ineq', 'fun': lambda v, i=i: H - v[3*i+1] - v[3*i+2]})
                # r_i >= 0 (we'll handle this with bounds)
            
            # Non-overlap: dist(i,j)^2 >= (r_i + r_j)^2
            # Equivalently: dist(i,j) - (r_i + r_j) >= 0
            # But dist is sqrt, so let's use: (x_i-x_j)^2 + (y_i-y_j)^2 - (r_i+r_j)^2 >= 0
            # This is not convex, but SLSQP might handle it.
            for i in range(n):
                for j in range(i+1, n):
                    cons.append({
                        'type': 'ineq',
                        'fun': lambda v, i=i, j=j: (v[3*i]-v[3*j])**2 + (v[3*i+1]-v[3*j+1])**2 - (v[3*i+2]+v[3*j+2])**2
                    })
            
            return cons
        
        # Bounds: x in [0, w], y in [0, H], r in [0, min(w,H)/2]
        max_r = min(w, H) / 2
        bounds = []
        for i in range(n):
            bounds.append((0, w))      # x_i
            bounds.append((0, H))      # y_i
            bounds.append((1e-6, max_r)) # r_i
        
        # Initial guess
        v0 = np.concatenate([x0, y0, r0])
        
        try:
            res = minimize(objective, v0, method='SLSQP', bounds=bounds,
                          constraints=constraints, options={'maxiter': 200, 'ftol': 1e-8})
            
            if res.success or res.fun < -best_score:
                r_final = res.x[2::3]
                score = np.sum(r_final)
                if score > best_score:
                    best_score = score
                    best_result = res.x
                    best_w = w
            return res
        except Exception:
            return None
    
    # Try different aspect ratios
    w_values = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]
    
    # For each aspect ratio, try a grid initialization
    for w in w_values:
        if time.time() > deadline:
            break
        H = 2.0 - w
        
        # 5x5 grid, use 21 circles
        side = 5
        # Centers of 5x5 grid
        xs = np.linspace(w/(2*side), w - w/(2*side), side)
        ys = np.linspace(H/(2*side), H - H/(2*side), side)
        
        # Take first 21 positions
        x0 = []
        y0 = []
        r0 = []
        count = 0
        for i in range(side):
            for j in range(side):
                if count < N:
                    x0.append(xs[i])
                    y0.append(ys[j])
                    # Initial radius: half the grid spacing, slightly smaller
                    dx = w / side
                    dy = H / side
                    r_init = 0.45 * min(dx, dy)
                    r0.append(r_init)
                    count += 1
        
        x0 = np.array(x0)
        y0 = np.array(y0)
        r0 = np.array(r0)
        
        evaluate_configuration(w, H, x0, r0)
        
        # Also try 4x6 grid
        side_x = 4
        side_y = 6
        xs = np.linspace(w/(2*side_x), w - w/(2*side_x), side_x)
        ys = np.linspace(H/(2*side_y), H - H/(2*side_y), side_y)
        
        x0 = []
        y0 = []
        r0 = []
        count = 0
        for i in range(side_x):
            for j in range(side_y):
                if count < N:
                    x0.append(xs[i])
                    y0.append(ys[j])
                    dx = w / side_x
                    dy = H / side_y
                    r_init = 0.45 * min(dx, dy)
                    r0.append(r_init)
                    count += 1
        
        x0 = np.array(x0)
        y0 = np.array(y0)
        r0 = np.array(r0)
        
        evaluate_configuration(w, H, x0, r0)
    
    # If we found a good result, format it
    if best_result is not None:
        x = best_result[0::3]
        y = best_result[1::3]
        r = best_result[2::3]
        result = np.column_stack([x, y, r])
        return result
    else:
        # Fallback: simple grid
        side = 5
        w = 1.0
        H = 1.0
        xs = np.linspace(w/(2*side), w - w/(2*side), side)
        ys = np.linspace(H/(2*side), H - H/(2*side), side)
        cells = []
        count = 0
        for i in range(side):
            for j in range(side):
                if count < N:
                    cells.append([xs[i], ys[j], 0.5/side - 1e-3])
                    count += 1
        return np.array(cells)

# EVOLVE-BLOCK-END
