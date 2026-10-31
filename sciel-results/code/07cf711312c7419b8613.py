import numpy as np
import time
from scipy.optimize import minimize
import sys

N = 21

def _is_valid(circles):
    """
    Check if the circles are valid.
    circles: (n, 3) array [cx, cy, r]
    Returns (is_valid, score)
    """
    if circles is None or len(circles) != N:
        return False, 0.0
    
    cx = circles[:, 0]
    cy = circles[:, 1]
    r = circles[:, 2]
    
    if np.any(r <= 0):
        return False, 0.0
        
    # Check bounding box
    left = np.min(cx - r)
    right = np.max(cx + r)
    bottom = np.min(cy - r)
    top = np.max(cy + r)
    
    w = right - left
    h = top - bottom
    
    if w + h > 2.0 + 1e-6:
        return False, 0.0
        
    # Check overlaps
    for i in range(N):
        for j in range(i + 1, N):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            dist_sq = dx * dx + dy * dy
            rad_sum = r[i] + r[j]
            if dist_sq < rad_sum * rad_sum - 1e-6:
                return False, 0.0
                
    score = float(np.sum(r))
    return True, score

def _objective(x):
    """
    Objective function to minimize.
    x: flattened array of [x1, y1, r1, ..., xN, yN, rN, w, h]
    """
    # Extract radii
    # Indices: i*3 + 2 are radii
    r = x[2::3]
    # We minimize negative sum of radii
    return -np.sum(r)

def _constraints_factory():
    """
    Generate the constraint functions.
    x: flattened array of [x1, y1, r1, ..., xN, yN, rN, w, h]
    """
    constraints = []
    
    # 1. Non-overlap constraints
    # For each pair i < j
    for i in range(N):
        for j in range(i + 1, N):
            # Indices
            ix, iy, ir = i*3, i*3+1, i*3+2
            jx, jy, jr = j*3, j*3+1, j*3+2
            
            def non_overlap(x, ix=ix, iy=iy, ir=ir, jx=jx, jy=jy, jr=jr):
                xi = x[ix]
                yi = x[iy]
                ri = x[ir]
                xj = x[jx]
                yj = x[jy]
                rj = x[jr]
                # (xi - xj)^2 + (yi - yj)^2 - (ri + rj)^2 >= 0
                return (xi - xj)**2 + (yi - yj)**2 - (ri + rj)**2
            
            constraints.append({'type': 'ineq', 'fun': non_overlap})
            
    # 2. Bounding box constraints
    # w = x[-2], h = x[-1]
    
    # x_i - r_i >= 0
    for i in range(N):
        ix, ir = i*3, i*3+2
        def left_bound(x, ix=ix, ir=ir):
            return x[ix] - x[ir]
        constraints.append({'type': 'ineq', 'fun': left_bound})
        
    # x_i + r_i <= w  =>  w - x_i - r_i >= 0
    for i in range(N):
        ix, ir = i*3, i*3+2
        def right_bound(x, ix=ix, ir=ir):
            return x[-2] - x[ix] - x[ir]
        constraints.append({'type': 'ineq', 'fun': right_bound})
        
    # y_i - r_i >= 0
    for i in range(N):
        iy, ir = i*3+1, i*3+2
        def bottom_bound(x, iy=iy, ir=ir):
            return x[iy] - x[ir]
        constraints.append({'type': 'ineq', 'fun': bottom_bound})
        
    # y_i + r_i <= h  =>  h - y_i - r_i >= 0
    for i in range(N):
        iy, ir = i*3+1, i*3+2
        def top_bound(x, iy=iy, ir=ir):
            return x[-1] - x[iy] - x[ir]
        constraints.append({'type': 'ineq', 'fun': top_bound})
        
    # 3. Perimeter constraint: w + h <= 2  =>  2 - w - h >= 0
    def perimeter_constraint(x):
        return 2.0 - x[-2] - x[-1]
    constraints.append({'type': 'ineq', 'fun': perimeter_constraint})
    
    # 4. w, h > 0 (handled by bounds, but good to ensure)
    
    return constraints

CONSTRAINTS = _constraints_factory()

def _initial_guess(w, h, seed=42):
    """
    Generate an initial guess for a given rectangle size w x h.
    """
    rng = np.random.default_rng(seed)
    x = np.zeros(N * 3 + 2)
    
    # Grid placement
    # Try to fit N circles in w x h
    # Simple grid: rows x cols
    # Optimize grid dimensions roughly based on aspect ratio
    aspect = w / h
    if aspect > 1:
        cols = int(np.sqrt(N * aspect))
        rows = int(np.ceil(N / cols))
    else:
        rows = int(np.sqrt(N / aspect))
        cols = int(np.ceil(N / rows))
    
    # Adjust to ensure rows * cols >= N
    while rows * cols < N:
        if w >= h:
            cols += 1
        else:
            rows += 1
            
    # Place circles
    dx = w / cols
    dy = h / rows
    
    idx = 0
    for r_idx in range(rows):
        for c_idx in range(cols):
            if idx >= N:
                break
            cx = (c_idx + 0.5) * dx
            cy = (r_idx + 0.5) * dy
            # Initial radius: half of min(dx, dy) with some margin
            r = 0.45 * min(dx, dy)
            
            x[idx*3] = cx
            x[idx*3+1] = cy
            x[idx*3+2] = r
            idx += 1
            
    # Fill remaining if any (should not happen if grid sized correctly)
    while idx < N:
        x[idx*3] = rng.uniform(0, w)
        x[idx*3+1] = rng.uniform(0, h)
        x[idx*3+2] = 0.05
        idx += 1
        
    x[-2] = w
    x[-1] = h
    
    return x

def construct():
    deadline = time.time() + 110.0  # Safety margin
    
    best_x = None
    best_score = 0.0
    
    # Define bounds for variables
    # x_i, y_i: [0, 2]
    # r_i: [0.001, 1.0]
    # w, h: [0.1, 1.9] (since w+h<=2)
    bounds = []
    for i in range(N):
        bounds.append((0.0, 2.0)) # x
        bounds.append((0.0, 2.0)) # y
        bounds.append((0.001, 1.0)) # r
    bounds.append((0.1, 1.9)) # w
    bounds.append((0.1, 1.9)) # h
    
    # List of initial aspect ratios to try
    # Square, slightly wide, slightly tall, more wide, more tall
    aspect_ratios = [1.0, 1.2, 1.5, 2.0, 0.8, 0.6, 0.5]
    
    seeds = [42, 123, 999, 2024, 555, 777]
    
    attempts = 0
    max_attempts = 50 # Cap to avoid infinite loop if time check fails
    
    while time.time() < deadline and attempts < max_attempts:
        attempts += 1
        
        # Pick an aspect ratio
        ar = aspect_ratios[attempts % len(aspect_ratios)]
        # Pick a seed
        seed = seeds[attempts % len(seeds)]
        
        # Calculate w, h based on ar and perimeter 4 (w+h=2)
        # w = ar * h
        # h * (ar + 1) = 2
        # h = 2 / (ar + 1)
        # w = 2 * ar / (ar + 1)
        h = 2.0 / (ar + 1.0)
        w = 2.0 * ar / (ar + 1.0)
        
        # Add some random perturbation to w, h to explore nearby
        if attempts > 6:
            noise = np.random.default_rng(seed + attempts).normal(0, 0.05, 2)
            w += noise[0]
            h += noise[1]
            # Ensure w+h <= 2
            if w + h > 2.0:
                scale = 2.0 / (w + h)
                w *= scale
                h *= scale
            # Ensure positive
            w = max(0.1, w)
            h = max(0.1, h)
        
        x0 = _initial_guess(w, h, seed=seed)
        
        # Run SLSQP
        try:
            res = minimize(
                _objective, 
                x0, 
                method='SLSQP', 
                bounds=bounds, 
                constraints=CONSTRAINTS,
                options={'maxiter': 200, 'ftol': 1e-8}
            )
        except Exception as e:
            continue
            
        if res.success:
            x_opt = res.x
            # Reconstruct circles array
            circles = np.zeros((N, 3))
            circles[:, 0] = x_opt[0::3]
            circles[:, 1] = x_opt[1::3]
            circles[:, 2] = x_opt[2::3]
            
            is_valid, score = _is_valid(circles)
            if is_valid and score > best_score:
                best_score = score
                best_x = x_opt.copy()
                
        # If we have a good solution, we can try to polish it with tighter tolerance
        if best_score > 0.8 and attempts % 10 == 0:
             # Use the best found so far as a start point
             if best_x is not None:
                 x0 = best_x.copy()
                 # Add tiny noise to escape local optima
                 noise = np.random.default_rng(attempts).normal(0, 0.001, x0.shape)
                 x0 += noise
                 # Ensure bounds
                 x0[0::3] = np.clip(x0[0::3], 0.0, 2.0)
                 x0[1::3] = np.clip(x0[1::3], 0.0, 2.0)
                 x0[2::3] = np.clip(x0[2::3], 0.001, 1.0)
                 x0[-2] = np.clip(x0[-2], 0.1, 1.9)
                 x0[-1] = np.clip(x0[-1], 0.1, 1.9)
                 
                 try:
                     res = minimize(
                         _objective, 
                         x0, 
                         method='SLSQP', 
                         bounds=bounds, 
                         constraints=CONSTRAINTS,
                         options={'maxiter': 500, 'ftol': 1e-10}
                     )
                     if res.success:
                         x_opt = res.x
                         circles = np.zeros((N, 3))
                         circles[:, 0] = x_opt[0::3]
                         circles[:, 1] = x_opt[1::3]
                         circles[:, 2] = x_opt[2::3]
                         
                         is_valid, score = _is_valid(circles)
                         if is_valid and score > best_score:
                             best_score = score
                             best_x = x_opt.copy()
                 except Exception:
                     pass

    # Final result
    if best_x is not None:
        circles = np.zeros((N, 3))
        circles[:, 0] = best_x[0::3]
        circles[:, 1] = best_x[1::3]
        circles[:, 2] = best_x[2::3]
        
        # Final validation
        is_valid, score = _is_valid(circles)
        if is_valid:
            return circles
            
    # Fallback: Simple grid
    # If optimization failed completely, return a valid grid
    w, h = 1.0, 1.0
    x0 = _initial_guess(w, h, seed=42)
    circles = np.zeros((N, 3))
    circles[:, 0] = x0[0::3]
    circles[:, 1] = x0[1::3]
    circles[:, 2] = x0[2::3]
    
    # Scale down radii to ensure validity if needed
    for _ in range(10):
        is_valid, score = _is_valid(circles)
        if is_valid:
            break
        circles[:, 2] *= 0.9
        
    return circles

