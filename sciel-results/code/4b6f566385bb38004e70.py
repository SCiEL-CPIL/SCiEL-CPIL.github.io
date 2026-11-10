import numpy as np
import time
from scipy.optimize import minimize

def _overlap_penalty(centers, sides, tol=1e-9):
    """
    Compute a smooth-ish penalty for overlaps and boundary violations.
    """
    n = len(sides)
    penalty = 0.0
    # Boundary penalties
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        # Distance to boundaries
        dx_min = cx - s/2
        dx_max = 1.0 - (cx + s/2)
        dy_min = cy - s/2
        dy_max = 1.0 - (cy + s/2)
        
        # Penalty if outside [0, 1]
        if dx_min < -tol:
            penalty += (dx_min + tol) ** 2
        if dx_max < -tol:
            penalty += (dx_max + tol) ** 2
        if dy_min < -tol:
            penalty += (dy_min + tol) ** 2
        if dy_max < -tol:
            penalty += (dy_max + tol) ** 2

    # Overlap penalties
    for i in range(n):
        for j in range(i + 1, n):
            cx_i, cy_i = centers[i]
            cx_j, cy_j = centers[j]
            s_i = sides[i]
            s_j = sides[j]
            
            # Required separation in x and y
            sep_x = abs(cx_i - cx_j) - (s_i + s_j) / 2.0
            sep_y = abs(cy_i - cy_j) - (s_i + s_j) / 2.0
            
            # If sep_x < 0, they overlap in x-projection
            # If sep_y < 0, they overlap in y-projection
            # Overlap area is proportional to max(0, -sep_x) * max(0, -sep_y)
            # But we want to penalize even if they are close.
            # Use a soft max or just square of negative separation.
            
            pen_x = max(0.0, -sep_x)
            pen_y = max(0.0, -sep_y)
            
            # The product is 0 if no overlap.
            # To make it differentiable-ish for Nelder-Mead (which doesn't need gradients),
            # the sharp corners at 0 are okay.
            penalty += pen_x * pen_y
            
            # Also add a linear penalty to push them apart more aggressively
            # if they are very close but not overlapping?
            # No, just area is sufficient for "hard" constraints usually.
            
    return penalty

def _objective(params, tol=1e-7):
    """
    Objective function for minimization.
    We want to maximize sum(sides), so we minimize -sum(sides) + penalty.
    """
    n = 10
    centers = params[:2*n].reshape(n, 2)
    sides = params[2*n:]
    
    # Ensure sides are positive
    sides = np.maximum(sides, 1e-6)
    
    # Clip centers to [0,1] roughly to help optimizer?
    # Better to let penalty handle it.
    
    penalty = _overlap_penalty(centers, sides, tol)
    sum_sides = np.sum(sides)
    
    # Large weight for penalty to ensure feasibility
    weight = 100.0
    return -sum_sides + weight * penalty

def _is_valid(centers, sides, tol=1e-7):
    """Check all constraints."""
    n = len(sides)
    if n != 10:
        return False
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
        return False
    if np.any(sides <= 0):
        return False
    
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        if cx - s/2 < -tol or cx + s/2 > 1.0 + tol:
            return False
        if cy - s/2 < -tol or cy + s/2 > 1.0 + tol:
            return False
            
    for i in range(n):
        for j in range(i+1, n):
            t = (sides[i] + sides[j]) / 2.0 - tol
            dx = abs(centers[i,0] - centers[j,0])
            dy = abs(centers[i,1] - centers[j,1])
            if dx < t and dy < t:
                return False
    return True

def _try_optimize(init_params, time_limit=5.0):
    """
    Run Nelder-Mead optimization from initial params.
    """
    start_time = time.time()
    options = {
        'maxiter': 500,
        'xatol': 1e-6,
        'fatol': 1e-6,
        'adaptive': True
    }
    
    try:
        res = minimize(_objective, init_params, method='Nelder-Mead', options=options)
        
        # Check time
        if time.time() - start_time > time_limit:
            return None
            
        opt_params = res.x
        centers = opt_params[:20].reshape(10, 2)
        sides = opt_params[20:]
        
        # Post-processing: clamp and slight shrink to ensure validity
        centers, sides = _repair(centers, sides)
        
        if _is_valid(centers, sides):
            return centers, sides
        else:
            return None
            
    except Exception:
        return None

def _repair(centers, sides):
    """
    Attempt to make the solution valid by clamping and shrinking.
    """
    centers = centers.copy()
    sides = sides.copy()
    
    # Clamp centers to valid range given sides
    for i in range(10):
        half = sides[i] / 2.0
        centers[i, 0] = np.clip(centers[i, 0], half, 1.0 - half)
        centers[i, 1] = np.clip(centers[i, 1], half, 1.0 - half)
        
    # If still invalid, shrink sides slightly
    if not _is_valid(centers, sides):
        for i in range(10):
            sides[i] *= 0.99
        centers, sides = _repair(centers, sides)
        
    return centers, sides

def _generate_initials():
    """
    Generate a set of diverse initial configurations.
    """
    initials = []
    
    # 1. 5x2 Grid
    centers = np.zeros((10, 2))
    sides = np.zeros(10)
    s = 0.2
    for i in range(10):
        row = i // 5
        col = i % 5
        centers[i] = [(col + 0.5) * s, (row + 0.5) * s]
        sides[i] = s
    initials.append(np.hstack([centers.flatten(), sides]))
    
    # 2. 3x3 Grid + 1 small in corner (rough)
    centers = np.zeros((10, 2))
    sides = np.zeros(10)
    s = 1.0/3.0
    for i in range(9):
        row = i // 3
        col = i % 3
        centers[i] = [(col + 0.5) * s, (row + 0.5) * s]
        sides[i] = s
    # 10th square in the gap? No, 3x3 fills the square.
    # So we need to shrink the 3x3 grid to fit a 10th?
    # Let's just use a 4x3 grid with one missing?
    # Or 2x5 grid is better.
    # Let's try a random valid packing generation.
    
    # 3. Random Packings
    rng = np.random.default_rng(42)
    for k in range(20):
        centers = np.zeros((10, 2))
        sides = np.zeros(10)
        # Place squares one by one
        for i in range(10):
            # Try to place a square of random size
            placed = False
            for _ in range(100):
                s = rng.uniform(0.1, 0.4)
                cx = rng.uniform(s/2, 1.0 - s/2)
                cy = rng.uniform(s/2, 1.0 - s/2)
                # Check if valid against placed squares
                valid = True
                for j in range(i):
                    t = (s + sides[j]) / 2.0
                    dx = abs(cx - centers[j, 0])
                    dy = abs(cy - centers[j, 1])
                    if dx < t and dy < t:
                        valid = False
                        break
                if valid:
                    centers[i] = [cx, cy]
                    sides[i] = s
                    placed = True
                    break
            if not placed:
                # If couldn't place, put a tiny one in a corner?
                # Just break and use what we have, padding with tiny squares
                centers[i] = [0.01, 0.01]
                sides[i] = 0.01
        initials.append(np.hstack([centers.flatten(), sides]))
        
    # 4. Known Best Packing Approximation
    # The best packing for 10 squares has a specific structure.
    # One large square in a corner, others filling the rest.
    # Let's try: 1 square of 0.5, 4 of 0.25, 5 of 0.15?
    # We'll let the optimizer figure it out from a "clustered" start.
    centers = np.zeros((10, 2))
    sides = np.zeros(10)
    sides[0] = 0.5
    centers[0] = [0.25, 0.25]
    for i in range(1, 10):
        sides[i] = 0.2
        # Place in remaining space roughly
        row = (i-1) // 4
        col = (i-1) % 4
        # Map to top/right region
        x = 0.6 + col * 0.1
        y = 0.6 + row * 0.1
        centers[i] = [x, y]
    initials.append(np.hstack([centers.flatten(), sides]))
    
    return initials

def squares_in_square10() -> tuple:
    start_time = time.time()
    deadline = start_time + 580.0
    
    initials = _generate_initials()
    
    best_centers = None
    best_sides = None
    best_obj = -1.0
    
    for init in initials:
        if time.time() > deadline:
            break
            
        # Run optimization
        result = _try_optimize(init, time_limit=10.0)
        
        if result is not None:
            centers, sides = result
            obj = np.sum(sides)
            if obj > best_obj:
                best_obj = obj
                best_centers = centers.copy()
                best_sides = sides.copy()
                
        # Also try a few random restarts from the best found so far?
        # If we have time, perturb the best solution and re-optimize.
        
    if best_centers is None:
        # Fallback to grid
        centers = np.zeros((10, 2))
        sides = np.zeros(10)
        s = 0.2
        for i in range(10):
            row = i // 5
            col = i % 5
            centers[i] = [(col + 0.5) * s, (row + 0.5) * s]
            sides[i] = s
        return centers, sides
        
    return best_centers, best_sides
