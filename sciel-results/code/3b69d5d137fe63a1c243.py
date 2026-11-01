import numpy as np
import scipy.optimize as opt
import time

N = 21

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    # Variables: [w, x1, y1, r1, ..., x21, y21, r21]
    # w is the width, h = 2 - w
    # Bounds for w: [0.5, 1.5] to keep aspect ratio reasonable (avoid degenerate rectangles)
    # Actually, w can range from ~0.5 to 1.5. If w is very small, h is large, but circles can't be too big.
    # Let's bound w between 0.4 and 1.6.
    
    def unpack(params):
        w = params[0]
        h = 2.0 - w
        # params[1:] are groups of (x, y, r)
        centers = params[1:].reshape(N, 3)
        return w, h, centers

    def objective(params):
        w, h, centers = unpack(params)
        xs = centers[:, 0]
        ys = centers[:, 1]
        rs = centers[:, 2]
        
        # Sum of radii to maximize (so we minimize negative)
        obj = -np.sum(rs)
        
        # Boundary penalties
        # Ensure circles are inside [0, w] x [0, h]
        # x_i >= r_i, x_i <= w - r_i
        # y_i >= r_i, y_i <= h - r_i
        pen_boundary = 0.0
        for i in range(N):
            if xs[i] < rs[i]:
                pen_boundary += (rs[i] - xs[i])**2
            if xs[i] > w - rs[i]:
                pen_boundary += (xs[i] - (w - rs[i]))**2
            if ys[i] < rs[i]:
                pen_boundary += (rs[i] - ys[i])**2
            if ys[i] > h - rs[i]:
                pen_boundary += (ys[i] - (h - rs[i]))**2
        obj += 100.0 * pen_boundary
        
        # Overlap penalties
        pen_overlap = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dist = np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)
                min_dist = rs[i] + rs[j]
                if dist < min_dist:
                    pen_overlap += (min_dist - dist)**2
        obj += 100.0 * pen_overlap
        
        return obj

    # Initial guess: Grid layout
    # Let's start with a square aspect ratio w=1.0, h=1.0
    # Arrange 21 circles in a 5x5 grid, taking the first 21? Or 4x6?
    # 5x5 = 25. 4x6 = 24.
    # Let's try a 5x5 grid and remove 4 corners or something, or just use a dense random start.
    # A better start: Sort by x then y in a grid.
    
    # Let's use a 5x5 grid for initialization, w=1.0, h=1.0
    # Spacing 0.2. Radius 0.09.
    # This is a valid starting point.
    
    init_params = np.zeros(1 + N*3)
    init_params[0] = 1.0 # w
    h_init = 1.0
    # 5x5 grid
    k = 0
    for i in range(5):
        for j in range(5):
            if k >= N:
                break
            x = (i + 0.5) * (1.0 / 5.0)
            y = (j + 0.5) * (1.0 / 5.0)
            r = 0.19 # Slightly less than 0.2 to allow movement
            init_params[1 + k*3] = x
            init_params[1 + k*3 + 1] = y
            init_params[1 + k*3 + 2] = r
            k += 1
        if k >= N:
            break
    
    # Bounds
    # w in [0.4, 1.6]
    # x in [0, 2] (max dimension)
    # y in [0, 2]
    # r in [0, 1]
    bounds = [(0.4, 1.6)]
    for i in range(N):
        bounds.append((0.0, 2.0)) # x
        bounds.append((0.0, 2.0)) # y
        bounds.append((0.0, 1.0)) # r

    # Run Differential Evolution
    # popsize=15, maxiter=50 might be fast enough for N=21
    # We need to be careful with time.
    
    best_result = None
    best_val = -np.inf
    
    def check_time():
        return time.time() - start_time > deadline

    try:
        # DE is stochastic, run it
        res = opt.differential_evolution(
            objective, 
            bounds, 
            maxiter=30, 
            popsize=10, 
            tol=0.01, 
            seed=42,
            polish=False # We will polish manually
        )
        
        if check_time():
            return np.array([]) # Should not happen yet
            
        best_params = res.x
        best_val = -res.fun
        
        # Local Polish with L-BFGS-B
        # We need to be careful that the penalty method allows L-BFGS-B to converge
        # to a local minimum of the penalized function.
        
        res_local = opt.minimize(
            objective, 
            best_params, 
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-10}
        )
        
        if check_time():
            # If we ran out of time, return the DE result if it's better
            pass
        else:
            if -res_local.fun > best_val:
                best_val = -res_local.fun
                best_params = res_local.x

    except Exception as e:
        # If optimization fails, fall back to initial guess or a simple grid
        pass

    # Convert best_params to output format
    # Output: (N, 3) array of [x, y, r]
    # We need to ensure the final result is valid (non-overlapping, inside).
    # The penalty might leave small overlaps. We should project or shrink radii slightly if invalid.
    
    w, h, centers = unpack(best_params)
    xs = centers[:, 0]
    ys = centers[:, 1]
    rs = centers[:, 2]
    
    # Validation and Correction
    # 1. Ensure inside bounds
    for i in range(N):
        # Clamp x, y
        xs[i] = np.clip(xs[i], rs[i], w - rs[i])
        ys[i] = np.clip(ys[i], rs[i], h - rs[i])
        
    # 2. Check for overlaps and shrink radii if necessary
    # This is a heuristic correction.
    # If overlaps exist, we can reduce all radii by a small factor.
    # Or we can just return the result as is if the penalty was high enough to prevent large overlaps.
    # Given the penalty weight of 100, overlaps should be minimal.
    
    # However, to be safe against the evaluator's strict "disjoint interiors" check,
    # we should verify and potentially shrink.
    
    valid = True
    for i in range(N):
        if xs[i] < rs[i] - 1e-9 or xs[i] > w - rs[i] + 1e-9:
            valid = False
        if ys[i] < rs[i] - 1e-9 or ys[i] > h - rs[i] + 1e-9:
            valid = False
        for j in range(i+1, N):
            dist = np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)
            if dist < rs[i] + rs[j] - 1e-9:
                valid = False
    
    if not valid:
        # Simple fallback: shrink all radii by 5% to ensure validity
        # This might lower the score but ensures validity.
        # Better: find max overlap and shrink by that amount?
        # For now, a small global shrink is safer than an invalid result.
        rs = rs * 0.95
        
        # Re-check validity after shrink
        valid = True
        for i in range(N):
            if xs[i] < rs[i] - 1e-9 or xs[i] > w - rs[i] + 1e-9:
                valid = False
            if ys[i] < rs[i] - 1e-9 or ys[i] > h - rs[i] + 1e-9:
                valid = False
            for j in range(i+1, N):
                dist = np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)
                if dist < rs[i] + rs[j] - 1e-9:
                    valid = False
        
        if not valid:
             # Extreme fallback: Grid
             side = 5
             cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
             return np.array(cells[:N])

    result = np.column_stack((xs, ys, rs))
    return result

# EVOLVE-BLOCK-END
