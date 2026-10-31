"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10 second safety margin
    
    best_score = 0.0
    best_result = None
    seed_result = None
    
    # Seed result for fallback
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
    seed_arr = np.array(cells[:N])
    # Check seed validity and score
    if _is_valid(seed_arr, 2.0, 2.0): # Seed uses square 1x1? No, seed uses side=5, so 5x5 grid in [0,1]?
        # The seed code: side=5. x in [0,1], y in [0,1]. Perimeter of 1x1 rect is 4.
        # So seed is in 1x1 box.
        seed_score = np.sum(seed_arr[:, 2])
        best_score = seed_score
        best_result = seed_arr
    else:
        # If seed is invalid (unlikely), we need a valid one.
        # But seed is simple grid in [0,1]x[0,1].
        # Perimeter 4 means w+h=2. 1x1 is valid.
        pass

    # Optimization
    # We will try different aspect ratios.
    # w + h = 2.
    # Let's try w in [0.5, 1.5].
    
    aspect_ratios = np.linspace(0.6, 1.4, 9)
    
    for w in aspect_ratios:
        if time.time() > deadline:
            break
        h = 2.0 - w
        
        # Initialize
        # Grid initialization
        rows = int(np.sqrt(N * h / w))
        cols = int(N / rows)
        if rows * cols < N:
            cols += 1
        
        # Adjust rows to fit N
        # Simple grid fill
        init_x = np.zeros(N)
        init_y = np.zeros(N)
        init_r = np.zeros(N)
        
        # Place circles in grid
        for i in range(N):
            r_idx = i // cols
            c_idx = i % cols
            # Center in cell
            init_x[i] = (c_idx + 0.5) * (w / cols)
            init_y[i] = (r_idx + 0.5) * (h / (int(np.ceil(N / cols))))
            # Initial radius estimate
            init_r[i] = 0.5 * min(w / cols, h / int(np.ceil(N / cols))) * 0.8

        # Ensure initial radii are small enough to not overlap in grid
        # Actually, L-BFGS with penalty will handle it.
        
        x0 = np.concatenate([init_x, init_y, init_r])
        
        # Bounds
        # x_i in [0, w], y_i in [0, h], r_i in [0, min(w,h)/2]
        bounds = []
        for i in range(N):
            bounds.append((0.0, w)) # x
            bounds.append((0.0, h)) # y
            bounds.append((1e-6, 1.0)) # r (max radius can't be > 1.0 in 2x1 box? max r is 1.0 if w=1,h=1)
        
        def objective_and_penalty(v):
            if time.time() > deadline:
                return 1e9
            x = v[:N]
            y = v[N:2*N]
            r = v[2*N:]
            
            # Boundary violations
            # x should be in [r, w-r], y in [r, h-r]
            # Penalty for being out of bounds
            pen_x = np.sum(np.maximum(0, r - x)**2 + np.maximum(0, x - (w - r))**2)
            pen_y = np.sum(np.maximum(0, r - y)**2 + np.maximum(0, y - (h - r))**2)
            
            # Overlap violations
            pen_overlap = 0.0
            for i in range(N):
                for j in range(i+1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    min_dist = r[i] + r[j]
                    if dist < min_dist:
                        pen_overlap += (min_dist - dist)**2
            
            total_penalty = pen_x + pen_y + 10.0 * pen_overlap
            sum_r = np.sum(r)
            
            return -sum_r + total_penalty

        # Run optimization
        # Use L-BFGS-B
        try:
            res = minimize(objective_and_penalty, x0, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-8})
            
            if res.success or res.fun < best_score * -1: # res.fun is -sum_r + pen
                # Extract result
                x = res.x[:N]
                y = res.x[N:2*N]
                r = res.x[2*N:]
                
                # Check validity strictly
                if _is_valid(np.column_stack([x, y, r]), w, h):
                    score = np.sum(r)
                    if score > best_score:
                        best_score = score
                        best_result = np.column_stack([x, y, r])
        except Exception:
            pass

    if best_result is not None:
        return best_result
    return seed_arr

def _is_valid(arr, w, h):
    x = arr[:, 0]
    y = arr[:, 1]
    r = arr[:, 2]
    
    # Check bounds
    if np.any(x < r - 1e-6) or np.any(x > w - r + 1e-6):
        return False
    if np.any(y < r - 1e-6) or np.any(y > h - r + 1e-6):
        return False
    if np.any(r < 1e-6):
        return False
        
    # Check overlaps
    for i in range(len(arr)):
        for j in range(i+1, len(arr)):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist_sq = dx*dx + dy*dy
            rad_sum = r[i] + r[j]
            if dist_sq < rad_sum*rad_sum - 1e-6:
                return False
    return True

# EVOLVE-BLOCK-END
