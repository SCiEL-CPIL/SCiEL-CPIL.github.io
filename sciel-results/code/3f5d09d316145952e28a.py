"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _penalized(params):
    """
    Penalized objective: minimize -sum(radii) + penalties for violations.
    params: [x1, y1, r1, x2, y2, r2, ...]
    """
    n = len(params) // 3
    xy = params[:2*n].reshape(n, 2)
    r = params[2*n:]
    
    # Objective: maximize sum of radii -> minimize -sum(r)
    obj = -np.sum(r)
    
    # Penalty for circles going outside [0,1]^2
    # We want x >= r, x <= 1-r, etc.
    # Violation = max(0, r - x) + max(0, r - (1-x)) + ...
    # Using squared penalty for smoothness
    viol_bound = 0.0
    for i in range(n):
        x, y = xy[i]
        ri = r[i]
        # Distance to boundaries
        d_left = x - ri
        d_right = 1.0 - x - ri
        d_bottom = y - ri
        d_top = 1.0 - y - ri
        # Penalize if any distance is negative
        if d_left < 0: viol_bound += d_left**2
        if d_right < 0: viol_bound += d_right**2
        if d_bottom < 0: viol_bound += d_bottom**2
        if d_top < 0: viol_bound += d_top**2
        
    # Penalty for overlapping circles
    viol_overlap = 0.0
    for i in range(n):
        for j in range(i+1, n):
            dx = xy[i, 0] - xy[j, 0]
            dy = xy[i, 1] - xy[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            min_dist = r[i] + r[j]
            if dist < min_dist:
                viol_overlap += (min_dist - dist)**2
    
    # Large penalty weight to ensure feasibility
    penalty_weight = 50.0
    obj += penalty_weight * (viol_bound + viol_overlap)
    
    return obj


def _greedy_init(num_candidates=100, seed=None):
    """
    Greedy initialization: place circles one by one at the position 
    that allows the largest possible radius.
    """
    if seed is not None:
        np.random.seed(seed)
        
    centers = []
    radii = []
    
    # Candidate grid for searching the best position
    # 100x100 grid gives 10,000 candidates. 
    # For 26 circles, this is 26 * 10000 * 26 dist checks ~ 6.7M ops.
    # Vectorized, this is fast.
    grid_x = np.linspace(0.01, 0.99, num_candidates)
    grid_y = np.linspace(0.01, 0.99, num_candidates)
    gx, gy = np.meshgrid(grid_x, grid_y)
    candidates = np.column_stack([gx.ravel(), gy.ravel()])
    
    for _ in range(N):
        if len(centers) == 0:
            # First circle: center of square is best
            best_x, best_y = 0.5, 0.5
            best_r = 0.5
        else:
            centers_arr = np.array(centers)
            radii_arr = np.array(radii)
            
            # Compute distances from all candidates to all existing centers
            # candidates: (M, 2), centers_arr: (K, 2)
            # dist: (M, K)
            dx = candidates[:, 0:1] - centers_arr[None, :, 0]
            dy = candidates[:, 1:2] - centers_arr[None, :, 1]
            dists = np.sqrt(dx**2 + dy**2)
            
            # Minimum distance to any existing circle, minus its radius
            # dists: (M, K), radii_arr: (K,)
            min_clearance = np.min(dists - radii_arr[None, :], axis=1)
            
            # Distance to boundaries for each candidate
            d_bound = np.min(np.column_stack([
                candidates[:, 0], 
                1.0 - candidates[:, 0], 
                candidates[:, 1], 
                1.0 - candidates[:, 1]
            ]), axis=1)
            
            # Max possible radius at each candidate
            max_radii = np.minimum(min_clearance, d_bound)
            
            # Find the candidate with the largest max radius
            best_idx = np.argmax(max_radii)
            best_x, best_y = candidates[best_idx]
            best_r = max_radii[best_idx]
            
            # If best_r is very small, we might be stuck. 
            # In a unit square with 26 circles, r should be ~0.09.
            # If greedy gets stuck, it means we've filled the space.
            
        # Shrink radius slightly to ensure strict non-overlap for the optimizer start
        # This helps the penalty method start in a feasible region
        final_r = max(0.001, best_r * 0.99)
        
        centers.append((best_x, best_y))
        radii.append(final_r)
        
    # Convert to array
    result = np.zeros((N, 3))
    for i in range(N):
        result[i, 0] = centers[i][0]
        result[i, 1] = centers[i][1]
        result[i, 2] = radii[i]
        
    return result


def construct():
    start_time = time.time()
    deadline = 145.0  # Leave 5s safety margin
    
    best_params = None
    best_score = -np.inf  # We want to minimize the penalized objective
    
    # Bounds: x, y in [0.01, 0.99], r in [0.001, 0.5]
    bounds = []
    for i in range(N):
        bounds.append((0.01, 0.99))  # x
        bounds.append((0.01, 0.99))  # y
    for i in range(N):
        bounds.append((0.001, 0.5))  # r
        
    # Try multiple trials with different greedy initializations
    # Greedy is deterministic for a given seed, so we vary the seed
    # or the candidate grid resolution/order
    
    np.random.seed(42)
    
    # We can also try a simple grid init as a fallback
    side = int(np.ceil(np.sqrt(N)))
    grid_init = np.zeros((N, 3))
    for i in range(N):
        gx = i % side
        gy = i // side
        grid_init[i, 0] = (gx + 0.5) / side
        grid_init[i, 1] = (gy + 0.5) / side
        grid_init[i, 2] = 0.5 / side - 0.005

    num_trials = 10
    for trial in range(num_trials):
        if time.time() > deadline - 10:  # Stop if less than 10s left
            break
            
        # Choose initialization
        if trial == 0:
            init = grid_init
        else:
            # Vary the greedy initialization
            # Use different seeds to shuffle the candidate selection?
            # Or just use the same greedy, it's usually robust.
            # Let's add some noise to the greedy result for diversity
            init = _greedy_init(num_candidates=100, seed=trial)
            # Add small random noise
            noise = np.random.normal(0, 0.01, init.shape)
            init[:, 0] = np.clip(init[:, 0] + noise[:, 0], 0.01, 0.99)
            init[:, 1] = np.clip(init[:, 1] + noise[:, 1], 0.01, 0.99)
            # Do not change radii too much, keep them close to greedy max
            # init[:, 2] = np.clip(init[:, 2] + noise[:, 2]*0.1, 0.001, 0.5)
            
        params = init.flatten()
        
        # Optimize
        try:
            result = minimize(
                _penalized,
                params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'maxfun': 500, 'ftol': 1e-8}
            )
            
            # Check if result is better
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x.copy()
                
        except Exception as e:
            #print(f"Trial {trial} failed: {e}")
            continue
        
    # If no valid result from optimization, fall back to grid
    if best_params is None:
        return grid_init
    
    # Extract and validate
    n = len(best_params) // 3
    xy = best_params[:2*n].reshape(n, 2)
    r = best_params[2*n:]
    
    result = np.column_stack([xy, r])
    
    # Final validation and clipping to ensure strict validity
    for i in range(n):
        x, y, ri = result[i]
        # Ensure circle is inside unit square
        # ri must be <= min(x, 1-x, y, 1-y)
        max_r = min(x, 1.0 - x, y, 1.0 - y)
        if ri > max_r:
            ri = max_r
        # Ensure ri is positive
        ri = max(ri, 0.001)
        result[i, 2] = ri
        result[i, 0] = np.clip(x, ri, 1.0 - ri)
        result[i, 1] = np.clip(y, ri, 1.0 - ri)
        
    # One final check for overlaps? 
    # The penalty method should have handled it, but clipping radii might cause overlaps.
    # If clipping causes overlaps, we should reduce radii further.
    # Let's do a quick pass to resolve overlaps by reducing radii.
    # This is a simple heuristic: if two circles overlap, reduce the smaller one.
    for _ in range(5): # Iterate a few times
        for i in range(n):
            for j in range(i+1, n):
                dx = result[i, 0] - result[j, 0]
                dy = result[i, 1] - result[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = result[i, 2] + result[j, 2]
                if dist < min_dist:
                    overlap = min_dist - dist
                    # Reduce the smaller radius
                    if result[i, 2] < result[j, 2]:
                        result[i, 2] = max(0.001, result[i, 2] - overlap/2)
                    else:
                        result[j, 2] = max(0.001, result[j, 2] - overlap/2)
                        
    return result
# EVOLVE-BLOCK-END
