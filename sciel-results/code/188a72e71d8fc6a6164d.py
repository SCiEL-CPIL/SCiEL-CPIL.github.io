import numpy as np
import time
import sys
from scipy.spatial.distance import cdist
from scipy.optimize import minimize

N = 200
TIME_LIMIT = 145.0  # Seconds, leaving margin before 150s kill
START_TIME = time.time()

def _check_time():
    return time.time() - START_TIME

def _compute_radius(centres):
    """Compute the packing radius for a set of centres.
    Returns r if valid, 0.0 if invalid.
    """
    if len(centres) != N:
        return 0.0
    if not np.all(np.isfinite(centres)):
        return 0.0
    # Check bounds: centres must be inside [-0.5, 0.5]^2
    if np.any(centres < -0.5) or np.any(centres > 0.5):
        return 0.0
    
    # Distance to walls
    dx = np.abs(centres[:, 0])
    dy = np.abs(centres[:, 1])
    wall_dists = np.min(np.stack([0.5 - dx, 0.5 - dy], axis=1), axis=1)
    min_wall = np.min(wall_dists)
    
    if min_wall <= 0:
        return 0.0

    # Minimum pairwise distance
    dists = cdist(centres, centres)
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)

    return min(min_dist / 2.0, min_wall)

def _generate_initial_config():
    """
    Generate a high-quality hexagonal initial configuration.
    We try to fit a hexagonal lattice into the square.
    """
    # Hexagonal lattice parameters
    # Let's try a range of spacings to find the best fit for N=200
    best_centres = None
    best_r = -1
    
    # Approximate density of hex packing is ~0.9069
    # Area of square = 1. Area of N circles = N * pi * r^2.
    # 200 * pi * r^2 <= 0.9069 * 1  => r^2 <= 0.9069 / (200 * pi) ~ 0.00144 => r ~ 0.038
    # Current best known is ~0.0366.
    
    # We will generate a hex grid with a specific spacing 's' (distance between centers)
    # and then center and clip it to the square, ensuring we have exactly N points.
    
    # Strategy: Generate a large hex grid, then select the N points closest to the center?
    # No, that might leave large empty spaces at the edges.
    # Better: Generate a hex grid that fits "roughly" and then adjust.
    
    # Let's use the 20x10 rectangular grid distorted into a hex shape as a baseline,
    # but also try a true hexagonal lattice generation.
    
    # Method 1: Distorted Rectangular Grid (Current approach, reliable)
    w, h = 20, 10
    xs = (np.arange(w) - (w-1)/2.0)
    ys = (np.arange(h) - (h-1)/2.0)
    
    # Scale to fit width 1 with some margin
    scale_x = 1.0 / (w - 1)
    scale_y = 1.0 / (h - 1)
    
    # Hex offset
    # In a hex lattice, vertical spacing is s * sqrt(3)/2
    # If we keep x-spacing = s, y-spacing should be s * sqrt(3)/2
    # Here we have fixed h=10, w=20.
    # Let's set s = scale_x. Then ideal y-spacing = scale_x * sqrt(3)/2.
    # Total height = (h-1) * scale_x * sqrt(3)/2.
    # If this height > 1, we scale down y.
    
    s = scale_x
    ideal_y_spacing = s * np.sqrt(3) / 2
    total_height = (h - 1) * ideal_y_spacing
    
    if total_height > 1.0:
        # Scale down y to fit
        scale_y = 1.0 / total_height
        y_spacing = 1.0 / (h - 1)
    else:
        y_spacing = ideal_y_spacing
        scale_y = 1.0 # Keep y as is, or center it
    
    # Generate points
    points = []
    for j in range(h):
        y = (j - (h-1)/2.0) * y_spacing
        # Offset for odd rows
        offset = 0
        if j % 2 != 0:
            offset = s / 2.0
            
        for i in range(w):
            x = (i - (w-1)/2.0) * s + offset
            points.append([x, y])
            
    grid = np.array(points)
    
    # Check bounds and clip/scale if necessary
    max_x = np.max(np.abs(grid[:, 0]))
    max_y = np.max(np.abs(grid[:, 1]))
    if max_x > 0.5 or max_y > 0.5:
        scale_down = 0.5 / max(max_x, max_y)
        grid *= scale_down
        
    r = _compute_radius(grid)
    if r > best_r:
        best_r = r
        best_centres = grid
        
    # Method 2: Try a few variations of hex spacing
    # We can vary the aspect ratio slightly
    for aspect in [0.9, 0.95, 1.0, 1.05, 1.1]:
        w_test, h_test = 20, 10
        s_test = 1.0 / (w_test - 1)
        y_spacing_test = s_test * np.sqrt(3) / 2 * aspect
        
        total_height_test = (h_test - 1) * y_spacing_test
        if total_height_test > 1.0:
            y_spacing_test = 1.0 / (h_test - 1)
            
        points = []
        for j in range(h_test):
            y = (j - (h_test-1)/2.0) * y_spacing_test
            offset = 0
            if j % 2 != 0:
                offset = s_test / 2.0
            for i in range(w_test):
                x = (i - (w_test-1)/2.0) * s_test + offset
                points.append([x, y])
                
        grid = np.array(points)
        max_x = np.max(np.abs(grid[:, 0]))
        max_y = np.max(np.abs(grid[:, 1]))
        if max_x > 0.5 or max_y > 0.5:
            scale_down = 0.5 / max(max_x, max_y)
            grid *= scale_down
            
        r = _compute_radius(grid)
        if r > best_r:
            best_r = r
            best_centres = grid
            
    return best_centres

def _soft_min_energy_and_grad(centres, beta):
    """
    Compute the soft-min energy and gradient.
    Objective: minimize (1/beta) * log(sum(exp(-beta * c_k)))
    where c_k are the constraints (pairwise dist/2, wall dist).
    This approximates -min(c_k).
    """
    centres = centres.reshape(N, 2)
    
    # Wall distances
    wx = 0.5 - np.abs(centres[:, 0])
    wy = 0.5 - np.abs(centres[:, 1])
    wall_dists = np.minimum(wx, wy)
    
    # Pairwise distances
    dists = cdist(centres, centres)
    np.fill_diagonal(dists, np.inf)
    pair_dists = dists / 2.0
    
    # Combine constraints
    # We need to compute weights w_k = exp(-beta * c_k) / sum(exp(-beta * c_j))
    # To avoid overflow/underflow, use log-sum-exp trick
    
    all_constraints = np.concatenate([wall_dists, pair_dists.ravel()])
    
    neg_beta_c = -beta * all_constraints
    max_val = np.max(neg_beta_c)
    exp_vals = np.exp(neg_beta_c - max_val)
    sum_exp = np.sum(exp_vals)
    weights = exp_vals / sum_exp
    
    # Objective value
    objective = (max_val + np.log(sum_exp)) / beta
    
    # Gradient
    grad = np.zeros_like(centres)
    
    # Wall gradients
    wall_weights = weights[:N]
    # c_wall_i = min(wx_i, wy_i)
    # If wx_i < wy_i, dc/dx = -sign(x_i), dc/dy = 0
    # If wy_i < wx_i, dc/dx = 0, dc/dy = -sign(y_i)
    # If equal, it's non-differentiable, we can average or pick one.
    for i in range(N):
        w = wall_weights[i]
        if w < 1e-12:
            continue
        if wx[i] < wy[i]:
            grad[i, 0] -= w * np.sign(centres[i, 0])
        elif wy[i] < wx[i]:
            grad[i, 1] -= w * np.sign(centres[i, 1])
        else:
            # Tie, push in both directions slightly?
            grad[i, 0] -= 0.5 * w * np.sign(centres[i, 0])
            grad[i, 1] -= 0.5 * w * np.sign(centres[i, 1])
            
    # Pairwise gradients
    # c_pair_ij = dist_ij / 2
    # d(dist)/dx_i = (x_i - x_j) / dist
    # So d(c_pair_ij)/dx_i = (x_i - x_j) / (2 * dist)
    
    pair_weights = weights[N:]
    idx_i, idx_j = np.triu_indices(N, k=1)
    
    # Vectorized gradient accumulation for pairs
    # This loop is O(N^2). For N=200, 19900 iterations.
    # We can try to vectorize this using matrix operations if possible, 
    # but the division by dist makes it tricky.
    # Given the small N, a loop is likely fast enough in Python if we avoid overhead.
    
    for k in range(len(idx_i)):
        i, j = idx_i[k], idx_j[k]
        w = pair_weights[k]
        if w < 1e-12:
            continue
            
        dx = centres[i, 0] - centres[j, 0]
        dy = centres[i, 1] - centres[j, 1]
        dist = np.sqrt(dx*dx + dy*dy)
        if dist < 1e-12:
            continue
            
        factor = w / (2.0 * dist)
        grad[i, 0] -= factor * dx
        grad[i, 1] -= factor * dy
        grad[j, 0] += factor * dx
        grad[j, 1] += factor * dy
        
    return objective, grad

def _local_push_refine(centres, steps=50):
    """
    Refine the configuration by pushing apart the most constrained pairs.
    """
    best_centres = centres.copy()
    best_r = _compute_radius(best_centres)
    
    for step in range(steps):
        if time.time() > START_TIME + TIME_LIMIT - 5:
            break
            
        r = _compute_radius(best_centres)
        if r <= best_r:
            break
            
        # Find all pairs with distance close to 2*r
        # And points close to walls
        
        dists = cdist(best_centres, best_centres)
        np.fill_diagonal(dists, np.inf)
        
        # Identify active constraints
        # Threshold: within 1% of the minimum
        threshold = 2.0 * r * 1.01
        
        moved = False
        step_size = 0.001
        
        # Pairwise pushes
        # We want to move i and j apart.
        # Force direction: from j to i for point i, from i to j for point j.
        
        # To avoid conflicting moves, we can accumulate displacements
        displacements = np.zeros_like(best_centres)
        
        for i in range(N):
            for j in range(i+1, N):
                d = dists[i, j]
                if d < threshold:
                    # They are too close
                    dx = best_centres[i, 0] - best_centres[j, 0]
                    dy = best_centres[i, 1] - best_centres[j, 1]
                    norm = np.sqrt(dx*dx + dy*dy)
                    if norm < 1e-6:
                        # Random push
                        angle = np.random.uniform(0, 2*np.pi)
                        dx = np.cos(angle)
                        dy = np.sin(angle)
                        norm = 1.0
                    else:
                        norm = 1.0 # normalized direction
                    
                    # Push strength proportional to violation
                    violation = threshold - d
                    strength = step_size * (violation / (2*r)) # normalized
                    
                    displacements[i, 0] += strength * dx
                    displacements[i, 1] += strength * dy
                    displacements[j, 0] -= strength * dx
                    displacements[j, 1] -= strength * dy
                    moved = True
        
        # Wall pushes
        for i in range(N):
            wx = 0.5 - np.abs(best_centres[i, 0])
            wy = 0.5 - np.abs(best_centres[i, 1])
            
            if wx < r * 1.01:
                # Close to left/right wall
                sign_x = np.sign(best_centres[i, 0])
                if sign_x == 0: sign_x = 1
                displacements[i, 0] += step_size * sign_x
                moved = True
                
            if wy < r * 1.01:
                # Close to top/bottom wall
                sign_y = np.sign(best_centres[i, 1])
                if sign_y == 0: sign_y = 1
                displacements[i, 1] += step_size * sign_y
                moved = True
        
        if not moved:
            break
            
        # Apply displacements
        new_centres = best_centres + displacements
        # Clip to bounds to ensure validity
        new_centres = np.clip(new_centres, -0.5, 0.5)
        
        new_r = _compute_radius(new_centres)
        if new_r > best_r:
            best_centres = new_centres
            best_r = new_r
        else:
            # If it didn't improve, reduce step size or break
            step_size *= 0.5
            if step_size < 1e-6:
                break
                
    return best_centres, best_r

def construct():
    start_time = time.time()
    time_budget = TIME_LIMIT - (start_time - START_TIME)
    
    # 1. Generate Initial Configuration
    centres = _generate_initial_config()
    current_r = _compute_radius(centres)
    
    if current_r == 0.0:
        # Emergency fallback
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
        
    best_centres = centres.copy()
    best_r = current_r
    
    # 2. Smooth Optimization (Soft-Min)
    if time_budget > 20:
        # Use a range of beta values, starting low and increasing
        betas = [10.0, 30.0, 100.0]
        
        for beta in betas:
            if time.time() > START_TIME + TIME_LIMIT - 10:
                break
                
            def obj_and_grad(x):
                if time.time() > START_TIME + TIME_LIMIT - 2:
                    return 1e10, np.zeros_like(x)
                c = x.reshape(N, 2)
                # Clip to avoid boundary issues during optimization
                c = np.clip(c, -0.499, 0.499)
                val, grad = _soft_min_energy_and_grad(c, beta)
                return val, grad
            
            try:
                result = minimize(
                    obj_and_grad,
                    best_centres.ravel(),
                    method='L-BFGS-B',
                    jac=True,
                    options={
                        'maxiter': 100,
                        'maxfun': 200,
                        'ftol': 1e-12,
                        'gtol': 1e-10,
                        'maxls': 30
                    }
                )
                
                new_centres = result.x.reshape(N, 2)
                new_centres = np.clip(new_centres, -0.5, 0.5)
                new_r = _compute_radius(new_centres)
                
                if new_r > best_r:
                    best_r = new_r
                    best_centres = new_centres
            except Exception as e:
                pass
    
    # 3. Local Push Refinement
    if time.time() < START_TIME + TIME_LIMIT - 5:
        best_centres, best_r = _local_push_refine(best_centres, steps=100)
        
    # 4. Final Validation
    final_r = _compute_radius(best_centres)
    if final_r <= 0:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
        
    return best_centres
