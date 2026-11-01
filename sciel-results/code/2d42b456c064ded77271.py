import numpy as np
import time
import sys
from scipy.optimize import minimize

N = 200
START_TIME = time.time()
TIME_LIMIT = 110.0  # Safety margin

def compute_soft_radius(c, beta):
    """
    Compute the soft-min radius and its gradient.
    c: (N, 2) array of centers.
    beta: sharpness parameter.
    Returns: -soft_radius, gradient
    """
    # Pairwise distances
    # c is (N, 2)
    # dist[i, j] = ||c[i] - c[j]||
    # Use broadcasting
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=-1))
    
    # Wall distances
    # dist_wall[i] = 0.5 - max(|c[i,0]|, |c[i,1]|)
    # This is non-smooth at |x|=|y|. 
    # Better wall distance: distance to boundary.
    # dist to left = 0.5 + c[i,0]
    # dist to right = 0.5 - c[i,0]
    # dist to bottom = 0.5 + c[i,1]
    # dist to top = 0.5 - c[i,1]
    # min_dist_wall = min(0.5+c_x, 0.5-c_x, 0.5+c_y, 0.5-c_y)
    # This is also non-smooth.
    # For soft-min, we can include all 4 wall distances for each point.
    
    # To keep it simple and smooth, let's use the distance to the closest wall.
    # Actually, the constraint is that the circle must be inside.
    # r <= min(dist_wall)
    # So we want to maximize min(all pair dists / 2, all wall dists).
    
    # Let's collect all distances into a single vector D.
    # Pairs: i < j
    i_idx, j_idx = np.triu_indices(N, k=1)
    pair_diff = c[i_idx] - c[j_idx]
    pair_dist = np.sqrt(np.sum(pair_diff**2, axis=1))
    
    # Walls
    # d_left = 0.5 + c[:, 0]
    # d_right = 0.5 - c[:, 0]
    # d_bottom = 0.5 + c[:, 1]
    # d_top = 0.5 - c[:, 1]
    wall_dists = np.array([
        0.5 + c[:, 0],
        0.5 - c[:, 0],
        0.5 + c[:, 1],
        0.5 - c[:, 1]
    ]).T # (N, 4)
    
    # Combine all distances
    # Note: pair distances contribute to r as dist/2.
    # Wall distances contribute to r as dist.
    # So we should scale pair distances by 0.5 before taking min?
    # No, r is the radius.
    # Condition: 2r <= dist_pair => r <= dist_pair / 2
    # Condition: r <= dist_wall
    # So we want to maximize min( dist_pair/2, dist_wall ).
    
    # Let's define effective distances:
    eff_pair_dist = pair_dist / 2.0
    eff_wall_dist = wall_dists.flatten()
    
    all_dists = np.concatenate([eff_pair_dist, eff_wall_dist])
    
    # Soft min: -1/beta * log(sum(exp(-beta * d)))
    # We want to maximize this, so minimize -soft_min.
    # f(c) = 1/beta * log(sum(exp(-beta * d)))
    
    # To avoid overflow, subtract max
    max_d = np.max(all_dists)
    # If beta is large, -beta * (d - max_d) is very negative for d < max_d.
    # exp(-beta * (d - max_d)) is safe if d <= max_d.
    # Let's compute log-sum-exp properly.
    # log(sum(exp(-beta * d))) = -beta * max_d + log(sum(exp(-beta * (d - max_d))))
    
    shifted_d = all_dists - max_d
    exp_terms = np.exp(-beta * shifted_d)
    sum_exp = np.sum(exp_terms)
    log_sum_exp = -beta * max_d + np.log(sum_exp)
    
    soft_min_val = -1.0 / beta * log_sum_exp
    # We want to minimize -soft_min_val
    obj_val = -soft_min_val
    
    # Gradient
    # d(obj)/dc = - d(soft_min)/dc
    # d(soft_min)/dc = 1/beta * (1/sum_exp) * sum( d(exp_term)/dc )
    # exp_term_k = exp(-beta * d_k)
    # d(exp_term_k)/dc = exp_term_k * (-beta * d(d_k)/dc)
    # So d(soft_min)/dc = 1/beta * 1/sum_exp * sum( exp_term_k * (-beta) * grad_d_k )
    #                 = -1/sum_exp * sum( exp_term_k * grad_d_k )
    
    # We need grad_d_k for each distance.
    # This is complex to vectorize directly for the sum.
    # Let's accumulate the gradient.
    
    grad_c = np.zeros_like(c)
    
    # Weights for each distance
    weights = exp_terms / sum_exp
    
    # Pair distances
    # eff_pair_dist[k] = 0.5 * ||c[i] - c[j]||
    # d(eff_pair_dist[k])/dc[i] = 0.5 * (c[i] - c[j]) / ||c[i] - c[j]||
    # d(eff_pair_dist[k])/dc[j] = 0.5 * (c[j] - c[i]) / ||c[i] - c[j]||
    
    pair_weights = weights[:N*(N-1)//2]
    
    # Vectorized gradient for pairs
    # diff is (num_pairs, 2)
    # norm is (num_pairs,)
    norms = pair_dist
    # Avoid division by zero
    norms_safe = np.where(norms > 1e-8, norms, 1.0)
    unit_diffs = pair_diff / norms_safe[:, np.newaxis]
    
    # grad for c[i] += 0.5 * unit_diff * weight
    # grad for c[j] -= 0.5 * unit_diff * weight
    
    # Use np.add.at for accumulation
    grad_pair = 0.5 * unit_diffs * pair_weights[:, np.newaxis]
    
    np.add.at(grad_c, i_idx, grad_pair)
    np.add.at(grad_c, j_idx, -grad_pair)
    
    # Wall distances
    # eff_wall_dist is flattened (N, 4)
    wall_weights = weights[N*(N-1)//2:]
    wall_weights_reshaped = wall_weights.reshape(N, 4)
    
    # d(0.5 + c_x)/dc_x = 1
    # d(0.5 - c_x)/dc_x = -1
    # d(0.5 + c_y)/dc_y = 1
    # d(0.5 - c_y)/dc_y = -1
    
    # grad_c[:, 0] += wall_weights[:, 0] * 1
    # grad_c[:, 0] += wall_weights[:, 1] * (-1)
    # grad_c[:, 1] += wall_weights[:, 2] * 1
    # grad_c[:, 1] += wall_weights[:, 3] * (-1)
    
    grad_c[:, 0] += wall_weights_reshaped[:, 0] - wall_weights_reshaped[:, 1]
    grad_c[:, 1] += wall_weights_reshaped[:, 2] - wall_weights_reshaped[:, 3]
    
    # Total gradient of soft_min is -1/sum_exp * sum(...)
    # We calculated sum(exp_term * grad_d) implicitly in the accumulation above?
    # No, the formula was:
    # d(soft_min)/dc = -1/sum_exp * sum( exp_term_k * grad_d_k )
    # My accumulation calculated sum( weight_k * grad_d_k ) where weight_k = exp_term_k / sum_exp.
    # So the accumulated grad_c is exactly d(soft_min)/dc.
    
    # We want to minimize -soft_min, so gradient is -grad_c.
    grad_obj = -grad_c
    
    return obj_val, grad_obj

def compute_exact_radius(c):
    """Compute the exact maximum radius for the given centers."""
    # Pairwise
    i_idx, j_idx = np.triu_indices(N, k=1)
    pair_diff = c[i_idx] - c[j_idx]
    pair_dist = np.sqrt(np.sum(pair_diff**2, axis=1))
    min_pair_dist = np.min(pair_dist)
    
    # Walls
    dist_left = 0.5 + c[:, 0]
    dist_right = 0.5 - c[:, 0]
    dist_bottom = 0.5 + c[:, 1]
    dist_top = 0.5 - c[:, 1]
    
    min_wall_dist = np.min(np.min([dist_left, dist_right, dist_bottom, dist_top], axis=0))
    
    r = min(min_pair_dist / 2.0, min_wall_dist)
    return r

def construct():
    global START_TIME, TIME_LIMIT
    
    best_c = None
    best_r = -1.0
    
    # Initializations
    # 1. Grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    init_grid = grid[:N]
    
    # 2. Hexagonal
    # Generate hex grid points
    hex_points = []
    side = 1.0
    # Approximate hex spacing
    # We want to fit 200 points.
    # Let's generate a large hex grid and pick the best 200.
    # Or just generate a hex grid that covers the square.
    # Hex grid generation:
    # dx = 1.0 / 15  # approx
    # dy = dx * np.sqrt(3) / 2
    # This is heuristic. Let's use a simple random init too.
    
    inits = [init_grid]
    
    # Random inits
    for _ in range(5):
        c_rand = np.random.uniform(-0.5, 0.5, (N, 2))
        inits.append(c_rand)
        
    # Hex init
    # Generate points on a hex lattice
    # Scale to fit in square
    # Let's try a few hex configurations
    for scale in [0.04, 0.05, 0.06]:
        points = []
        # Generate grid
        for i in range(-20, 21):
            for j in range(-20, 21):
                x = i * scale + (j % 2) * scale / 2
                y = j * scale * np.sqrt(3) / 2
                if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                    points.append([x, y])
        points = np.array(points)
        if len(points) >= N:
            # Pick N closest to center? Or random?
            # Random selection might be better to break symmetry
            idx = np.random.choice(len(points), N, replace=False)
            inits.append(points[idx])
        else:
            inits.append(points) # Pad if necessary? No, N is fixed.
            
    # Optimization
    beta_schedule = [10.0, 20.0, 50.0, 100.0, 200.0]
    
    for init in inits:
        if time.time() - START_TIME > TIME_LIMIT:
            break
            
        c = init.copy()
        
        # Run optimization with increasing beta
        for beta in beta_schedule:
            if time.time() - START_TIME > TIME_LIMIT:
                break
                
            # Define objective and gradient
            def obj_func(c_flat):
                c_2d = c_flat.reshape(N, 2)
                val, grad = compute_soft_radius(c_2d, beta)
                return val, grad.reshape(-1)
                
            res = minimize(obj_func, c.flatten(), method='L-BFGS-B', jac=True, 
                           options={'maxiter': 100, 'gtol': 1e-8, 'ftol': 1e-8})
            
            c = res.x.reshape(N, 2)
            
            # Check time
            if time.time() - START_TIME > TIME_LIMIT:
                break
        
        # Evaluate exact radius
        r = compute_exact_radius(c)
        
        if r > best_r:
            best_r = r
            best_c = c.copy()
            
    # Final polish with Nelder-Mead or Powell on the exact objective?
    # L-BFGS-B on soft-min is good. 
    # Let's do a few more L-BFGS-B steps with high beta on the best result.
    
    if best_c is not None and time.time() - START_TIME < TIME_LIMIT - 10:
        c = best_c.copy()
        for beta in [500.0, 1000.0]:
            if time.time() - START_TIME > TIME_LIMIT - 5:
                break
            def obj_func(c_flat):
                c_2d = c_flat.reshape(N, 2)
                val, grad = compute_soft_radius(c_2d, beta)
                return val, grad.reshape(-1)
                
            res = minimize(obj_func, c.flatten(), method='L-BFGS-B', jac=True, 
                           options={'maxiter': 50, 'gtol': 1e-10, 'ftol': 1e-10})
            c = res.x.reshape(N, 2)
            
        r = compute_exact_radius(c)
        if r > best_r:
            best_r = r
            best_c = c
            
    # Ensure validity
    if best_c is None:
        return init_grid
        
    # Clip to bounds just in case (though optimizer should handle it)
    # Actually, if a point goes out, the wall distance becomes negative, 
    # and the soft-min will penalize it heavily. 
    # But we should ensure the returned centers are strictly inside or on the boundary?
    # The task says "inside the square [-1/2, 1/2]^2".
    # "A centre on or outside the square ... makes the candidate invalid."
    # So we must ensure |c_ik| < 0.5.
    # My wall distance calculation used 0.5 - |c|. If c is outside, this is negative.
    # The optimizer minimizes -soft_min. If dist is negative, exp(-beta * neg) is huge.
    # So it should push them in.
    # However, floating point errors might leave them slightly outside.
    # Let's project them back if they are slightly outside.
    
    # Check and fix
    for i in range(N):
        for k in range(2):
            if abs(best_c[i, k]) >= 0.5:
                best_c[i, k] = np.sign(best_c[i, k]) * (0.5 - 1e-9)
                
    return best_c

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

N = 200
START_TIME = time.time()
TIME_LIMIT = 110.0  # Safety margin

def compute_soft_radius(c, beta):
    i_idx, j_idx = np.triu_indices(N, k=1)
    pair_diff = c[i_idx] - c[j_idx]
    pair_dist = np.sqrt(np.sum(pair_diff**2, axis=1))
    
    wall_dists = np.array([
        0.5 + c[:, 0],
        0.5 - c[:, 0],
        0.5 + c[:, 1],
        0.5 - c[:, 1]
    ]).T # (N, 4)
    
    eff_pair_dist = pair_dist / 2.0
    eff_wall_dist = wall_dists.flatten()
    
    all_dists = np.concatenate([eff_pair_dist, eff_wall_dist])
    
    max_d = np.max(all_dists)
    shifted_d = all_dists - max_d
    exp_terms = np.exp(-beta * shifted_d)
    sum_exp = np.sum(exp_terms)
    log_sum_exp = -beta * max_d + np.log(sum_exp)
    
    soft_min_val = -1.0 / beta * log_sum_exp
    obj_val = -soft_min_val
    
    grad_c = np.zeros_like(c)
    weights = exp_terms / sum_exp
    
    pair_weights = weights[:N*(N-1)//2]
    norms = pair_dist
    norms_safe = np.where(norms > 1e-8, norms, 1.0)
    unit_diffs = pair_diff / norms_safe[:, np.newaxis]
    
    grad_pair = 0.5 * unit_diffs * pair_weights[:, np.newaxis]
    
    np.add.at(grad_c, i_idx, grad_pair)
    np.add.at(grad_c, j_idx, -grad_pair)
    
    wall_weights = weights[N*(N-1)//2:]
    wall_weights_reshaped = wall_weights.reshape(N, 4)
    
    grad_c[:, 0] += wall_weights_reshaped[:, 0] - wall_weights_reshaped[:, 1]
    grad_c[:, 1] += wall_weights_reshaped[:, 2] - wall_weights_reshaped[:, 3]
    
    grad_obj = -grad_c
    return obj_val, grad_obj

def compute_exact_radius(c):
    i_idx, j_idx = np.triu_indices(N, k=1)
    pair_diff = c[i_idx] - c[j_idx]
    pair_dist = np.sqrt(np.sum(pair_diff**2, axis=1))
    min_pair_dist = np.min(pair_dist)
    
    dist_left = 0.5 + c[:, 0]
    dist_right = 0.5 - c[:, 0]
    dist_bottom = 0.5 + c[:, 1]
    dist_top = 0.5 - c[:, 1]
    
    min_wall_dist = np.min(np.min([dist_left, dist_right, dist_bottom, dist_top], axis=0))
    
    r = min(min_pair_dist / 2.0, min_wall_dist)
    return r

def construct():
    global START_TIME, TIME_LIMIT
    
    best_c = None
    best_r = -1.0
    
    # Initializations
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    init_grid = grid[:N]
    
    inits = [init_grid]
    
    for _ in range(5):
        c_rand = np.random.uniform(-0.5, 0.5, (N, 2))
        inits.append(c_rand)
        
    for scale in [0.04, 0.05, 0.06]:
        points = []
        for i in range(-20, 21):
            for j in range(-20, 21):
                x = i * scale + (j % 2) * scale / 2
                y = j * scale * np.sqrt(3) / 2
                if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                    points.append([x, y])
        points = np.array(points)
        if len(points) >= N:
            idx = np.random.choice(len(points), N, replace=False)
            inits.append(points[idx])
        else:
            # If not enough, pad with random
            while len(points) < N:
                points = np.vstack([points, np.random.uniform(-0.5, 0.5, (1, 2))])
            inits.append(points)
            
    beta_schedule = [10.0, 20.0, 50.0, 100.0, 200.0]
    
    for init in inits:
        if time.time() - START_TIME > TIME_LIMIT:
            break
            
        c = init.copy()
        
        for beta in beta_schedule:
            if time.time() - START_TIME > TIME_LIMIT:
                break
                
            def obj_func(c_flat):
                c_2d = c_flat.reshape(N, 2)
                val, grad = compute_soft_radius(c_2d, beta)
                return val, grad.reshape(-1)
                
            res = minimize(obj_func, c.flatten(), method='L-BFGS-B', jac=True, 
                           options={'maxiter': 100, 'gtol': 1e-8, 'ftol': 1e-8})
            
            c = res.x.reshape(N, 2)
            
            if time.time() - START_TIME > TIME_LIMIT:
                break
        
        r = compute_exact_radius(c)
        
        if r > best_r:
            best_r = r
            best_c = c.copy()
            
    if best_c is not None and time.time() - START_TIME < TIME_LIMIT - 10:
        c = best_c.copy()
        for beta in [500.0, 1000.0]:
            if time.time() - START_TIME > TIME_LIMIT - 5:
                break
            def obj_func(c_flat):
                c_2d = c_flat.reshape(N, 2)
                val, grad = compute_soft_radius(c_2d, beta)
                return val, grad.reshape(-1)
                
            res = minimize(obj_func, c.flatten(), method='L-BFGS-B', jac=True, 
                           options={'maxiter': 50, 'gtol': 1e-10, 'ftol': 1e-10})
            c = res.x.reshape(N, 2)
            
        r = compute_exact_radius(c)
        if r > best_r:
            best_r = r
            best_c = c
            
    if best_c is None:
        return init_grid
        
    for i in range(N):
        for k in range(2):
            if abs(best_c[i, k]) >= 0.5:
                best_c[i, k] = np.sign(best_c[i, k]) * (0.5 - 1e-9)
                
    return best_c
# EVOLVE-BLOCK-END
