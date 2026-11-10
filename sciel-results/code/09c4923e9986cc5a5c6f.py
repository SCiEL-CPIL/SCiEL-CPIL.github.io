import numpy as np
import time

N = 100
TOTAL_TIME_LIMIT = 140.0  # Safety margin from 150s limit

def _compute_radius(c):
    """
    Compute the largest common radius for given centers.
    c: (N, 3) array of centers.
    Returns: min(min_pair_dist/2, min_wall_dist)
    """
    c = np.asarray(c, dtype=np.float64)
    n = len(c)
    if n < 2:
        return 0.0
    
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Distance to walls
    wall_dists = 0.5 - np.abs(c)
    min_wall_dist = np.min(wall_dists)
    
    return 0.5 * min(min_pair_dist, min_wall_dist)

def _soft_min_surrogate(c, beta=50.0):
    """
    Smooth approximation of the radius using soft-min (log-sum-exp).
    We want to maximize r = min(d_ij/2, wall_dist_i)
    This is equivalent to maximizing t subject to t <= d_ij/2 and t <= wall_dist_i for all constraints.
    
    Using a soft-min surrogate:
    r_soft = -1/beta * log( sum_j exp(-beta * d_ij/2) + sum_i exp(-beta * wall_dist_i) ) / (N*(N-1)/2 + N)
    
    Actually, a simpler and more effective approach for optimization is to use a penalty function
    or just optimize the soft-min directly.
    
    Let's use:
    S = sum of exp(-beta * constraint_k) for all constraints k
    r_soft = -1/beta * log(S) / num_constraints
    
    As beta -> inf, this approaches the true min.
    """
    c = np.asarray(c, dtype=np.float64)
    n = len(c)
    
    # Pairwise distances / 2
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    pair_terms = dists / 2.0
    
    # Wall distances
    wall_dists = 0.5 - np.abs(c)
    
    # Combine all constraints
    # We need to be careful with numerical stability in log-sum-exp
    # Let's compute the log-sum-exp of -beta * constraint
    # To avoid overflow, subtract the max
    
    # Pair constraints: N*(N-1)/2 values
    # Wall constraints: N*3 values (actually N*3, but we take min over all)
    
    # For efficiency, let's just compute the raw sum if beta is small, or use log-sum-exp properly.
    # Given N=100, number of constraints is ~5000+300 = 5300.
    
    # Let's compute the values we want to exponentiate: -beta * constraint
    # We want to maximize the minimum constraint, so we want to maximize -log(sum(exp(-beta * constraint)))
    
    # To be numerically stable:
    # Let A_k = -beta * constraint_k
    # log_sum_exp(A) = max(A) + log(sum(exp(A - max(A))))
    
    # We can compute this for pairs and walls separately, then combine.
    # Or just concatenate.
    
    # For speed, let's try a simpler approach if beta is not too large.
    # But let's do it properly.
    
    # Pairwise part
    pair_vals = -beta * pair_terms
    max_pair = np.max(pair_vals)
    sum_pair_exp = np.sum(np.exp(pair_vals - max_pair))
    log_sum_pair = max_pair + np.log(sum_pair_exp)
    
    # Wall part
    wall_vals = -beta * wall_dists
    max_wall = np.max(wall_vals)
    sum_wall_exp = np.sum(np.exp(wall_vals - max_wall))
    log_sum_wall = max_wall + np.log(sum_wall_exp)
    
    # Combine: log(sum_pair_exp * exp(max_pair) + sum_wall_exp * exp(max_wall))
    # This is log(exp(log_sum_pair) + exp(log_sum_wall))
    # = max(log_sum_pair, log_sum_wall) + log(exp(log_sum_pair - max) + exp(log_sum_wall - max))
    
    max_log = np.max([log_sum_pair, log_sum_wall])
    combined = np.exp(log_sum_pair - max_log) + np.exp(log_sum_wall - max_log)
    log_sum_total = max_log + np.log(combined)
    
    # The soft-min is -1/beta * log_sum_total
    # Note: The true min is the minimum of the constraints.
    # The soft-min approximation is: -1/beta * log( sum(exp(-beta * c_k)) )
    # This is close to the min when beta is large.
    
    r_soft = -1.0 / beta * log_sum_total
    
    return r_soft

def _gradient_soft_min(c, beta=50.0):
    """
    Compute gradient of the soft-min surrogate.
    """
    c = np.asarray(c, dtype=np.float64)
    n = len(c)
    
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    pair_terms = dists / 2.0
    
    wall_dists = 0.5 - np.abs(c)
    
    # Gradients of the soft-min
    # d/dx_k [ -1/beta * log( sum_j exp(-beta * c_j(x)) ) ]
    # = -1/beta * [ 1/sum_j exp(-beta c_j) * sum_j exp(-beta c_j) * (-beta) * dc_j/dx_k ]
    # = sum_j [ exp(-beta c_j) / sum_l exp(-beta c_l) ] * dc_j/dx_k
    
    # So the gradient is a weighted sum of the gradients of the individual constraints.
    # The weight for constraint j is w_j = exp(-beta * c_j) / sum_l exp(-beta * c_l)
    
    # Let's compute the weights.
    # We need to be careful with numerical stability.
    
    # Pairwise weights
    pair_vals = -beta * pair_terms
    max_pair = np.max(pair_vals)
    exp_pair = np.exp(pair_vals - max_pair)
    sum_exp_pair = np.sum(exp_pair)
    # Weights for pairs: w_ij = exp_pair[i,j] / (sum_exp_pair + sum_exp_wall * exp(max_wall - max_pair))
    # This is getting complicated. Let's just compute the total sum.
    
    # Wall weights
    wall_vals = -beta * wall_dists
    max_wall = np.max(wall_vals)
    exp_wall = np.exp(wall_vals - max_wall)
    sum_exp_wall = np.sum(exp_wall)
    
    # To combine, we need to scale them to a common max.
    max_total = np.max([max_pair, max_wall])
    scale_pair = np.exp(max_pair - max_total)
    scale_wall = np.exp(max_wall - max_total)
    
    total_sum = scale_pair * sum_exp_pair + scale_wall * sum_exp_wall
    
    # Now compute the gradient
    grad = np.zeros_like(c)
    
    # Pairwise gradient contribution
    # dc_ij/dc_i = (c_i - c_j) / |c_i - c_j| = diff[i,j] / dists[i,j]
    # dc_ij/dc_j = (c_j - c_i) / |c_j - c_i| = -diff[i,j] / dists[i,j]
    
    # Weight for pair (i,j) is w_ij = exp_pair[i,j] * scale_pair / total_sum
    # But we need to account for the fact that each pair appears twice (i,j) and (j,i).
    # Actually, in the sum, we sum over all i!=j.
    # So the gradient contribution from pair (i,j) to c_i is w_ij * (c_i - c_j)/d_ij
    # And to c_j is w_ij * (c_j - c_i)/d_ij
    
    # Let's compute the weighted sum.
    # w_pair[i,j] = exp_pair[i,j] * scale_pair / total_sum
    w_pair = exp_pair * (scale_pair / total_sum)
    
    # Gradient from pairs:
    # grad_i += sum_j w_pair[i,j] * diff[i,j] / dists[i,j]
    # Note: diff[i,j] / dists[i,j] is the unit vector from j to i.
    
    # To vectorize:
    # unit_diff = diff / dists[:,:,None]
    # grad_pair = sum_j w_pair[i,j] * unit_diff[i,j]
    
    unit_diff = np.zeros_like(diff)
    mask = dists > 1e-8
    unit_diff[mask] = diff[mask] / dists[mask, np.newaxis]
    
    # w_pair has shape (n, n)
    # grad_pair[i] = sum_j w_pair[i,j] * unit_diff[i,j]
    grad_pair = np.sum(w_pair[:, :, np.newaxis] * unit_diff, axis=1)
    
    # Wall gradient contribution
    # c_wall_i = 0.5 - |c_i|
    # dc_wall_i / dc_i = -sign(c_i)
    # w_wall[i] = exp_wall[i] * scale_wall / total_sum
    w_wall = exp_wall * (scale_wall / total_sum)
    
    # grad_wall_i = w_wall[i] * (-sign(c_i))
    grad_wall = -w_wall * np.sign(c)
    
    # Total gradient
    grad = grad_pair + grad_wall
    
    return grad

def _optimize_gradient(c_init, time_limit, beta_start=50.0, beta_end=500.0):
    """
    Optimize using gradient ascent on the soft-min surrogate.
    """
    c = c_init.copy()
    n = len(c)
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    start_time = time.time()
    
    # Learning rate
    lr = 0.001
    lr_decay = 0.999
    
    # Beta schedule
    # Start with small beta (smoother), end with large beta (sharper)
    
    while time.time() - start_time < time_limit:
        # Compute current beta (linear interpolation based on time)
        elapsed = time.time() - start_time
        progress = min(1.0, elapsed / time_limit)
        beta = beta_start + (beta_end - beta_start) * progress
        
        # Compute gradient
        grad = _gradient_soft_min(c, beta)
        
        # Gradient ascent
        c_new = c + lr * grad
        
        # Clip to cube
        c_new = np.clip(c_new, -0.5, 0.5)
        
        # Evaluate
        r_new = _compute_radius(c_new)
        
        if r_new > best_r:
            c = c_new
            best_c = c_new.copy()
            best_r = r_new
            lr *= lr_decay
        else:
            # If no improvement, reduce learning rate
            lr *= 0.99
            # Maybe also try a small random perturbation
            if lr < 1e-6:
                break
        
    return best_c, best_r

def _optimize_force_directed(c_init, time_limit):
    """
    Vectorized Force-Directed Method with Simulated Annealing.
    """
    c = c_init.copy()
    n = len(c)
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    start_time = time.time()
    
    step_size = 0.01
    temperature = 0.01
    step_decay = 0.995
    temp_decay = 0.998
    
    while time.time() - start_time < time_limit:
        
        # 1. Compute Forces
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        dists_safe = np.where(dists > 1e-8, dists, 1e-8)
        
        inv_dists2 = 1.0 / (dists_safe * dists_safe)
        inv_dists2[np.diag_indices(n)] = 0.0
        
        forces = np.sum(diff * inv_dists2[:, :, np.newaxis], axis=1)
        
        # 2. Wall Constraints
        wall_dist = 0.5 - np.abs(c)
        threshold = 0.1
        mask = wall_dist < threshold
        wall_force = np.where(mask, (threshold - wall_dist) * np.sign(c) * 10.0, 0.0)
        forces += wall_force
        
        # 3. Normalize Forces
        force_norms = np.linalg.norm(forces, axis=1, keepdims=True)
        max_force = 1.0
        forces = np.where(force_norms > max_force, forces / force_norms * max_force, forces)
        
        # 4. Add Noise
        noise = np.random.randn(n, 3) * temperature
        total_move = step_size * forces + noise
        
        # 5. Update Positions
        c_new = c + total_move
        
        # 6. Clip to Cube
        c_new = np.clip(c_new, -0.5, 0.5)
        
        # 7. Evaluate
        r_new = _compute_radius(c_new)
        
        if r_new > best_r:
            c = c_new
            best_c = c_new.copy()
            best_r = r_new
            step_size *= 0.99
            temperature *= 0.99
        else:
            step_size *= 0.999
            temperature *= 0.999
            
            if step_size < 1e-6 and temperature < 1e-6:
                break
                
    return best_c, best_r

def construct():
    """
    Construct centers for 100 equal spheres in unit cube.
    """
    start_time = time.time()
    
    best_c = None
    best_r = -1.0
    
    # Strategy 1: Grid initialization
    nx, ny, nz = 4, 5, 5
    ticks_x = (np.arange(nx) + 0.5) / nx - 0.5
    ticks_y = (np.arange(ny) + 0.5) / ny - 0.5
    ticks_z = (np.arange(nz) + 0.5) / nz - 0.5
    
    grid_points = np.array([(x, y, z) for z in ticks_z for y in ticks_y for x in ticks_x])
    c_grid = grid_points[:N].copy()
    
    # Run gradient optimization on grid start
    t_left = TOTAL_TIME_LIMIT - (time.time() - start_time)
    if t_left > 10:
        c_opt, r_opt = _optimize_gradient(c_grid, t_left / 2.0)
        if r_opt > best_r:
            best_r = r_opt
            best_c = c_opt.copy()
    
    # Run force-directed on grid start
    t_left = TOTAL_TIME_LIMIT - (time.time() - start_time)
    if t_left > 5:
        c_opt, r_opt = _optimize_force_directed(c_grid, t_left / 2.0)
        if r_opt > best_r:
            best_r = r_opt
            best_c = c_opt.copy()
    
    # Strategy 2: Random initialization
    c_rand = np.random.uniform(-0.4, 0.4, (N, 3))
    t_left = TOTAL_TIME_LIMIT - (time.time() - start_time)
    if t_left > 5:
        c_opt, r_opt = _optimize_gradient(c_rand, t_left / 2.0)
        if r_opt > best_r:
            best_r = r_opt
            best_c = c_opt.copy()
    
    # Strategy 3: Perturbed Grid
    c_perturbed = c_grid + np.random.randn(N, 3) * 0.05
    c_perturbed = np.clip(c_perturbed, -0.5, 0.5)
    t_left = TOTAL_TIME_LIMIT - (time.time() - start_time)
    if t_left > 5:
        c_opt, r_opt = _optimize_force_directed(c_perturbed, t_left / 2.0)
        if r_opt > best_r:
            best_r = r_opt
            best_c = c_opt.copy()
    
    # Strategy 4: Final polish on best solution
    t_left = TOTAL_TIME_LIMIT - (time.time() - start_time)
    if t_left > 2.0 and best_c is not None:
        c_opt, r_opt = _optimize_gradient(best_c, t_left - 0.5)
        if r_opt > best_r:
            best_r = r_opt
            best_c = c_opt.copy()
    
    if best_c is None:
        return c_grid
        
    return best_c
