"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize as opt


def _radius(centers):
    """Compute the maximum common radius for given centers (vectorized)."""
    n = len(centers)
    if n < 2:
        wall = 0.5 - np.max(np.abs(centers), axis=1)
        return np.min(wall)
    # Pairwise distances (upper triangle)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    d2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(d2, np.inf)
    min_pair_dist = np.sqrt(np.min(d2))
    # Distance to walls
    wall = 0.5 - np.max(np.abs(centers), axis=1)
    min_wall = np.min(wall)
    return min(min_pair_dist / 2.0, min_wall)


def _soft_radius(centers, beta):
    """
    Compute a smooth approximation of the radius using log-sum-exp.
    We want to maximize min(pairwise_dist/2, wall_dist).
    Equivalent to minimizing -min(...).
    -min(x) = logsumexp(-x) / beta  (as beta -> inf)
    Here x_i are the 'constraints' (distances).
    We define the 'energy' as the soft-min of the distances.
    soft_min(d) = -(1/beta) * log(sum(exp(-beta * d_i)))
    We want to maximize soft_min(d).
    So we minimize -soft_min(d) = (1/beta) * log(sum(exp(-beta * d_i))).
    
    The distances d are:
    1. d_ij = |c_i - c_j| / 2 for i < j
    2. d_ik = 0.5 - |c_ik| for each point i and coord k (distance to wall)
    
    Note: The score is min( min_{i<j} |c_i-c_j|/2, min_i min_k (0.5 - |c_ik|) ).
    So we have a set of distances D. We want max(min(D)).
    Let f(centers) = soft_min(D, beta).
    We minimize -f(centers).
    """
    n = len(centers)
    
    # 1. Pairwise distances / 2
    # diff shape (n, n, 3)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    d2 = np.sum(diff * diff, axis=2)
    # Mask diagonal
    mask = ~np.eye(n, dtype=bool)
    # We only need upper triangle to avoid double counting, but for logsumexp it doesn't matter much 
    # except for the magnitude of the sum. To be consistent with min, we should consider each pair once.
    # Let's extract upper triangle indices
    iu = np.triu_indices(n, k=1)
    d_pair = np.sqrt(d2[iu]) / 2.0
    
    # 2. Wall distances
    # For each point i, distance to each of the 6 walls is 0.5 - |c_ik|
    # But the constraint is min_k (0.5 - |c_ik|).
    # So for point i, the distance to the boundary is min(0.5-c_ix, 0.5+c_ix, ...)
    # Which is 0.5 - max(|c_ix|, |c_iy|, |c_iz|).
    # Let's compute this.
    abs_c = np.abs(centers)
    max_abs = np.max(abs_c, axis=1)
    d_wall = 0.5 - max_abs
    
    # Combine all distances
    # We want soft_min of the union of d_pair and d_wall.
    # soft_min = -1/beta * log( sum(exp(-beta * d)) )
    # To avoid overflow/underflow, we use the log-sum-exp trick.
    # Let d_min = min(all_d).
    # sum(exp(-beta * d)) = exp(-beta * d_min) * sum(exp(-beta * (d - d_min)))
    # log(...) = -beta * d_min + log(sum(exp(-beta * (d - d_min))))
    
    all_d = np.concatenate([d_pair, d_wall])
    
    if len(all_d) == 0:
        return 0.0
        
    d_min = np.min(all_d)
    # Subtract d_min to stabilize
    shifted = all_d - d_min
    # exp(-beta * shifted)
    # If beta is large, exp(-beta * shifted) can underflow for large shifted, which is good (becomes 0).
    # But if shifted is 0, it's 1.
    exp_term = np.exp(-beta * shifted)
    log_sum = np.log(np.sum(exp_term))
    
    soft_min_val = -1.0 / beta * (log_sum - beta * d_min)
    # Wait:
    # log(sum(exp(-beta d))) = log(exp(-beta d_min) sum(exp(-beta (d-d_min))))
    # = -beta d_min + log(sum(exp(-beta (d-d_min))))
    # soft_min = -1/beta * ( -beta d_min + log_sum_shifted )
    # = d_min - 1/beta * log_sum_shifted
    
    return d_min - (1.0 / beta) * log_sum


def _negative_soft_radius(centers, beta):
    """Objective to minimize: negative soft radius."""
    return -_soft_radius(centers, beta)


def _grad_soft_radius(centers, beta):
    """
    Compute gradient of the soft radius w.r.t. centers.
    We want d(-soft_radius)/d(centers) for minimization, or d(soft_radius)/d(centers) for maximization.
    Let's compute gradient of soft_radius to maximize it.
    
    soft_r = d_min - (1/beta) * log(sum(exp(-beta * (d_k - d_min))))
    This is tricky because d_min changes with centers.
    
    Alternative formulation:
    soft_r = -1/beta * log( sum_k exp(-beta * d_k) )
    Let S = sum_k exp(-beta * d_k)
    soft_r = -1/beta * log(S)
    d(soft_r)/d(x) = -1/beta * (1/S) * dS/dx
    dS/dx = sum_k exp(-beta * d_k) * (-beta) * d(d_k)/dx
    d(soft_r)/d(x) = -1/beta * (1/S) * (-beta) * sum_k exp(-beta * d_k) * d(d_k)/dx
    d(soft_r)/d(x) = (1/S) * sum_k exp(-beta * d_k) * d(d_k)/dx
    Let w_k = exp(-beta * d_k) / S. These are the soft-min weights, sum to 1.
    d(soft_r)/d(x) = sum_k w_k * d(d_k)/dx
    
    So we just need the gradients of the distances d_k.
    
    d_k are:
    1. Pairwise: d_ij = 0.5 * |c_i - c_j|
       d(d_ij)/d(c_i) = 0.5 * (c_i - c_j) / |c_i - c_j|
       d(d_ij)/d(c_j) = 0.5 * (c_j - c_i) / |c_i - c_j|
       
    2. Wall: d_ik = 0.5 - max(|c_ix|, |c_iy|, |c_iz|)
       Let m_i = max(|c_ix|, |c_iy|, |c_iz|)
       d(d_ik)/d(c_i) = - d(m_i)/d(c_i)
       d(m_i)/d(c_i) is the sign of the coordinate that is max.
       If |c_ix| is the unique max, d(m_i)/d(c_ix) = sign(c_ix), others 0.
       If there are ties, we can average or pick one. For smoothness, we can use a soft-max for the max as well?
       Or just use the hard max gradient (subgradient).
       Since we are using soft-min on the outside, the inner hard max might still be non-smooth.
       However, the set of ties has measure zero.
       
    Let's implement this.
    """
    n = len(centers)
    grad = np.zeros_like(centers)
    
    # 1. Pairwise contributions
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    d2 = np.sum(diff * diff, axis=2)
    # Avoid division by zero
    d2_safe = np.where(d2 < 1e-12, 1e-12, d2)
    dist = np.sqrt(d2_safe)
    # Unit vectors
    # u_ij = (c_i - c_j) / |c_i - c_j|
    # Only for i < j
    iu = np.triu_indices(n, k=1)
    d_pair = 0.5 * dist[iu]
    
    # Weights for soft min
    # We need the weights w_k for ALL distances (pairwise and wall).
    # Let's compute all distances and weights first.
    
    # Pairwise distances
    d_pair_vals = 0.5 * dist[iu]
    
    # Wall distances
    abs_c = np.abs(centers)
    max_abs = np.max(abs_c, axis=1)
    d_wall_vals = 0.5 - max_abs
    
    all_d = np.concatenate([d_pair_vals, d_wall_vals])
    
    # Compute weights
    # w_k = exp(-beta * d_k) / sum(exp(-beta * d_j))
    # Use log-sum-exp trick
    d_min = np.min(all_d)
    shifted = all_d - d_min
    exp_term = np.exp(-beta * shifted)
    sum_exp = np.sum(exp_term)
    weights = exp_term / sum_exp
    
    # Split weights
    w_pair = weights[:len(d_pair_vals)]
    w_wall = weights[len(d_pair_vals):]
    
    # Accumulate gradients for pairwise
    # d(d_ij)/d(c_i) = 0.5 * u_ij
    # d(d_ij)/d(c_j) = -0.5 * u_ij
    # u_ij = (c_i - c_j) / dist_ij
    # dist is stored in dist[iu]
    # diff[iu] gives c_i - c_j for i<j
    
    # We need to scatter the gradients.
    # For each pair (i, j):
    # grad[i] += w_pair[k] * 0.5 * (c_i - c_j) / dist
    # grad[j] -= w_pair[k] * 0.5 * (c_i - c_j) / dist
    
    # Vectorized approach:
    # diff_iu = diff[iu] shape (num_pairs, 3)
    # dist_iu = dist[iu] shape (num_pairs,)
    # u_iu = diff_iu / dist_iu[:, np.newaxis]
    
    diff_iu = diff[iu]
    dist_iu = dist[iu]
    u_iu = diff_iu / dist_iu[:, np.newaxis]
    
    # Weighted unit vectors
    w_u = u_iu * (w_pair[:, np.newaxis] * 0.5)
    
    # Scatter to grad
    # indices i and j
    i_idx, j_idx = iu
    # grad[i] += w_u
    # np.add.at(grad, i_idx, w_u)
    # grad[j] -= w_u
    # np.add.at(grad, j_idx, -w_u)
    
    np.add.at(grad, i_idx, w_u)
    np.add.at(grad, j_idx, -w_u)
    
    # Accumulate gradients for wall
    # d(d_wall_i)/d(c_i) = - d(max_abs_i)/d(c_i)
    # d(max_abs_i)/d(c_i) is sign of the coord with max abs value.
    # Let's find the index of the max abs value for each point.
    # If there are ties, we can average the signs? Or just pick the first one?
    # For smoothness, let's use a soft-max on the abs values?
    # max_abs = soft_max(abs_c, beta_inner) ?
    # But the task says "soft-min of pairwise and boundary distances".
    # The boundary distance is min_k (0.5 - |c_ik|).
    # This is 0.5 - max_k |c_ik|.
    # So the non-smoothness is in the max.
    # Let's approximate max with soft-max.
    # max(x) ≈ (1/beta_inner) log(sum(exp(beta_inner * x)))
    # Let's use beta_inner = beta (same scale).
    
    # Recalculate wall distance with soft-max for smoothness?
    # If we use soft-max for max_abs, then d_wall = 0.5 - soft_max(abs_c).
    # Then the gradient is well-defined everywhere.
    
    # Let's re-evaluate the soft radius using this smoother definition.
    # This changes the objective slightly, but makes it fully differentiable.
    # Is it worth it?
    # The hard max gradient is 0 almost everywhere except at ties.
    # L-BFGS-B might handle the subgradient okay, but smooth is better.
    
    # Let's implement the soft-max version for the wall distance gradient.
    # max_abs_soft = (1/beta) * log(sum(exp(beta * abs_c)))
    # d(max_abs_soft)/d(c_i_k) = (1/beta) * (1/sum) * exp(beta * abs_c_ik) * beta * sign(c_ik)
    # = (exp(beta * abs_c_ik) / sum(exp(beta * abs_c))) * sign(c_ik)
    
    # Let's compute these weights.
    # Note: This changes the value of d_wall used in the soft-min.
    # To be consistent, we should use the soft-max value for d_wall in the soft-min calculation too.
    # Let's restart the gradient calculation with this assumption.
    
    # --- Revised Gradient Calculation ---
    
    # 1. Soft-max for abs values
    # abs_c shape (n, 3)
    # To avoid overflow, subtract max
    max_abs_hard = np.max(abs_c, axis=1, keepdims=True)
    exp_abs = np.exp(beta * (abs_c - max_abs_hard))
    sum_exp_abs = np.sum(exp_abs, axis=1, keepdims=True)
    # Weights for soft-max
    w_abs = exp_abs / sum_exp_abs
    # soft_max_val = max_abs_hard + (1/beta) * log(sum_exp_abs)
    # But we only need the gradient of the max.
    # grad_max_abs = w_abs * sign(c)
    sign_c = np.sign(centers)
    grad_max_abs = w_abs * sign_c
    
    # d(d_wall)/d(c) = -grad_max_abs
    # So contribution to grad(soft_r) from wall is:
    # sum_i w_wall_i * (-grad_max_abs_i)
    
    # We need to re-compute the soft-min weights using the SOFT wall distances?
    # Yes, for consistency.
    # d_wall_soft = 0.5 - (max_abs_hard + (1/beta) * log(sum_exp_abs))
    # Note: max_abs_hard is per row.
    # log_sum_abs = np.log(sum_exp_abs)
    # d_wall_soft = 0.5 - max_abs_hard[:, 0] - (1/beta) * log_sum_abs[:, 0]
    
    # Let's recompute all_d with soft wall distances.
    # d_pair_vals is unchanged (pairwise distance is smooth except at 0, but we assume non-zero).
    # d_wall_soft shape (n,)
    
    # We need to recompute the soft-min weights based on the new d_wall_soft.
    # This requires recalculating d_min, shifted, etc.
    
    # This is getting complex. Let's simplify.
    # The difference between hard max and soft max is small if beta is large.
    # For the purpose of the gradient, using the hard max subgradient is often sufficient
    # if we assume the optimizer doesn't get stuck exactly on a tie.
    # However, to be robust, let's stick to the hard max subgradient for now,
    # but handle ties by averaging.
    
    # Actually, the previous calculation of w_wall used d_wall_vals from hard max.
    # And I calculated grad_max_abs using hard max? No, I used a soft-max formula in the thought process
    # but the code above `w_abs` uses soft-max.
    # Let's use the soft-max gradient for the wall part, but keep the soft-min value
    # based on hard-max wall distances? That's inconsistent.
    
    # Let's do it properly:
    # 1. Compute soft wall distances d_wall_soft.
    # 2. Compute soft-min weights based on d_pair and d_wall_soft.
    # 3. Compute gradient using these weights and the gradients of d_pair and d_wall_soft.
    
    # Step 1: Soft wall distances
    # We already have w_abs and sum_exp_abs.
    # log_sum_abs = np.log(sum_exp_abs)
    # d_wall_soft = 0.5 - (max_abs_hard[:, 0] + (1/beta) * log_sum_abs[:, 0])
    
    # Let's recompute d_wall_vals to be d_wall_soft.
    log_sum_abs = np.log(sum_exp_abs)
    d_wall_vals = 0.5 - (max_abs_hard[:, 0] + (1.0 / beta) * log_sum_abs[:, 0])
    
    # Step 2: Recompute soft-min weights
    all_d = np.concatenate([d_pair_vals, d_wall_vals])
    d_min = np.min(all_d)
    shifted = all_d - d_min
    exp_term = np.exp(-beta * shifted)
    sum_exp = np.sum(exp_term)
    weights = exp_term / sum_exp
    
    w_pair = weights[:len(d_pair_vals)]
    w_wall = weights[len(d_pair_vals):]
    
    # Step 3: Gradient
    grad = np.zeros_like(centers)
    
    # Pairwise part (same as before)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    d2 = np.sum(diff * diff, axis=2)
    d2_safe = np.where(d2 < 1e-12, 1e-12, d2)
    dist = np.sqrt(d2_safe)
    iu = np.triu_indices(n, k=1)
    diff_iu = diff[iu]
    dist_iu = dist[iu]
    u_iu = diff_iu / dist_iu[:, np.newaxis]
    w_u = u_iu * (w_pair[:, np.newaxis] * 0.5)
    i_idx, j_idx = iu
    np.add.at(grad, i_idx, w_u)
    np.add.at(grad, j_idx, -w_u)
    
    # Wall part
    # d(soft_r)/d(c_i) += w_wall_i * d(d_wall_soft_i)/d(c_i)
    # d(d_wall_soft_i)/d(c_i) = - grad_max_abs_soft_i
    # grad_max_abs_soft_i = w_abs_i * sign(c_i)
    # So contribution is -w_wall_i * w_abs_i * sign(c_i)
    
    # w_wall shape (n,)
    # w_abs shape (n, 3)
    # sign_c shape (n, 3)
    
    wall_grad = -w_wall[:, np.newaxis] * w_abs * sign_c
    grad += wall_grad
    
    return grad


def _soft_min_radius(centers, beta):
    """
    Compute the soft-min radius value for optimization.
    This should match the value used in the gradient calculation.
    """
    n = len(centers)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    d2 = np.sum(diff * diff, axis=2)
    d2_safe = np.where(d2 < 1e-12, 1e-12, d2)
    dist = np.sqrt(d2_safe)
    iu = np.triu_indices(n, k=1)
    d_pair_vals = 0.5 * dist[iu]
    
    # Wall distances (soft)
    abs_c = np.abs(centers)
    max_abs_hard = np.max(abs_c, axis=1, keepdims=True)
    exp_abs = np.exp(beta * (abs_c - max_abs_hard))
    sum_exp_abs = np.sum(exp_abs, axis=1, keepdims=True)
    log_sum_abs = np.log(sum_exp_abs)
    d_wall_vals = 0.5 - (max_abs_hard[:, 0] + (1.0 / beta) * log_sum_abs[:, 0])
    
    all_d = np.concatenate([d_pair_vals, d_wall_vals])
    
    if len(all_d) == 0:
        return 0.0
        
    d_min = np.min(all_d)
    shifted = all_d - d_min
    exp_term = np.exp(-beta * shifted)
    log_sum = np.log(np.sum(exp_term))
    
    soft_min_val = d_min - (1.0 / beta) * log_sum
    
    return soft_min_val


def _objective(x, beta):
    """Objective function for L-BFGS-B."""
    centers = x.reshape(N, 3)
    # Penalty for going out of bounds
    # L-BFGS-B handles bounds, so we don't need a penalty if we use bounds.
    # But let's add a small penalty just in case or to push away from walls if needed?
    # No, the wall distance is part of the objective.
    # If a point is outside, the wall distance becomes negative, which is bad.
    # L-BFGS-B with bounds [-0.5, 0.5] should keep them inside.
    return -_soft_min_radius(centers, beta)


def _gradient(x, beta):
    """Gradient function for L-BFGS-B."""
    centers = x.reshape(N, 3)
    g = -_grad_soft_radius(centers, beta)
    return g.flatten()


def _bcc_init(n_points):
    """Generate initial points on a distorted BCC lattice."""
    points = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                ox = -0.5 + i * (1.0/3.0)
                oy = -0.5 + j * (1.0/3.0)
                oz = -0.5 + k * (1.0/3.0)
                
                points.append([ox, oy, oz])
                points.append([ox + 1.0/6.0, oy + 1.0/6.0, oz + 1.0/6.0])
                
    points = np.array(points)
    return points[:n_points].copy()


def _fcc_init(n_points):
    """Generate initial points on an FCC-like lattice."""
    # FCC lattice points
    # Corners and face centers.
    # Let's just use a grid and add noise, or a specific pattern.
    # For now, let's use a simple grid initialization as a fallback.
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:n_points].copy()


def _optimize_smooth(centers, time_budget):
    """
    Optimize using L-BFGS-B with an adaptive beta schedule.
    Start with low beta (smooth) and increase to high beta (sharp).
    """
    start = time.time()
    centers = centers.copy()
    
    # Adaptive beta schedule
    # Start beta_low, end beta_high
    beta_low = 50.0
    beta_high = 500.0
    
    # Number of phases
    n_phases = 5
    betas = np.linspace(beta_low, beta_high, n_phases)
    
    best_r = -np.inf
    best_centers = centers.copy()
    
    # We will run L-BFGS-B for each beta, using the result of the previous as start.
    # Time budget per phase? Or total budget?
    # Let's allocate time evenly, but check deadline.
    
    for i, beta in enumerate(betas):
        if time.time() - start > time_budget:
            break
            
        x0 = centers.flatten()
        bounds = [(-0.5, 0.5)] * (N * 3)
        
        # Max iterations per phase
        max_iter = 500
        
        try:
            result = opt.minimize(
                _objective, x0,
                jac=lambda x: _gradient(x, beta),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iter, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            
            if result.success or result.status == 0:
                new_centers = result.x.reshape(N, 3)
                # Ensure validity
                if np.all(np.isfinite(new_centers)) and np.all(np.abs(new_centers) <= 0.5 + 1e-9):
                    new_centers = np.clip(new_centers, -0.5, 0.5)
                    r = _radius(new_centers)
                    if r > best_r:
                        best_r = r
                        best_centers = new_centers.copy()
                    # Update centers for next phase
                    centers = new_centers.copy()
        except Exception as e:
            pass
            
    return best_centers, best_r


def _sa_optimize(centers, time_budget):
    """Simulated Annealing optimization to escape local minima."""
    start = time.time()
    centers = centers.copy()
    n = len(centers)
    current_r = _radius(centers)
    best_r = current_r
    best_centers = centers.copy()
    
    T = 0.005
    T_min = 1e-5
    step_size = 0.01
    decay = 0.9995
    
    while time.time() - start < time_budget:
        i = np.random.randint(n)
        
        delta = np.random.normal(0, step_size, 3)
        new_center = centers[i].copy()
        new_center += delta
        new_center = np.clip(new_center, -0.5, 0.5)
        
        new_centers = centers.copy()
        new_centers[i] = new_center
        new_r = _radius(new_centers)
        
        if new_r >= current_r:
            centers = new_centers
            current_r = new_r
            if current_r > best_r:
                best_r = current_r
                best_centers = centers.copy()
        else:
            if np.random.rand() < np.exp((new_r - current_r) / T):
                centers = new_centers
                current_r = new_r
                if current_r > best_r:
                    best_r = current_r
                    best_centers = centers.copy()
        
        T *= decay
        step_size *= decay
        if T < T_min:
            T = T_min
            step_size = max(step_size, 0.001)
            
    return best_centers, best_r


def _coordinate_descent(centers, time_budget):
    """Coordinate Descent polish."""
    start = time.time()
    centers = centers.copy()
    n = len(centers)
    current_r = _radius(centers)
    best_r = current_r
    best_centers = centers.copy()
    
    while time.time() - start < time_budget:
        improved = False
        for i in range(n):
            for d in range(3):
                val = centers[i, d]
                
                # Use golden section or simple grid?
                # Simple grid is fast enough for 1D.
                candidates = np.linspace(-0.5, 0.5, 50)
                best_val = val
                best_r_local = current_r
                
                for c_val in candidates:
                    new_centers = centers.copy()
                    new_centers[i, d] = c_val
                    r = _radius(new_centers)
                    if r > best_r_local:
                        best_r_local = r
                        best_val = c_val
                
                if best_val != val:
                    centers[i, d] = best_val
                    current_r = best_r_local
                    improved = True
                    if current_r > best_r:
                        best_r = current_r
                        best_centers = centers.copy()
        
        if not improved:
            break
            
    return best_centers, best_r


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    # Initial configuration: BCC lattice
    centers = _bcc_init(N)
    
    best_centers = centers.copy()
    best_r = _radius(best_centers)

    # Phase 1: Smooth Optimization with Adaptive Beta
    # This is the main improvement: using the soft-min objective with L-BFGS-B.
    if time.time() - start_time < deadline - 30:
        budget = min(60.0, deadline - 30 - (time.time() - start_time))
        sm_centers, sm_r = _optimize_smooth(centers, budget)
        if sm_r > best_r:
            best_r = sm_r
            best_centers = sm_centers.copy()

    # Phase 2: Simulated Annealing to escape any local minima found by gradient method
    if time.time() - start_time < deadline - 20:
        budget = min(30.0, deadline - 20 - (time.time() - start_time))
        sa_centers, sa_r = _sa_optimize(best_centers, budget)
        if sa_r > best_r:
            best_r = sa_r
            best_centers = sa_centers.copy()
        
        # Run smooth optimization again on the SA result?
        if time.time() - start_time < deadline - 10:
            budget2 = min(20.0, deadline - 10 - (time.time() - start_time))
            sm2_centers, sm2_r = _optimize_smooth(best_centers, budget2)
            if sm2_r > best_r:
                best_r = sm2_r
                best_centers = sm2_centers.copy()

    # Phase 3: Coordinate Descent Polish
    if time.time() - start_time < deadline - 5:
        budget = min(10.0, deadline - 5 - (time.time() - start_time))
        cd_centers, cd_r = _coordinate_descent(best_centers, budget)
        if cd_r > best_r:
            best_r = cd_r
            best_centers = cd_centers.copy()

    # Final validity check
    if not np.all(np.isfinite(best_centers)) or np.any(np.abs(best_centers) > 0.5):
        m = 4
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_centers = grid[:N].copy()

    return best_centers
# EVOLVE-BLOCK-END
