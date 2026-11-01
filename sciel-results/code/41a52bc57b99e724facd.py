"""Improved packing of 50 equal spheres in a unit cube using smooth surrogate optimization."""
import numpy as np
import time

N = 50
TIME_LIMIT = 140.0  # Safety margin before 150s kill

def _compute_radius(centers):
    """Compute the minimum radius for non-overlapping spheres inside [-0.5, 0.5]^3."""
    n = len(centers)
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Wall distances
    wall_dists = 0.5 - np.abs(centers)
    min_wall_dist = np.min(wall_dists)
    
    r = min(min_pair_dist / 2.0, min_wall_dist)
    return r

def _soft_min_radius_and_grad(centers, beta):
    """
    Compute the soft-min radius and its gradient.
    r_soft = -1/beta * log( sum(exp(-beta * constraints)) )
    constraints: d_ij/2 for pairs, w_i for walls.
    """
    n = len(centers)
    # 1. Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    # We only need upper triangle
    iu = np.triu_indices(n, k=1)
    pair_dists = dists[iu] / 2.0
    
    # Avoid division by zero in gradient calculation
    # If dist is 0, gradient is undefined, but exp(-beta*0) is 1.
    # We handle this by setting gradient to 0 if dist is very small.
    pair_grads = np.zeros_like(centers) # Accumulate gradients for centers
    pair_weights = np.exp(-beta * pair_dists)
    
    # Gradient of d_ij/2 w.r.t c_i is (c_i - c_j) / d_ij
    # We accumulate contributions to each center
    # For each pair (i, j):
    # grad_c_i += weight * (c_i - c_j) / d_ij
    # grad_c_j += weight * (c_j - c_i) / d_ij
    
    idx_i, idx_j = iu
    vecs = centers[idx_i] - centers[idx_j]
    dists_full = np.sqrt(np.sum(vecs**2, axis=1))
    
    # Safe division
    inv_dists = np.zeros_like(dists_full)
    mask = dists_full > 1e-10
    inv_dists[mask] = 1.0 / dists_full[mask]
    
    grad_vecs = vecs * inv_dists[:, np.newaxis]
    
    # Accumulate using np.add.at
    np.add.at(pair_grads, idx_i, pair_weights[:, np.newaxis] * grad_vecs)
    np.add.at(pair_grads, idx_j, -pair_weights[:, np.newaxis] * grad_vecs)
    
    # 2. Wall distances
    # w_i = min_k (0.5 - |c_ik|)
    # We treat each coordinate's distance to the nearest wall as a constraint?
    # Actually, the constraint is min_k (0.5 - |c_ik|).
    # To make it smooth, we can use a soft-min over the 3 coordinates for each sphere?
    # Or simply treat the 3 coordinate-wise distances as separate constraints?
    # If we treat w_i = min(0.5-|x|, 0.5-|y|, 0.5-|z|), the soft-min over coordinates is better.
    
    # Let's define constraints for walls as:
    # For each sphere i, and each axis k, constraint is 0.5 - |c_ik|.
    # But the radius is limited by the MIN of these.
    # So we add all 3*N wall constraints to the soft-min sum.
    
    wall_dists = 0.5 - np.abs(centers) # Shape (N, 3)
    wall_weights = np.exp(-beta * wall_dists)
    
    # Gradient of (0.5 - |c_ik|) w.r.t c_ik is -sign(c_ik)
    wall_grads = -np.sign(centers)
    
    # Accumulate wall gradients
    # pair_grads currently holds pair contributions. Let's create a total grad array.
    total_grads = pair_grads.copy()
    
    # Add wall contributions
    # For each sphere i, and axis k:
    # total_grads[i, k] += wall_weights[i, k] * wall_grads[i, k]
    total_grads += wall_weights * wall_grads
    
    # 3. Compute Soft Min Value
    # Sum of all exponentials
    sum_exp = np.sum(pair_weights) + np.sum(wall_weights)
    if sum_exp == 0:
        return 0.0, np.zeros_like(centers)
    
    r_soft = -1.0 / beta * np.log(sum_exp)
    
    # 4. Compute Gradient
    # grad_r_soft = (1/sum_exp) * ( sum(pair_weights * grad_pair) + sum(wall_weights * grad_wall) )
    # Note: The gradients accumulated in total_grads are already weighted by exp(-beta*constraint)
    # Wait, the formula is:
    # d/dx (-1/beta log(sum e^{-beta x_k})) = (1/sum e^{-beta x_k}) * sum(e^{-beta x_k} * dx_k/dx)
    # So we need to divide the accumulated weighted gradients by the sum of weights.
    
    grad_r_soft = total_grads / sum_exp
    
    return r_soft, grad_r_soft

def _initial_configuration_5x5x2():
    """Generate a 5x5x2 grid configuration."""
    xy_ticks = (np.arange(5) + 0.5) / 5.0 - 0.5
    z_ticks = np.array([-0.25, 0.25])
    
    centers = []
    for z in z_ticks:
        for y in xy_ticks:
            for x in xy_ticks:
                centers.append([x, y, z])
    
    return np.array(centers)

def _optimize_smooth(centers, beta_start=10.0, beta_end=200.0, steps=200, lr=0.01, deadline=None):
    """
    Optimize centers using gradient ascent on the soft-min radius.
    Anneals beta from beta_start to beta_end.
    """
    centers = centers.copy()
    n_steps = steps
    
    # Schedule betas
    betas = np.linspace(beta_start, beta_end, n_steps)
    
    for step in range(n_steps):
        if deadline and time.time() > deadline:
            break
            
        beta = betas[step]
        r, grad = _soft_min_radius_and_grad(centers, beta)
        
        # Gradient Ascent Step
        # Adaptive learning rate?
        # Simple fixed with decay
        current_lr = lr * (1.0 - step / n_steps) + 0.001
        
        centers += current_lr * grad
        
        # Project to box
        centers = np.clip(centers, -0.5, 0.5)
        
        # Optional: Small random jitter to escape flat regions?
        # if step % 10 == 0:
        #     centers += np.random.normal(0, 0.001, centers.shape)
        #     centers = np.clip(centers, -0.5, 0.5)
        
    return centers

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_centers = _initial_configuration_5x5x2()
    best_radius = _compute_radius(best_centers)
    
    rng = np.random.default_rng(42)
    
    # Number of restarts depends on available time
    # Estimate time per restart: ~1-2 seconds for 200 steps?
    # Let's do a quick test or just allocate a budget.
    # 50 spheres, 1225 pairs. Gradient calc is O(N^2).
    # 200 steps * 50 restarts = 10000 gradient calcs.
    # Each gradient calc is fast in NumPy.
    
    max_restarts = 100
    restarts_done = 0
    
    for restart in range(max_restarts):
        if time.time() > deadline - 5.0: # Leave 5s for final check
            break
            
        # Initialize
        if restart == 0:
            centers = _initial_configuration_5x5x2()
        else:
            # Jitter the best known solution or the grid?
            # Jittering the best known solution allows local improvements.
            # Jittering the grid allows finding different basins.
            if restart % 2 == 0:
                centers = _initial_configuration_5x5x2().copy()
                centers += rng.normal(0, 0.02, size=centers.shape)
            else:
                centers = best_centers.copy()
                centers += rng.normal(0, 0.01, size=centers.shape)
            
            centers = np.clip(centers, -0.49, 0.49)
        
        # Optimize
        # Use annealing: start with low beta (smooth), end with high beta (sharp)
        # More steps for later restarts if time permits? No, keep consistent.
        centers_opt = _optimize_smooth(centers, beta_start=5.0, beta_end=100.0, steps=300, lr=0.05, deadline=deadline)
        
        # Final polish with very high beta to sharpen
        # Only if time permits
        if time.time() < deadline - 2.0:
            centers_opt = _optimize_smooth(centers_opt, beta_start=100.0, beta_end=500.0, steps=100, lr=0.01, deadline=deadline)
        
        radius = _compute_radius(centers_opt)
        
        if radius > best_radius:
            best_radius = radius
            best_centers = centers_opt.copy()
        
        restarts_done += 1
        
        # Break early if we hit a very high score?
        # Best known is 0.13595...
        # If we are close, maybe do more restarts with smaller jitters.
    
    # Final verification
    final_radius = _compute_radius(best_centers)
    if final_radius < 0.1: # Sanity check
        best_centers = _initial_configuration_5x5x2()
    
    return best_centers

if __name__ == "__main__":
    centers = construct()
    r = _compute_radius(centers)
    print(f"Radius: {r}")
