import numpy as np
import time
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START
def _compute_radius(c):
    """Compute the common radius for centres c (N,3) in cube [-0.5,0.5]^3."""
    if len(c) == 0:
        return 0.0
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    # Wall distances
    wall_dists = 0.5 - np.abs(c)
    min_wall = np.min(wall_dists)
    
    return min(min_pair / 2.0, min_wall)

def _soft_min_objective_and_grad(c, beta):
    """
    Compute the soft-min objective and its gradient.
    Objective: (1/beta) * log(sum(exp(-beta * d)))
    This approximates -min(d) as beta -> inf.
    We want to minimize this, which maximizes min(d).
    """
    c = np.asarray(c).reshape(-1, 3)
    n = c.shape[0]
    
    # Calculate all relevant distances
    # Pairwise
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, np.inf)
    
    # We only need i < j to avoid double counting
    i_idx, j_idx = np.triu_indices(n, k=1)
    pair_dists = dists[i_idx, j_idx]
    
    # Wall distances
    # For each point i and coordinate k, d = 0.5 - |c_ik|
    # There are n*3 such distances
    wall_dists = 0.5 - np.abs(c) # shape (n, 3)
    wall_dists_flat = wall_dists.flatten()
    
    # Combine all distances
    # d is a vector of length M = n(n-1)/2 + 3n
    all_dists = np.concatenate([pair_dists, wall_dists_flat])
    
    # For numerical stability, subtract max d before exp
    d_max = np.max(all_dists)
    exp_terms = np.exp(-beta * (all_dists - d_max))
    
    S = np.sum(exp_terms)
    if S == 0:
        S = 1e-300
        
    obj = (1.0 / beta) * (np.log(S) - beta * d_max)
    
    # Probabilities p_d = exp(-beta * d) / S
    p_d = exp_terms / S
    
    # Gradient calculation
    grad = np.zeros_like(c)
    
    # Pairwise contribution
    # d_ij = 0.5 * ||c_i - c_j||
    # dd_ij/dc_i = (c_i - c_j) / (2 * ||c_i - c_j||)
    # dd_ij/dc_j = (c_j - c_i) / (2 * ||c_i - c_j||)
    # Note: grad_obj = - sum p_d * dd/dc
    # So grad_c_i += - p_ij * (c_i - c_j) / (2 * dist)
    
    pair_diff = c[i_idx] - c[j_idx]
    pair_norms = np.linalg.norm(pair_diff, axis=1)
    # Avoid division by zero
    pair_norms_safe = np.where(pair_norms < 1e-10, 1e-10, pair_norms)
    
    # Gradient factor for each pair
    # g_factor = - p_ij / (2 * pair_norm)
    g_factors = -p_d[:len(pair_dists)] / (2.0 * pair_norms_safe)
    
    # Accumulate gradients
    # c_i gets g_factor * (c_i - c_j)
    # c_j gets g_factor * (c_j - c_i) = -g_factor * (c_i - c_j)
    
    # Using np.add.at for accumulation
    np.add.at(grad, i_idx, g_factors[:, np.newaxis] * pair_diff)
    np.add.at(grad, j_idx, -g_factors[:, np.newaxis] * pair_diff)
    
    # Wall contribution
    # d_ik = 0.5 - |c_ik|
    # dd_ik/dc_ik = -sign(c_ik)
    # grad_obj += - p_ik * (-sign(c_ik)) = p_ik * sign(c_ik)
    
    signs = np.sign(c)
    wall_grads = p_d[len(pair_dists):].reshape(n, 3) * signs
    
    grad += wall_grads
    
    return obj, grad

def _initial_grid():
    """Create a 5x5x2 grid initialization."""
    m = 5
    ticks_xy = (np.arange(m) + 0.5) / m - 0.5
    ticks_z = np.array([-0.25, 0.25])
    
    grid = []
    for z in ticks_z:
        for y in ticks_xy:
            for x in ticks_xy:
                grid.append([x, y, z])
    return np.array(grid)

def _refine(c_init, time_limit, seed=0):
    """
    Run L-BFGS-B optimization with beta annealing.
    """
    c = c_init.copy()
    rng = np.random.RandomState(seed)
    start_time = time.time()
    
    # Beta schedule: start low, increase to sharpen the minimum
    betas = [10, 50, 100, 200, 500, 1000]
    current_c = c.copy()
    
    for beta in betas:
        if time.time() - start_time > time_limit:
            break
            
        # Define objective and gradient functions for scipy
        def fun(x):
            c_local = x.reshape(-1, 3)
            obj, grad = _soft_min_objective_and_grad(c_local, beta)
            return obj, grad.flatten()
            
        # Initial gradient
        _, g0 = _soft_min_objective_and_grad(current_c, beta)
        
        # Run optimizer
        try:
            res = minimize(fun, current_c.flatten(), method='L-BFGS-B', jac=True,
                           options={'maxiter': 1000, 'maxfun': 1000, 'disp': False, 'maxls': 50})
            if res.success:
                current_c = res.x.reshape(-1, 3)
        except Exception:
            pass
            
        # Check time
        if time.time() - start_time > time_limit:
            break
            
    return current_c

def construct():
    start_time = time.time()
    deadline = 140.0 # Safety margin
    
    # Initial configuration
    c0 = _initial_grid()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    
    # Perform multiple restarts
    # We'll try to do as many restarts as time allows.
    # Each restart takes some time. Let's estimate ~10-20s per restart.
    # So we can do ~5-10 restarts.
    
    seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    
    for seed in seeds:
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining < 5.0: # Need at least 5s for a decent run
            break
            
        # Create perturbed initial configuration
        # Small perturbation around the grid
        rng = np.random.RandomState(seed)
        noise_scale = 0.05
        c_init = c0 + rng.normal(0, noise_scale, c0.shape)
        # Ensure inside bounds
        c_init = np.clip(c_init, -0.49, 0.49)
        
        # Run refinement
        c_refined = _refine(c_init, min(remaining - 1.0, 20.0), seed=seed)
        
        # Evaluate
        r_refined = _compute_radius(c_refined)
        
        if r_refined > best_r:
            best_r = r_refined
            best_c = c_refined.copy()
            
        # Optional: Use the best found as a new seed for next iteration?
        # Or stick to grid-based perturbations for diversity?
        # Let's mix: 50% from grid, 50% from best found
        if seed % 2 == 0:
             c0 = best_c.copy() # Update base for next perturbations if it's a good one
             # But maybe we want to keep the original grid for some diversity?
             # Let's just use best_c for even seeds, original grid for odd seeds.
             # Actually, let's just keep using the original grid for perturbations to ensure diversity,
             # but if we find a significantly better one, we can perturb it too.
             # For simplicity, let's just perturb the original grid every time, 
             # but with different noise scales.
    
    # Final check
    if not np.all(np.isfinite(best_c)):
        return c0
        
    # Ensure strictly inside
    best_c = np.clip(best_c, -0.49999, 0.49999)
    
    return best_c
# EVOLVE-BLOCK-END
