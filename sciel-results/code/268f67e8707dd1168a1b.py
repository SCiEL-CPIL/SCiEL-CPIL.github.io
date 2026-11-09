"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, LinearConstraint

def _score(centers):
    """Compute the packing radius for given centers."""
    # Check validity
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Check bounds
    if np.any(centers < -0.5) or np.any(centers > 0.5):
        return 0.0
    # Compute pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    # Mask diagonal
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist)
    # Compute boundary distances
    boundary = np.min(np.minimum(0.5 - centers, centers + 0.5), axis=1)
    min_boundary = np.min(boundary)
    r = min(min_pair / 2.0, min_boundary)
    return r

def _objective(centers_flat):
    """Objective to minimize: negative of a smoothed max-min proxy."""
    centers = centers_flat.reshape(N, 3)
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    
    # Boundary squared distances
    boundary = np.minimum(0.5 - centers, centers + 0.5)
    boundary_sq = np.sum(boundary**2, axis=1)
    
    # Use log-sum-exp to approximate the minimum
    # We want to maximize min(dist_sq, boundary_sq * 4)  [since r = min(d/2, b), so r^2 = min(d^2/4, b^2)]
    # Let's work with r^2 for smoothness: r^2 = min(d^2/4, b^2)
    pair_sq_over_4 = dist_sq / 4.0
    # Combine all constraints: we want min of all these
    all_vals = np.concatenate([pair_sq_over_4.flatten(), boundary_sq])
    
    # Log-sum-exp trick to approximate minimum
    # min(x) ≈ -log(sum(exp(-x/t))) * t  for small t
    t = 0.01
    neg_vals = -all_vals / t
    max_neg = np.max(neg_vals)
    log_sum = max_neg + np.log(np.sum(np.exp(neg_vals - max_neg)))
    min_approx = -log_sum * t
    
    # Also add a penalty for going out of bounds (shouldn't happen with constraints, but safety)
    bound_penalty = np.sum(np.maximum(0, np.abs(centers) - 0.5)**2)
    
    return -min_approx + bound_penalty

def _gradient(centers_flat):
    """Gradient of the objective."""
    centers = centers_flat.reshape(N, 3)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    
    boundary = np.minimum(0.5 - centers, centers + 0.5)
    boundary_sq = np.sum(boundary**2, axis=1)
    
    pair_sq_over_4 = dist_sq / 4.0
    all_vals = np.concatenate([pair_sq_over_4.flatten(), boundary_sq])
    
    t = 0.01
    neg_vals = -all_vals / t
    max_neg = np.max(neg_vals)
    exp_vals = np.exp(neg_vals - max_neg)
    sum_exp = np.sum(exp_vals)
    weights = exp_vals / sum_exp
    
    # Gradient of -log(sum(exp(-x/t))) * t = -t * d/dx [log(sum(exp(-x/t)))]
    # = -t * [sum(exp(-x_i/t) * (-1/t) * dx_i/dx) / sum(exp(-x_i/t))]
    # = sum(weights_i * x_i_grad)  where x_i_grad is gradient of the i-th value w.r.t. centers
    
    # This is getting complex. Let's use numerical gradient instead for reliability.
    return None  # Signal to use numerical gradient

def construct():
    t_start = time.time()
    deadline = t_start + 110.0  # Safety margin before 120s limit
    
    best_centers = None
    best_score = 0.0
    
    # Generate multiple random initializations
    np.random.seed(42)
    
    # Start with a reasonable initial configuration
    # Use a jittered grid as a good starting point
    m = 4  # 4x4x4 = 64 points, we'll use 50
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    
    # Add small random jitter
    initial = grid[:N].copy()
    initial += np.random.normal(0, 0.01, size=initial.shape)
    initial = np.clip(initial, -0.49, 0.49)
    
    best_centers = initial.copy()
    best_score = _score(best_centers)
    
    # Try to improve using gradient-based optimization
    # Use L-BFGS-B with bounds
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    max_iterations = 2000
    
    # Try a few different initializations
    n_restarts = 5
    
    for restart in range(n_restarts):
        if time.time() > deadline - 30:
            break
        
        if restart == 0:
            x0 = initial.copy()
        else:
            # Random initialization with some structure
            # Use a random subset of a finer grid
            m_r = 5
            ticks_r = (np.arange(m_r) + 0.5) / m_r - 0.5
            grid_r = np.array([(x, y, z) for z in ticks_r for y in ticks_r for x in ticks_r])
            # Randomly select N points
            idx = np.random.choice(len(grid_r), N, replace=False)
            x0 = grid_r[idx].copy()
            x0 += np.random.normal(0, 0.02, size=x0.shape)
            x0 = np.clip(x0, -0.49, 0.49)
        
        x0_flat = x0.flatten()
        
        # Use L-BFGS-B with numerical gradient
        result = minimize(
            _objective,
            x0_flat,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iterations, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        if result.success or result.fun < _objective(x0_flat):
            new_centers = result.x.reshape(N, 3)
            new_score = _score(new_centers)
            if new_score > best_score:
                best_score = new_score
                best_centers = new_centers.copy()
        
        # If we have good score, do a local refinement
        if best_score > 0.13:
            # Try a few more local optimizations from the best point
            if time.time() > deadline - 20:
                break
            for _ in range(3):
                if time.time() > deadline - 10:
                    break
                x0_flat = best_centers.flatten()
                # Small perturbation
                x0_pert = x0_flat + np.random.normal(0, 0.005, size=x0_flat.shape)
                x0_pert = np.clip(x0_pert.reshape(N, 3), -0.49, 0.49).flatten()
                
                result = minimize(
                    _objective,
                    x0_pert,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': max_iterations, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                if result.success or result.fun < _objective(x0_pert):
                    new_centers = result.x.reshape(N, 3)
                    new_score = _score(new_centers)
                    if new_score > best_score:
                        best_score = new_score
                        best_centers = new_centers.copy()
    
    # Final validation and cleanup
    # Ensure all points are strictly inside the cube
    best_centers = np.clip(best_centers, -0.499999, 0.499999)
    
    # Verify the score one more time
    final_score = _score(best_centers)
    if final_score <= 0:
        # Fallback to a valid configuration
        m = 4
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_centers = grid[:N]
    
    return best_centers
# EVOLVE-BLOCK-END
