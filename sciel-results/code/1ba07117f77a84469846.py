"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_radius(centers):
    """Compute the packing radius for given centers."""
    if not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(centers < -0.5) or np.any(centers > 0.5):
        return 0.0
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    # Boundary distances
    dist_x = 0.5 - np.abs(centers[:, 0])
    dist_y = 0.5 - np.abs(centers[:, 1])
    min_bound = np.min(np.minimum(dist_x, dist_y))
    
    return min(min_pair, min_bound)

def _soft_min_obj(x, n, beta):
    """Soft-min objective for optimization. x is flattened centers."""
    centers = x.reshape(n, 2)
    
    # Pairwise distances / 2
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    pair_vals = dists / 2.0
    
    # Boundary distances
    dist_x = 0.5 - np.abs(centers[:, 0])
    dist_y = 0.5 - np.abs(centers[:, 1])
    bound_vals = np.minimum(dist_x, dist_y)
    
    # Combine all values
    all_vals = np.concatenate([pair_vals[~np.eye(n, dtype=bool)], bound_vals])
    
    # Soft min
    min_val = np.min(all_vals)
    # Use a smoothed minimum
    log_sum = np.log(np.sum(np.exp(-beta * all_vals)))
    soft_min = -np.log(np.sum(np.exp(-beta * all_vals))) / beta + min_val / beta
    # Actually, let's use a simpler approach: -log-sum-exp scaled
    # soft_min ≈ (1/beta) * (-log(sum(exp(-beta * x))))
    # As beta -> inf, this approaches min(x)
    soft_min_val = -np.log(np.sum(np.exp(-beta * all_vals))) / beta
    
    return -soft_min_val  # We minimize the negative, so maximizing soft_min

def _initial_grid(n):
    """Create an initial grid configuration for n circles."""
    # Start with a grid that approximates the optimal arrangement
    # 7x7 grid = 49 points, plus 1 more
    m = 7
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid_49 = np.array([(x, y) for y in ticks for x in ticks])
    
    if n == 49:
        return grid_49
    
    # For 50 points, add one more point
    # Place the 50th point at a location that gives good spacing
    # Try placing it near the center but offset
    extra = np.array([[0.0, 0.0]])
    centers = np.vstack([grid_49, extra])
    
    # Add small random perturbation to break symmetry
    np.random.seed(42)
    centers = centers + np.random.normal(0, 0.01, size=centers.shape)
    
    return centers[:n]

def _initial_jittered(n):
    """Create a jittered grid initial configuration."""
    m = 7
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid_49 = np.array([(x, y) for y in ticks for x in ticks])
    
    if n == 49:
        return grid_49
    
    extra = np.array([[0.0, 0.0]])
    centers = np.vstack([grid_49, extra])
    
    # Larger jitter
    np.random.seed(123)
    centers = centers + np.random.normal(0, 0.03, size=centers.shape)
    
    return centers[:n]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin from 120s limit
    
    best_centers = None
    best_r = 0.0
    
    # Try multiple initial configurations
    initial_configs = [
        _initial_grid(N),
        _initial_jittered(N),
    ]
    
    # Also try a few random perturbations of the grid
    np.random.seed(0)
    for seed_val in [1, 2, 3]:
        np.random.seed(seed_val)
        cfg = _initial_grid(N)
        cfg = cfg + np.random.normal(0, 0.05, size=cfg.shape)
        initial_configs.append(cfg)
    
    for init_centers in initial_configs:
        if time.time() > deadline - 10.0:
            break
        
        # First, do a quick simulated annealing to improve the initial config
        centers = init_centers.copy()
        r = _compute_radius(centers)
        
        # Simulated Annealing
        temp = 0.05
        cooling = 0.9995
        step_size = 0.02
        sa_steps = 5000
        
        for sa_step in range(sa_steps):
            if time.time() > deadline - 5.0:
                break
            
            # Pick a random circle to move
            idx = np.random.randint(N)
            old_pos = centers[idx].copy()
            
            # Random perturbation
            delta = np.random.normal(0, step_size, size=2)
            new_pos = old_pos + delta
            
            # Keep within bounds
            new_pos = np.clip(new_pos, -0.49, 0.49)
            
            centers[idx] = new_pos
            new_r = _compute_radius(centers)
            
            # Accept if better, or with probability based on temperature
            if new_r > r:
                r = new_r
            else:
                delta_r = r - new_r
                if delta_r < temp:
                    prob = np.exp(-delta_r / temp)
                    if np.random.random() > prob:
                        centers[idx] = old_pos
                        new_r = r  # revert
                    else:
                        r = new_r
                else:
                    centers[idx] = old_pos
                    new_r = r
            
            temp *= cooling
        
        if r > best_r:
            best_r = r
            best_centers = centers.copy()
        
        # Now try gradient-based refinement using scipy
        if time.time() > deadline - 10.0:
            break
        
        # Use scipy minimize with the soft-min objective
        # Flatten centers
        x0 = best_centers.flatten()
        
        # Use a moderate beta for soft-min
        beta = 50.0
        
        try:
            result = minimize(
                _soft_min_obj,
                x0,
                args=(N, beta),
                method='L-BFGS-B',
                bounds=[(-0.49, 0.49)] * (2 * N),
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            refined_centers = result.x.reshape(N, 2)
            refined_r = _compute_radius(refined_centers)
            
            if refined_r > best_r:
                best_r = refined_r
                best_centers = refined_centers
        except Exception:
            pass
        
        # Do another round of SA starting from the refined config
        if time.time() > deadline - 5.0:
            break
        
        centers = best_centers.copy()
        r = best_r
        temp = 0.02
        cooling = 0.999
        step_size = 0.01
        sa_steps2 = 3000
        
        for sa_step in range(sa_steps2):
            if time.time() > deadline - 2.0:
                break
            
            idx = np.random.randint(N)
            old_pos = centers[idx].copy()
            
            delta = np.random.normal(0, step_size, size=2)
            new_pos = old_pos + delta
            new_pos = np.clip(new_pos, -0.49, 0.49)
            
            centers[idx] = new_pos
            new_r = _compute_radius(centers)
            
            if new_r > r:
                r = new_r
            else:
                delta_r = r - new_r
                if delta_r < temp:
                    prob = np.exp(-delta_r / temp)
                    if np.random.random() > prob:
                        centers[idx] = old_pos
                        new_r = r
                    else:
                        r = new_r
                else:
                    centers[idx] = old_pos
                    new_r = r
            
            temp *= cooling
        
        if r > best_r:
            best_r = r
            best_centers = centers.copy()
    
    # Ensure validity: strictly inside the square
    if best_centers is not None:
        eps = 1e-9
        best_centers = np.clip(best_centers, -0.5 + eps, 0.5 - eps)
    else:
        # Fallback: return a valid grid
        best_centers = _initial_grid(N)
        best_centers = np.clip(best_centers, -0.5 + 1e-9, 0.5 - 1e-9)
    
    return best_centers

# EVOLVE-BLOCK-END
