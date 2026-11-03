"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_radius(centers):
    """Compute the maximum common radius for given centers."""
    if len(centers) < 2:
        min_dist_to_wall = np.min(0.5 - np.abs(centers))
        return float(min_dist_to_wall)
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    min_wall_dist = np.min(0.5 - np.abs(centers))
    
    return float(min(min_pair_dist / 2.0, min_wall_dist))

def _soft_min_radius(centers, temperature=0.01):
    """Smooth surrogate for the radius using log-sum-exp soft-min."""
    if len(centers) < 2:
        return float(np.min(0.5 - np.abs(centers)))
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    np.fill_diagonal(sq_dists, np.inf)
    
    # Soft-min of pairwise distances / 2
    # Use log-sum-exp trick: soft_min(x) = -log(sum(exp(-x/T)))/log(n) approximately
    # Actually: soft_min = -T * log(sum(exp(-x/T)))
    # For stability, subtract max first
    x_pair = sq_dists / 4.0  # (d/2)^2, we want soft min of d/2
    # Work with squared distances for efficiency, then take sqrt at end
    # soft_min of d/2 where d = sqrt(sq_d)
    # Let's use: soft_min(d/2) ≈ sqrt(soft_min(d^2/4))
    
    # Pairwise: soft min of distances
    max_sq = np.max(sq_dists[np.isfinite(sq_dists)])
    exp_terms = np.exp(-(sq_dists - max_sq) / (4.0 * temperature**2))
    np.fill_diagonal(exp_terms, 0.0)
    sum_exp = np.sum(exp_terms)
    if sum_exp > 0:
        soft_min_sq_pair = max_sq - 4.0 * temperature**2 * np.log(sum_exp)
    else:
        soft_min_sq_pair = max_sq
    soft_min_pair = np.sqrt(max(soft_min_sq_pair, 0.0))
    
    # Wall distances
    wall_dists = 0.5 - np.abs(centers)
    max_wall = np.max(wall_dists)
    exp_wall = np.exp(-(wall_dists - max_wall) / temperature**2)
    sum_wall = np.sum(exp_wall)
    if sum_wall > 0:
        soft_min_wall_sq = max_wall**2 - 2.0 * max_wall * temperature**2 * np.log(sum_wall) / max_wall if max_wall > 0 else 0
        # Simpler: soft_min of wall_dists directly
        soft_min_wall = max_wall - temperature**2 * np.log(sum_wall)
    else:
        soft_min_wall = max_wall
    
    # Combine: soft min of (soft_min_pair, soft_min_wall)
    # Use another soft-min
    vals = np.array([soft_min_pair, soft_min_wall])
    max_v = np.max(vals)
    exp_v = np.exp(-(vals - max_v) / temperature**2)
    soft_min_combined = max_v - temperature**2 * np.log(np.sum(exp_v))
    
    return float(soft_min_combined)

def _objective_and_grad(x, temperature=0.01):
    """Objective: maximize soft-min radius. Returns (-radius, gradient)."""
    centers = x.reshape(N, 3)
    
    # Compute soft-min radius
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    np.fill_diagonal(sq_dists, np.inf)
    
    T2 = temperature**2
    
    # Pairwise soft-min
    finite_mask = np.isfinite(sq_dists)
    if np.any(finite_mask):
        max_sq = np.max(sq_dists[finite_mask])
        exp_terms = np.zeros_like(sq_dists)
        exp_terms[finite_mask] = np.exp(-(sq_dists[finite_mask] - max_sq) / (4.0 * T2))
        sum_exp = np.sum(exp_terms)
        if sum_exp > 0:
            log_sum = np.log(sum_exp)
            soft_min_sq_pair = max_sq - 4.0 * T2 * log_sum
        else:
            soft_min_sq_pair = max_sq
    else:
        soft_min_sq_pair = 0.0
    
    soft_min_sq_pair = max(soft_min_sq_pair, 0.0)
    soft_min_pair = np.sqrt(soft_min_sq_pair)
    
    # Wall soft-min
    wall_dists = 0.5 - np.abs(centers)
    max_wall = np.max(wall_dists)
    exp_wall = np.exp(-(wall_dists - max_wall) / T2)
    sum_wall = np.sum(exp_wall)
    if sum_wall > 0:
        soft_min_wall = max_wall - T2 * np.log(sum_wall)
    else:
        soft_min_wall = max_wall
    
    # Combined soft-min
    vals = np.array([soft_min_pair, soft_min_wall])
    max_v = np.max(vals)
    exp_v = np.exp(-(vals - max_v) / T2)
    sum_v = np.sum(exp_v)
    soft_min_combined = max_v - T2 * np.log(sum_v)
    
    # Gradient computation
    grad = np.zeros_like(centers)
    
    # Pairwise gradient
    if np.any(finite_mask) and sum_exp > 0:
        weights = exp_terms / sum_exp  # (N, N) matrix
        sum_w = np.sum(weights, axis=1)  # (N,)
        weighted_sum_x = weights @ centers  # (N, 3)
        
        grad_pair_sq = 4.0 * (centers * sum_w[:, np.newaxis] - weighted_sum_x)
        
        if soft_min_sq_pair > 1e-12:
            grad_pair = grad_pair_sq / (2.0 * soft_min_pair)
        else:
            grad_pair = np.zeros_like(centers)
        
        signs = np.sign(centers)
        signs[centers == 0] = 1.0
        
        d_sum_wall_dx = exp_wall * signs / T2  # (N, 3)
        d_soft_min_wall_dx = -T2 * d_sum_wall_dx / sum_wall
        
        exp_pair_v = np.exp(-(soft_min_pair - max_v) / T2)
        exp_wall_v = np.exp(-(soft_min_wall - max_v) / T2)
        
        grad_combined = (exp_pair_v / sum_v) * grad_pair + (exp_wall_v / sum_v) * d_soft_min_wall_dx
        
        grad = grad_combined
    
    # We want to MAXIMIZE soft_min_combined, so minimize -soft_min_combined
    return -soft_min_combined, -grad

def _optimize_from_init(init_centers, temperature=0.01, max_iter=5000, time_limit=20.0):
    """Optimize from initial centers using L-BFGS-B."""
    start_time = time.time()
    x0 = init_centers.flatten()
    
    def obj(x):
        if time.time() - start_time > time_limit:
            return 1e10, np.zeros_like(x)
        val, grad = _objective_and_grad(x, temperature)
        return val, grad
    
    result = minimize(
        obj, x0,
        method='L-BFGS-B',
        bounds=[(-0.5, 0.5)] * (N * 3),
        options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
    )
    
    return result.x.reshape(N, 3)

def _polish_critical_pairs(centers, time_limit=5.0):
    """Final high-precision polish focusing only on critical (closest) pairs."""
    start_time = time.time()
    centers = centers.copy()
    
    for iteration in range(100):
        if time.time() - start_time > time_limit:
            break
        
        # Find the minimum radius constraint
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        min_pair_dist = np.min(dists)
        min_pair_idx = np.unravel_index(np.argmin(dists), dists.shape)
        
        min_wall_dist = np.min(0.5 - np.abs(centers))
        min_wall_idx = np.unravel_index(np.argmin(0.5 - np.abs(centers)), (N, 3))
        
        r = min(min_pair_dist / 2.0, min_wall_dist)
        
        # If pair distance is limiting, push those two apart
        if min_pair_dist / 2.0 < min_wall_dist:
            i, j = min_pair_idx
            if i > j:
                i, j = j, i
            d = centers[i] - centers[j]
            dist = np.linalg.norm(d)
            if dist < 1e-10:
                d = np.array([1e-6, 0, 0])
                dist = 1e-6
            # Push apart along the connecting line
            push = (r - dist/2.0) * d / dist * 0.5
            centers[i] += push
            centers[j] -= push
        else:
            # Wall is limiting, push that sphere inward
            i, k = min_wall_idx
            wall = 0.5 - np.abs(centers[i, k])
            if wall < r:
                sign = np.sign(centers[i, k])
                if sign == 0:
                    sign = 1.0
                centers[i, k] += (r - wall) * sign
        
        # Re-clip to cube
        centers = np.clip(centers, -0.5, 0.5)
    
    return centers

def _grid_init(n):
    """Generate grid initial configuration."""
    m = int(np.ceil(n ** (1.0/3.0)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:n]

def _jitter_init(n, seed=None, jitter=0.01):
    """Generate jittered grid initial configuration to break symmetry."""
    grid = _grid_init(n)
    rng = np.random.RandomState(seed)
    noise = rng.uniform(-jitter, jitter, size=grid.shape)
    return np.clip(grid + noise, -0.49, 0.49)

def _random_init(n, seed=None):
    """Generate random initial configuration inside the cube."""
    rng = np.random.RandomState(seed)
    return rng.uniform(-0.45, 0.45, size=(n, 3))

def construct():
    """Construct the optimal packing of N spheres in the unit cube."""
    start_time = time.time()
    deadline = start_time + 140.0  # 10 second safety margin
    
    best_centers = None
    best_radius = 0.0
    
    # Initial grid
    grid = _grid_init(N)
    r_grid = _compute_radius(grid)
    best_radius = r_grid
    best_centers = grid.copy()
    
    # Stage 1: Coarse optimization with larger temperature
    temps = [0.05, 0.03, 0.02, 0.01, 0.005, 0.002]
    
    inits = [grid]
    # Add jittered grids to help escape symmetric local minima
    for seed in range(4):
        inits.append(_jitter_init(N, seed=seed, jitter=0.02))
    for seed in range(4):
        inits.append(_random_init(N, seed=seed))
    
    for temp in temps:
        if time.time() > deadline - 40.0:
            break
        
        for init in inits:
            if time.time() > deadline - 30.0:
                break
            
            try:
                time_remaining = deadline - time.time() - 25.0
                if time_remaining < 5.0:
                    break
                optimized = _optimize_from_init(init, temperature=temp, 
                                                max_iter=2000, time_limit=min(12.0, time_remaining))
                r_opt = _compute_radius(optimized)
                if r_opt > best_radius:
                    best_radius = r_opt
                    best_centers = optimized.copy()
            except Exception:
                continue
        
        # Update inits with best found so far
        inits = [best_centers] + inits[:2]
    
    # Stage 2: Fine polishing from best with very small temperature
    if time.time() < deadline - 15.0:
        fine_temps = [0.001, 0.0005, 0.0002, 0.0001, 0.00005]
        for temp in fine_temps:
            if time.time() > deadline - 8.0:
                break
            
            try:
                time_remaining = deadline - time.time() - 5.0
                if time_remaining < 3.0:
                    break
                optimized = _optimize_from_init(best_centers, temperature=temp,
                                                max_iter=3000, time_limit=min(8.0, time_remaining))
                r_opt = _compute_radius(optimized)
                if r_opt > best_radius:
                    best_radius = r_opt
                    best_centers = optimized.copy()
            except Exception:
                continue
    
    # Stage 3: Final high-precision polish using L-BFGS with very small temperature
    if time.time() < deadline - 3.0:
        try:
            time_remaining = deadline - time.time() - 1.0
            if time_remaining > 2.0:
                # Use a very small temperature to sharpen the soft-min approximation
                optimized = _optimize_from_init(best_centers, temperature=0.00001,
                                                max_iter=5000, time_limit=min(2.5, time_remaining))
                r_opt = _compute_radius(optimized)
                if r_opt > best_radius:
                    best_radius = r_opt
                    best_centers = optimized.copy()
        except Exception:
            pass
    
    # Final validation and clipping
    if best_centers is None:
        best_centers = _grid_init(N)
    
    best_centers = np.clip(best_centers, -0.4999999, 0.4999999)
    
    return best_centers
# EVOLVE-BLOCK-END
