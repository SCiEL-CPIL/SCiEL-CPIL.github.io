import numpy as np
import time
from scipy.spatial.distance import cdist

N = 100
DEADLINE = 115.0

def _compute_radius(centers):
    """Compute the largest common radius for given centers."""
    if not np.all(np.isfinite(centers)):
        return 0.0
    if centers.shape != (N, 3):
        return 0.0
    wall_dists = 0.5 - np.abs(centers)
    min_wall = wall_dists.min()
    if min_wall < -1e-10:
        return 0.0
    dists = cdist(centers, centers)
    np.fill_diagonal(dists, np.inf)
    min_pair = dists.min() / 2.0
    return min(min_wall, min_pair)

def _force_directed_relax(centers, max_iters, step_size, decay, start_time, deadline):
    """Vectorized force-directed relaxation with early termination."""
    centers = centers.copy()
    for it in range(max_iters):
        if time.time() > deadline:
            break
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dist_sq, 1.0)
        
        # Force magnitude: repulsion ~ 1/d^2, only for close pairs
        force_mag = 0.02 / dist_sq
        np.fill_diagonal(force_mag, 0.0)
        
        # Only apply forces from pairs closer than some threshold to save computation
        # Actually we need all for correctness, but we can cap the force
        force_mag = np.minimum(force_mag, 1.0)
        
        unit = diff / np.sqrt(dist_sq)[:, :, np.newaxis]
        forces = np.sum(force_mag[:, :, np.newaxis] * unit, axis=1)
        
        # Wall repulsion
        for k in range(3):
            wall_d = np.maximum(0.5 - np.abs(centers[:, k]), 1e-6)
            forces[:, k] -= 0.02 / (wall_d**2) * np.sign(centers[:, k])
        
        centers += step_size * forces
        centers = np.clip(centers, -0.49, 0.49)
        step_size *= decay
    return centers

def _local_refine(centers, maxiter=30, beta=150.0):
    """Scipy L-BFGS-B refinement using smooth-min objective."""
    try:
        from scipy.optimize import minimize
        
        def objective(x):
            c = x.reshape(N, 3)
            c = np.clip(c, -0.499, 0.499)
            diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
            dist_sq = np.sum(diff**2, axis=2)
            np.fill_diagonal(dist_sq, 1.0)
            dist = np.sqrt(dist_sq)
            np.fill_diagonal(dist, np.inf)
            d_half = dist[~np.isinf(dist)] / 2
            wall_d = (0.5 - np.abs(c)).flatten()
            all_d = np.concatenate([d_half, wall_d])
            
            max_d = np.max(all_d)
            shifted = all_d - max_d
            shifted = np.clip(shifted, -40.0, 0.0)
            log_sum = np.log(np.sum(np.exp(-beta * shifted)))
            smooth_min = max_d - log_sum / beta
            return -smooth_min
        
        res = minimize(objective, centers.flatten(), method='L-BFGS-B',
                      bounds=[(-0.499, 0.499)] * (N*3),
                      options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 20})
        
        c_opt = res.x.reshape(N, 3)
        c_opt = np.clip(c_opt, -0.499, 0.499)
        return c_opt
    except:
        return centers

def construct():
    start_time = time.time()
    rng = np.random.RandomState(42)
    
    # Initial: 5x5x5 grid, select 100 points
    m = 5
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    idx = rng.choice(len(grid), N, replace=False)
    best_config = grid[idx].astype(float).copy()
    best_r = _compute_radius(best_config)
    
    # Phase 1: Multiple restarts with force-directed relaxation
    # Each restart: random init -> force-directed relax -> local refine
    num_restarts = 40
    for restart in range(num_restarts):
        if time.time() - start_time > DEADLINE - 25:
            break
        
        # Random initial positions with some structure
        if restart % 3 == 0:
            # Perturbed grid
            centers = grid[idx] + rng.uniform(-0.05, 0.05, size=(N, 3))
            centers = np.clip(centers, -0.4, 0.4)
        elif restart % 3 == 1:
            # Uniform random
            centers = rng.uniform(-0.4, 0.4, size=(N, 3))
        else:
            # Clustered random (two groups)
            c1 = rng.uniform(-0.3, -0.05, size=(50, 3))
            c2 = rng.uniform(0.05, 0.3, size=(50, 3))
            centers = np.vstack([c1, c2])
        
        # Force-directed relaxation (moderate)
        centers = _force_directed_relax(centers, max_iters=800, step_size=0.005, 
                                         decay=0.9995, start_time=start_time, 
                                         deadline=DEADLINE - 25)
        
        # Local refinement
        centers = _local_refine(centers, maxiter=25, beta=100.0)
        
        r = _compute_radius(centers)
        if r > best_r:
            best_r = r
            best_config = centers.copy()
    
    # Phase 2: Deep refinement on best configuration
    if time.time() - start_time < DEADLINE - 8:
        # Short force-directed polish from best config
        centers = _force_directed_relax(best_config, max_iters=2000, step_size=0.003,
                                         decay=0.9998, start_time=start_time,
                                         deadline=DEADLINE - 8)
        # Deep local refinement with high beta
        centers = _local_refine(centers, maxiter=50, beta=200.0)
        
        r = _compute_radius(centers)
        if r > best_r:
            best_r = r
            best_config = centers.copy()
    
    # Phase 3: Final high-precision polish
    if time.time() - start_time < DEADLINE - 3:
        centers = _local_refine(best_config, maxiter=40, beta=400.0)
        r = _compute_radius(centers)
        if r > best_r:
            best_r = r
            best_config = centers.copy()
    
    # Final safety: ensure validity
    best_config = np.clip(best_config, -0.5, 0.5)
    return best_config
