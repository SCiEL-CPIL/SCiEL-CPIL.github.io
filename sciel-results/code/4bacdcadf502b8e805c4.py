import numpy as np
import time
from scipy.spatial.distance import cdist

N = 100
DEADLINE = 145.0

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
    """Vectorized force-directed relaxation."""
    centers = centers.copy()
    for it in range(max_iters):
        if time.time() > deadline:
            break
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, 1.0)
        
        force_mag = 0.05 / (dist**2)
        np.fill_diagonal(force_mag, 0.0)
        
        unit = diff / dist[:, :, np.newaxis]
        forces = np.sum(force_mag[:, :, np.newaxis] * unit, axis=1)
        
        for k in range(3):
            # Push spheres away from walls (toward center)
            wall_d = np.maximum(0.5 - np.abs(centers[:, k]), 1e-8)
            # sign should push AWAY from wall: if center > 0 and near +0.5, push negative
            forces[:, k] -= 0.05 / (wall_d**2) * np.sign(centers[:, k])
        
        centers += step_size * forces
        centers = np.clip(centers, -0.49, 0.49)
        step_size *= decay
    return centers

def _local_refine(centers, maxiter=50, beta=100.0):
    """Scipy L-BFGS-B refinement using smooth-min objective."""
    try:
        from scipy.optimize import minimize
        
        def objective(x):
            c = x.reshape(N, 3)
            c = np.clip(c, -0.5, 0.5)
            diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
            dist = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dist, np.inf)
            d_half = dist[~np.isinf(dist)] / 2
            wall_d = (0.5 - np.abs(c)).flatten()
            all_d = np.concatenate([d_half, wall_d])
            
            max_d = np.max(all_d)
            shifted = all_d - max_d
            # Clip to avoid overflow in exp
            shifted = np.clip(shifted, -50.0, 0.0)
            log_sum = np.log(np.sum(np.exp(-beta * shifted)))
            smooth_min = max_d - log_sum / beta
            return -smooth_min
        
        res = minimize(objective, centers.flatten(), method='L-BFGS-B',
                      bounds=[(-0.5, 0.5)] * (N*3),
                      options={'maxiter': maxiter, 'ftol': 1e-10, 'gtol': 1e-8})
        
        c_opt = res.x.reshape(N, 3)
        c_opt = np.clip(c_opt, -0.5, 0.5)
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
    # Each restart: random init -> force-directed relax -> quick local refine
    num_restarts = 30
    for restart in range(num_restarts):
        if time.time() - start_time > DEADLINE - 30:
            break
        
        # Random initial positions
        centers = rng.uniform(-0.4, 0.4, size=(N, 3))
        
        # Force-directed relaxation (shorter per restart)
        centers = _force_directed_relax(centers, max_iters=2000, step_size=0.003, 
                                         decay=0.9997, start_time=start_time, 
                                         deadline=DEADLINE - 30)
        
        # Quick local refinement
        centers = _local_refine(centers, maxiter=20)
        
        r = _compute_radius(centers)
        if r > best_r:
            best_r = r
            best_config = centers.copy()
    
    # Phase 2: Final deep refinement on best configuration
    if time.time() - start_time < DEADLINE - 5:
        # First do a longer force-directed phase from the best config
        centers = _force_directed_relax(best_config, max_iters=5000, step_size=0.002,
                                         decay=0.9999, start_time=start_time,
                                         deadline=DEADLINE - 10)
        # Then deep local refinement with high beta for precision
        centers = _local_refine(centers, maxiter=100, beta=200.0)
        
        r = _compute_radius(centers)
        if r > best_r:
            best_r = r
            best_config = centers.copy()
    
    # Phase 3: Final "push the bottleneck" refinement
    if time.time() - start_time < DEADLINE - 2:
        centers = _local_refine(best_config, maxiter=80, beta=500.0)
        r = _compute_radius(centers)
        if r > best_r:
            best_r = r
            best_config = centers.copy()
    
    # Final safety: ensure validity
    best_config = np.clip(best_config, -0.5, 0.5)
    return best_config
