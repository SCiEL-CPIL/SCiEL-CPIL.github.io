import numpy as np
import scipy.optimize as opt
import time

N = 50

def _compute_radius(centers):
    """
    Computes the exact maximum radius for the given centers.
    Returns 0.0 if invalid (out of bounds or coincident).
    """
    if not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(centers < -0.5) or np.any(centers > 0.5):
        return 0.0
    
    # Pairwise distances
    # Use broadcasting for speed
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    
    # Mask diagonal
    mask = ~np.eye(N, dtype=bool)
    min_pair_dist_sq = np.min(dists_sq[mask])
    
    if min_pair_dist_sq < 1e-12:
        return 0.0
        
    min_pair_dist = np.sqrt(min_pair_dist_sq)
    
    # Boundary distances
    dx = 0.5 - np.abs(centers[:, 0])
    dy = 0.5 - np.abs(centers[:, 1])
    min_wall_dist = np.min(np.minimum(dx, dy))
    
    return np.min([min_pair_dist / 2.0, min_wall_dist])

def _soft_min(values, alpha):
    """
    Smooth approximation of min(values).
    values: 1D numpy array of positive floats.
    """
    if len(values) == 0:
        return 0.0
    v = np.asarray(values)
    v_min = np.min(v)
    # To avoid numerical issues with large alpha, shift by v_min
    shifted = v - v_min
    # exp(-alpha * shifted)
    # If alpha * shifted is large, exp is 0.
    # We can clip shifted to avoid overflow in exp if alpha is huge?
    # Actually, exp(-large) is 0, which is fine.
    # exp(-negative) would be overflow, but shifted >= 0.
    exponents = -alpha * shifted
    log_sum = np.log(np.sum(np.exp(exponents)))
    return v_min - log_sum / alpha

def _objective_smooth(centers_flat, alpha):
    centers = centers_flat.reshape(-1, 2)
    
    # Pairwise distances / 2
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    mask = ~np.eye(N, dtype=bool)
    pair_dists = np.sqrt(dists_sq[mask]) / 2.0
    
    # Wall distances
    dx = 0.5 - np.abs(centers[:, 0])
    dy = 0.5 - np.abs(centers[:, 1])
    wall_dists = np.minimum(dx, dy)
    
    # Combine
    all_dists = np.concatenate([pair_dists, wall_dists])
    
    # We want to maximize the soft-min, so minimize negative soft-min
    return -_soft_min(all_dists, alpha)

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    np.random.seed(42)
    
    best_radius = 0.0
    best_centers = None
    
    # Initial configurations
    inits = []
    
    # 1. 7x7 grid + 1 center
    m = 7
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    init1 = grid[:49]
    init1 = np.vstack([init1, [[0.0, 0.0]]])
    inits.append(init1.copy())
    
    # 2. 8x8 grid subset
    m = 8
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid8 = np.array([(x, y) for y in ticks for x in ticks])
    inits.append(grid8[:50].copy())
    
    # 3. Random inits
    for _ in range(10):
        rand = np.random.uniform(-0.45, 0.45, (N, 2))
        inits.append(rand)
        
    # Optimization parameters
    alphas = [10, 50, 100, 200, 500, 1000]
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    for init in inits:
        if time.time() > deadline:
            break
            
        current = init.copy()
        
        for alpha in alphas:
            if time.time() > deadline:
                break
                
            try:
                res = opt.minimize(
                    _objective_smooth,
                    current.flatten(),
                    args=(alpha,),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                current = res.x.reshape(-1, 2)
            except Exception:
                pass
                
        r = _compute_radius(current)
        if r > best_radius:
            best_radius = r
            best_centers = current.copy()
            
        # If we found a very good one, maybe do a few more random perturbations around it?
        # For now, just proceed to next init.
        
    if best_centers is None:
        # Fallback
        return inits[0]
        
    return best_centers
