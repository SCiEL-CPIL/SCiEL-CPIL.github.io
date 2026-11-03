"""Pack 50 equal circles in a unit square, maximize common radius."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 140.0  # seconds, safety margin before 150s kill

def _radius(centers):
    """Compute the max common radius for given centers."""
    # Wall radius = min of (0.5 - |coord|)
    wall_r = np.min(0.5 - np.abs(centers))
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist)
    return min(wall_r, min_pair / 2.0)

def _soft_min_radius(centers, temperature=0.01):
    """Smooth surrogate for the max common radius using log-sum-exp soft-min."""
    wall_r = 0.5 - np.abs(centers)
    wall_vals = wall_r.ravel()
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    pair_vals = np.sqrt(dist_sq) / 2.0
    pair_vals_flat = pair_vals.ravel()
    
    # Combine all constraints into one array
    all_vals = np.concatenate([wall_vals, pair_vals_flat])
    
    # Soft-min via log-sum-exp: min(x) ≈ -log(sum(exp(x/T))) * T
    # For numerical stability, shift by min
    m = np.min(all_vals)
    if temperature < 1e-10:
        return m
    s = np.sum(np.exp((all_vals - m) / temperature))
    soft_min = m - temperature * np.log(s)
    return soft_min

def _smooth_objective(x, temperature=0.01):
    """Smooth negative radius for gradient-based minimization."""
    centers = x.reshape(N, 2)
    r = _soft_min_radius(centers, temperature)
    return -r

def _smooth_gradient(x, temperature=0.01):
    """Numerical gradient of smooth objective (central differences)."""
    centers = x.reshape(N, 2)
    eps = 1e-7
    grad = np.zeros_like(x)
    for i in range(len(x)):
        xp = x.copy()
        xm = x.copy()
        xp[i] += eps
        xm[i] -= eps
        grad[i] = (_smooth_objective(xp, temperature) - _smooth_objective(xm, temperature)) / (2 * eps)
    return grad

def _objective(x):
    """Negative radius for minimization."""
    centers = x.reshape(N, 2)
    r = _radius(centers)
    return -r

def _make_grid(n_per_side, N):
    """Create a grid of N points in [-0.5, 0.5]^2."""
    # Use n_per_side x n_per_side grid, take first N
    ticks = (np.arange(n_per_side) + 0.5) / n_per_side - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()

def _jitter(centers, scale):
    """Add random jitter to break symmetries."""
    return centers + np.random.normal(0, scale, size=centers.shape)

def _local_refine(centers, iters, step_size, time_limit):
    """Simple iterative local optimization: move points to maximize min radius."""
    centers = centers.copy()
    best_r = _radius(centers)
    t0 = time.time()
    
    for it in range(iters):
        if time.time() - t0 > time_limit:
            break
        
        r = best_r
        for i in range(N):
            if time.time() - t0 > time_limit:
                break
            best_i = centers[i].copy()
            best_r_i = r
            
            for _ in range(5):
                trial = centers.copy()
                trial[i] = best_i + np.random.normal(0, step_size, size=2)
                trial[i] = np.clip(trial[i], -0.5 + 0.01, 0.5 - 0.01)
                r_trial = _radius(trial)
                if r_trial > best_r_i:
                    best_r_i = r_trial
                    best_i = trial[i]
            
            if best_r_i > r:
                centers[i] = best_i
                r = best_r_i
        
        if r > best_r:
            best_r = r
    
    return centers, best_r

def _smooth_optimize(centers, time_limit):
    """Use smooth surrogate with L-BFGS-B to refine the packing."""
    t0 = time.time()
    x0 = centers.ravel().copy()
    
    # Start with a moderate temperature, anneal down
    temperatures = [0.05, 0.02, 0.005, 0.001, 0.0001]
    best_x = x0.copy()
    best_r = _radius(centers)
    
    for temp in temperatures:
        if time.time() - t0 > time_limit:
            break
        
        # Animate temperature: start higher, decrease
        def obj(x):
            return _smooth_objective(x, temp)
        
        result = minimize(
            obj, best_x, method='L-BFGS-B',
            bounds=[(-0.499, 0.499)] * (2 * N),
            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8,
                     'maxls': 50}
        )
        
        if result.fun < -best_r or _radius(result.x.reshape(N, 2)) > best_r:
            r_check = _radius(result.x.reshape(N, 2))
            if r_check > best_r:
                best_r = r_check
                best_x = result.x.copy()
        
        if time.time() - t0 > time_limit:
            break
    
    return best_x.reshape(N, 2), best_r

def _polish(centers, time_limit):
    """High-precision polish: coordinate-wise exact maximization of min distance."""
    t0 = time.time()
    centers = centers.copy()
    best_r = _radius(centers)
    
    for sweep in range(200):
        if time.time() - t0 > time_limit:
            break
        
        improved = False
        for i in range(N):
            if time.time() - t0 > time_limit:
                break
            
            # For point i, find the best position that maximizes min distance
            # to all other points and walls. Use a fine local search.
            best_pos = centers[i].copy()
            best_local_r = best_r
            
            # Coarse-to-fine local search around current position
            for scale in [0.05, 0.02, 0.008, 0.003, 0.001]:
                for dx in np.linspace(-scale, scale, 11):
                    for dy in np.linspace(-scale, scale, 11):
                        if time.time() - t0 > time_limit:
                            break
                        trial = centers.copy()
                        trial[i] = best_pos + np.array([dx, dy])
                        trial[i] = np.clip(trial[i], -0.499, 0.499)
                        r_trial = _radius(trial)
                        if r_trial > best_local_r:
                            best_local_r = r_trial
                            best_pos = trial[i].copy()
                
                if time.time() - t0 > time_limit:
                    break
            
            if best_local_r > best_r:
                centers[i] = best_pos
                best_r = best_local_r
                improved = True
        
        if not improved:
            break
    
    return centers, best_r

def construct():
    global DEADLINE
    start_time = time.time()
    
    # Start with a 7x7 grid (49 points) + 1 center
    # Actually, use 7x7 = 49, add 50th at center, but that's a duplicate.
    # Better: use a 7x7 grid but shift one point or use 7x7 with 50th inserted.
    # Let's use a 7x7 grid and place the 50th point by optimizing its position.
    
    # Initial: 7x7 grid
    centers = _make_grid(7, 50)  # This takes first 50 of 49x49... no, 7x7=49
    
    # Fix: _make_grid(7, 50) returns 50 points from a 7x7 grid (49 points) + need 1 more
    # Let me fix _make_grid to handle N > n_per_side^2
    
    # Actually, let's use a different approach:
    # Start with 7x7 grid (49 points), then add 1 point at center (0,0)
    # But (0,0) is already in the grid for 7x7? Let's check.
    # 7x7 grid: ticks = (0,1,2,3,4,5,6)/7 - 0.5 = -0.5, -4/7, -3/7, -2/7, -1/7, 0, 1/7, 2/7, 3/7, 4/7, 0.5
    # Wait, (np.arange(7) + 0.5)/7 - 0.5 = [0.5/7 - 0.5, 1.5/7 - 0.5, ...] 
    # = [-3/7, -2/7, -1/7, 0, 1/7, 2/7, 3/7]
    # So (0,0) is at index (3,3). We have a duplicate if we add another.
    
    # Better approach: use 7x7 grid and perturb the 50th point away.
    # Or use a 7x7 grid and optimize all 50 points.
    
    # Let's use a different initialization: 7x7 grid with 50th point placed
    # by finding the best empty spot.
    
    centers_49 = _make_grid(7, 49)
    # Add 50th point at (0,0) but offset slightly
    centers = np.vstack([centers_49, np.array([0.0, 0.0])])
    # Now we have 50 points with one duplicate. Jitter all.
    centers = _jitter(centers, 0.01)
    
    best_centers = centers.copy()
    best_r = _radius(centers)
    
    # Try a few different initial configurations
    configs = []
    configs.append(centers.copy())
    
    # Try 8x8 grid with 50 points
    centers_8x8 = _make_grid(8, 50)
    centers_8x8 = _jitter(centers_8x8, 0.01)
    configs.append(centers_8x8)
    
    # Try random initial points
    for seed in range(5):
        np.random.seed(seed)
        centers_rand = np.random.uniform(-0.45, 0.45, size=(N, 2))
        configs.append(centers_rand)
    
    # Local refine each config
    for i, cfg in enumerate(configs):
        if time.time() - start_time > DEADLINE - 30:
            break
        time_left = DEADLINE - 30 - (time.time() - start_time)
        if time_left <= 0:
            break
        refined, r = _local_refine(cfg, iters=100, step_size=0.02, time_limit=time_left / len(configs))
        if r > best_r:
            best_r = r
            best_centers = refined
    
    # Global optimization if time remains
    if time.time() - start_time < DEADLINE - 10:
        time_left = DEADLINE - 10 - (time.time() - start_time)
        try:
            from scipy.optimize import differential_evolution
            # Use differential evolution on the best configuration
            # This is expensive, so use small population
            x0 = best_centers.ravel()
            bounds = [(-0.5, 0.5)] * (2 * N)
            result = differential_evolution(
                _objective, bounds,
                maxiter=50, tol=1e-6,
                popsize=15, seed=42,
                update_interval=1,
                polish=False
            )
            if result.fun < -best_r:
                best_centers = result.x.reshape(N, 2)
                best_r = -result.fun
        except Exception:
            pass
    
    return best_centers
