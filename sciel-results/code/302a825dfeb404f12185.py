"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.spatial import cKDTree

N = 50


def _compute_radius(centers):
    """Compute the maximum common radius for given centers."""
    centers = np.asarray(centers, dtype=np.float64)
    n = centers.shape[0]
    if n < 2:
        # Wall constraint only
        return np.min(0.5 - np.abs(centers))
    
    # Pairwise distances
    diff = centers[:, None, :] - centers[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_dist = np.min(dist)
    
    # Wall constraint
    wall_min = np.min(0.5 - np.abs(centers))
    
    return min(min_dist / 2.0, wall_min)


def _energy(centers, target_r, beta=50.0):
    """Smooth surrogate for the radius using log-sum-exp style soft-min."""
    centers = np.asarray(centers, dtype=np.float64)
    n = centers.shape[0]
    
    # Pairwise distances
    diff = centers[:, None, :] - centers[None, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    
    # Soft-min of distances
    # Use a soft approximation: r_pairs = 0.5 * min(dist)
    # Approximate min via soft-min with temperature
    # Actually, let's use a penalty-based approach
    # Penalize pairs that are too close and walls that are too close
    
    # Wall penalty
    wall_violations = np.maximum(0.0, np.abs(centers) - (0.5 - target_r))
    wall_pen = np.sum(wall_violations ** 2)
    
    # Pair penalty
    dist = np.sqrt(dist_sq)
    np.fill_diagonal(dist, np.inf)
    pair_violations = np.maximum(0.0, (2 * target_r - dist) / (2 * target_r))
    # Only count violations where dist < 2*target_r
    pair_pen = np.sum(pair_violations ** 2)
    
    return wall_pen + pair_pen


def construct():
    """Place 50 equal spheres in a unit cube to maximize the common radius."""
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Initial best known radius from the problem statement
    best_r = 0.0
    best_centers = None
    
    # Strategy: Start from a good grid and use gradient-free local optimization
    # with multiple restarts
    
    # Generate initial configurations from different grid structures
    configs = []
    
    # 1. Simple cubic grid (4x4x4 = 64, take 50)
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    configs.append(grid[:N])
    
    # 2. Slightly perturbed grid
    rng = np.random.default_rng(42)
    perturbed = grid[:N].copy()
    perturbed += rng.normal(0, 0.02, size=perturbed.shape)
    # Clamp to cube
    perturbed = np.clip(perturbed, -0.48, 0.48)
    configs.append(perturbed)
    
    # 3. Random configuration
    rand_config = rng.uniform(-0.45, 0.45, size=(N, 3))
    configs.append(rand_config)
    
    # 4. Body-centered cubic arrangement (approximate)
    # FCC-like structure
    a = 1.0 / 4.0  # lattice parameter
    positions = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                positions.append([i * a, j * a, k * a])
                # Add face-centered positions
    # Take a subset
    if len(positions) >= N:
        configs.append(np.array(positions[:N]))
    
    # Evaluate each config
    for config in configs:
        if time.time() > deadline - 20:
            break
        r = _compute_radius(config)
        if r > best_r:
            best_r = r
            best_centers = config.copy()
    
    # Local optimization using Nelder-Mead or similar
    # We'll use a simple stochastic local search with annealing
    
    if best_centers is None:
        # Fallback: return simple grid
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        return grid[:N]
    
    # Local optimization phase
    current_centers = best_centers.copy()
    current_r = best_r
    
    # Simulated annealing / stochastic local search
    rng_opt = np.random.default_rng(123)
    temperature = 0.01
    cooling = 0.999
    max_iter = 5000
    
    for it in range(max_iter):
        if time.time() > deadline - 5:
            break
        
        # Propose a move: perturb one or two random centers
        n_perturb = rng_opt.integers(1, 3)
        indices = rng_opt.choice(N, size=n_perturb, replace=False)
        
        new_centers = current_centers.copy()
        for idx in indices:
            step = temperature * rng_opt.standard_normal(3)
            new_centers[idx] += step
            # Keep within cube
            new_centers[idx] = np.clip(new_centers[idx], -0.48, 0.48)
        
        new_r = _compute_radius(new_centers)
        
        # Accept if better, or with probability exp((new_r - current_r) / temperature)
        if new_r >= current_r or (temperature > 1e-6 and rng_opt.random() < np.exp((new_r - current_r) / temperature)):
            current_centers = new_centers
            current_r = new_r
            if new_r > best_r:
                best_r = new_r
                best_centers = new_centers.copy()
        
        temperature *= cooling
    
    # Final polish: use scipy.optimize.minimize with a smooth objective
    if time.time() < deadline - 10:
        from scipy.optimize import minimize
        
        # Flatten centers
        x0 = best_centers.flatten()
        
        def objective(x):
            c = x.reshape(N, 3)
            # Smooth penalty for violations
            # Wall constraint
            wall_v = np.maximum(0, np.abs(c) - 0.5 + best_r)
            wall_pen = np.sum(wall_v ** 2)
            
            # Pair constraint
            diff = c[:, None, :] - c[None, :, :]
            dist = np.sqrt(np.sum(diff ** 2, axis=2))
            np.fill_diagonal(dist, np.inf)
            pair_v = np.maximum(0, 2 * best_r - dist)
            pair_pen = np.sum(pair_v ** 2)
            
            return wall_pen + pair_pen
        
        # Try a few local minimizations
        for trial in range(3):
            if time.time() > deadline - 5:
                break
            
            x0_trial = best_centers.flatten()
            if trial > 0:
                x0_trial += rng_opt.normal(0, 0.005, size=x0_trial.shape)
            
            res = minimize(objective, x0_trial, method='L-BFGS-B', 
                          bounds=[(-0.49, 0.49)] * (3 * N),
                          options={'maxiter': 500, 'ftol': 1e-12})
            
            if res.success:
                c_opt = res.x.reshape(N, 3)
                r_opt = _compute_radius(c_opt)
                if r_opt > best_r:
                    best_r = r_opt
                    best_centers = c_opt
    
    # Ensure valid output
    best_centers = np.clip(best_centers, -0.4999, 0.4999)
    
    return best_centers
