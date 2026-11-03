"""Pack 50 equal circles in a unit square, maximize common radius."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 140.0  # safety margin before 150s kill


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


def _objective(x):
    """Negative radius for minimization."""
    centers = x.reshape(N, 2)
    r = _radius(centers)
    return -r


def _make_grid(n_per_side, N):
    """Create a grid of N points in [-0.5, 0.5]^2."""
    ticks = (np.arange(n_per_side) + 0.5) / n_per_side - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()


def _jitter(centers, scale):
    """Add random jitter to break symmetries."""
    return centers + np.random.normal(0, scale, size=centers.shape)


def _local_refine(centers, time_limit):
    """Iterative local optimization using constraint-aware moves."""
    centers = centers.copy()
    best_r = _radius(centers)
    t0 = time.time()
    
    # Number of iterations to attempt
    max_iters = 200
    
    for it in range(max_iters):
        if time.time() - t0 > time_limit:
            break
        
        improved = False
        r = best_r
        
        for i in range(N):
            if time.time() - t0 > time_limit:
                break
            
            # Find the current radius and identify constraints
            # Compute all distances from point i to others and to walls
            dists = np.linalg.norm(centers - centers[i], axis=1)
            np.fill_diagonal(dists, np.inf)
            nearest_j = np.argmin(dists)
            min_pair_dist = dists[nearest_j]
            
            # Distance to walls
            wall_dists = 0.5 - np.abs(centers[i])
            nearest_wall = np.argmin(wall_dists)
            min_wall_dist = wall_dists[nearest_wall]
            
            # Determine which constraint is binding
            # If pair distance is limiting, move away from nearest neighbor
            # If wall is limiting, move away from wall
            
            current_r = min(min_wall_dist, min_pair_dist / 2.0)
            
            # Try to improve by moving point i
            best_i = centers[i].copy()
            best_r_i = current_r
            
            # Generate candidate moves
            # Move away from nearest neighbor
            if min_pair_dist < 2 * min_wall_dist:
                # Pair constraint is tighter
                direction = centers[i] - centers[nearest_j]
                norm = np.linalg.norm(direction)
                if norm > 1e-10:
                    direction /= norm
                else:
                    direction = np.array([1.0, 0.0])
            else:
                # Wall constraint is tighter
                # Move away from the nearest wall
                if nearest_wall == 0:  # x coordinate
                    if centers[i][0] < 0:
                        direction = np.array([1.0, 0.0])
                    else:
                        direction = np.array([-1.0, 0.0])
                else:  # y coordinate
                    if centers[i][1] < 0:
                        direction = np.array([0.0, 1.0])
                    else:
                        direction = np.array([0.0, -1.0])
            
            # Try multiple step sizes in the identified direction
            for step in [0.01, 0.005, 0.002, 0.001]:
                trial = centers.copy()
                trial[i] = best_i + direction * step
                # Keep in bounds
                trial[i] = np.clip(trial[i], -0.499, 0.499)
                r_trial = _radius(trial)
                if r_trial > best_r_i:
                    best_r_i = r_trial
                    best_i = trial[i]
            
            # Also try some random perturbations
            for _ in range(3):
                trial = centers.copy()
                trial[i] = best_i + np.random.normal(0, 0.005, size=2)
                trial[i] = np.clip(trial[i], -0.499, 0.499)
                r_trial = _radius(trial)
                if r_trial > best_r_i:
                    best_r_i = r_trial
                    best_i = trial[i]
            
            if best_r_i > r:
                centers[i] = best_i
                r = best_r_i
                improved = True
        
        if r > best_r:
            best_r = r
        else:
            # No improvement in this full pass, try smaller steps
            if not improved:
                break
    
    return centers, best_r


def _polish(centers, time_limit):
    """High-precision polish using scipy minimize with smooth surrogate."""
    t0 = time.time()
    centers = centers.copy()
    best_r = _radius(centers)
    
    # Use a smooth surrogate: soft-min of pairwise distances and wall distances
    # r = min(min_pair/2, min_wall)
    # Approximate with: r_smooth = -logsumexp(-min_pair/2, -min_wall) / alpha
    # But simpler: just use the exact objective with SLSQP
    
    def _obj_smooth(x):
        c = x.reshape(N, 2)
        wall_r = np.min(0.5 - np.abs(c))
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dist, np.inf)
        min_pair = np.min(dist)
        return -min(wall_r, min_pair / 2.0)
    
    # Try SLSQP from current best
    x0 = centers.ravel()
    bounds = [(-0.499, 0.499)] * (2 * N)
    
    try:
        result = minimize(_obj_smooth, x0, method='SLSQP', bounds=bounds,
                          options={'maxiter': 100, 'ftol': 1e-12})
        if time.time() - t0 < time_limit and result.fun < -best_r:
            best_centers = result.x.reshape(N, 2)
            best_r = -result.fun
            centers = best_centers
    except Exception:
        pass
    
    # Additional coordinate descent polish
    t1 = time.time()
    if time_limit - (t1 - t0) > 2:
        refined, r2 = _local_refine(centers, time_limit - (t1 - t0))
        if r2 > best_r:
            best_r = r2
            centers = refined
    
    return centers, best_r


def construct():
    start_time = time.time()
    
    # Initial configuration: 7x7 grid (49 points) + 1 extra point
    centers_49 = _make_grid(7, 49)
    
    # Find best position for 50th point
    best_50th = np.array([0.0, 0.0])
    best_min_dist = 0.0
    for y in np.linspace(-0.45, 0.45, 15):
        for x in np.linspace(-0.45, 0.45, 15):
            candidate = np.array([x, y])
            dists = np.linalg.norm(centers_49 - candidate, axis=1)
            min_d = np.min(dists)
            if min_d > best_min_dist:
                best_min_dist = min_d
                best_50th = candidate
    
    centers_init = np.vstack([centers_49, best_50th])
    centers_init = _jitter(centers_init, 0.003)
    
    best_centers = centers_init.copy()
    best_r = _radius(centers_init)
    
    # Try multiple initializations
    configs = [centers_init.copy()]
    
    # 8x8 grid, take 50 points
    grid_8 = _make_grid(8, 64)
    configs.append(grid_8[:50].copy())
    
    # Random starts with different seeds
    for seed in range(3):
        np.random.seed(seed * 1000 + 42)
        configs.append(np.random.uniform(-0.45, 0.45, size=(N, 2)))
    
    # Local refine each config
    for i, cfg in enumerate(configs):
        if time.time() - start_time > DEADLINE - 25:
            break
        time_left = DEADLINE - 25 - (time.time() - start_time)
        if time_left <= 5:
            break
        refined, r = _local_refine(cfg, time_left)
        if r > best_r:
            best_r = r
            best_centers = refined
    
    # Final polish with high-precision methods
    if time.time() - start_time < DEADLINE - 10:
        time_left = DEADLINE - 10 - (time.time() - start_time)
        if time_left > 5:
            best_centers, best_r = _polish(best_centers, time_left)
    
    return best_centers
