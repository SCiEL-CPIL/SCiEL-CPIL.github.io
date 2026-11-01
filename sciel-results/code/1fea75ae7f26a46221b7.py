"""Model-and-optimize approach for 50 equal circles in a unit square."""
import numpy as np
import time
import scipy.optimize as opt
from scipy.spatial import cKDTree

N = 50
DEADLINE = 142.0  # Safety margin before 150s kill

def _compute_radius(centers):
    """Compute the true maximum radius for a given configuration."""
    centers = np.asarray(centers, dtype=np.float64)
    if centers.shape != (N, 2) or not np.all(np.isfinite(centers)):
        return 0.0
    
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    min_wall = np.min(np.minimum(wall_dists_x, wall_dists_y))
    
    if min_wall < 0:
        return 0.0
    
    # Use cKDTree for efficient nearest-neighbor distance
    tree = cKDTree(centers)
    # Query each point's nearest neighbor (excluding itself)
    dists, idx = tree.query(centers, k=2)
    min_pair = np.min(dists[:, 1]) / 2.0
    
    return min(min_pair, min_wall)

def _soft_min_objective(x, tau=0.008):
    """
    Objective: minimize -soft_min(minimum distance to constraint boundary).
    x: flattened array of 2*N coordinates
    """
    centers = x.reshape(-1, 2)
    
    # Wall distances
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    walls = np.minimum(wall_dists_x, wall_dists_y)
    
    # Pairwise distances using cKDTree
    tree = cKDTree(centers)
    dists, _ = tree.query(centers, k=2)
    pair_dists = dists[:, 1] / 2.0
    
    # Combine all constraint distances
    all_dists = np.concatenate([walls, pair_dists])
    
    # Soft-min using log-sum-exp
    min_d = np.min(all_dists)
    shifted = all_dists - min_d
    log_sum = np.log(np.sum(np.exp(-shifted / tau)))
    soft_min = min_d - tau * log_sum
    
    return -soft_min

def _grad_soft_min(x, tau=0.008):
    """
    Compute gradient of -soft_min objective.
    """
    centers = x.reshape(-1, 2)
    
    # Wall distances
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    walls = np.minimum(wall_dists_x, wall_dists_y)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    pair_dists = dists / 2.0
    
    # Upper triangle pairwise distances
    iu, ju = np.triu_indices(N, k=1)
    pair_upper = pair_dists[iu, ju]
    
    all_dists = np.concatenate([walls, pair_upper])
    
    # Soft-min weights: w_k = exp(-d_k/tau) / sum(exp(-d_j/tau))
    min_d = np.min(all_dists)
    shifted = all_dists - min_d
    exp_vals = np.exp(-shifted / tau)
    weights = exp_vals / np.sum(exp_vals)
    
    grad = np.zeros_like(x)
    
    # Wall gradient contributions
    for i in range(N):
        w_wall = weights[i]
        if walls[i] == wall_dists_x[i]:
            grad[2*i] += w_wall * np.sign(centers[i, 0])
        if walls[i] == wall_dists_y[i]:
            grad[2*i+1] += w_wall * np.sign(centers[i, 1])
    
    # Pairwise gradient contributions
    for k in range(len(iu)):
        w_pair = weights[N + k]
        i, j = iu[k], ju[k]
        d = dists[i, j]
        if d > 1e-12:
            factor = 0.5 * w_pair / d
            diff_ij = centers[i] - centers[j]
            grad[2*i:2*i+2] -= factor * diff_ij
            grad[2*j:2*j+2] += factor * diff_ij
    
    return grad

def _violation_objective(x, r_target):
    """
    Objective: minimize sum of squared violations for target radius r_target.
    A violation occurs when a circle is too close to another or to the wall.
    """
    centers = x.reshape(-1, 2)
    
    # Wall distances
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    walls = np.minimum(wall_dists_x, wall_dists_y)
    wall_viol = np.maximum(0, r_target - walls)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    pair_dists = dists / 2.0
    pair_viol = np.maximum(0, r_target - pair_dists)
    
    # Sum of squared violations
    return np.sum(wall_viol**2) + np.sum(pair_viol**2)

def _violation_grad(x, r_target):
    """
    Gradient of violation objective.
    """
    centers = x.reshape(-1, 2)
    grad = np.zeros_like(x)
    
    # Wall gradient
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    walls = np.minimum(wall_dists_x, wall_dists_y)
    
    for i in range(N):
        if walls[i] < r_target:
            viol = r_target - walls[i]
            if walls[i] == wall_dists_x[i]:
                grad[2*i] += 2 * viol * np.sign(centers[i, 0])
            if walls[i] == wall_dists_y[i]:
                grad[2*i+1] += 2 * viol * np.sign(centers[i, 1])
    
    # Pairwise gradient
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    pair_dists = dists / 2.0
    
    for i in range(N):
        for j in range(i+1, N):
            d = dists[i, j]
            if d < 2 * r_target and d > 1e-12:
                viol = r_target - d/2
                factor = 2 * viol / d
                diff_ij = centers[i] - centers[j]
                grad[2*i:2*i+2] -= factor * diff_ij
                grad[2*j:2*j+2] += factor * diff_ij
    
    return grad

def _initial_grid():
    """Create initial hexagonal packing layout with multiple variants."""
    best_grid = None
    best_r = 0.0
    
    # Try different row/column arrangements with hexagonal offset
    for rows in range(6, 10):
        cols = int(np.ceil(N / rows))
        if rows * cols < N:
            continue
        
        for seed in [12345, 54321, 99999, 77777, 11111]:
            grid = np.zeros((N, 2))
            idx = 0
            for i in range(rows):
                for j in range(cols):
                    if idx >= N:
                        break
                    x = (j + 0.5) / cols - 0.5
                    y = (i + 0.5) / rows - 0.5
                    if i % 2 == 1:
                        x += 0.5 / cols
                    grid[idx] = [x, y]
                    idx += 1
            
            if idx < N:
                continue
            
            # Add small jitter to break symmetry
            rng = np.random.default_rng(seed)
            grid = grid + rng.normal(0, 0.004, grid.shape)
            grid = np.clip(grid, -0.49, 0.49)
            
            r = _compute_radius(grid)
            if r > best_r:
                best_r = r
                best_grid = grid.copy()
    
    if best_grid is None:
        # Fallback to simple grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        best_grid = np.array([(x, y) for y in ticks for x in ticks])[:N].copy()
        rng = np.random.default_rng(12345)
        best_grid = best_grid + rng.normal(0, 0.005, grid.shape)
        best_grid = np.clip(best_grid, -0.49, 0.49)
    
    return best_grid

def _simulated_annealing(centers, rng, time_budget):
    """Simulated annealing to improve packing."""
    start_time = time.time()
    current = centers.copy()
    current_r = _compute_radius(current)
    best = current.copy()
    best_r = current_r
    
    T_start = 0.025
    T_end = 1e-8
    max_steps = 1000000
    
    for step in range(max_steps):
        if time.time() - start_time > time_budget:
            break
        
        # Exponential cooling
        T = T_start * (T_end / T_start) ** (step / max_steps)
        
        # Move a random circle
        idx = rng.integers(0, N)
        step_size = 0.01 * (1.0 + 0.3 * np.cos(np.pi * step / max_steps))
        delta = rng.normal(0, step_size, 2)
        
        new = current.copy()
        new[idx] += delta
        new = np.clip(new, -0.5, 0.5)
        
        new_r = _compute_radius(new)
        
        if new_r > current_r:
            current = new
            current_r = new_r
            if current_r > best_r:
                best_r = current_r
                best = current.copy()
        else:
            delta_r = new_r - current_r
            if delta_r > -2.5 * T:
                prob = np.exp(delta_r / T)
                if rng.random() < prob:
                    current = new
                    current_r = new_r
    
    return best, best_r

def _lbfgs_softmin(x0, tau, time_budget):
    """Run L-BFGS-B optimization on the soft-min objective."""
    start_time = time.time()
    
    def obj_and_grad(x):
        if time.time() - start_time > time_budget:
            return 0.0, np.zeros_like(x)
        f = _soft_min_objective(x, tau)
        g = _grad_soft_min(x, tau)
        return f, g
    
    try:
        result = opt.minimize(
            obj_and_grad, x0, jac=True, method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * (2 * N),
            options={'maxiter': 400, 'ftol': 1e-15, 'gtol': 1e-11,
                     'maxfun': 1500, 'disp': False}
        )
        polished = result.x.reshape(-1, 2)
        polished = np.clip(polished, -0.5, 0.5)
        return polished
    except Exception:
        return x0.reshape(-1, 2)

def _lbfgs_violation(x0, r_target, time_budget):
    """Run L-BFGS-B optimization on the violation objective for target radius."""
    start_time = time.time()
    
    def obj_and_grad(x):
        if time.time() - start_time > time_budget:
            return 0.0, np.zeros_like(x)
        f = _violation_objective(x, r_target)
        g = _violation_grad(x, r_target)
        return f, g
    
    try:
        result = opt.minimize(
            obj_and_grad, x0, jac=True, method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * (2 * N),
            options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10,
                     'maxfun': 1000, 'disp': False}
        )
        polished = result.x.reshape(-1, 2)
        polished = np.clip(polished, -0.5, 0.5)
        return polished
    except Exception:
        return x0.reshape(-1, 2)

def _binary_search_radius(centers, lo, hi, time_budget):
    """Binary search for maximum feasible radius with L-BFGS violation minimization."""
    start_time = time.time()
    best = centers.copy()
    best_r = lo
    
    for iteration in range(18):
        if time.time() - start_time > time_budget:
            break
        
        mid = (lo + hi) / 2
        x0 = best.flatten().copy()
        
        # Use L-BFGS to minimize violation for target radius mid
        trial = _lbfgs_violation(x0, mid, time_budget=min(10, deadline - time.time() - 5))
        trial_r = _compute_radius(trial)
        
        if trial_r >= mid - 1e-10:
            lo = mid
            best = trial
            best_r = trial_r
        else:
            hi = mid
    
    return best, best_r

def construct():
    global deadline
    start_time = time.time()
    deadline = start_time + DEADLINE

    # Initial configuration
    best_centers = _initial_grid()
    best_radius = _compute_radius(best_centers)
    
    rng = np.random.default_rng(777)
    
    # Phase 1: Simulated Annealing
    sa_budget = min(45.0, (deadline - time.time()) * 0.4)
    if sa_budget > 5:
        best_centers, best_radius = _simulated_annealing(best_centers, rng, sa_budget)
    
    # Phase 2: L-BFGS optimization with soft-min (multiple tau values)
    if time.time() < deadline - 35:
        x0 = best_centers.flatten()
        for tau in [0.012, 0.008, 0.005, 0.003]:
            if time.time() > deadline - 20:
                break
            trial = _lbfgs_softmin(x0, tau=tau, time_budget=min(10, deadline - time.time() - 15))
            trial_r = _compute_radius(trial)
            if trial_r > best_radius:
                best_radius = trial_r
                best_centers = trial
            x0 = best_centers.flatten()
    
    # Phase 3: High-precision soft-min polish
    if time.time() < deadline - 20:
        trial = _lbfgs_softmin(best_centers.flatten(), tau=0.002, time_budget=min(12, deadline - time.time() - 10))
        trial_r = _compute_radius(trial)
        if trial_r > best_radius:
            best_radius = trial_r
            best_centers = trial
    
    # Phase 4: Binary search with violation minimization
    if time.time() < deadline - 10:
        lo = best_radius
        hi = min(0.075, best_radius * 1.04)
        best_centers, best_radius = _binary_search_radius(
            best_centers, lo, hi, time_budget=min(20, deadline - time.time() - 5)
        )
    
    # Final validation
    if not np.all(np.isfinite(best_centers)) or best_centers.shape != (N, 2):
        return _initial_grid()
    
    return best_centers
