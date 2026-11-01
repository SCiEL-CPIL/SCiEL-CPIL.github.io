"""Model-and-optimize approach for 50 equal circles in a unit square."""
import numpy as np
import time
import scipy.optimize as opt
from scipy.spatial import cKDTree

N = 50
DEADLINE = 145.0  # Safety margin before 150s kill

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
    Compute gradient of -soft_min objective numerically (for safety)
    or analytically.
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
        # d(wall_i)/d(centers[i,0]) = -sign(centers[i,0]) if x is limiting
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

def _initial_grid():
    """Create initial grid layout with slight jitter."""
    # Use a hexagonal-like arrangement for better packing
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    grid = grid[:N].copy()
    
    rng = np.random.default_rng(12345)
    grid = grid + rng.normal(0, 0.005, grid.shape)
    grid = np.clip(grid, -0.49, 0.49)
    
    return grid

def _hex_grid():
    """Create hexagonal lattice initialization."""
    # For 50 circles, use a hexagonal arrangement
    # 7 columns: 4,5,4,5,4,5,3 = 30, need more
    # Try 8 columns with varying rows
    # Better: use a proper hex packing
    points = []
    # Hex lattice with spacing s
    s = 0.135
    rows = 7
    for row in range(rows):
        y = (row - 3) * s * np.sqrt(3) / 2
        if row % 2 == 0:
            x_start = -3 * s
            n_cols = 7
        else:
            x_start = -3 * s + s/2
            n_cols = 7
        for col in range(n_cols):
            x = x_start + col * s
            points.append([x, y])
    
    points = np.array(points[:N])
    # Center the arrangement
    cx, cy = points[:, 0].mean(), points[:, 1].mean()
    points[:, 0] -= cx
    points[:, 1] -= cy
    # Scale to fit
    max_extent = np.max(np.abs(points))
    if max_extent > 0.45:
        points *= 0.45 / max_extent
    
    rng = np.random.default_rng(42)
    points += rng.normal(0, 0.003, points.shape)
    points = np.clip(points, -0.49, 0.49)
    
    return points

def _simulated_annealing(centers, rng, time_budget):
    """
    Simulated annealing to optimize the minimum radius directly.
    Proposes random moves and accepts based on improvement in min radius.
    """
    start_time = time.time()
    current = centers.copy()
    best = current.copy()
    best_r = _compute_radius(current)
    
    # Temperature schedule
    T_start = 0.02
    T_end = 0.0001
    n_steps = 30000
    cooling = (T_end / T_start) ** (1.0 / max(n_steps, 1))
    T = T_start
    
    for step in range(n_steps):
        if time.time() - start_time > time_budget:
            break
        
        # Propose a random move
        idx = rng.integers(0, N)
        move = rng.normal(0, T, 2)
        trial = current.copy()
        trial[idx] += move
        trial[idx] = np.clip(trial[idx], -0.49, 0.49)
        
        trial_r = _compute_radius(trial)
        delta = trial_r - best_r
        
        # Accept if better or with probability exp(delta/T)
        if delta > 0 or rng.random() < np.exp(delta / max(T, 1e-10)):
            current = trial
            if trial_r > best_r:
                best_r = trial_r
                best = trial.copy()
        
        T *= cooling
    
    return best, best_r

def _force_directed_relax(centers, rng, time_budget):
    """
    Force-directed relaxation to spread circles.
    Uses repulsive forces between circles and from walls.
    """
    start_time = time.time()
    current = centers.copy()
    best = current.copy()
    best_r = _compute_radius(current)
    
    # Parameters
    k_repulse = 1.0
    k_wall = 1.0
    damping = 0.8
    momentum = np.zeros_like(current)
    step_size = 0.01
    
    for step in range(5000):
        if time.time() - start_time > time_budget:
            break
            
        forces = np.zeros_like(current)
        
        # Pairwise repulsive forces
        # O(N^2) is fine for N=50
        for i in range(N):
            for j in range(i + 1, N):
                diff = current[i] - current[j]
                dist = np.linalg.norm(diff)
                if dist > 1e-6:
                    # Force magnitude ~ 1/dist^2
                    mag = k_repulse / (dist * dist)
                    unit = diff / dist
                    forces[i] += mag * unit
                    forces[j] -= mag * unit
                else:
                    # Random jitter if overlapping
                    forces[i] += rng.normal(0, 0.1, 2)
                    forces[j] -= rng.normal(0, 0.1, 2)
        
        # Wall repulsive forces
        for i in range(N):
            # x-walls
            wx_left = current[i, 0] + 0.5
            wx_right = 0.5 - current[i, 0]
            if wx_left < 0.1:
                forces[i, 0] += k_wall / (wx_left * wx_left + 1e-6)
            if wx_right < 0.1:
                forces[i, 0] -= k_wall / (wx_right * wx_right + 1e-6)
            
            # y-walls
            wy_bottom = current[i, 1] + 0.5
            wy_top = 0.5 - current[i, 1]
            if wy_bottom < 0.1:
                forces[i, 1] += k_wall / (wy_bottom * wy_bottom + 1e-6)
            if wy_top < 0.1:
                forces[i, 1] -= k_wall / (wy_top * wy_top + 1e-6)
                
        # Update positions
        momentum = damping * momentum + forces
        current += step_size * momentum
        current = np.clip(current, -0.5, 0.5)
        
        # Anneal step size
        step_size *= 0.995
        
        r_curr = _compute_radius(current)
        if r_curr > best_r:
            best_r = r_curr
            best = current.copy()
            
    return best, best_r

def _lbfgs_optimize(x0, tau=0.008, time_budget=20.0):
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
            options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10,
                     'maxfun': 1000, 'disp': False}
        )
        polished = result.x.reshape(-1, 2)
        polished = np.clip(polished, -0.5, 0.5)
        return polished
    except Exception:
        return x0.reshape(-1, 2)

def _high_precision_polish(centers, time_budget):
    """
    High-precision polish using very tight tolerances and exact local solve.
    Uses a smaller tau for the soft-min to get closer to the true min.
    """
    start_time = time.time()
    x0 = centers.flatten().copy()
    
    def obj_and_grad(x):
        if time.time() - start_time > time_budget:
            return 0.0, np.zeros_like(x)
        f = _soft_min_objective(x, tau=0.001)
        g = _grad_soft_min(x, tau=0.001)
        return f, g
    
    try:
        result = opt.minimize(
            obj_and_grad, x0, jac=True, method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * (2 * N),
            options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-12,
                     'maxfun': 2000, 'disp': False}
        )
        polished = result.x.reshape(-1, 2)
        polished = np.clip(polished, -0.5, 0.5)
        return polished
    except Exception:
        return centers

def _binary_search_radius(centers, lo, hi, time_budget):
    """Binary search for maximum feasible radius with gradient polishing."""
    start_time = time.time()
    best = centers.copy()
    best_r = lo
    
    for _ in range(20):
        if time.time() - start_time > time_budget:
            break
        
        mid = (lo + hi) / 2
        trial = best.copy()
        
        # Use gradient descent to minimize violation for radius mid
        x = trial.flatten().copy()
        lr = 0.002
        for _ in range(100):
            if time.time() - start_time > time_budget:
                break
            c = x.reshape(-1, 2)
            
            # Compute violation
            diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dists, np.inf)
            pair_viol = np.sum(np.maximum(0, mid - dists / 2.0))
            
            wall_x = 0.5 - np.abs(c[:, 0])
            wall_y = 0.5 - np.abs(c[:, 1])
            wall_viol = np.sum(np.maximum(0, mid - np.minimum(wall_x, wall_y)))
            
            v = pair_viol + wall_viol
            if v < 1e-12:
                break
            
            # Gradient
            grad = np.zeros_like(x)
            for i in range(N):
                for j in range(i+1, N):
                    d = dists[i, j]
                    if d < 2 * mid and d > 1e-12:
                        factor = 0.5 / d
                        diff_ij = c[i] - c[j]
                        grad[2*i:2*i+2] -= factor * diff_ij
                        grad[2*j:2*j+2] += factor * diff_ij
                if wall_x[i] < mid:
                    grad[2*i] += np.sign(c[i, 0])
                if wall_y[i] < mid:
                    grad[2*i+1] += np.sign(c[i, 1])
            
            x -= lr * grad
            x = np.clip(x, -0.5, 0.5)
        
        trial = x.reshape(-1, 2)
        trial_r = _compute_radius(trial)
        
        if trial_r >= mid - 1e-9:
            lo = mid
            best = trial
            best_r = trial_r
        else:
            hi = mid
    
    return best, best_r

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Initial configuration: try both grid and hex
    best_centers = _hex_grid()
    best_radius = _compute_radius(best_centers)
    
    grid_centers = _initial_grid()
    grid_radius = _compute_radius(grid_centers)
    if grid_radius > best_radius:
        best_centers = grid_centers
        best_radius = grid_radius
    
    rng = np.random.default_rng(777)
    
    # Phase 1: Force-Directed Relaxation
    fd_budget = min(25.0, (deadline - time.time()) * 0.3)
    if fd_budget > 5:
        trial_centers, trial_r = _force_directed_relax(best_centers, rng, fd_budget)
        if trial_r > best_radius:
            best_radius = trial_r
            best_centers = trial_centers
    
    # Phase 2: Simulated Annealing for global exploration
    sa_budget = min(20.0, (deadline - time.time()) * 0.25)
    if sa_budget > 5:
        trial_centers, trial_r = _simulated_annealing(best_centers, rng, sa_budget)
        if trial_r > best_radius:
            best_radius = trial_r
            best_centers = trial_centers
    
    # Phase 3: L-BFGS optimization with soft-min
    if time.time() < deadline - 35:
        x0 = best_centers.flatten()
        # Try with different tau values
        for tau in [0.01, 0.008, 0.005]:
            if time.time() > deadline - 20:
                break
            trial = _lbfgs_optimize(x0, tau=tau, time_budget=min(10, deadline - time.time() - 10))
            trial_r = _compute_radius(trial)
            if trial_r > best_radius:
                best_radius = trial_r
                best_centers = trial
            x0 = best_centers.flatten()
    
    # Phase 4: Binary search with gradient polish
    if time.time() < deadline - 15:
        lo = best_radius
        hi = min(0.08, best_radius * 1.05)
        best_centers, best_radius = _binary_search_radius(
            best_centers, lo, hi, time_budget=min(10, deadline - time.time() - 5)
        )
    
    # Phase 5: High-precision polish
    if time.time() < deadline - 5:
        trial = _high_precision_polish(best_centers, time_budget=min(5, deadline - time.time() - 2))
        trial_r = _compute_radius(trial)
        if trial_r > best_radius:
            best_radius = trial_r
            best_centers = trial
    
    # Final validation
    if not np.all(np.isfinite(best_centers)) or best_centers.shape != (N, 2):
        return _initial_grid()
    
    return best_centers
