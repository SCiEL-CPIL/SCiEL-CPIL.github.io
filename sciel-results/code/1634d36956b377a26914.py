"""
Fifty equal circles in a unit square - smooth optimization approach.
Uses a soft-min (log-sum-exp) surrogate for the minimum distance objective
and L-BFGS for optimization, combined with Simulated Annealing for global search.
"""
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

    tree = cKDTree(centers)
    dists, idx = tree.query(centers, k=2)
    min_pair = np.min(dists[:, 1]) / 2.0

    return min(min_pair, min_wall)


def _soft_min_objective(x, tau=0.01):
    """
    Objective function to minimize: -soft_min(min_distance).
    We want to maximize the minimum distance, so we minimize the negative soft-min.
    soft_min(d) = -tau * log(sum(exp(-d_i/tau)))
    """
    c = x.reshape(-1, 2)
    
    # Wall distances
    wall_x = 0.5 - np.abs(c[:, 0])
    wall_y = 0.5 - np.abs(c[:, 1])
    walls = np.minimum(wall_x, wall_y)
    
    # Pairwise distances (half distance is radius)
    tree = cKDTree(c)
    dists, _ = tree.query(c, k=2)
    pair_dists = dists[:, 1] / 2.0
    
    # Combine all distances
    all_dists = np.concatenate([walls, pair_dists])
    
    # Soft min
    min_d = np.min(all_dists)
    shifted = all_dists - min_d
    log_sum = np.log(np.sum(np.exp(-shifted / tau)))
    soft_min = min_d - tau * log_sum
    
    return -soft_min


def _soft_min_gradient(x, tau=0.01):
    """
    Gradient of the soft-min objective.
    """
    c = x.reshape(-1, 2)
    grad = np.zeros_like(x)
    
    # Wall distances
    wall_x = 0.5 - np.abs(c[:, 0])
    wall_y = 0.5 - np.abs(c[:, 1])
    walls = np.minimum(wall_x, wall_y)
    
    # Pairwise distances
    tree = cKDTree(c)
    dists, idx = tree.query(c, k=2)
    pair_dists = dists[:, 1] / 2.0
    
    all_dists = np.concatenate([walls, pair_dists])
    min_d = np.min(all_dists)
    shifted = all_dists - min_d
    exp_vals = np.exp(-shifted / tau)
    weights = exp_vals / np.sum(exp_vals)
    
    # Gradient of walls
    for i in range(N):
        w = weights[i]
        if w < 1e-15:
            continue
        # d(wall_x)/dx = -sign(x), d(wall_y)/dy = -sign(y)
        # We want gradient of -soft_min, so sign flips
        # soft_min = min_d - tau * log_sum
        # d(soft_min)/dx = sum(w_i * d(d_i)/dx)
        # d(-soft_min)/dx = -sum(w_i * d(d_i)/dx)
        
        # For wall_x: d(wall_x)/dx = -sign(x)
        # So contribution to grad[-soft_min] is -w * (-sign(x)) = w * sign(x)
        if walls[i] == wall_x[i]:
            grad[2*i] += w * np.sign(c[i, 0])
        # For wall_y: d(wall_y)/dy = -sign(y)
        # So contribution to grad[-soft_min] is -w * (-sign(y)) = w * sign(y)
        if walls[i] == wall_y[i]:
            grad[2*i+1] += w * np.sign(c[i, 1])
            
    # Gradient of pair distances
    for i in range(N):
        w = weights[N + i]
        if w < 1e-15:
            continue
        j = idx[i, 1]
        d = dists[i, 1]
        if d < 1e-12:
            continue
        # pair_dist = d/2
        # d(pair_dist)/dc_i = 0.5 * (c_i - c_j) / d
        # Contribution to grad[-soft_min] = -w * 0.5 * (c_i - c_j) / d
        diff = c[i] - c[j]
        factor = -0.5 * w / d
        grad[2*i:2*i+2] += factor * diff
        grad[2*j:2*j+2] -= factor * diff
        
    return grad


def _initial_hexagonal():
    """Create initial hexagonal packing layout."""
    row_counts = [8, 7, 8, 7, 8, 7, 5]
    points = []
    total_rows = len(row_counts)
    vert_spacing = 0.9 / (total_rows - 1)

    for row_idx, count in enumerate(row_counts):
        y = -0.45 + row_idx * vert_spacing
        horiz_spacing = 0.9 / count if count > 1 else 0
        x_start = -0.45 + (0.9 - horiz_spacing * (count - 1)) / 2
        if row_idx % 2 == 1:
            x_offset = horiz_spacing / 2
        else:
            x_offset = 0
        for i in range(count):
            x = x_start + i * horiz_spacing + x_offset
            points.append([x, y])

    points = np.array(points[:N])
    rng = np.random.default_rng(12345)
    points = points + rng.normal(0, 0.005, points.shape)
    points = np.clip(points, -0.49, 0.49)
    return points


def _simulated_annealing(centers, rng, time_budget):
    """
    Simulated annealing to improve the configuration.
    Uses the soft-min objective for acceptance criteria.
    """
    start_time = time.time()
    current = centers.copy()
    best = current.copy()
    best_obj = _soft_min_objective(current.flatten(), tau=0.01)
    best_r = _compute_radius(current)

    T_start = 0.005
    T_end = 1e-6
    n_steps = 100000
    cooling = (T_end / T_start) ** (1.0 / max(n_steps, 1))
    T = T_start

    for step in range(n_steps):
        if time.time() - start_time > time_budget:
            break

        # Pick a random circle to move
        i = rng.integers(N)
        
        # Move size scales with temperature
        move_size = 0.001 + 0.01 * (T / T_start)
        delta = rng.normal(0, move_size, 2)
        
        trial = current.copy()
        trial[i] += delta
        trial[i, 0] = np.clip(trial[i, 0], -0.49, 0.49)
        trial[i, 1] = np.clip(trial[i, 1], -0.49, 0.49)
        
        curr_obj = _soft_min_objective(current.flatten(), tau=0.01)
        trial_obj = _soft_min_objective(trial.flatten(), tau=0.01)
        
        delta_obj = trial_obj - curr_obj
        
        if delta_obj <= 0 or rng.random() < np.exp(-delta_obj / max(T, 1e-12)):
            current = trial
            if trial_obj < best_obj:
                best_obj = trial_obj
                best = trial.copy()
                
        T *= cooling

    # Update best radius
    best_r = _compute_radius(best)
    return best, best_r


def _lbfgs_polish(centers, time_budget, tau=0.01, maxiter=500):
    """
    L-BFGS polish using the soft-min objective.
    """
    start_time = time.time()
    x0 = centers.flatten().copy()
    
    # Adaptive tau: start larger for global-ish search, decrease for precision
    # We'll run multiple passes with decreasing tau if time permits
    
    def _obj(x):
        return _soft_min_objective(x, tau=tau)
        
    def _grad(x):
        return _soft_min_gradient(x, tau=tau)
        
    try:
        result = opt.minimize(
            _obj, x0, jac=_grad, method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * (2 * N),
            options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-10,
                     'maxfun': maxiter * 2, 'disp': False}
        )
        polished = result.x.reshape(-1, 2)
        polished = np.clip(polished, -0.5, 0.5)
        return polished
    except Exception:
        return centers


def _high_precision_polish(centers, time_budget):
    """
    High precision polish with very small tau and tight tolerances.
    """
    start_time = time.time()
    
    # Use a very small tau to approximate the true min
    tau = 0.001
    
    def _obj(x):
        return _soft_min_objective(x, tau=tau)
        
    def _grad(x):
        return _soft_min_gradient(x, tau=tau)
        
    x0 = centers.flatten().copy()
    
    try:
        result = opt.minimize(
            _obj, x0, jac=_grad, method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * (2 * N),
            options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-12,
                     'maxfun': 2000, 'disp': False}
        )
        polished = result.x.reshape(-1, 2)
        polished = np.clip(polished, -0.5, 0.5)
        return polished
    except Exception:
        return centers


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE

    # Initial configuration
    best_centers = _initial_hexagonal()
    best_radius = _compute_radius(best_centers)

    rng = np.random.default_rng(777)

    # Phase 1: Simulated Annealing
    # Allocate ~30% of time to SA
    sa_budget = min(45.0, (deadline - time.time()) * 0.3)
    if sa_budget > 5:
        best_centers, sa_r = _simulated_annealing(best_centers, rng, sa_budget)
        if sa_r > best_radius:
            best_radius = sa_r

    # Phase 2: L-BFGS Polish (General)
    # Allocate ~30% of remaining time
    remaining = deadline - time.time()
    lbfgs_budget = min(30.0, remaining * 0.3)
    if lbfgs_budget > 5:
        trial = _lbfgs_polish(best_centers, lbfgs_budget, tau=0.01, maxiter=300)
        trial_r = _compute_radius(trial)
        if trial_r > best_radius:
            best_radius = trial_r
            best_centers = trial

    # Phase 3: High Precision Polish
    # Allocate remaining time
    remaining = deadline - time.time() - 5
    if remaining > 10:
        trial = _high_precision_polish(best_centers, remaining)
        trial_r = _compute_radius(trial)
        if trial_r > best_radius:
            best_radius = trial_r
            best_centers = trial

    # Final validation
    if not np.all(np.isfinite(best_centers)) or best_centers.shape != (N, 2):
        return _initial_hexagonal()

    return best_centers
