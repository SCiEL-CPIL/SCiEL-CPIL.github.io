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
    
    tree = cKDTree(centers)
    dists, _ = tree.query(centers, k=2)
    min_pair = np.min(dists[:, 1]) / 2.0
    
    return min(min_pair, min_wall)

def _force_directed_relax(centers, target_r, max_iter=5000, time_budget=10.0):
    """
    Force-directed relaxation: push overlapping circles apart and keep them in bounds.
    Uses iterative pairwise repulsion for circles closer than 2*target_r.
    """
    start_time = time.time()
    c = centers.copy()
    
    for it in range(max_iter):
        if time.time() - start_time > time_budget:
            break
        
        # Compute all pairwise distances
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Find violating pairs
        mask = dists < 2 * target_r
        if not np.any(mask):
            break
        
        # Compute forces: repulsion for close pairs
        forces = np.zeros_like(c)
        iu, ju = np.triu_indices(N, k=1)
        for i, j in zip(iu[mask[iu, ju]], ju[mask[iu, ju]]):
            d = dists[i, j]
            if d > 1e-12:
                # Repulsion strength proportional to overlap
                overlap = 2 * target_r - d
                strength = 0.5 * overlap / d
                diff_ij = c[i] - c[j]
                forces[i] += strength * diff_ij
                forces[j] -= strength * diff_ij
        
        # Wall forces: push circles away from walls
        for i in range(N):
            for axis in range(2):
                coord = c[i, axis]
                dist_to_wall = 0.5 - abs(coord)
                if dist_to_wall < target_r:
                    # Push away from wall
                    push = (target_r - dist_to_wall) * 0.5
                    forces[i, axis] += np.sign(coord) * push
        
        # Update positions with adaptive step size
        max_force = np.max(np.abs(forces))
        if max_force > 1e-10:
            step = 0.1 / max_force
            c += step * forces
            c = np.clip(c, -0.5 + 1e-9, 0.5 - 1e-9)
    
    return c

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
    Compute gradient of -soft_min objective analytically (vectorized).
    """
    centers = x.reshape(-1, 2)
    
    # Wall distances
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    walls = np.minimum(wall_dists_x, wall_dists_y)
    
    # Pairwise distances (vectorized)
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
    
    # Wall gradient contributions (vectorized)
    wall_weights = weights[:N]
    sx = np.sign(centers[:, 0])
    sy = np.sign(centers[:, 1])
    grad[0::2] += wall_weights * sx
    grad[1::2] += wall_weights * sy
    
    # Pairwise gradient contributions (vectorized)
    pair_weights = weights[N:]
    d_vals = dists[iu, ju]
    valid = d_vals > 1e-12
    factors = np.zeros(len(iu))
    factors[valid] = 0.5 * pair_weights[valid] / d_vals[valid]
    
    # For each pair (i,j) with weight w:
    # grad[i] -= factor * (c[i] - c[j])
    # grad[j] += factor * (c[i] - c[j])
    diff_ij = centers[iu] - centers[ju]  # shape (M, 2)
    weighted_diff = factors[:, np.newaxis] * diff_ij  # shape (M, 2)
    
    # Scatter subtract from i indices
    np.add.at(grad, (iu * 2), -weighted_diff[:, 0])
    np.add.at(grad, (iu * 2 + 1), -weighted_diff[:, 1])
    # Scatter add to j indices
    np.add.at(grad, (ju * 2), weighted_diff[:, 0])
    np.add.at(grad, (ju * 2 + 1), weighted_diff[:, 1])
    
    return grad

def _initial_grid():
    """Create initial hexagonal grid layout."""
    rows = [7, 8, 7, 8, 7, 8, 5]
    spacing_x = 1.0 / 8.5
    spacing_y = spacing_x * np.sqrt(3) / 2
    
    centers = []
    y_pos = -0.5 + spacing_y * 0.5
    for r_idx, n_cols in enumerate(rows):
        offset = 0.5 * spacing_x if r_idx % 2 == 1 else 0.0
        x_start = -0.5 + spacing_x * 0.5 + offset
        for c in range(n_cols):
            x = x_start + c * spacing_x
            y = y_pos + r_idx * spacing_y
            centers.append([x, y])
    
    centers = np.array(centers[:N])
    
    rng = np.random.default_rng(12345)
    centers = centers + rng.normal(0, 0.002, centers.shape)
    centers = np.clip(centers, -0.49, 0.49)
    
    return centers

def _hex_init(seed=42):
    """Hexagonal packing with different row configurations."""
    rng = np.random.default_rng(seed)
    
    # Try different hexagonal arrangements
    best = None
    best_r = 0
    
    for attempt in range(5):
        # Random number of rows between 6 and 9
        n_rows = rng.integers(6, 10)
        # Distribute 50 circles among rows
        base = N // n_rows
        extra = N % n_rows
        rows = [base + (1 if i < extra else 0) for i in range(n_rows)]
        
        # Hexagonal spacing
        max_cols = max(rows)
        spacing_x = 1.0 / (max_cols + 0.5)
        spacing_y = spacing_x * np.sqrt(3) / 2
        
        total_height = (n_rows - 1) * spacing_y
        y_start = -0.5 + 0.5 * spacing_y
        if total_height > 1.0 - spacing_y:
            continue
        
        centers = []
        for r_idx, n_cols in enumerate(rows):
            offset = 0.5 * spacing_x if r_idx % 2 == 1 else 0.0
            x_start = -0.5 + spacing_x * 0.5 + offset
            for c in range(n_cols):
                x = x_start + c * spacing_x
                y = y_start + r_idx * spacing_y
                centers.append([x, y])
        
        if len(centers) != N:
            continue
        
        centers = np.array(centers)
        centers = centers + rng.normal(0, 0.003, centers.shape)
        centers = np.clip(centers, -0.49, 0.49)
        
        r = _compute_radius(centers)
        if r > best_r:
            best_r = r
            best = centers
    
    return best if best is not None else _initial_grid()

def _random_init(seed=99):
    """Random initial positions with minimum spacing check."""
    rng = np.random.default_rng(seed)
    
    for attempt in range(10):
        centers = rng.uniform(-0.45, 0.45, size=(N, 2))
        r = _compute_radius(centers)
        if r > 0.05:
            return centers
    
    return _initial_grid()

def _simulated_annealing(centers, rng, time_budget):
    """
    Simulated annealing with adaptive step size and better temperature schedule.
    """
    start_time = time.time()
    current = centers.copy()
    best = current.copy()
    best_r = _compute_radius(current)
    
    # Temperature schedule with geometric cooling
    T_start = 0.025
    T_end = 0.00005
    n_steps = 30000
    
    # Track recent improvements for adaptive step size
    recent_improvements = []
    base_step = T_start
    
    for step in range(n_steps):
        if time.time() - start_time > time_budget:
            break
        
        t_frac = step / n_steps
        T = T_start * (T_end / T_start) ** t_frac
        
        # Adaptive step size: reduce if many rejections, increase if stagnating
        if step % 100 == 0 and len(recent_improvements) > 50:
            recent_improvements = recent_improvements[-100:]
            improvement_rate = np.mean(recent_improvements)
            if improvement_rate < 0.01:
                base_step *= 0.8
            elif improvement_rate > 0.05:
                base_step = min(base_step * 1.1, T_start)
        
        # Generate perturbation with adaptive scale
        n_perturb = rng.integers(1, 4)
        idx = rng.integers(0, N, size=n_perturb)
        step_scale = max(T, base_step * 0.3)
        perturbation = rng.normal(0, step_scale, size=(n_perturb, 2))
        
        trial = current.copy()
        trial[idx] += perturbation
        trial = np.clip(trial, -0.5, 0.5)
        
        trial_r = _compute_radius(trial)
        
        # Accept if better than current, or with probability based on delta from current
        delta = trial_r - _compute_radius(current)
        if delta > 0 or rng.random() < np.exp(-delta / max(T, 1e-10)):
            current = trial
            if trial_r > best_r:
                best_r = trial_r
                best = trial.copy()
            recent_improvements.append(1.0 if trial_r > best_r else 0.0)
        else:
            recent_improvements.append(0.0)
    
    return best, best_r

def _lbfgs_optimize(x0, tau=0.008, time_budget=20.0):
    """Run L-BFGS-B optimization on the soft-min objective with tighter tolerances."""
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
            options={'maxiter': 1000, 'ftol': 1e-18, 'gtol': 1e-14,
                     'maxfun': 5000, 'disp': False, 'maxls': 50}
        )
        polished = result.x.reshape(-1, 2)
        polished = np.clip(polished, -0.5, 0.5)
        return polished
    except Exception:
        return x0.reshape(-1, 2)

def _high_precision_polish(centers, time_budget):
    """
    High-precision polish using very small tau, force-directed relaxation, and tight tolerances.
    """
    start_time = time.time()
    current = centers.copy()
    best_r = _compute_radius(current)
    
    # Phase 1: L-BFGS with decreasing tau
    for tau in [0.003, 0.002, 0.001, 0.0005]:
        if time.time() - start_time > time_budget:
            break
        
        x0 = current.flatten()
        remaining = time_budget - (time.time() - start_time)
        if remaining < 5:
            break
        trial = _lbfgs_optimize(x0, tau=tau, time_budget=min(15, remaining * 0.7))
        trial_r = _compute_radius(trial)
        
        if trial_r > best_r:
            best_r = trial_r
            current = trial
    
    # Phase 2: Force-directed relaxation to push toward target radius
    if time.time() - start_time < time_budget - 5:
        target_r = best_r * 1.001
        remaining = time_budget - (time.time() - start_time)
        if remaining > 3:
            relaxed = _force_directed_relax(current, target_r, max_iter=3000, time_budget=min(10, remaining))
            relaxed_r = _compute_radius(relaxed)
            if relaxed_r > best_r:
                best_r = relaxed_r
                current = relaxed
    
    # Phase 3: Final L-BFGS polish with very tight settings
    if time.time() - start_time < time_budget - 2:
        x0 = current.flatten()
        remaining = time_budget - (time.time() - start_time)
        if remaining > 2:
            trial = _lbfgs_optimize(x0, tau=0.0003, time_budget=remaining)
            trial_r = _compute_radius(trial)
            if trial_r > best_r:
                best_r = trial_r
                current = trial
    
    return current, best_r

def _binary_search_radius(centers, lo, hi, time_budget):
    """Binary search for maximum feasible radius with force-directed relaxation."""
    start_time = time.time()
    best = centers.copy()
    best_r = lo
    
    for _ in range(20):
        if time.time() - start_time > time_budget:
            break
        
        mid = (lo + hi) / 2
        
        # Use force-directed relaxation to try to achieve radius mid
        trial = _force_directed_relax(best, mid, max_iter=2000, time_budget=min(5, time_budget - (time.time() - start_time)))
        trial_r = _compute_radius(trial)
        
        if trial_r >= mid - 1e-10:
            lo = mid
            best = trial
            best_r = trial_r
        else:
            hi = mid
    
    return best, best_r

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Try multiple initializations and keep the best
    best_centers = None
    best_radius = 0.0
    
    inits = [
        _initial_grid(),
        _hex_init(42),
        _hex_init(123),
        _random_init(99),
    ]
    
    for init in inits:
        r = _compute_radius(init)
        if r > best_radius:
            best_radius = r
            best_centers = init
    
    rng = np.random.default_rng(777)
    
    # Phase 1: Simulated Annealing on best initialization
    sa_budget = min(55.0, (deadline - time.time()) * 0.45)
    if sa_budget > 5:
        best_centers, best_radius = _simulated_annealing(best_centers, rng, sa_budget)
    
    # Phase 2: L-BFGS optimization with soft-min
    if time.time() < deadline - 45:
        x0 = best_centers.flatten()
        for tau in [0.01, 0.008, 0.005, 0.003, 0.002]:
            if time.time() > deadline - 25:
                break
            remaining = deadline - time.time() - 10
            if remaining < 5:
                break
            trial = _lbfgs_optimize(x0, tau=tau, time_budget=min(15, remaining))
            trial_r = _compute_radius(trial)
            if trial_r > best_radius:
                best_radius = trial_r
                best_centers = trial
            x0 = best_centers.flatten()
    
    # Phase 3: High-precision polish
    if time.time() < deadline - 15:
        remaining = deadline - time.time() - 5
        if remaining > 5:
            best_centers, best_radius = _high_precision_polish(best_centers, remaining)
    
    # Phase 4: Binary search with force-directed relaxation
    if time.time() < deadline - 5:
        lo = best_radius
        hi = min(0.08, best_radius * 1.05)
        remaining = deadline - time.time() - 2
        if remaining > 3:
            best_centers, best_radius = _binary_search_radius(
                best_centers, lo, hi, time_budget=remaining
            )
    
    # Final validation
    if not np.all(np.isfinite(best_centers)) or best_centers.shape != (N, 2):
        return _initial_grid()
    
    return best_centers
