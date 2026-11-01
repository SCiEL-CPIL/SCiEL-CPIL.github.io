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
    Compute gradient of -soft_min objective analytically.
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

def _initial_grid():
    """Create initial hexagonal grid layout."""
    # Hexagonal packing: rows offset by half spacing
    # For 50 circles, use 7 rows: 7,8,7,8,7,8,5 = 50
    rows = [7, 8, 7, 8, 7, 8, 5]
    spacing_x = 1.0 / 8.5  # spacing between columns
    spacing_y = spacing_x * np.sqrt(3) / 2  # hexagonal row spacing
    
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
    
    # Add small jitter to break symmetry
    rng = np.random.default_rng(12345)
    centers = centers + rng.normal(0, 0.002, centers.shape)
    centers = np.clip(centers, -0.49, 0.49)
    
    return centers

def _simulated_annealing(centers, rng, time_budget):
    """
    Simulated annealing to maximize the minimum radius.
    Directly optimizes the min-radius objective.
    """
    start_time = time.time()
    current = centers.copy()
    best = current.copy()
    best_r = _compute_radius(current)
    
    # Temperature schedule
    T_start = 0.02
    T_end = 0.0001
    n_steps = 20000
    
    for step in range(n_steps):
        if time.time() - start_time > time_budget:
            break
        
        # Anneal temperature
        t_frac = step / n_steps
        T = T_start * (T_end / T_start) ** t_frac
        
        # Generate perturbation
        n_perturb = rng.integers(1, 5)
        idx = rng.integers(0, N, size=n_perturb)
        perturbation = rng.normal(0, T, size=(n_perturb, 2))
        
        trial = current.copy()
        trial[idx] += perturbation
        trial = np.clip(trial, -0.5, 0.5)
        
        trial_r = _compute_radius(trial)
        
        # Accept if better, or with probability exp(-(delta_r)/T)
        delta = trial_r - best_r
        if delta > 0 or rng.random() < np.exp(-delta / max(T, 1e-10)):
            current = trial
            if trial_r > best_r:
                best_r = trial_r
                best = trial.copy()
    
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
            options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-12,
                     'maxfun': 2000, 'disp': False}
        )
        polished = result.x.reshape(-1, 2)
        polished = np.clip(polished, -0.5, 0.5)
        return polished
    except Exception:
        return x0.reshape(-1, 2)

def _high_precision_polish(centers, time_budget):
    """
    High-precision polish using very small tau and tight tolerances.
    """
    start_time = time.time()
    current = centers.copy()
    best_r = _compute_radius(current)
    
    # Use very small tau for near-exact min optimization
    for tau in [0.003, 0.002, 0.001]:
        if time.time() - start_time > time_budget:
            break
        
        x0 = current.flatten()
        trial = _lbfgs_optimize(x0, tau=tau, time_budget=min(10, deadline - time.time() - 5) if 'deadline' in dir() else 10)
        trial_r = _compute_radius(trial)
        
        if trial_r > best_r:
            best_r = trial_r
            current = trial
    
    return current, best_r

def _binary_search_radius(centers, lo, hi, time_budget):
    """Binary search for maximum feasible radius with gradient polishing."""
    start_time = time.time()
    best = centers.copy()
    best_r = lo
    
    for _ in range(25):
        if time.time() - start_time > time_budget:
            break
        
        mid = (lo + hi) / 2
        trial = best.copy()
        
        # Use gradient descent to minimize violation for radius mid
        x = trial.flatten().copy()
        lr = 0.001
        for _ in range(200):
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
            if v < 1e-14:
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
    
    # Initial configuration
    best_centers = _initial_grid()
    best_radius = _compute_radius(best_centers)
    
    rng = np.random.default_rng(777)
    
    # Phase 1: Simulated Annealing
    sa_budget = min(50.0, (deadline - time.time()) * 0.4)
    if sa_budget > 5:
        best_centers, best_radius = _simulated_annealing(best_centers, rng, sa_budget)
    
    # Phase 2: L-BFGS optimization with soft-min
    if time.time() < deadline - 40:
        x0 = best_centers.flatten()
        # Try with different tau values
        for tau in [0.01, 0.008, 0.005, 0.003]:
            if time.time() > deadline - 20:
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
    
    # Phase 4: Binary search with gradient polish
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
