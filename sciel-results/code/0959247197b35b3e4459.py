"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import scipy.optimize as opt

N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize as opt

N = 50

def _compute_radius(centers):
    """
    Compute the true maximum radius for a given configuration.
    """
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    min_wall = np.min(np.minimum(wall_dists_x, wall_dists_y))
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    return min(min_pair, min_wall)

def _violation(centers, r):
    """
    Compute total violation for radius r.
    Violation = sum of max(0, r - d/2) for pairs + sum of max(0, r - wall_dist) for walls.
    Lower is better. Zero means feasible.
    """
    centers = np.asarray(centers)
    n = len(centers)
    
    # Pairwise violations
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    pair_violation = np.sum(np.maximum(0, r - dists / 2.0))
    
    # Wall violations
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    wall_dists = np.minimum(wall_dists_x, wall_dists_y)
    wall_violation = np.sum(np.maximum(0, r - wall_dists))
    
    return pair_violation + wall_violation

def _grad_violation(centers, r):
    """
    Compute gradient of violation with respect to centers.
    """
    centers = np.asarray(centers)
    n = len(centers)
    grad = np.zeros_like(centers)
    
    # Pairwise gradients
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # For each pair (i,j) where dist/2 < r, add gradient
    # d/dr max(0, r - d/2) = 1 if d/2 < r else 0
    # d/dx_i max(0, r - d/2) = -0.5 * (x_i - x_j)/d * (1 if d/2 < r)
    
    active = dists / 2.0 < r
    active[~np.eye(n, dtype=bool)] = active[~np.eye(n, dtype=bool)]
    
    # Only consider upper triangle to avoid double counting
    for i in range(n):
        for j in range(i+1, n):
            if dists[i, j] / 2.0 < r:
                d = dists[i, j]
                if d > 1e-12:
                    factor = 0.5 / d
                    grad[i] -= factor * (centers[i] - centers[j])
                    grad[j] += factor * (centers[i] - centers[j])
    
    # Wall gradients
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    
    for i in range(n):
        if wall_dists_x[i] < r:
            # d/dx_i max(0, r - (0.5 - |x_i|)) = sign(x_i) if 0.5 - |x_i| < r
            grad[i, 0] += np.sign(centers[i, 0])
        if wall_dists_y[i] < r:
            grad[i, 1] += np.sign(centers[i, 1])
    
    return grad

def _soft_min_obj_grad(x, tau):
    """
    Soft-min objective and gradient: minimize -soft_min(all distances).
    x: flattened array of 2*N coordinates
    tau: temperature parameter for soft-min
    """
    centers = x.reshape(-1, 2)
    n = len(centers)
    
    # 1. Compute all distances
    # Walls
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    walls = np.minimum(wall_dists_x, wall_dists_y)
    wall_active_x = wall_dists_x <= wall_dists_y
    
    # Pairs
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    pair_dists = dists / 2.0
    # Extract upper triangle
    i_idx, j_idx = np.triu_indices(n, k=1)
    pair_vals = pair_dists[i_idx, j_idx]
    pair_diffs = diff[i_idx, j_idx]
    pair_dists_norm = np.linalg.norm(pair_diffs, axis=1)
    
    all_dists = np.concatenate([walls, pair_vals])
    
    # 2. Compute Soft-Min and Weights
    min_d = np.min(all_dists)
    shifted = all_dists - min_d
    exp_terms = np.exp(-shifted / tau)
    sum_exp = np.sum(exp_terms)
    weights = exp_terms / sum_exp
    
    soft_min_val = min_d - tau * np.log(sum_exp)
    obj_val = -soft_min_val
    
    # 3. Compute Gradient
    grad = np.zeros_like(centers)
    
    # Wall gradients
    w_walls = weights[:n]
    for i in range(n):
        w = w_walls[i]
        if w < 1e-12: continue
        if wall_active_x[i]:
            grad[i, 0] -= w * (-np.sign(centers[i, 0]))
        else:
            grad[i, 1] -= w * (-np.sign(centers[i, 1]))
            
    # Pair gradients
    w_pairs = weights[n:]
    with np.errstate(divide='ignore', invalid='ignore'):
        factors = 0.5 / pair_dists_norm
        factors = np.where(pair_dists_norm > 1e-12, factors, 0.0)
        
    contrib = w_pairs[:, np.newaxis] * factors[:, np.newaxis] * pair_diffs
    
    np.add.at(grad, i_idx, -contrib)
    np.add.at(grad, j_idx, contrib)
    
    return obj_val, grad.flatten()

def _sa_search(centers, rng, budget_time):
    """
    Simulated annealing to improve packing.
    """
    start_time = time.time()
    current = centers.copy()
    current_r = _compute_radius(current)
    best = current.copy()
    best_r = current_r
    
    T_start = 0.03
    T_end = 1e-7
    max_steps = 1000000
    
    for step in range(max_steps):
        if time.time() - start_time > budget_time:
            break
        
        # Temperature schedule
        T = T_start * (T_end / T_start) ** (step / max_steps)
        
        # Random perturbation
        idx = rng.integers(0, N)
        step_size = 0.015 * (1.0 + 0.3 * np.cos(np.pi * step / max_steps))
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
            if delta_r > -5 * T:
                prob = np.exp(delta_r / T)
                if rng.random() < prob:
                    current = new
                    current_r = new_r
    
    return best, best_r

def _gradient_polish(centers, r_target, max_iter=500):
    """
    Use gradient descent to minimize violation for a target radius.
    """
    x = centers.flatten().copy()
    lr = 0.005
    
    for i in range(max_iter):
        if time.time() > _deadline:
            break
        
        v = _violation(x.reshape(-1, 2), r_target)
        if v < 1e-10:
            break
        
        g = _grad_violation(x.reshape(-1, 2), r_target).flatten()
        x -= lr * g
        x = np.clip(x, -0.5, 0.5)
    
    return x.reshape(-1, 2)

# Global deadline
_deadline = 0.0

def construct():
    global _deadline
    start_time = time.time()
    _deadline = start_time + 145.0  # 5s safety margin
    
    # Initial grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    best_centers = grid[:N].copy()
    
    # Add small random jitter to break symmetry
    rng = np.random.default_rng(42)
    best_centers = best_centers + rng.normal(0, 0.01, best_centers.shape)
    best_centers = np.clip(best_centers, -0.49, 0.49)
    
    best_radius = _compute_radius(best_centers)
    
    # Phase 1: Simulated Annealing
    remaining = _deadline - time.time()
    sa_budget = min(80.0, remaining * 0.6)
    best_centers, best_radius = _sa_search(best_centers, rng, sa_budget)
    
    # Phase 2: Binary search on radius with violation minimization
    remaining = _deadline - time.time()
    if remaining > 30:
        lo = best_radius
        hi = 0.08  # Upper bound
        
        # Try to find better radius via binary search
        for _ in range(15):
            if time.time() > _deadline - 10:
                break
            
            mid = (lo + hi) / 2
            # Try to find feasible configuration for radius mid
            # Use gradient descent to minimize violation
            trial = best_centers.copy()
            trial = _gradient_polish(trial, mid, max_iter=200)
            
            trial_r = _compute_radius(trial)
            if trial_r >= mid - 1e-9:
                lo = mid
                best_centers = trial
                best_radius = trial_r
            else:
                hi = mid
    
    # Phase 3: Final polish with gradient-based optimization
    remaining = _deadline - time.time()
    if remaining > 15:
        # Use soft-min objective with L-BFGS-B and analytical gradient
        x0 = best_centers.flatten()
        
        # Stage 1: Coarse polish
        tau_1 = 0.02
        def obj1(x):
            return _soft_min_obj_grad(x, tau_1)[0]
        def jac1(x):
            return _soft_min_obj_grad(x, tau_1)[1]
        
        try:
            result = opt.minimize(
                obj1, x0, jac=jac1, method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (2 * N),
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if result.success or result.fun < obj1(x0):
                polished = result.x.reshape(-1, 2)
                polished = np.clip(polished, -0.5, 0.5)
                polished_r = _compute_radius(polished)
                if polished_r > best_radius:
                    best_centers = polished
                    best_radius = polished_r
                    x0 = polished.flatten()
        except Exception:
            pass
            
        # Stage 2: Fine polish
        remaining = _deadline - time.time()
        if remaining > 10:
            tau_2 = 0.005
            def obj2(x):
                return _soft_min_obj_grad(x, tau_2)[0]
            def jac2(x):
                return _soft_min_obj_grad(x, tau_2)[1]
            
            try:
                result = opt.minimize(
                    obj2, x0, jac=jac2, method='L-BFGS-B',
                    bounds=[(-0.5, 0.5)] * (2 * N),
                    options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10}
                )
                if result.success or result.fun < obj2(x0):
                    polished = result.x.reshape(-1, 2)
                    polished = np.clip(polished, -0.5, 0.5)
                    polished_r = _compute_radius(polished)
                    if polished_r > best_radius:
                        best_centers = polished
                        best_radius = polished_r
                        x0 = polished.flatten()
            except Exception:
                pass
                
        # Stage 3: Ultra-fine polish
        remaining = _deadline - time.time()
        if remaining > 5:
            tau_3 = 0.001
            def obj3(x):
                return _soft_min_obj_grad(x, tau_3)[0]
            def jac3(x):
                return _soft_min_obj_grad(x, tau_3)[1]
            
            try:
                result = opt.minimize(
                    obj3, x0, jac=jac3, method='L-BFGS-B',
                    bounds=[(-0.5, 0.5)] * (2 * N),
                    options={'maxiter': 100, 'ftol': 1e-15, 'gtol': 1e-12}
                )
                if result.success or result.fun < obj3(x0):
                    polished = result.x.reshape(-1, 2)
                    polished = np.clip(polished, -0.5, 0.5)
                    polished_r = _compute_radius(polished)
                    if polished_r > best_radius:
                        best_centers = polished
                        best_radius = polished_r
            except Exception:
                pass
    
    # Final check
    if not np.all(np.isfinite(best_centers)):
        return grid[:N]
        
    return best_centers
# EVOLVE-BLOCK-END
