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
HALF = 0.5

def _compute_radius(centers):
    """
    Compute the true maximum radius for a given configuration.
    """
    centers = np.asarray(centers)
    wall_dists_x = HALF - np.abs(centers[:, 0])
    wall_dists_y = HALF - np.abs(centers[:, 1])
    min_wall = np.min(np.minimum(wall_dists_x, wall_dists_y))
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    return min(min_pair, min_wall)

def _violation_and_grad(centers, r):
    """
    Compute total violation and its gradient for radius r.
    Vectorized for speed.
    """
    centers = np.asarray(centers)
    n = len(centers)
    
    # Pairwise: diff[i,j] = centers[i] - centers[j]
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # Pairwise violation: sum over i<j of max(0, r - dists[i,j]/2)
    # Only upper triangle to avoid double counting
    iu = np.triu_indices(n, k=1)
    pair_dists = dists[iu]
    pair_active = pair_dists / 2.0 < r
    pair_viol = np.sum(np.maximum(0, r - pair_dists / 2.0))
    
    # Wall distances
    wall_dists_x = HALF - np.abs(centers[:, 0])
    wall_dists_y = HALF - np.abs(centers[:, 1])
    wall_dists = np.minimum(wall_dists_x, wall_dists_y)
    wall_viol = np.sum(np.maximum(0, r - wall_dists))
    
    total_viol = pair_viol + wall_viol
    
    # Gradient computation (vectorized)
    grad = np.zeros_like(centers)
    
    # Pairwise gradient contribution
    # For pair (i,j) with dist < 2r:
    #   d/dx_i = -0.5 * (x_i - x_j) / dist
    #   d/dx_j = +0.5 * (x_i - x_j) / dist
    # We accumulate using the full diff matrix
    pair_factor = np.zeros((n, n, 2))
    active_mask = (dists / 2.0 < r) & (dists > 1e-12)
    np.fill_diagonal(active_mask, False)
    
    # factor[i,j] = 0.5/dists[i,j] * diff[i,j] if active else 0
    with np.errstate(divide='ignore', invalid='ignore'):
        inv_dists = np.where(dists > 1e-12, 1.0 / dists, 0.0)
    pair_factor = active_mask[:, :, np.newaxis] * (0.5 * inv_dists[:, :, np.newaxis]) * diff
    
    # grad[i] = -sum_j pair_factor[i,j]  (from pairs where i is first)
    # grad[j] = +sum_i pair_factor[i,j]  (from pairs where j is second)
    # Since pair_factor[i,j] = 0.5/d * (x_i - x_j), and we want:
    #   contribution to grad[i] from pair (i,j): -0.5/d * (x_i - x_j) = -pair_factor[i,j]
    #   contribution to grad[j] from pair (i,j): +0.5/d * (x_i - x_j) = +pair_factor[i,j]
    # So: grad[i] -= sum_j pair_factor[i,j]
    #     grad[j] += sum_i pair_factor[i,j]
    # Equivalently: grad = -row_sum + col_sum where we use the full matrix
    # But since pair_factor is antisymmetric (pair_factor[i,j] = -pair_factor[j,i]),
    # grad[i] = sum_j (-pair_factor[i,j]) for j != i, but we need to be careful.
    
    # Actually, let's just do it directly:
    # For each i, grad[i] gets contribution from all j:
    #   if j > i and active: grad[i] += -pair_factor[i,j], grad[j] += pair_factor[i,j]
    #   if j < i and active: grad[i] += pair_factor[j,i] (since pair_factor[j,i] = -pair_factor[i,j])
    # This is getting complex. Let's use a simpler approach:
    # grad[i] = -sum_{j!=i, active} 0.5/dists[i,j] * (centers[i]-centers[j])
    #          + wall_grad[i]
    
    # Vectorized: for each i, sum over j of active_mask[i,j] * 0.5/dists[i,j] * diff[i,j]
    # This gives sum_j 0.5/d * (x_i - x_j) for active pairs, which is what we need with a negative sign
    pair_grad_contrib = np.sum(active_mask[:, :, np.newaxis] * (0.5 * inv_dists[:, :, np.newaxis]) * diff, axis=1)
    grad -= pair_grad_contrib
    
    # Wall gradient
    # d/dx_i max(0, r - (0.5 - |x_i|)) = sign(x_i) if 0.5 - |x_i| < r
    wx_active = wall_dists_x < r
    wy_active = wall_dists_y < r
    grad[wx_active, 0] += np.sign(centers[wx_active, 0])
    grad[wy_active, 1] += np.sign(centers[wy_active, 1])
    
    return total_viol, grad

def _pgd_feasible(centers, r_target, max_iter=300, lr=0.002):
    """
    Projected Gradient Descent to minimize violation for target radius.
    Returns (centers, violation) at the end.
    """
    x = centers.copy()
    v, g = _violation_and_grad(x, r_target)
    
    for it in range(max_iter):
        if time.time() > _deadline:
            break
        if v < 1e-10:
            break
        
        # Adaptive learning rate
        step = lr / (1.0 + 0.1 * it)
        x_new = x - step * g
        x_new = np.clip(x_new, -HALF, HALF)
        
        v_new, g_new = _violation_and_grad(x_new, r_target)
        
        if v_new < v:
            x = x_new
            v = v_new
            g = g_new
        else:
            # Halve learning rate
            lr *= 0.5
    
    return x, v

def _soft_min_obj_and_grad(x, tau):
    """
    Soft-min objective and gradient.
    Minimizing -soft_min(distances) = maximizing soft_min(distances).
    """
    centers = x.reshape(-1, 2)
    n = len(centers)
    
    # Wall distances
    wall_dists_x = HALF - np.abs(centers[:, 0])
    wall_dists_y = HALF - np.abs(centers[:, 1])
    walls = np.minimum(wall_dists_x, wall_dists_y)
    
    # Pairwise distances (only upper triangle)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    iu = np.triu_indices(n, k=1)
    pair_dists = dists[iu] / 2.0
    
    all_dists = np.concatenate([walls, pair_dists])
    n_dists = len(all_dists)
    
    # Soft-min: min_d - tau * log(sum(exp(-(d - min_d)/tau)))
    min_d = np.min(all_dists)
    shifted = all_dists - min_d
    exp_terms = np.exp(-shifted / tau)
    log_sum = np.log(np.sum(exp_terms))
    soft_min_val = min_d - tau * log_sum
    
    obj = -soft_min_val
    
    # Gradient of soft_min w.r.t. each distance d_k:
    # d(soft_min)/d(d_k) = exp(-(d_k - min_d)/tau) / sum(exp(-(d_j - min_d)/tau))
    weights = exp_terms / np.sum(exp_terms)
    
    # Now chain rule: d(soft_min)/dx = sum_k weights[k] * d(d_k)/dx
    grad = np.zeros_like(centers)
    
    # Wall distances contribution
    # walls[i] = min(0.5 - |x_i|, 0.5 - |y_i|)
    # This is non-smooth, so we use a soft approximation or just handle the min
    # For gradient, we use the subgradient of the min
    wx_active = wall_dists_x <= wall_dists_y
    wy_active = ~wx_active
    
    # d(walls[i])/dx_i = -sign(x_i) if wx_active else 0
    # d(walls[i])/dy_i = -sign(y_i) if wy_active else 0
    wall_grad_x = np.zeros(n)
    wall_grad_y = np.zeros(n)
    wall_grad_x[wx_active] = -np.sign(centers[wx_active, 0])
    wall_grad_y[wy_active] = -np.sign(centers[wy_active, 1])
    
    # Wall weights
    wall_weights = weights[:n]
    grad[:, 0] += wall_weights * wall_grad_x
    grad[:, 1] += wall_weights * wall_grad_y
    
    # Pairwise distances contribution
    # pair_dists[k] = dists[i,j] / 2 for (i,j) in upper triangle
    # d(dists[i,j])/dx_i = (x_i - x_j) / dists[i,j]
    # d(dists[i,j])/dx_j = (x_j - x_i) / dists[i,j]
    pair_weights = weights[n:]
    
    for k, (i, j) in enumerate(zip(iu[0], iu[1])):
        d = dists[i, j]
        if d > 1e-12:
            w = pair_weights[k]
            diff_ij = centers[i] - centers[j]
            grad[i] += w * diff_ij / d
            grad[j] -= w * diff_ij / d
    
    # obj = -soft_min, so grad_obj = -grad_soft_min
    grad = -grad
    
    return obj, grad

def _sa_search(centers, rng, budget_time):
    """
    Simulated annealing to improve packing.
    """
    start_time = time.time()
    current = centers.copy()
    current_r = _compute_radius(current)
    best = current.copy()
    best_r = current_r
    
    T_start = 0.025
    T_end = 1e-8
    max_steps = 2000000
    
    for step in range(max_steps):
        if time.time() - start_time > budget_time:
            break
        
        # Geometric cooling
        T = T_start * (T_end / T_start) ** (step / max_steps)
        
        # Random perturbation - move one circle
        idx = rng.integers(0, N)
        # Step size decreases as T decreases
        step_size = 0.01 * (1.0 + 0.5 * T / T_start)
        delta = rng.normal(0, step_size, 2)
        
        new = current.copy()
        new[idx] += delta
        new = np.clip(new, -HALF + 1e-10, HALF - 1e-10)
        
        new_r = _compute_radius(new)
        
        if new_r >= current_r:
            current = new
            current_r = new_r
            if current_r > best_r:
                best_r = current_r
                best = current.copy()
        else:
            delta_r = new_r - current_r
            if delta_r > -10 * T:
                prob = np.exp(delta_r / T)
                if rng.random() < prob:
                    current = new
                    current_r = new_r
    
    return best, best_r

# Global deadline
_deadline = 0.0

def construct():
    global _deadline
    start_time = time.time()
    _deadline = start_time + 145.0  # 5s safety margin
    
    rng = np.random.default_rng(42)
    
    # Create initial grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    base_centers = grid[:N].copy()
    
    best_centers = base_centers.copy()
    best_radius = _compute_radius(best_centers)
    
    # Phase 1: Multiple SA restarts from jittered grids
    sa_budget_total = 70.0
    n_restarts = 3
    sa_budget_each = sa_budget_total / n_restarts
    
    for restart in range(n_restarts):
        if time.time() > _deadline - 60:
            break
        
        # Jitter the grid
        jitter_scale = 0.008 * (restart + 1)
        trial = base_centers + rng.normal(0, jitter_scale, base_centers.shape)
        trial = np.clip(trial, -HALF + 0.001, HALF - 0.001)
        
        trial_r = _compute_radius(trial)
        if trial_r > best_radius:
            best_centers = trial.copy()
            best_radius = trial_r
        
        # Run SA
        remaining = _deadline - time.time() - 50.0
        sa_budget = min(sa_budget_each, remaining)
        if sa_budget < 5:
            break
        
        sa_result, sa_r = _sa_search(trial, rng, sa_budget)
        if sa_r > best_radius:
            best_centers = sa_result.copy()
            best_radius = sa_r
    
    # Phase 2: Binary search on radius with PGD feasibility
    remaining = _deadline - time.time()
    if remaining > 40:
        lo = best_radius
        hi = 0.075  # Conservative upper bound
        
        for bs_iter in range(20):
            if time.time() > _deadline - 20:
                break
            
            mid = (lo + hi) / 2
            
            # Try multiple starting points for PGD
            trial = best_centers.copy()
            trial, viol = _pgd_feasible(trial, mid, max_iter=200, lr=0.003)
            trial_r = _compute_radius(trial)
            
            if trial_r >= mid - 1e-8:
                lo = mid
                best_centers = trial.copy()
                best_radius = trial_r
            else:
                hi = mid
    
    # Phase 3: Final polish with soft-min L-BFGS-B
    remaining = _deadline - time.time()
    if remaining > 10:
        x0 = best_centers.flatten()
        
        # Try multiple tau values
        for tau in [0.005, 0.01, 0.02]:
            if time.time() > _deadline - 5:
                break
            
            def obj(x, tau=tau):
                val, _ = _soft_min_obj_and_grad(x, tau)
                return val
            
            def jac(x, tau=tau):
                _, g = _soft_min_obj_and_grad(x, tau)
                return g
            
            try:
                result = opt.minimize(
                    obj, x0, jac=jac, method='L-BFGS-B',
                    bounds=[(-HALF, HALF)] * (2 * N),
                    options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 20}
                )
                if result.fun < obj(x0):
                    polished = np.clip(result.x.reshape(-1, 2), -HALF, HALF)
                    polished_r = _compute_radius(polished)
                    if polished_r > best_radius:
                        best_centers = polished
                        best_radius = polished_r
                        x0 = polished.flatten()
            except Exception:
                pass
    
    # Final validation
    if not np.all(np.isfinite(best_centers)):
        return base_centers
    
    # Ensure all centers are strictly inside
    best_centers = np.clip(best_centers, -HALF + 1e-10, HALF - 1e-10)
    
    return best_centers
# EVOLVE-BLOCK-END
