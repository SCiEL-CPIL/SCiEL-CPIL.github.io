"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import scipy.optimize as opt
from scipy.spatial import cKDTree

N = 50
_DEADLINE = 0.0

def _compute_radius(centers):
    """Compute the true maximum radius for a given configuration using KDTree."""
    centers = np.asarray(centers, dtype=np.float64)
    if centers.shape != (N, 2) or not np.all(np.isfinite(centers)):
        return 0.0
    
    # Wall distances
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    min_wall = np.min(np.minimum(wall_dists_x, wall_dists_y))
    
    # Pairwise distances via KDTree
    tree = cKDTree(centers)
    # Query each point's nearest neighbor (excluding self)
    dists, _ = tree.query(centers, k=2)
    min_pair = np.min(dists[:, 1]) / 2.0
    
    return min(min_pair, min_wall)

def _compute_radius_fast(centers):
    """Even faster radius computation - no KDTree overhead for small N."""
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    min_wall = np.min(np.minimum(wall_dists_x, wall_dists_y))
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    return min(min_pair, min_wall)

def _soft_min_obj(x, tau):
    """Soft-min objective: minimize -soft_min(all distances)."""
    centers = x.reshape(-1, 2)
    n = len(centers)
    
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    walls = np.minimum(wall_dists_x, wall_dists_y)
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    pair_dists = dists / 2.0
    
    all_dists = np.concatenate([walls, pair_dists[np.triu_indices(n, k=1)]])
    
    min_d = np.min(all_dists)
    shifted = all_dists - min_d
    log_sum = np.log(np.sum(np.exp(-shifted / tau)))
    soft_min = min_d - tau * log_sum
    
    return -soft_min

def _soft_min_grad(x, tau):
    """Gradient of soft-min objective."""
    centers = x.reshape(-1, 2)
    n = len(centers)
    grad = np.zeros_like(x)
    
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    walls = np.minimum(wall_dists_x, wall_dists_y)
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    pair_dists = dists / 2.0
    
    all_dists = np.concatenate([walls, pair_dists[np.triu_indices(n, k=1)]])
    min_d = np.min(all_dists)
    shifted = all_dists - min_d
    weights = np.exp(-shifted / tau) / np.sum(np.exp(-shifted / tau))
    
    # Gradient of -soft_min = -d(soft_min)/dx
    # soft_min = min_d - tau * log(sum(exp(-d/tau)))
    # d(soft_min)/dx_i = -tau * sum_j (w_j * (-1/tau) * d(d_j)/dx_i) = sum_j w_j * d(d_j)/dx_i
    # So d(-soft_min)/dx_i = -sum_j w_j * d(d_j)/dx_i
    
    # Wall contributions
    for i in range(n):
        # dx wall
        if centers[i, 0] >= 0:
            grad[2*i] += weights[i] * 1.0
        else:
            grad[2*i] += weights[i] * (-1.0)
        # dy wall
        if centers[i, 1] >= 0:
            grad[2*i+1] += weights[i] * 1.0
        else:
            grad[2*i+1] += weights[i] * (-1.0)
    
    # Pair contributions
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            w = weights[n + idx]
            d = dists[i, j]
            if d > 1e-12:
                factor = w * 0.5 / d
                grad[2*i]   -= factor * (centers[i,0] - centers[j,0])
                grad[2*i+1] -= factor * (centers[i,1] - centers[j,1])
                grad[2*j]   += factor * (centers[i,0] - centers[j,0])
                grad[2*j+1] += factor * (centers[i,1] - centers[j,1])
            idx += 1
    
    return -grad  # We want gradient of -soft_min

def _sa_search(centers, rng, budget_time, T_start=0.025, T_end=1e-8):
    """Simulated annealing to improve packing."""
    start_time = time.time()
    current = centers.copy()
    current_r = _compute_radius_fast(current)
    best = current.copy()
    best_r = current_r
    
    max_steps = 2000000
    steps_done = 0
    
    for step in range(max_steps):
        if time.time() - start_time > budget_time:
            break
        steps_done = step + 1
        
        T = T_start * (T_end / T_start) ** (step / max_steps)
        
        # Adaptive step size
        step_size = 0.012 * (1.0 + 0.5 * np.cos(np.pi * step / max_steps))
        
        idx = rng.integers(0, N)
        delta = rng.normal(0, step_size, 2)
        
        new = current.copy()
        new[idx] += delta
        new = np.clip(new, -0.5, 0.5)
        
        new_r = _compute_radius_fast(new)
        
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

def _force_relax(centers, iterations=2000, time_budget=10.0):
    """Force-directed relaxation: push overlapping circles apart, pull toward center if near wall."""
    start_time = time.time()
    current = centers.copy()
    
    for it in range(iterations):
        if time.time() - start_time > time_budget:
            break
        
        forces = np.zeros_like(current)
        
        # Pairwise repulsion
        for i in range(N):
            for j in range(i+1, N):
                d_vec = current[i] - current[j]
                d = np.linalg.norm(d_vec)
                if d < 0.14:  # Within 2x target radius
                    if d > 1e-12:
                        f = (0.14 - d) / d
                        forces[i] += f * d_vec
                        forces[j] -= f * d_vec
        
        # Wall repulsion
        for i in range(N):
            for axis in range(2):
                if current[i, axis] > 0.36:
                    forces[i, axis] -= (current[i, axis] - 0.36) * 2.0
                elif current[i, axis] < -0.36:
                    forces[i, axis] += (-0.36 - current[i, axis]) * 2.0
        
        lr = 0.01 * (1.0 - it / iterations)
        current += lr * forces
        current = np.clip(current, -0.5, 0.5)
    
    return current

def _gradient_polish(centers, r_target, max_iter=300, lr_init=0.003):
    """Gradient descent to minimize violation for a target radius."""
    x = centers.flatten().copy()
    lr = lr_init
    start_time = time.time()
    
    for i in range(max_iter):
        if time.time() > _DEADLINE - 5:
            break
        
        c = x.reshape(-1, 2)
        
        # Compute violation
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        pair_v = np.sum(np.maximum(0, r_target - dists / 2.0))
        
        wall_dists_x = 0.5 - np.abs(c[:, 0])
        wall_dists_y = 0.5 - np.abs(c[:, 1])
        wall_dists = np.minimum(wall_dists_x, wall_dists_y)
        wall_v = np.sum(np.maximum(0, r_target - wall_dists))
        
        total_v = pair_v + wall_v
        if total_v < 1e-12:
            break
        
        # Compute gradient
        grad = np.zeros_like(c)
        
        for i in range(N):
            for j in range(i+1, N):
                d = dists[i, j]
                if d / 2.0 < r_target and d > 1e-12:
                    factor = 0.5 / d
                    grad[i] -= factor * (c[i] - c[j])
                    grad[j] += factor * (c[i] - c[j])
        
        for i in range(N):
            if wall_dists_x[i] < r_target:
                grad[i, 0] += np.sign(c[i, 0])
            if wall_dists_y[i] < r_target:
                grad[i, 1] += np.sign(c[i, 1])
        
        x -= lr * grad.flatten()
        x = np.clip(x, -0.5, 0.5)
        
        # Reduce learning rate
        if i % 50 == 0:
            lr *= 0.7
    
    return x.reshape(-1, 2)

def construct():
    global _DEADLINE
    start_time = time.time()
    _DEADLINE = start_time + 145.0
    
    rng = np.random.default_rng(42)
    
    # Multiple initial configurations
    best_centers = None
    best_radius = 0.0
    
    # Grid-based initializations with different jitter
    m = int(np.ceil(np.sqrt(N)))  # m=8
    ticks = (np.arange(m) + 0.5) / m - 0.5
    
    # Try several different starting points
    n_starts = 4
    for s in range(n_starts):
        if time.time() > _DEADLINE - 30:
            break
        
        if s == 0:
            grid = np.array([(x, y) for y in ticks for x in ticks])
            centers = grid[:N].copy()
            centers += rng.normal(0, 0.008, centers.shape)
        elif s == 1:
            # Slightly different grid offset
            ticks2 = (np.arange(m) + 0.3) / m - 0.5
            grid2 = np.array([(x, y) for y in ticks2 for x in ticks2])
            centers = grid2[:N].copy()
            centers += rng.normal(0, 0.005, centers.shape)
        elif s == 2:
            # Random perturbation of grid
            grid = np.array([(x, y) for y in ticks for x in ticks])
            centers = grid[:N].copy()
            centers += rng.normal(0, 0.02, centers.shape)
        else:
            # More random
            centers = rng.uniform(-0.45, 0.45, (N, 2))
        
        centers = np.clip(centers, -0.49, 0.49)
        
        # Force relaxation first
        remaining = _DEADLINE - time.time()
        relax_budget = min(5.0, remaining * 0.05)
        if relax_budget > 1:
            centers = _force_relax(centers, iterations=1000, time_budget=relax_budget)
        
        # SA
        remaining = _DEADLINE - time.time()
        sa_budget = min(25.0, remaining * 0.35)
        if sa_budget > 3:
            centers, r = _sa_search(centers, rng, sa_budget)
        else:
            r = _compute_radius_fast(centers)
        
        if r > best_radius:
            best_radius = r
            best_centers = centers.copy()
    
    # Binary search on radius
    remaining = _DEADLINE - time.time()
    if remaining > 25:
        lo = best_radius
        hi = 0.078
        
        for _ in range(12):
            if time.time() > _DEADLINE - 15:
                break
            
            mid = (lo + hi) / 2
            trial = best_centers.copy()
            trial = _gradient_polish(trial, mid, max_iter=150)
            
            trial_r = _compute_radius_fast(trial)
            if trial_r >= mid - 1e-9:
                lo = mid
                best_centers = trial
                best_radius = trial_r
            else:
                hi = mid
    
    # Final L-BFGS-B polish with very tight soft-min
    remaining = _DEADLINE - time.time()
    if remaining > 10:
        x0 = best_centers.flatten()
        
        def obj(x):
            return _soft_min_obj(x, 0.005)
        
        def obj_grad(x):
            return _soft_min_grad(x, 0.005)
        
        try:
            result = opt.minimize(
                obj, x0, jac=obj_grad, method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (2 * N),
                options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 20}
            )
            if result.fun < obj(x0):
                polished = result.x.reshape(-1, 2)
                polished = np.clip(polished, -0.5, 0.5)
                polished_r = _compute_radius_fast(polished)
                if polished_r > best_radius:
                    best_centers = polished
                    best_radius = polished_r
        except Exception:
            pass
        
        # Second polish with even tighter tau
        remaining = _DEADLINE - time.time()
        if remaining > 5:
            x0 = best_centers.flatten()
            
            def obj2(x):
                return _soft_min_obj(x, 0.002)
            
            def obj2_grad(x):
                return _soft_min_grad(x, 0.002)
            
            try:
                result2 = opt.minimize(
                    obj2, x0, jac=obj2_grad, method='L-BFGS-B',
                    bounds=[(-0.5, 0.5)] * (2 * N),
                    options={'maxiter': 50, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 10}
                )
                if result2.fun < obj2(x0):
                    polished2 = result2.x.reshape(-1, 2)
                    polished2 = np.clip(polished2, -0.5, 0.5)
                    polished2_r = _compute_radius_fast(polished2)
                    if polished2_r > best_radius:
                        best_centers = polished2
                        best_radius = polished2_r
            except Exception:
                pass
    
    # Final verification
    if not np.all(np.isfinite(best_centers)):
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
    
    return best_centers
