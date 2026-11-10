import numpy as np
import time
from scipy.optimize import minimize

N = 100
CUBE_HALF = 0.5

def _compute_radius(centers):
    """Compute the exact common radius for a set of centers."""
    # Wall clearance: min over i,k of (0.5 - |c_ik|)
    wall_clear = 0.5 - np.max(np.abs(centers), axis=1)
    min_wall = np.min(wall_clear)
    
    # Inter-sphere clearance: min over i<j of |c_i - c_j| / 2
    if len(centers) > 1:
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt((diff ** 2).sum(axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
    else:
        min_pair = min_wall
    
    return min(min_pair, min_wall)

def _make_initial_guess(seed):
    """
    Generate diverse initial guesses.
    Strategy: 
    1. Jittered 5x5x5 grid (first 100 points)
    2. Random uniform points
    3. Jittered 4x5x5 grid (100 points exactly)
    """
    rng = np.random.RandomState(seed)
    
    if seed % 3 == 0:
        # 5x5x5 grid jittered
        m = 5
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        grid = grid[:N]
        jitter = rng.uniform(-0.02, 0.02, size=(N, 3))
        grid = grid + jitter
    elif seed % 3 == 1:
        # 4x5x5 grid exactly 100 points
        mx, my, mz = 4, 5, 5
        tx = (np.arange(mx) + 0.5) / mx - 0.5
        ty = (np.arange(my) + 0.5) / my - 0.5
        tz = (np.arange(mz) + 0.5) / mz - 0.5
        grid = np.array([(x, y, z) for z in tz for y in ty for x in tx])
        jitter = rng.uniform(-0.01, 0.01, size=(N, 3))
        grid = grid + jitter
    else:
        # Random uniform
        grid = rng.uniform(-0.4, 0.4, size=(N, 3))
        
    # Ensure strictly inside
    grid = np.clip(grid, -0.49, 0.49)
    return grid.flatten()

def _objective(x, lam):
    """
    Objective function for epigraph formulation.
    x = [c_1, ..., c_N, t]
    Maximize t subject to constraints.
    Minimize -t + lam * sum(violations^2)
    """
    c = x[:N*3].reshape(N, 3)
    t = x[N*3]
    
    # Wall constraints: 0.5 - |c_ik| >= t  => |c_ik| <= 0.5 - t
    # Violation: max(0, |c_ik| - (0.5 - t))
    abs_c = np.abs(c)
    max_coord = np.max(abs_c, axis=1)
    wall_viol = np.maximum(0.0, max_coord - (0.5 - t))
    wall_pen = np.sum(wall_viol ** 2)
    
    # Pair constraints: |c_i - c_j| >= 2t
    # Violation: max(0, 2t - |c_i - c_j|)
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=2)
    dists = np.sqrt(dists_sq)
    np.fill_diagonal(dists, np.inf)
    
    pair_viol = np.maximum(0.0, 2.0 * t - dists)
    # Only sum upper triangle to avoid double counting, or just sum all and divide? 
    # Summing all is fine, just scales the penalty.
    pair_pen = 0.5 * np.sum(pair_viol ** 2) # 0.5 to compensate for double counting
    
    return -t + lam * (wall_pen + pair_pen)

def _gradient(x, lam):
    """
    Analytical gradient for the objective.
    """
    c = x[:N*3].reshape(N, 3)
    t = x[N*3]
    
    grad_c = np.zeros_like(c)
    grad_t = -1.0
    
    # Wall constraints
    abs_c = np.abs(c)
    max_coord = np.max(abs_c, axis=1)
    # Index of max coordinate for each sphere
    max_idx = np.argmax(abs_c, axis=1)
    
    wall_viol = np.maximum(0.0, max_coord - (0.5 - t))
    if np.any(wall_viol > 0):
        # d/d c_ik max(0, |c_ik| - 0.5 + t)
        # If |c_ik| > 0.5 - t:
        # grad_c = 2 * lam * wall_viol * sign(c_ik) for max coord
        # grad_t = 2 * lam * wall_viol * (-1) = -2 * lam * wall_viol
        
        # Apply to grad_c
        for i in range(N):
            if wall_viol[i] > 0:
                k = max_idx[i]
                grad_c[i, k] += 2.0 * lam * wall_viol[i] * np.sign(c[i, k])
        grad_t -= 2.0 * lam * np.sum(wall_viol)
        
    # Pair constraints
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=2)
    dists = np.sqrt(dists_sq)
    np.fill_diagonal(dists, np.inf)
    
    pair_viol = np.maximum(0.0, 2.0 * t - dists)
    
    # Mask for active constraints
    active = pair_viol > 0
    
    if np.any(active):
        # Gradient of 0.5 * sum (2t - d_ij)^2
        # d/d t: sum 2 * (2t - d_ij) * 2 = 4 * sum pair_viol
        grad_t += 4.0 * lam * np.sum(pair_viol)
        
        # d/d c_i: sum_j 2 * (2t - d_ij) * (-diff_ij / d_ij)
        # d/d c_j: sum_i 2 * (2t - d_ij) * (diff_ij / d_ij)
        
        # Vectorized gradient calculation
        # G_ij = 2 * lam * pair_viol_ij * (diff_ij / dists_ij)
        # grad_c[i] = - sum_j G_ij
        # grad_c[j] = sum_i G_ij
        
        # To avoid division by zero, we only care about active ones where dist > 0
        # For active ones, dist < 2t, so dist > 0 unless t=0 and dist=0 (degenerate)
        safe_dists = np.where(dists > 1e-12, dists, 1.0)
        
        # Factor = 2 * lam * pair_viol / dists
        factor = np.zeros_like(dists)
        factor[active] = 2.0 * lam * pair_viol[active] / safe_dists[active]
        
        # grad_c[i] = - sum_j factor_ij * diff_ij
        # diff_ij = c_i - c_j
        # sum_j factor_ij * (c_i - c_j)
        
        # Use matrix multiplication
        # grad_c = - (factor * diff) .sum(axis=1) ? No, factor is scalar per pair.
        # term_ij = factor_ij * diff_ij
        # grad_c[i] = - sum_j term_ij
        
        terms = factor[:, :, np.newaxis] * diff
        grad_c -= np.sum(terms, axis=1)
        
    return np.concatenate([grad_c.flatten(), [grad_t]])

def construct():
    start_time = time.time()
    time_limit = 140.0  # Safety margin
    
    bounds = [(-0.5, 0.5)] * (N * 3) + [(0.0, 0.5)]
    
    best_r = 0.0
    best_c = None
    
    # Multi-start optimization
    num_seeds = 15
    
    for seed in range(num_seeds):
        if time.time() - start_time > time_limit - 5:
            break
            
        x0 = _make_initial_guess(seed)
        # Initial t estimate
        c0 = x0.reshape(N, 3)
        r0 = _compute_radius(c0)
        x0 = np.concatenate([x0, [r0 * 0.9]]) # Start slightly below current radius
        
        # Adaptive penalty method
        lam = 10.0
        x_current = x0.copy()
        
        # Run a few iterations with increasing lambda
        for lam_iter in range(5):
            if time.time() - start_time > time_limit - 5:
                break
                
            def obj(x):
                return _objective(x, lam)
            def grad(x):
                return _gradient(x, lam)
                
            try:
                res = minimize(
                    obj,
                    x_current,
                    jac=grad,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 100, 'maxfev': 1000, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                x_current = res.x
            except Exception:
                pass
                
            # Check if constraints are satisfied
            c_test = x_current[:N*3].reshape(N, 3)
            t_test = x_current[N*3]
            r_test = _compute_radius(c_test)
            
            if r_test > best_r:
                best_r = r_test
                best_c = c_test.copy()
                
            # Check violation
            # If violation is small, we can increase lambda or stop
            wall_clear = 0.5 - np.max(np.abs(c_test), axis=1)
            min_wall = np.min(wall_clear)
            diff = c_test[:, np.newaxis, :] - c_test[np.newaxis, :, :]
            dists = np.sqrt((diff ** 2).sum(axis=2))
            np.fill_diagonal(dists, np.inf)
            min_pair = np.min(dists) / 2.0
            actual_r = min(min_pair, min_wall)
            
            if abs(actual_r - t_test) < 1e-6:
                # Constraints met, break early for this seed
                break
                
            lam *= 2.0
            
    # Final Polish:
    # Use the best configuration found and run a final optimization with a very high lambda
    if best_c is not None and time.time() - start_time < time_limit - 10:
        x_polish = np.concatenate([best_c.flatten(), [best_r * 1.01]])
        lam_polish = 1000.0
        
        def obj_polish(x):
            return _objective(x, lam_polish)
        def grad_polish(x):
            return _gradient(x, lam_polish)
            
        try:
            res_polish = minimize(
                obj_polish,
                x_polish,
                jac=grad_polish,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'maxfev': 2000, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            c_polish = res_polish.x[:N*3].reshape(N, 3)
            r_polish = _compute_radius(c_polish)
            if r_polish > best_r:
                best_r = r_polish
                best_c = c_polish.copy()
        except Exception:
            pass

    # Fallback
    if best_c is None:
        m = 5
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_c = grid[:N]

    return best_c
