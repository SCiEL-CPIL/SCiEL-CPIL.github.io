"""
Packing solver: multi-start gradient ascent on the minimum-pair-distance objective.
Combines the epigraph/soft-min approach with iterative radius targeting.
"""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 100
HALF = 0.5
EPS = 1e-12


def _score(c):
    """Compute the packing score (minimum radius) for centre array c of shape (N,3)."""
    if c.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(c)):
        return 0.0
    
    # Boundary constraint: distance to nearest face
    bound = HALF - np.max(np.abs(c), axis=1)
    r_bound = np.min(bound)
    
    if r_bound < 0:
        return 0.0

    # Pairwise distances
    diff = c[:, None, :] - c[None, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(d, np.inf)
    r_pair = 0.5 * np.min(d)
    
    return min(r_bound, r_pair)


def _soft_min_objective(c, target_r, temperature=0.01):
    """
    Soft-min based objective for epigraph formulation.
    We want to maximize r = min(bound, pair_r).
    Equivalent to minimizing -r.
    Using soft-min: -soft_min(all_r, t)
    soft_min(x, t) = -t * log(sum(exp(-x_i / t)))
    """
    # Boundary radii
    bound = HALF - np.max(np.abs(c), axis=1)  # shape (N,)
    
    # Pairwise radii
    diff = c[:, None, :] - c[None, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(d, np.inf)
    pair_r = 0.5 * d  # shape (N, N)
    
    # Collect all constraint values
    all_r = np.concatenate([bound, pair_r[np.triu_indices(N, 1)]])
    
    # Soft min
    # To avoid overflow/underflow, shift by min
    min_r = np.min(all_r)
    shifted = all_r - min_r
    log_sum = np.log(np.sum(np.exp(-shifted / temperature)))
    soft_min_val = -temperature * log_sum + min_r
    
    # We want to maximize r, so minimize -soft_min_val
    # But we also have the constraint that r <= target_r implicitly? 
    # Actually, for the epigraph method, we usually fix r and minimize violations.
    # Here we are using a direct soft-max approach.
    # Let's stick to the subgradient method from the current program as it was working,
    # but maybe the soft-min is smoother.
    
    return -soft_min_val


def _soft_min_grad(c, target_r, temperature=0.01):
    """Gradient of the soft-min objective."""
    # Boundary radii
    bound = HALF - np.max(np.abs(c), axis=1)
    
    # Pairwise radii
    diff = c[:, None, :] - c[None, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(d, np.inf)
    pair_r = 0.5 * d
    
    all_r = np.concatenate([bound, pair_r[np.triu_indices(N, 1)]])
    
    min_r = np.min(all_r)
    shifted = all_r - min_r
    exp_term = np.exp(-shifted / temperature)
    sum_exp = np.sum(exp_term)
    
    # Weights for soft-min
    # d(soft_min)/dx_i = (1/sum) * exp(-x_i/t) * (-1/t) ... wait.
    # soft_min = -t * log(sum(exp(-x/t)))
    # d(soft_min)/dx_i = -t * (1/sum) * exp(-x_i/t) * (-1/t) = (1/sum) * exp(-x_i/t)
    weights = exp_term / sum_exp
    
    grad = np.zeros_like(c)
    
    # Boundary gradients
    w_bound = weights[:N]
    for i in range(N):
        if w_bound[i] < 1e-9:
            continue
        abs_c = np.abs(c[i])
        max_abs = np.max(abs_c)
        active_dim = np.where(np.abs(abs_c - max_abs) < 1e-12)[0]
        for k in active_dim:
            # d(bound_i)/dc_ik = -sign(c_ik) if active
            grad[i, k] += w_bound[i] * (-np.sign(c[i, k]))
            
    # Pairwise gradients
    w_pair = weights[N:]
    i_idx, j_idx = np.triu_indices(N, 1)
    for idx in range(len(w_pair)):
        if w_pair[idx] < 1e-9:
            continue
        i, j = i_idx[idx], j_idx[idx]
        diff_ij = c[i] - c[j]
        d_ij = np.sqrt(np.sum(diff_ij * diff_ij))
        if d_ij < EPS:
            continue
        unit = diff_ij / d_ij
        # d(pair_r_ij)/dc_i = 0.5 * unit
        # d(pair_r_ij)/dc_j = -0.5 * unit
        grad[i] += w_pair[idx] * 0.5 * unit
        grad[j] -= w_pair[idx] * 0.5 * unit
        
    # Objective is -soft_min, so grad_obj = -grad_soft_min
    return -grad


def _violation_objective(x, R):
    """
    Objective for fixed radius R: sum of squared violations.
    Violation for pair (i,j): max(0, 2R - d_ij)^2
    Violation for wall i: max(0, R - wall_dist_i)^2
    """
    centers = x.reshape(-1, 3)
    n = len(centers)
    energy = 0.0
    grad = np.zeros_like(centers)
    
    # 1. Pairwise terms
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2) + 1e-15)
    np.fill_diagonal(dists, np.inf)
    
    mask = dists < 2.0 * R
    if np.any(mask):
        i_idx, j_idx = np.triu_indices(n, k=1)
        d_ij = dists[i_idx, j_idx]
        m_ij = mask[i_idx, j_idx]
        
        if np.any(m_ij):
            v_ij = 2.0 * R - d_ij[m_ij]
            energy += np.sum(v_ij**2)
            
            # Gradient
            # Force on i from j: -2 * v * (c_i - c_j) / d
            for k in range(len(m_ij)):
                if not m_ij[k]:
                    continue
                i = i_idx[k]
                j = j_idx[k]
                v = v_ij[k]
                d = d_ij[k]
                factor = -2.0 * v / d
                vec = centers[i] - centers[j]
                force = factor * vec
                grad[i] += force
                grad[j] -= force

    # 2. Wall terms
    abs_centers = np.abs(centers)
    max_abs = np.max(abs_centers, axis=1)
    wall_dists = 0.5 - max_abs
    
    mask_wall = wall_dists < R
    if np.any(mask_wall):
        v_wall = (R - wall_dists)[mask_wall]
        energy += np.sum(v_wall**2)
        
        indices_wall = np.where(mask_wall)[0]
        for i in indices_wall:
            v = R - wall_dists[i]
            max_val = max_abs[i]
            grad_wall_i = np.zeros(3)
            for k in range(3):
                if np.isclose(abs_centers[i, k], max_val):
                    grad_wall_i[k] = np.sign(centers[i, k])
                    break
            
            force_wall = 2.0 * v * grad_wall_i
            grad[i] += force_wall
            
    return energy, grad.flatten()


def _optimize_for_radius(init_centers, target_R, max_iter=200):
    """
    Attempts to optimize the configuration to fit within radius target_R.
    """
    start_time = time.time()
    
    def objective_and_grad(x):
        val, g = _violation_objective(x, target_R)
        return val, g

    bounds = [(-0.5, 0.5)] * (N * 3)
    
    try:
        res = minimize(
            objective_and_grad,
            init_centers.flatten(),
            jac=True,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'disp': False, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        return res.x.reshape(-1, 3), res.fun
    except Exception:
        return init_centers, np.inf


def _gradient_ascent_polish(c, target_r, steps=100, lr=0.01):
    """
    Simple gradient ascent on the soft-min objective to polish the solution.
    """
    c = c.copy()
    for _ in range(steps):
        obj = _soft_min_objective(c, target_r, temperature=0.005)
        grad = _soft_min_grad(c, target_r, temperature=0.005)
        # We are minimizing -soft_min, so we want to move in direction of -grad
        # But grad is already for -soft_min? 
        # In _soft_min_grad, we return -grad_soft_min.
        # So obj = -soft_min. grad = -grad_soft_min.
        # To minimize obj, we move c <- c - lr * grad.
        c = c - lr * grad
        c = np.clip(c, -HALF + 1e-6, HALF - 1e-6)
    return c


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    best_score = -np.inf
    best_c = None
    
    # Initial guess: simple grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    init_c = grid[:N].copy()
    
    # Strategy 1: Iterative Radius Targeting (from Inspiration)
    # Start with a low radius and increase it
    current_r = _score(init_c)
    if current_r < 0.05:
        current_r = 0.05
        
    current_c = init_c.copy()
    
    # Run radius targeting for a fixed number of iterations or time
    max_radius_steps = 50
    for step in range(max_radius_steps):
        if time.time() > deadline - 10.0:
            break
            
        target_R = current_r * 1.005
        # Don't go too high initially
        if target_R > 0.115:
            target_R = current_r * 1.001
            
        optimized_c, energy = _optimize_for_radius(current_c, target_R, max_iter=100)
        
        r_achieved = _score(optimized_c)
        
        if r_achieved > best_score:
            best_score = r_achieved
            best_c = optimized_c.copy()
            
        if energy < 1e-6 and r_achieved >= target_R * 0.99:
            # Successfully fit the radius, increase it
            current_c = optimized_c
            current_r = r_achieved
        else:
            # Failed to fit, try to recover or break
            # If we are stuck, try a smaller step or break
            if r_achieved < current_r:
                # We got worse, revert
                current_c = best_c if best_c is not None else current_c
                # Try to polish best_c
                if best_c is not None and time.time() < deadline - 5.0:
                    polished = _gradient_ascent_polish(best_c, best_score, steps=50, lr=0.005)
                    r_polished = _score(polished)
                    if r_polished > best_score:
                        best_score = r_polished
                        best_c = polished.copy()
                break
            else:
                # We are at the limit, try to polish
                if time.time() < deadline - 5.0:
                    polished = _gradient_ascent_polish(current_c, current_r, steps=100, lr=0.005)
                    r_polished = _score(polished)
                    if r_polished > best_score:
                        best_score = r_polished
                        best_c = polished.copy()
                break

    # Strategy 2: Multi-start with gradient ascent (from Current)
    n_restarts = 5
    rng = np.random.RandomState(42)
    
    for restart in range(n_restarts):
        if time.time() > deadline - 10.0:
            break
            
        if restart == 0:
            c = init_c.copy()
        else:
            c = init_c + 0.05 * rng.randn(N, 3)
            if restart % 2 == 0:
                c = rng.uniform(-0.4, 0.4, (N, 3))
                
        c = np.clip(c, -HALF + 0.01, HALF - 0.01)
        
        # Use gradient ascent with soft-min
        # Estimate current radius
        r_est = _score(c)
        
        # Polish
        c_polished = _gradient_ascent_polish(c, r_est, steps=200, lr=0.01)
        score_polished = _score(c_polished)
        
        if score_polished > best_score:
            best_score = score_polished
            best_c = c_polished.copy()
            
        # Try to increase radius from this point
        target_R = score_polished * 1.01
        for _ in range(10):
            if time.time() > deadline - 5.0:
                break
            opt_c, energy = _optimize_for_radius(c_polished, target_R, max_iter=50)
            r_new = _score(opt_c)
            if r_new > best_score:
                best_score = r_new
                best_c = opt_c.copy()
            if energy < 1e-6 and r_new >= target_R * 0.99:
                c_polished = opt_c
                target_R *= 1.005
            else:
                break

    # Final polish
    if best_c is not None and time.time() < deadline:
        polished = _gradient_ascent_polish(best_c, best_score, steps=200, lr=0.005)
        score_polished = _score(polished)
        if score_polished > best_score:
            best_score = score_polished
            best_c = polished.copy()

    # Ensure validity: clip to cube
    if best_c is None:
        return init_c[:N]
    
    best_c = np.clip(best_c, -HALF, HALF)
    return best_c
