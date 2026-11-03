"""
Strategy: Deterministic Gradient-Based Optimization using Log-Sum-Exp Soft-Min.

The current program uses Simulated Annealing (SA), which is a stochastic local search.
This program replaces that with a deterministic, differentiable approach:
1. Approximate the "min" operation in the score function using a Log-Sum-Exp (soft-min)
   transformation. This creates a smooth, differentiable objective function.
2. Use L-BFGS-B (a quasi-Newton method) to maximize this smooth objective.
3. Perform multiple restarts from different high-quality initializations (hexagonal lattice
   variants) to explore different basins of attraction.

Why this is different:
- SA relies on random moves and temperature schedules. It can get stuck in local minima
  and requires many evaluations to converge.
- Gradient-based methods use the shape of the landscape to take large, directed steps.
  For smooth approximations of packing problems, these often converge to higher quality
  solutions faster and more reliably than random perturbations.
"""

import numpy as np
import time
import scipy.optimize as opt
from scipy.special import logsumexp

N = 300
TIME_LIMIT = 115.0  # Safety margin for 150s kill switch

def _hex_init(rotation=0.0, jitter=0.0, spacing=None):
    """
    Generate a hexagonal lattice initialization.
    Returns an array of shape (N, 2).
    """
    if spacing is None:
        spacing = 0.105
    
    # Generate points in a grid large enough to cover the unit circle
    k = int(1.1 / spacing) + 2
    points = []
    for i in range(-k, k + 1):
        for j in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            points.append([x, y])
    
    points = np.array(points)
    
    # Rotate points
    if rotation != 0.0:
        c, s = np.cos(rotation), np.sin(rotation)
        R = np.array([[c, -s], [s, c]])
        points = points @ R.T
        
    # Add jitter if requested
    if jitter > 0.0:
        points += np.random.normal(0, jitter, points.shape)
        
    # Filter points strictly inside unit circle (with small margin)
    radii = np.linalg.norm(points, axis=1)
    mask = radii < 0.99
    points = points[mask]
    radii = radii[mask]
    
    # Sort by radius (closest to center first) and take N
    idx = np.argsort(radii)
    points = points[idx]
    
    if len(points) < N:
        # Fallback: if we don't have enough points (shouldn't happen), 
        # pad with random points
        theta = np.random.uniform(0, 2 * np.pi, N - len(points))
        rho = np.sqrt(np.random.uniform(0, 0.5, N - len(points)))
        pad = np.column_stack((rho * np.cos(theta), rho * np.sin(theta)))
        points = np.vstack([points, pad])
    else:
        points = points[:N]
        
    return points

def _objective_and_grad(flat_x, beta=50.0):
    """
    Objective: Minimize negative soft-min of distances.
    Returns (objective_value, gradient_flat).
    """
    points = flat_x.reshape(N, 2)
    
    # Calculate pairwise distances
    diffs = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diffs, axis=-1)
    np.fill_diagonal(dists, np.inf)
    
    # Values for pairs: dists / 2
    v_pairs = dists / 2.0
    
    # Boundary
    radii = np.linalg.norm(points, axis=1)
    v_bounds = 1.0 - radii
    
    # Log weights
    log_w_pairs = -beta * v_pairs
    log_w_bounds = -beta * v_bounds
    
    mask_diag = np.eye(N, dtype=bool)
    
    # Compute log_S using logsumexp
    log_w_pairs_valid = np.where(mask_diag, -np.inf, log_w_pairs)
    all_log_w = np.concatenate([log_w_pairs_valid.flatten(), log_w_bounds])
    log_S = logsumexp(all_log_w)
    
    # Weights
    w_pairs = np.exp(log_w_pairs - log_S)
    w_pairs[mask_diag] = 0.0
    w_bounds = np.exp(log_w_bounds - log_S)
    
    # Objective value: minimize log_S / beta
    obj = log_S / beta
    
    # Gradient
    grad = np.zeros_like(points)
    
    # Pair contributions
    with np.errstate(divide='ignore', invalid='ignore'):
        factor_pairs = np.where(dists > 1e-8, w_pairs * 0.5 / dists, 0.0)
        
    sum_factor = np.sum(factor_pairs, axis=1)
    sum_factor_c = factor_pairs @ points
    grad_pairs = points * sum_factor[:, np.newaxis] - sum_factor_c
    
    # Boundary contributions
    with np.errstate(divide='ignore', invalid='ignore'):
        grad_bound_factors = np.where(radii > 1e-8, w_bounds / radii, 0.0)
        
    grad_bounds = -points * grad_bound_factors[:, np.newaxis]
    
    grad = grad_pairs + grad_bounds
    grad_flat = grad.flatten()
    
    return obj, grad_flat


def _compute_true_score(points):
    """Compute the actual packing radius (hard min)."""
    diffs = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diffs, axis=-1)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    radii = np.linalg.norm(points, axis=1)
    min_bound_dist = np.min(1.0 - radii)
    return min(min_pair_dist / 2.0, min_bound_dist)


def _active_set_polish(points, max_iter=50, tol=1e-12):
    """
    Final polish: focus optimization on the most constraining pairs and boundary points.
    Uses a higher beta and only considers the tightest constraints.
    """
    current = points.copy()
    best_score = _compute_true_score(current)
    
    for iteration in range(max_iter):
        if time.time() > TIME_LIMIT - 3.0:
            break
            
        score = _compute_true_score(current)
        if score <= best_score:
            break
        best_score = score
        
        # Find the tightest constraints
        diffs = current[:, np.newaxis, :] - current[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=-1)
        np.fill_diagonal(dists, np.inf)
        radii = np.linalg.norm(current, axis=1)
        bound_dists = 1.0 - radii
        
        # The bottleneck radius
        r_pair = np.min(dists) / 2.0
        r_bound = np.min(bound_dists)
        r_bottleneck = min(r_pair, r_bound)
        
        # Select active set: pairs with distance < r_bottleneck * 1.5 + small_margin
        # and boundary points with dist < r_bottleneck * 1.5 + small_margin
        margin = r_bottleneck * 0.3 + 1e-6
        active_pairs = dists < (r_bottleneck + margin) * 2.0
        active_bounds = bound_dists < (r_bottleneck + margin)
        
        # Build a reduced problem with only active constraints
        # Use very high beta for sharpness
        beta = 200.0
        
        # Objective using only active constraints
        def active_obj_grad(flat_x):
            pts = flat_x.reshape(N, 2)
            
            # Recompute distances (expensive but needed)
            d = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            ds = np.linalg.norm(d, axis=-1)
            np.fill_diagonal(ds, np.inf)
            
            rad = np.linalg.norm(pts, axis=1)
            bd = 1.0 - rad
            
            # Only active constraints
            v_p = np.where(active_pairs, ds / 2.0, np.inf)
            v_b = np.where(active_bounds, bd, np.inf)
            
            log_wp = -beta * v_p
            log_wb = -beta * v_b
            
            log_wp_valid = np.where(np.eye(N, dtype=bool), -np.inf, log_wp)
            all_lw = np.concatenate([log_wp_valid.flatten(), log_wb])
            
            # Handle all-inf case
            finite_mask = np.isfinite(all_lw)
            if not np.any(finite_mask):
                return 0.0, np.zeros_like(flat_x)
            
            log_S = logsumexp(all_lw)
            
            wp = np.exp(log_wp - log_S)
            wp[np.eye(N, dtype=bool)] = 0.0
            wb = np.exp(log_wb - log_S)
            
            obj = log_S / beta
            
            grad = np.zeros_like(pts)
            
            with np.errstate(divide='ignore', invalid='ignore'):
                fp = np.where(ds > 1e-8, wp * 0.5 / ds, 0.0)
            
            sf = np.sum(fp, axis=1)
            sfc = fp @ pts
            gp = pts * sf[:, np.newaxis] - sfc
            
            with np.errstate(divide='ignore', invalid='ignore'):
                gb_f = np.where(rad > 1e-8, wb / rad, 0.0)
            
            gb = -pts * gb_f[:, np.newaxis]
            
            grad = gp + gb
            return obj, grad.flatten()
        
        try:
            res = opt.minimize(
                active_obj_grad,
                current.flatten(),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 20, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 10}
            )
            new_points = res.x.reshape(N, 2)
            
            new_score = _compute_true_score(new_points)
            if new_score > best_score:
                best_score = new_score
                current = new_points
            else:
                break
        except Exception:
            break
    
    return current, best_score

def construct():
    start_time = time.time()
    best_points = None
    best_score = -np.inf
    
    # Define multiple different initializations to restart from
    inits = [
        _hex_init(rotation=0.0, jitter=0.0),
        _hex_init(rotation=np.pi/6, jitter=0.0),
        _hex_init(rotation=0.0, jitter=0.01),
        _hex_init(rotation=np.pi/3, jitter=0.01),
        _hex_init(rotation=np.pi/12, jitter=0.005),
    ]
    
    # Beta annealing schedule: start smooth, end sharp
    beta_schedule = [20.0, 50.0, 100.0]
    
    for i, init_points in enumerate(inits):
        if time.time() - start_time > TIME_LIMIT - 10.0:
            break
            
        x0 = init_points.flatten()
        
        # Multi-phase beta annealing
        current_x = x0.copy()
        
        for phase, beta in enumerate(beta_schedule):
            if time.time() - start_time > TIME_LIMIT - 10.0:
                break
                
            # Create closure with fixed beta
            def f_and_g(x, _beta=beta):
                return _objective_and_grad(x, _beta)
            
            try:
                res = opt.minimize(
                    f_and_g,
                    current_x,
                    method='L-BFGS-B',
                    jac=True,
                    options={
                        'maxiter': 100,
                        'ftol': 1e-10,
                        'gtol': 1e-6,
                        'maxls': 20
                    }
                )
                current_x = res.x
            except Exception:
                break
            
            # Check time after each phase
            if time.time() - start_time > TIME_LIMIT - 10.0:
                break
        
        x_opt = current_x.reshape(N, 2)
        
        # Calculate true score
        score = _compute_true_score(x_opt)
        
        if score > best_score:
            best_score = score
            best_points = x_opt.copy()
            
        if time.time() - start_time > TIME_LIMIT - 15.0:
            break
    
    # Final active-set polish on the best solution
    if best_points is not None and time.time() - start_time < TIME_LIMIT - 5.0:
        polished, polished_score = _active_set_polish(best_points, max_iter=30)
        if polished_score > best_score:
            best_score = polished_score
            best_points = polished
    
    # Final validity check
    if best_points is None:
        best_points = inits[0]
        
    radii = np.linalg.norm(best_points, axis=1)
    valid = np.isfinite(best_points).all(axis=1) & (radii < 1.0)
    if not np.all(valid):
        theta = np.random.uniform(0, 2 * np.pi, N)
        rho = np.sqrt(np.random.uniform(0, 0.9, N))
        best_points = np.column_stack((rho * np.cos(theta), rho * np.sin(theta)))
        
    return best_points

# Helper for testing
if __name__ == "__main__":
    pts = construct()
    # Calculate score
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diffs, axis=-1)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    radii = np.linalg.norm(pts, axis=1)
    min_bound_dist = np.min(1.0 - radii)
    score = min(min_pair_dist / 2.0, min_bound_dist)
    print(f"Score: {score}")
