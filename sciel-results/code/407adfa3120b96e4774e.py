"""
Improved program for packing 300 circles in a unit circle.
Uses a hybrid approach:
1. Force-Directed Relaxation to find a valid packing for a given radius.
2. Binary search on radius to find the approximate maximum radius.
3. Soft-Min Gradient Descent for final high-precision polishing.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 300

# EVOLVE-BLOCK-START

def _compute_score(points):
    """Compute the minimum radius for a given set of points."""
    n = len(points)
    if n < 2:
        return min(1.0 - np.linalg.norm(points[0])) if n == 1 else 0.0
    
    # Compute all pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists) / 2.0
    
    # Compute distance to boundary
    boundary_dists = 1.0 - np.linalg.norm(points, axis=1)
    min_boundary = np.min(boundary_dists)
    
    return min(min_pair_dist, min_boundary)

def _hex_lattice_init(n):
    """
    Initialize points in a hexagonal lattice clipped to the unit circle.
    """
    # Estimate spacing for hexagonal packing
    # Area per point ~ pi / n
    # Hex unit cell area ~ (sqrt(3)/2) * s^2
    s = np.sqrt(2.0 * np.pi / (n * np.sqrt(3.0)))
    
    # Generate hex lattice points
    max_k = int(1.0 / (s * 0.5)) + 3
    candidates = []
    for j in range(-max_k, max_k + 1):
        for i in range(-max_k, max_k + 1):
            # Standard hex lattice: offset every other row
            x = s * (i + 0.5 * (j % 2))
            y = s * j * np.sqrt(3.0) / 2.0
            if x*x + y*y <= 0.999:
                candidates.append([x, y])
    
    candidates = np.array(candidates)
    
    if len(candidates) >= n:
        # Select n closest to center
        radii = np.linalg.norm(candidates, axis=1)
        indices = np.argsort(radii)[:n]
        points = candidates[indices].copy()
    else:
        # Fallback: random points inside unit circle
        points = np.random.rand(n, 2) * 2 - 1
        norms = np.linalg.norm(points, axis=1)
        mask = norms > 1.0
        points[mask] = points[mask] / norms[mask, np.newaxis] * 0.95
    
    # Add small perturbation to break symmetry
    points += np.random.normal(0, 1e-5, points.shape)
    
    return points

def _ring_init(n):
    """
    Initialize points in concentric rings.
    """
    points = []
    # Center
    points.append([0.0, 0.0])
    remaining = n - 1
    ring_idx = 1
    while remaining > 0:
        # Estimate points in this ring
        # Circumference ~ 2 * pi * r
        # Spacing ~ s (from hex init)
        s = np.sqrt(2.0 * np.pi / (n * np.sqrt(3.0)))
        r = ring_idx * s
        # Max points in ring without overlap
        max_points = int(2.0 * np.pi * r / s)
        # Take min of remaining and max_points
        num_points = min(remaining, max_points)
        if num_points <= 0:
             num_points = 1 # Force at least 1 to avoid infinite loop if calculation fails
        
        angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        # Add slight radial jitter to avoid perfect symmetry issues
        r_jitter = r * (1 + np.random.normal(0, 0.01, num_points))
        
        for i in range(num_points):
            x = r_jitter[i] * np.cos(angles[i])
            y = r_jitter[i] * np.sin(angles[i])
            points.append([x, y])
            remaining -= 1
        ring_idx += 1
    
    points = np.array(points)
    # Ensure inside unit circle
    norms = np.linalg.norm(points, axis=1)
    mask = norms > 0.99
    points[mask] = points[mask] / norms[mask, np.newaxis] * 0.99
    
    # Add small perturbation
    points += np.random.normal(0, 1e-5, points.shape)
    
    return points

def _run_optimization(points, time_limit=30.0, beta_schedule=None, max_iter_base=200):
    """
    Run L-BFGS-B optimization with a schedule of beta values.
    """
    start_time = time.time()
    if beta_schedule is None:
        beta_schedule = [10.0, 50.0, 200.0, 1000.0, 5000.0, 20000.0]
    
    current_points = points.copy()
    current_score = _compute_score(current_points)
    
    for beta in beta_schedule:
        if time.time() - start_time > time_limit:
            break
        
        try:
            # Adjust max_iter based on beta: higher beta needs more precision
            # But we are limited by time. L-BFGS stops when tol is met.
            if beta >= 10000:
                opts = {
                    'maxiter': 500,
                    'gtol': 1e-10,
                    'ftol': 1e-14,
                    'maxls': 100,
                    'disp': False
                }
            elif beta >= 1000:
                opts = {
                    'maxiter': 300,
                    'gtol': 1e-9,
                    'ftol': 1e-13,
                    'maxls': 80,
                    'disp': False
                }
            else:
                opts = {
                    'maxiter': 200,
                    'gtol': 1e-8,
                    'ftol': 1e-12,
                    'maxls': 50,
                    'disp': False
                }
            
            res = minimize(
                lambda x: _objective_and_grad(x, beta)[0],
                current_points.flatten(),
                jac=lambda x: _objective_and_grad(x, beta)[1],
                method='L-BFGS-B',
                options=opts
            )
            
            candidate = res.x.reshape(-1, 2)
            candidate_score = _compute_score(candidate)
            
            if candidate_score > current_score:
                current_points = candidate
                current_score = candidate_score
            else:
                # If it didn't improve, we might have diverged or stuck.
                # Keep the previous best.
                pass
                
        except Exception:
            continue
    
    return current_points, current_score

def _soft_min_and_grad(values, beta):
    """
    Compute soft-min and its gradient weights using log-sum-exp trick.
    """
    values = np.array(values)
    z = -beta * values
    
    # Log-sum-exp for numerical stability
    max_z = np.max(z)
    exp_vals = np.exp(z - max_z)
    sum_exp = np.sum(exp_vals)
    log_sum_exp = max_z + np.log(sum_exp)
    
    soft_min_val = -log_sum_exp / beta
    weights = exp_vals / sum_exp
    
    return soft_min_val, weights

def _objective_and_grad(x, beta):
    """
    Compute the negative soft-min objective and its gradient.
    """
    p = x.reshape(-1, 2)
    n = len(p)
    
    # Calculate pairwise distances
    diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    
    # Boundary distances
    norms = np.linalg.norm(p, axis=1)
    boundary_dists = 1.0 - norms
    
    # Constraint values: d_ij/2 for pairs, 1-|p_i| for boundary
    iu = np.triu_indices(n, k=1)
    pair_vals = dists[iu] / 2.0
    all_vals = np.concatenate([pair_vals, boundary_dists])
    
    # Soft-min
    r_soft, weights = _soft_min_and_grad(all_vals, beta)
    objective = -r_soft
    
    # Gradient
    grad = np.zeros_like(p)
    
    # Pair contributions
    w_pairs = weights[:len(pair_vals)]
    i_idx, j_idx = iu
    safe_pair_dists = np.where(pair_vals > 1e-12, pair_vals, 1.0)
    coeff = w_pairs * 0.25 / safe_pair_dists
    vecs = p[i_idx] - p[j_idx]
    
    np.add.at(grad, i_idx, coeff[:, np.newaxis] * vecs)
    np.add.at(grad, j_idx, -coeff[:, np.newaxis] * vecs)
    
    # Boundary contributions
    w_bounds = weights[len(pair_vals):]
    safe_norms = np.where(norms > 1e-12, norms, 1.0)
    bound_grad_vecs = -p / safe_norms[:, np.newaxis]
    grad += w_bounds[:, np.newaxis] * bound_grad_vecs
    
    # Gradient of objective = -gradient of soft_min
    grad_obj = -grad
    
    return objective, grad_obj

# _local_optimize is replaced by _run_optimization

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Strategy: Multiple restarts with different initializations
    # 1. Hex lattice
    # 2. Ring lattice
    # 3. Hex lattice with larger jitter
    
    best_points = None
    best_score = 0.0
    
    initializers = [
        _hex_lattice_init,
        _ring_init,
        lambda n: _hex_lattice_init(n) + np.random.normal(0, 0.01, (n, 2))
    ]
    
    # Allocate time to each run. 
    # We have ~145s. 
    # Let's do 3 runs of ~45s each, leaving 10s for final polish/validation if needed, 
    # or just use the best result.
    # Actually, the optimization itself does the polishing.
    
    num_runs = 3
    time_per_run = (deadline - time.time()) / num_runs - 5.0 # 5s buffer
    
    for i, init_func in enumerate(initializers):
        if time.time() > deadline - 10:
            break
            
        points = init_func(N)
        
        # Ensure points are within bounds after jitter
        norms = np.linalg.norm(points, axis=1)
        mask = norms > 0.99
        points[mask] = points[mask] / norms[mask, np.newaxis] * 0.99
        
        # Run optimization
        # We use a slightly shorter time limit here to ensure we don't overshoot
        run_time = min(time_per_run, deadline - time.time() - 10)
        
        if run_time < 5:
            continue
            
        opt_points, opt_score = _run_optimization(
            points, 
            time_limit=run_time,
            beta_schedule=[10.0, 50.0, 200.0, 1000.0, 5000.0, 20000.0]
        )
        
        if opt_score > best_score:
            best_score = opt_score
            best_points = opt_points
    
    # Final fallback if no optimization succeeded
    if best_points is None:
        best_points = _hex_lattice_init(N)
        
    # Final validation and clamping
    norms = np.linalg.norm(best_points, axis=1)
    if np.any(norms > 1.0):
        for i in range(len(best_points)):
            if norms[i] > 1.0:
                best_points[i] = best_points[i] / norms[i] * (1.0 - 1e-12)
        
    # Ensure all values are finite
    if not np.all(np.isfinite(best_points)):
        best_points = np.nan_to_num(best_points, nan=0.0, posinf=0.9, neginf=-0.9)
        
    return best_points

# EVOLVE-BLOCK-END
