"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = 145.0  # safety margin before 150s kill

def _compute_radius(c):
    """Compute the achievable common radius for centres c (N,3)."""
    n = c.shape[0]
    # Wall distances
    wall = 0.5 - np.max(np.abs(c), axis=1)
    min_wall = np.min(wall)
    # Pairwise distances - vectorized for 100 points
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_dist = np.min(dist)
    return min(min_dist / 2.0, min_wall)

def _soft_min_radius(c, beta=50.0):
    """
    Smooth surrogate for the radius using soft-min.
    Returns a differentiable approximation of min(pairwise_dists/2, wall_dists).
    """
    n = c.shape[0]
    # Wall distances
    wall = 0.5 - np.max(np.abs(c), axis=1)
    
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    
    # Soft-min of pairwise distances/2
    # Use log-sum-exp trick for numerical stability
    half_dists_sq = dist_sq / 4.0  # (d/2)^2
    # We want soft_min of sqrt(half_dists_sq), approximate with soft_min of half_dists_sq then sqrt
    # Actually, let's work with the squared values for smoothness
    # soft_min(x_i) ≈ -1/beta * log(sum(exp(-beta*x_i)))
    
    # For pairwise: soft_min of (d_ij/2) over i<j
    # Use the squared version for better conditioning
    mask = ~np.eye(n, dtype=bool)
    valid_half_dists_sq = np.where(mask, half_dists_sq, np.inf)
    
    # Soft-min of half_dists_sq
    min_half_sq = np.min(valid_half_dists_sq)
    exp_terms = np.exp(-beta * (valid_half_dists_sq - min_half_sq))
    soft_min_half_sq = min_half_sq - np.log(np.sum(exp_terms)) / beta
    
    # Wall soft-min
    exp_wall = np.exp(-beta * (wall - np.min(wall)))
    soft_min_wall = np.min(wall) - np.log(np.sum(exp_wall)) / beta
    
    # Combine: soft_min of the two soft-mins
    # Approximate by taking the minimum of the two soft-min values (still somewhat smooth)
    # Better: use another soft-min
    all_vals = np.array([soft_min_half_sq, soft_min_wall])
    min_val = np.min(all_vals)
    exp_all = np.exp(-beta * (all_vals - min_val))
    combined_soft_min = min_val - np.log(np.sum(exp_all)) / beta
    
    return np.sqrt(combined_soft_min) if combined_soft_min > 0 else 0.0

def _soft_min_radius_grad(c, beta=50.0):
    """Gradient of the soft-min radius surrogate."""
    n = c.shape[0]
    grad = np.zeros_like(c)
    
    # Wall gradient
    wall = 0.5 - np.max(np.abs(c), axis=1)
    # Gradient of max(|c_ik|) w.r.t. c is sign(c) at the max component
    abs_c = np.abs(c)
    max_idx = np.argmax(abs_c, axis=1)
    signs = np.sign(c[np.arange(n), max_idx])
    wall_grad = -np.zeros_like(c)
    wall_grad[np.arange(n), max_idx] = signs
    
    # Soft-min of wall
    min_wall = np.min(wall)
    exp_wall = np.exp(-beta * (wall - min_wall))
    w_wall = exp_wall / np.sum(exp_wall)
    soft_min_wall = min_wall - np.log(np.sum(exp_wall)) / beta
    
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    half_dists_sq = dist_sq / 4.0
    
    mask = ~np.eye(n, dtype=bool)
    valid_half_dists_sq = np.where(mask, half_dists_sq, np.inf)
    
    # Soft-min of half_dists_sq
    min_half_sq = np.min(valid_half_dists_sq)
    exp_terms = np.exp(-beta * (valid_half_dists_sq - min_half_sq))
    w_pair = np.where(mask, exp_terms / np.sum(exp_terms), 0.0)
    soft_min_half_sq = min_half_sq - np.log(np.sum(exp_terms)) / beta
    
    # Gradient of soft_min_half_sq w.r.t. c
    # d(soft_min_half_sq)/d(c_i) = sum_j w_pair[i,j] * d(half_dists_sq[i,j])/d(c_i)
    # d(half_dists_sq[i,j])/d(c_i) = (c_i - c_j)
    # So grad_i = sum_j w_pair[i,j] * (c_i - c_j)
    pair_grad = np.sum(w_pair[:, :, np.newaxis] * diff, axis=1)
    
    # Combine gradients
    min_val = np.min([soft_min_half_sq, soft_min_wall])
    exp_all = np.exp(-beta * (np.array([soft_min_half_sq, soft_min_wall]) - min_val))
    w_pair_combined = exp_all[0] / np.sum(exp_all)
    w_wall_combined = exp_all[1] / np.sum(exp_all)
    
    # Chain rule: d(sqrt(x))/dx = 1/(2*sqrt(x))
    sqrt_val = np.sqrt(max(combined_soft_min := (soft_min_half_sq if soft_min_half_sq < soft_min_wall else soft_min_wall), 1e-12))
    chain = 1.0 / (2.0 * sqrt_val)
    
    grad = chain * (w_pair_combined * pair_grad + w_wall_combined * wall_grad)
    
    return grad

def _relax_spheres(c, r, max_iter=3000, tol=1e-9):
    """
    Relax spheres to satisfy radius r using vectorized repulsion.
    Returns (c_new, converged)
    """
    c = c.copy()
    n = c.shape[0]
    limit = 0.5 - r
    lr = 0.3
    
    for it in range(max_iter):
        # Pairwise differences and distances
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=2))
        
        # Avoid division by zero
        dist_safe = np.where(dist > 1e-10, dist, 1.0)
        
        # Overlap: positive if dist < 2r
        overlap = np.where(dist < 2*r, 2*r - dist, 0.0)
        
        # Force on each sphere from all others
        force = np.sum(diff * (overlap / dist_safe)[:, :, np.newaxis], axis=1)
        
        # Update positions with adaptive learning rate
        c += lr * force
        
        # Project to walls
        c = np.clip(c, -limit, limit)
        
        # Check convergence
        max_force = np.max(np.abs(force))
        if max_force < tol:
            return c, True
        
        # Adapt learning rate
        if it % 100 == 0 and it > 0:
            lr *= 0.95
            
    return c, False

def _make_grid(nx, ny, nz, jitter=0.0):
    """Create a grid of points in the unit cube centered at origin."""
    xs = (np.arange(nx) + 0.5) / nx - 0.5
    ys = (np.arange(ny) + 0.5) / ny - 0.5
    zs = (np.arange(nz) + 0.5) / nz - 0.5
    grid = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    if jitter > 0:
        grid = grid + np.random.uniform(-jitter, jitter, grid.shape)
    return grid

def _polish_with_scipy(c, r_target, time_budget=5.0):
    """
    Use scipy minimize to polish the configuration for a target radius.
    Maximizes the soft-min radius surrogate.
    """
    t0 = time.time()
    n = c.shape[0]
    c_flat = c.flatten().copy()
    
    def objective(x):
        if time.time() - t0 > time_budget:
            return 0.0
        c_x = x.reshape(n, 3)
        # Penalize going outside bounds
        limit = 0.5 - r_target
        outside = np.sum(np.maximum(0, np.abs(c_x) - limit)) ** 2
        # Maximize soft-min radius, so minimize negative
        r_soft = _soft_min_radius(c_x, beta=30.0)
        return -r_soft + 100.0 * outside
    
    def gradient(x):
        if time.time() - t0 > time_budget:
            return np.zeros_like(x)
        c_x = x.reshape(n, 3)
        grad_soft = _soft_min_radius_grad(c_x, beta=30.0)
        # Add wall penalty gradient
        limit = 0.5 - r_target
        outside_grad = np.zeros_like(c_x)
        for i in range(n):
            for k in range(3):
                if c_x[i, k] > limit:
                    outside_grad[i, k] = 200.0 * (c_x[i, k] - limit)
                elif c_x[i, k] < -limit:
                    outside_grad[i, k] = 200.0 * (c_x[i, k] + limit)
        return (-grad_soft + outside_grad).flatten()
    
    try:
        result = minimize(
            objective, c_flat, jac=gradient,
            method='L-BFGS-B',
            bounds=[(-0.5 + r_target, 0.5 - r_target)] * (n * 3),
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        c_new = result.x.reshape(n, 3)
        c_new = np.clip(c_new, -(0.5 - r_target), 0.5 - r_target)
        return c_new
    except Exception:
        return c

def construct():
    t0 = time.time()
    rng = np.random.RandomState(42)
    
    best_c = None
    best_r = 0.0
    
    # Try multiple grid configurations and optimizations
    grid_configs = [
        (5, 5, 4),
        (4, 5, 5),
        (5, 4, 5),
    ]
    
    for nx, ny, nz in grid_configs:
        if time.time() - t0 > _DEADLINE:
            break
        
        for jitter in [0.0, 0.002, 0.005, 0.01]:
            if time.time() - t0 > _DEADLINE:
                break
            
            c0 = _make_grid(nx, ny, nz, jitter=jitter)
            c0 = np.clip(c0, -0.49, 0.49)
            
            r_current = _compute_radius(c0)
            if best_r is None or r_current > best_r:
                best_r = r_current
                best_c = c0.copy()
            
            # Incrementally increase radius
            r_target = r_current * 1.003
            max_r_target = 0.1125  # Slightly above known best 0.11138
            
            while time.time() - t0 < _DEADLINE and r_target <= max_r_target:
                c_test, converged = _relax_spheres(c0, r_target, max_iter=2000)
                
                r_actual = _compute_radius(c_test)
                
                if r_actual >= r_target - 1e-9 and r_actual > best_r:
                    best_r = r_actual
                    best_c = c_test.copy()
                    c0 = c_test.copy()
                    r_target *= 1.003
                elif r_actual > best_r:
                    # Even if not fully converged, if we improved, keep it
                    best_r = r_actual
                    best_c = c_test.copy()
                    c0 = c_test.copy()
                    r_target = r_actual * 1.002
                else:
                    # Failed to improve, try smaller increment
                    r_target = (r_target + best_r) / 2.0
                    if r_target <= best_r + 1e-10:
                        break
            
            # After grid optimization, try random perturbations
            if best_c is not None and time.time() - t0 < _DEADLINE:
                for _ in range(3):
                    if time.time() - t0 > _DEADLINE:
                        break
                    c_pert = best_c + rng.randn(*best_c.shape) * 0.008
                    c_pert = np.clip(c_pert, -(0.5 - best_r), 0.5 - best_r)
                    r_pert = _compute_radius(c_pert)
                    if r_pert > best_r:
                        best_r = r_pert
                        best_c = c_pert.copy()
    
    # Final polish with scipy if we have time
    if best_c is not None and time.time() - t0 < _DEADLINE - 10:
        r_target = best_r * 1.001
        c_polished = _polish_with_scipy(best_c, r_target, time_budget=min(8.0, _DEADLINE - (time.time() - t0)))
        r_polished = _compute_radius(c_polished)
        if r_polished > best_r:
            best_r = r_polished
            best_c = c_polished.copy()
        
        # Try another polish with slightly higher target
        if time.time() - t0 < _DEADLINE - 5:
            r_target2 = best_r * 1.0005
            c_polished2 = _polish_with_scipy(best_c, r_target2, time_budget=min(5.0, _DEADLINE - (time.time() - t0)))
            r_polished2 = _compute_radius(c_polished2)
            if r_polished2 > best_r:
                best_r = r_polished2
                best_c = c_polished2.copy()
    
    # Ensure valid output
    if best_c is None:
        best_c = _make_grid(5, 5, 4)
    
    # Final safety: clip to ensure inside cube
    min_r = _compute_radius(best_c)
    limit = 0.5 - min_r
    best_c = np.clip(best_c, -limit, limit)
    
    return best_c
# EVOLVE-BLOCK-END
