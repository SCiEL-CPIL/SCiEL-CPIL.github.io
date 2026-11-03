"""
Optimization of 100 equal spheres in a unit cube.
Uses a barrier method with L-BFGS-B optimization to maximize the minimum radius.
"""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

def _compute_radius(centers):
    """
    Computes the maximum common radius for a given set of centers.
    Centers: (N, 3) array.
    Cube: [-0.5, 0.5]^3.
    """
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
        
    # Wall constraints: distance from center to nearest wall
    max_abs = np.max(np.abs(centers), axis=1)
    wall_dists = 0.5 - max_abs
    min_wall_dist = np.min(wall_dists)
    
    if min_wall_dist < 0:
        return 0.0

    # Pairwise constraints
    dists = cdist(centers, centers)
    # Exclude self-distance (diagonal)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    r = min(min_pair_dist / 2.0, min_wall_dist)
    return r

def _soft_min_radius(x, beta=100.0):
    """
    Smooth approximation of the minimum radius using log-sum-exp.
    Returns a smooth surrogate for the objective with analytical gradient.
    Vectorized implementation for speed.
    """
    centers = x.reshape(-1, 3)
    n = len(centers)
    
    # Pairwise distances (upper triangle)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2) + 1e-15
    dists = np.sqrt(dists_sq)
    np.fill_diagonal(dists, np.inf)
    
    i_idx, j_idx = np.triu_indices(n, k=1)
    pair_dists = dists[i_idx, j_idx] / 2.0
    
    # Wall distances
    max_abs = np.max(np.abs(centers), axis=1)
    wall_dists = 0.5 - max_abs
    
    # Combine all constraints into a single array
    all_dists = np.concatenate([pair_dists, wall_dists])
    
    # Soft min via log-sum-exp
    min_d = np.min(all_dists)
    if min_d < 1e-10:
        return 0.0, np.zeros_like(x)
    
    shifted = all_dists - min_d
    weights = np.exp(-beta * shifted)
    S = np.sum(weights)
    if S < 1e-300:
        return 0.0, np.zeros_like(x)
    w_norm = weights / S
    
    soft_min = min_d - np.log(S) / beta
    
    # Compute gradient using vectorized operations
    grad = np.zeros_like(x)
    
    # Pairwise distance gradients (vectorized)
    pair_w = w_norm[:len(i_idx)]
    pair_d = pair_dists
    valid = pair_d > 1e-10
    
    if np.any(valid):
        # Vector = centers[i] - centers[j] for valid pairs
        vecs = centers[i_idx[valid]] - centers[j_idx[valid]]
        # Factor = -w / (4 * pair_d)
        factors = -pair_w[valid] / (4.0 * pair_d[valid])
        
        # Accumulate gradients
        # grad[i] += factor * vec
        # grad[j] -= factor * vec
        # Using np.add.at for accumulation
        np.add.at(grad, i_idx[valid], factors[:, None] * vecs)
        np.add.at(grad, j_idx[valid], -factors[:, None] * vecs)
    
    # Wall distance gradients (vectorized)
    wall_w = w_norm[len(i_idx):]
    abs_centers = np.abs(centers)
    max_vals = max_abs
    
    # For each center, find which coordinate is the max
    # Use argmax to find the dominant coordinate
    max_coord_idx = np.argmax(abs_centers, axis=1)
    
    # Create a sparse-like update for wall gradients
    # grad[i, k] += w * (-sign(centers[i, k]))
    # We can use advanced indexing
    rows = np.arange(n)
    cols = max_coord_idx
    signs = np.sign(centers[rows, cols])
    updates = wall_w * (-signs)
    
    # Accumulate into grad
    grad[rows, cols] += updates
    
    return -soft_min, grad

def _energy_and_gradient(x, R):
    """
    Calculates the repulsion energy and gradient for a target radius R.
    Energy is the sum of squared violations of the distance constraints.
    Uses vectorized operations for speed.
    """
    centers = x.reshape(-1, 3)
    n = len(centers)
    energy = 0.0
    grad = np.zeros_like(centers)
    
    # 1. Pairwise terms
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2) + 1e-15)
    np.fill_diagonal(dists, np.inf)
    
    # Vectorized computation for pairs
    i_idx, j_idx = np.triu_indices(n, k=1)
    d_ij = dists[i_idx, j_idx]
    v_ij = np.maximum(0, 2.0 * R - d_ij)
    
    if np.any(v_ij > 0):
        energy += np.sum(v_ij**2)
        
        # Gradient: for each violating pair, add force
        # dE/dc_i = -2 * v_ij * (c_i - c_j) / d_ij
        # dE/c_j = 2 * v_ij * (c_i - c_j) / d_ij
        # factor = -2 * v / d
        # force = factor * vec
        # grad[i] += force, grad[j] -= force
        
        valid = v_ij > 0
        if np.any(valid):
            vecs = centers[i_idx[valid]] - centers[j_idx[valid]]
            d_valid = d_ij[valid]
            v_valid = v_ij[valid]
            
            # Avoid division by zero
            d_safe = np.where(d_valid > 1e-10, d_valid, 1e-10)
            factors = -2.0 * v_valid / d_safe
            forces = factors[:, None] * vecs
            
            np.add.at(grad, i_idx[valid], forces)
            np.add.at(grad, j_idx[valid], -forces)

    # 2. Wall terms - vectorized
    abs_centers = np.abs(centers)
    max_abs = np.max(abs_centers, axis=1)
    wall_dists = 0.5 - max_abs
    v_wall = np.maximum(0, R - wall_dists)
    
    if np.any(v_wall > 0):
        energy += np.sum(v_wall**2)
        
        # Gradient for wall constraint:
        # d/dx (R - (0.5 - |x|))^2 = 2 * v * sign(x) for the dominant coordinate
        # Only update the coordinate with max absolute value
        
        valid_wall = v_wall > 0
        if np.any(valid_wall):
            idxs = np.where(valid_wall)[0]
            max_coord_idx = np.argmax(abs_centers[idxs], axis=1)
            rows = idxs
            cols = max_coord_idx
            signs = np.sign(centers[rows, cols])
            # dE/dx = 2 * v * sign(x)
            # We want to minimize energy, so we move x in direction -grad
            # grad = 2 * v * sign(x)
            updates = 2.0 * v_wall[idxs] * signs
            grad[rows, cols] += updates
            
    return energy, grad.flatten()

def _polish_centers(centers, max_iter=500, time_budget=10.0):
    """
    High-precision polish pass to maximize the minimum radius.
    Uses a smooth objective with tight tolerances and analytical gradient.
    Multiple stages with increasing beta for convergence.
    """
    start_time = time.time()
    current = centers.copy()
    
    # Stage 1: Moderate beta for broad convergence
    def objective_and_grad_beta1(x):
        val, g = _soft_min_radius(x, beta=50.0)
        return val, g

    bounds = [(-0.5, 0.5)] * (N * 3)
    
    try:
        res = minimize(
            objective_and_grad_beta1,
            current.flatten(),
            jac=True,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter//2, 'disp': False, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 2000}
        )
        current = res.x.reshape(-1, 3)
        current = np.clip(current, -0.499999, 0.499999)
    except Exception:
        pass
    
    if time.time() - start_time > time_budget * 0.6:
        return current
    
    # Stage 2: Higher beta for fine convergence
    def objective_and_grad_beta2(x):
        val, g = _soft_min_radius(x, beta=100.0)
        return val, g
    
    try:
        res = minimize(
            objective_and_grad_beta2,
            current.flatten(),
            jac=True,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter//2, 'disp': False, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 3000}
        )
        current = res.x.reshape(-1, 3)
        current = np.clip(current, -0.499999, 0.499999)
    except Exception:
        pass
    
    if time.time() - start_time > time_budget * 0.85:
        return current
    
    # Stage 3: Very high beta for final precision
    def objective_and_grad_beta3(x):
        val, g = _soft_min_radius(x, beta=200.0)
        return val, g
    
    try:
        res = minimize(
            objective_and_grad_beta3,
            current.flatten(),
            jac=True,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 100, 'disp': False, 'ftol': 1e-15, 'gtol': 1e-13, 'maxfun': 1000}
        )
        current = res.x.reshape(-1, 3)
        current = np.clip(current, -0.499999, 0.499999)
    except Exception:
        pass
    
    return current

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin
    
    best_radius = 0.0
    best_centers = None
    
    rng = np.random.RandomState(42)
    
    # Generate initial candidates
    # 1. Jittered Grid
    # A 5x4x5 grid is 100 points.
    # Spacing: x: 1/5=0.2, y: 1/4=0.25, z: 1/5=0.2
    # Radius approx min(0.1, 0.125, 0.1) = 0.1
    # We can jitter to improve.
    
    def make_jittered_grid(nx, ny, nz, noise_scale=0.01):
        xs = np.linspace(-0.5 + 0.5/nx, 0.5 - 0.5/nx, nx)
        ys = np.linspace(-0.5 + 0.5/ny, 0.5 - 0.5/ny, ny)
        zs = np.linspace(-0.5 + 0.5/nz, 0.5 - 0.5/nz, nz)
        
        grid = []
        for z in zs:
            for y in ys:
                for x in xs:
                    grid.append([x, y, z])
        centers = np.array(grid)
        
        # Add noise
        noise = rng.normal(0, noise_scale, size=centers.shape)
        centers = centers + noise
        centers = np.clip(centers, -0.45, 0.45)
        
        return centers

    inits = []
    
    # Try a few grid configurations
    # 5x4x5
    inits.append(make_jittered_grid(5, 4, 5, 0.02))
    inits.append(make_jittered_grid(5, 4, 5, 0.05))
    inits.append(make_jittered_grid(5, 4, 5, 0.1))
    
    # 4x5x5
    inits.append(make_jittered_grid(4, 5, 5, 0.02))
    inits.append(make_jittered_grid(4, 5, 5, 0.05))
    
    # 5x5x4
    inits.append(make_jittered_grid(5, 5, 4, 0.02))
    inits.append(make_jittered_grid(5, 5, 4, 0.05))
    
    # Random
    for _ in range(5):
        inits.append(rng.uniform(-0.45, 0.45, size=(N, 3)))
    
    for init in inits:
        if time.time() > deadline - 20.0:
            break
            
        # Polish the initialization
        # Use a generous time budget for the first few, less for later
        time_budget = min(15.0, deadline - 10.0 - time.time())
        if time_budget < 2.0:
            break
            
        polished = _polish_centers(init, max_iter=500, time_budget=time_budget)
        
        r = _compute_radius(polished)
        
        if r > best_radius:
            best_radius = r
            best_centers = polished.copy()
            
        # Early stopping if we reach a very high radius
        if best_radius > 0.111:
            break

    # Final safety check
    if best_centers is None:
        # Fallback to a simple grid
        xs = np.linspace(-0.4, 0.4, 5)
        ys = np.linspace(-0.4, 0.4, 4)
        zs = np.linspace(-0.4, 0.4, 5)
        grid = []
        for z in zs:
            for y in ys:
                for x in xs:
                    grid.append([x, y, z])
        best_centers = np.array(grid)
    else:
        best_centers = np.clip(best_centers, -0.49999, 0.49999)
        if not np.all(np.isfinite(best_centers)):
             # Fallback
             xs = np.linspace(-0.4, 0.4, 5)
             ys = np.linspace(-0.4, 0.4, 4)
             zs = np.linspace(-0.4, 0.4, 5)
             grid = []
             for z in zs:
                 for y in ys:
                     for x in xs:
                         grid.append([x, y, z])
             best_centers = np.array(grid)
        
    return best_centers
# EVOLVE-BLOCK-END
