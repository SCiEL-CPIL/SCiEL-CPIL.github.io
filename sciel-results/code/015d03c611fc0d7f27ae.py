"""
Optimization of 100 equal spheres in a unit cube.
Uses a barrier method with L-BFGS-B optimization to maximize the minimum radius.
"""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

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
    # d_i = min(0.5 - x, x - (-0.5), 0.5 - y, ...) = 0.5 - max(|x|, |y|, |z|)
    max_abs = np.max(np.abs(centers), axis=1)
    wall_dists = 0.5 - max_abs
    min_wall_dist = np.min(wall_dists)
    
    if min_wall_dist < 0:
        return 0.0

    # Pairwise constraints
    # Calculate all pairwise distances efficiently
    # Using broadcasting might be memory heavy for N=100 (100x100x3 is fine)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    # Set diagonal to infinity to exclude self-distance
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # The radius is half the minimum pairwise distance, limited by wall distance
    r = min(min_pair_dist / 2.0, min_wall_dist)
    return r

def _energy_and_gradient(x, R, beta=100.0):
    """
    Calculates a smooth surrogate energy and gradient for a target radius R.
    Uses soft-min via log-sum-exp for pairwise and wall constraints.
    """
    centers = x.reshape(-1, 3)
    n = len(centers)
    
    # 1. Pairwise terms - soft-min approach
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    dists = np.sqrt(sq_dists + 1e-15)
    np.fill_diagonal(dists, np.inf)
    np.fill_diagonal(sq_dists, np.inf)
    
    # Only consider upper triangle to avoid double counting
    i_idx, j_idx = np.triu_indices(n, k=1)
    v_ij = 2.0 * R - dists[i_idx, j_idx]
    d_ij = dists[i_idx, j_idx]
    
    # Soft-min: use log-sum-exp on violations (only where v > 0)
    # Energy = sum of soft violations
    # For smoothness, use: E_pair = sum_i (1/beta) * log(sum_j exp(beta * max(0, v_ij)))
    # But for gradient simplicity, use: E = sum(v_ij^2) for v_ij > 0, with smooth transition
    # Better: use a smooth max function
    # smooth_max(0, v) = (1/beta) * log(1 + exp(beta*v))
    # Its derivative: sigmoid(beta*v)
    
    # Clip for numerical stability
    v_clipped = np.clip(v_ij, -50, 50)
    exp_v = np.exp(beta * v_clipped)
    smooth_v = (1.0 / beta) * np.log1p(exp_v)  # smooth max(0, v)
    
    # Energy: sum of smooth violations squared (or just sum)
    # Use sum of smooth_v for simplicity, gradient is sigmoid(beta*v)
    energy_pair = np.sum(smooth_v)
    
    # Gradient: d/dx_i of smooth_max(0, 2R - d_ij)
    # = sigmoid(beta * (2R - d_ij)) * (-1) * d(d_ij)/d(x_i)
    # d(d_ij)/d(x_i) = (x_i - x_j) / d_ij
    sigmoid_v = 1.0 / (1.0 + np.exp(-beta * v_clipped))
    
    # factor[k] = sigmoid_v[k] / d_ij[k]
    with np.errstate(divide='ignore', invalid='ignore'):
        factors = sigmoid_v / d_ij
    factors[d_ij < 1e-10] = 0.0
    
    # For each pair (i,j), force on i = -factor * (x_i - x_j), force on j = +factor * (x_i - x_j)
    # Wait: dE/dx_i = sigmoid(beta*v) * (-1) * (x_i - x_j)/d_ij
    # So gradient contribution to x_i is: -factor * (x_i - x_j)
    # Gradient contribution to x_j is: +factor * (x_i - x_j)
    
    grad = np.zeros_like(centers)
    diff_ij = diff[i_idx, j_idx]  # x_i - x_j
    # grad[i] += -factor * diff_ij
    # grad[j] += +factor * diff_ij
    
    # Vectorized accumulation
    # For each k, i = i_idx[k], j = j_idx[k]
    # grad[i_idx[k], :] -= factors[k] * diff_ij[k]
    # grad[j_idx[k], :] += factors[k] * diff_ij[k]
    
    # Use np.add.at for vectorized accumulation
    np.add.at(grad, i_idx, -factors[:, np.newaxis] * diff_ij)
    np.add.at(grad, j_idx, factors[:, np.newaxis] * diff_ij)
    
    energy = energy_pair

    # 2. Wall terms - soft approach
    abs_centers = np.abs(centers)
    max_abs = np.max(abs_centers, axis=1)
    wall_dists = 0.5 - max_abs
    
    v_wall = R - wall_dists
    v_wall_clipped = np.clip(v_wall, -50, 50)
    exp_wall = np.exp(beta * v_wall_clipped)
    smooth_wall = (1.0 / beta) * np.log1p(exp_wall)
    
    energy += np.sum(smooth_wall)
    
    # Gradient for wall: sigmoid(beta * v_wall) * d(v_wall)/d(centers)
    # v_wall = R - (0.5 - max(|c|)) = R - 0.5 + max(|c|)
    # d(v_wall)/d(c_ik) = sign(c_ik) if c_ik is the max abs component
    sigmoid_wall = 1.0 / (1.0 + np.exp(-beta * v_wall_clipped))
    
    # For each center, find which coordinate has max abs value
    # If tie, distribute gradient evenly (subgradient)
    max_idx = np.argmax(abs_centers, axis=1)
    
    for i in range(n):
        if sigmoid_wall[i] < 1e-10:
            continue
        k = max_idx[i]
        # d(v_wall)/d(c_ik) = sign(c_ik)
        grad[i, k] += sigmoid_wall[i] * np.sign(centers[i, k])
    
    return energy, grad.flatten()

def _optimize_for_radius(init_centers, target_R, max_iter=200, beta=100.0):
    """
    Attempts to optimize the configuration to fit within radius target_R.
    Returns the optimized centers.
    """
    def objective_and_grad(x):
        val, g = _energy_and_gradient(x, target_R, beta=beta)
        return val, g

    # L-BFGS-B requires bounds. Centers must be in [-0.5, 0.5]
    # Use tighter bounds to keep spheres inside with margin for radius
    margin = target_R * 0.5
    bounds = [(-0.5 + margin, 0.5 - margin)] * (N * 3)
    
    try:
        res = minimize(
            objective_and_grad,
            init_centers.flatten(),
            jac=True, # We return tuple (f, g)
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'disp': False, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        return res.x.reshape(-1, 3), res.fun
    except Exception:
        return init_centers, np.inf

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_radius = 0.0
    best_centers = None
    
    # Strategy 1: Grid Initialization
    # 100 points. 5x4x5 grid? 
    # 5*4*5 = 100.
    # x: 5, y: 4, z: 5
    # Spacing: 1/5, 1/4, 1/5?
    # Centered in [-0.5, 0.5]
    
    def make_grid(nx, ny, nz):
        xs = np.linspace(-0.5 + 0.5/nx, 0.5 - 0.5/nx, nx)
        ys = np.linspace(-0.5 + 0.5/ny, 0.5 - 0.5/ny, ny)
        zs = np.linspace(-0.5 + 0.5/nz, 0.5 - 0.5/nz, nz)
        
        grid = []
        for z in zs:
            for y in ys:
                for x in xs:
                    grid.append([x, y, z])
        return np.array(grid)

    # 5x4x5 = 100
    grid_centers = make_grid(5, 4, 5)
    
    # Strategy 2: Random Initialization
    rng = np.random.RandomState(42)
    # Start near center to avoid wall collisions initially
    random_centers = rng.uniform(-0.4, 0.4, size=(N, 3))
    
    inits = [grid_centers, random_centers]
    
    # Add a few more random inits if time permits
    for i in range(5):
        rc = rng.uniform(-0.4, 0.4, size=(N, 3))
        inits.append(rc)

    for init in inits:
        if time.time() > deadline - 10.0:
            break
            
        # 1. Calculate initial radius
        r_est = _compute_radius(init)
        
        # 2. Iteratively increase target radius
        # Start slightly higher than estimated
        target_R = r_est * 1.01
        if target_R < 0.05: # Fallback for bad inits
            target_R = 0.05
            
        current_centers = init.copy()
        max_iter_per_step = 100
        beta = 100.0
        
        while time.time() < deadline - 15.0:
            # Try to optimize for target_R
            optimized_centers, energy = _optimize_for_radius(current_centers, target_R, max_iter=max_iter_per_step, beta=beta)
            
            # Check if we achieved a valid packing for this radius
            # Energy should be close to 0 (smooth surrogate)
            if energy < 1e-8:
                # Validate the radius actually achieved
                r_achieved = _compute_radius(optimized_centers)
                
                if r_achieved > best_radius:
                    best_radius = r_achieved
                    best_centers = optimized_centers.copy()
                    
                # Try to increase the radius
                # Increase by a small factor
                target_R *= 1.005
                current_centers = optimized_centers
                
                # If we've reached a known good value or higher, slow down or stop
                if target_R > 0.12: 
                     target_R *= 1.001
            else:
                # Failed to fit this radius.
                # Try with lower beta for smoother landscape
                if beta > 10:
                    beta = max(10.0, beta * 0.5)
                    continue
                else:
                    break
        
        # Update best with current optimized centers even if energy wasn't 0 (maybe it's still better than before)
        r_final = _compute_radius(current_centers)
        if r_final > best_radius:
            best_radius = r_final
            best_centers = current_centers.copy()

    # Final safety check
    if best_centers is None:
        best_centers = grid_centers
    else:
        # Ensure centers are strictly inside
        best_centers = np.clip(best_centers, -0.4999, 0.4999)
    
    # Final high-precision polish pass
    if time.time() < deadline - 2.0:
        r_current = _compute_radius(best_centers)
        if r_current > 0:
            # Try to polish with very tight tolerances
            target_R_polish = r_current * 1.002
            for beta_polish in [500.0, 1000.0]:
                if time.time() > deadline - 1.0:
                    break
                polished, e = _optimize_for_radius(
                    best_centers, target_R_polish, 
                    max_iter=300, beta=beta_polish
                )
                r_polished = _compute_radius(polished)
                if r_polished > best_radius:
                    best_radius = r_polished
                    best_centers = polished.copy()
                # Try to push a bit higher
                target_R_polish = r_polished * 1.001
            
            # Final clip
            best_centers = np.clip(best_centers, -0.4999, 0.4999)
    
    return best_centers
# EVOLVE-BLOCK-END
