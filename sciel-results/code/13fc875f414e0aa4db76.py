"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_radius(centers):
    """Compute the maximum common radius for given centers."""
    if len(centers) < 2:
        min_dist_to_wall = np.min(0.5 - np.abs(centers))
        return min_dist_to_wall
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    min_wall_dist = np.min(0.5 - np.abs(centers))
    return min(min_pair_dist / 2.0, min_wall_dist)

def _smooth_min_distance(centers, beta=50.0):
    """
    Compute a smooth surrogate for the minimum distance problem using log-sum-exp.
    Returns (smooth_min_pair_dist, smooth_min_wall_dist, gradient).
    """
    n = len(centers)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    dists = np.sqrt(dists_sq)
    
    # Smooth min of pair distances using log-sum-exp (for min, we use -logsumexp(-x/beta))
    # min(x) ≈ -1/beta * log(sum(exp(-x/beta)))
    # For numerical stability, shift by min value
    min_d = np.min(dists)
    # Use a subset of the closest pairs to keep computation manageable
    # Find indices of the K closest pairs
    K = 50
    flat_dists = dists.flatten()
    # Get K smallest values (excluding diagonal which is inf)
    # We'll use a different approach: compute gradient contribution from all pairs
    # with a soft-min weighting
    
    # Soft-min: f(x) = -1/beta * log(sum_j exp(-x_j/beta))
    # f'(x_i) = -exp(-x_i/beta) / sum_j exp(-x_j/beta)
    
    # For numerical stability, let m = min(dists)
    # f = -1/beta * (m + log(sum_j exp(-(dists_j - m)/beta)))
    # f'(x_i) = -exp(-(dists_i - m)/beta) / sum_j exp(-(dists_j - m)/beta)
    
    shifted = -dists / beta + min_d / beta
    # Clip for numerical stability
    shifted = np.clip(shifted, -500, 0)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    
    # Soft min value
    smooth_min_pair = min_d - np.log(sum_exp) / beta
    
    # Gradient of soft min wrt each center
    # d(smooth_min_pair)/d(c_i) = sum_j w_ij * d(dists_ij)/d(c_i)
    # where w_ij = -exp_shifted[i,j] / sum_exp (for i != j)
    # d(dists_ij)/d(c_i) = (c_i - c_j) / dists_ij
    
    weights = -exp_shifted / sum_exp
    weights = np.where(dists < np.inf, weights, 0.0)  # Zero out diagonal
    
    # Gradient contribution from pair distances
    # grad_i = sum_j weights[i,j] * (c_i - c_j) / dists[i,j]
    unit_vecs = diff / np.where(dists > 0, dists, 1.0)[:, :, np.newaxis]
    unit_vecs = np.where(dists < np.inf, unit_vecs, 0.0)
    
    pair_grad = np.sum(weights[:, :, np.newaxis] * unit_vecs, axis=1)
    
    # Wall distances
    dw = np.zeros((n, 6))
    dw[:, 0] = 0.5 - centers[:, 0]
    dw[:, 1] = 0.5 + centers[:, 0]
    dw[:, 2] = 0.5 - centers[:, 1]
    dw[:, 3] = 0.5 + centers[:, 1]
    dw[:, 4] = 0.5 - centers[:, 2]
    dw[:, 5] = 0.5 + centers[:, 2]
    
    # Smooth min of wall distances
    min_w = np.min(dw)
    shifted_w = -dw / beta + min_w / beta
    shifted_w = np.clip(shifted_w, -500, 0)
    exp_shifted_w = np.exp(shifted_w)
    sum_exp_w = np.sum(exp_shifted_w)
    
    smooth_min_wall = min_w - np.log(sum_exp_w) / beta
    
    # Gradient of soft min wall distance
    weights_w = -exp_shifted_w / sum_exp_w
    
    # Gradient contributions from each wall
    wall_grad = np.zeros_like(centers)
    # dw[:,0] = 0.5 - c_x, d/d(c_x) = -1
    # dw[:,1] = 0.5 + c_x, d/d(c_x) = +1
    # dw[:,2] = 0.5 - c_y, d/d(c_y) = -1
    # dw[:,3] = 0.5 + c_y, d/d(c_y) = +1
    # dw[:,4] = 0.5 - c_z, d/d(c_z) = -1
    # dw[:,5] = 0.5 + c_z, d/d(c_z) = +1
    
    wall_grad[:, 0] += -weights_w[:, 0] * (-1) + weights_w[:, 1] * (1)
    wall_grad[:, 1] += -weights_w[:, 2] * (-1) + weights_w[:, 3] * (1)
    wall_grad[:, 2] += -weights_w[:, 4] * (-1) + weights_w[:, 5] * (1)
    
    total_grad = pair_grad + wall_grad
    
    return smooth_min_pair, smooth_min_wall, total_grad

def _energy_and_grad(centers, target_r):
    """Compute energy and gradient for repulsion model."""
    n = len(centers)
    energy = 0.0
    grad = np.zeros_like(centers)
    
    # Pairwise repulsion
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # Mask for pairs closer than 2*target_r
    mask = dists < 2.0 * target_r
    violations = np.where(mask, 2.0 * target_r - dists, 0.0)
    energy += np.sum(violations**2)
    
    unit_vecs = diff / dists[:, :, np.newaxis]
    coeff = np.where(mask, -2.0 * (2.0 * target_r - dists), 0.0)
    pair_grad = np.sum(coeff[:, :, np.newaxis] * unit_vecs, axis=1)
    grad += pair_grad
    
    # Wall repulsion
    dw = np.zeros((n, 6))
    dw[:, 0] = 0.5 - centers[:, 0]
    dw[:, 1] = 0.5 + centers[:, 0]
    dw[:, 2] = 0.5 - centers[:, 1]
    dw[:, 3] = 0.5 + centers[:, 1]
    dw[:, 4] = 0.5 - centers[:, 2]
    dw[:, 5] = 0.5 + centers[:, 2]
    
    min_wall_dist = np.min(dw, axis=1)
    wall_violations = np.maximum(0, target_r - min_wall_dist)
    energy += np.sum(wall_violations**2)
    
    min_idx = np.argmin(dw, axis=1)
    wall_grad = np.zeros_like(centers)
    violated = wall_violations > 0
    
    for i in range(n):
        if violated[i]:
            idx = min_idx[i]
            factor = -2.0 * (target_r - min_wall_dist[i])
            if idx == 0: wall_grad[i, 0] += -factor * (-1)
            elif idx == 1: wall_grad[i, 0] += -factor * (1)
            elif idx == 2: wall_grad[i, 1] += -factor * (-1)
            elif idx == 3: wall_grad[i, 1] += -factor * (1)
            elif idx == 4: wall_grad[i, 2] += -factor * (-1)
            elif idx == 5: wall_grad[i, 2] += -factor * (1)
            
    grad += wall_grad
    
    return energy, grad

def _sgld_optimize(centers_init, max_steps=2000, start_temp=0.01, end_temp=0.001,
                   start_lr=0.001, end_lr=0.0001, beta_schedule=None):
    """
    Stochastic Gradient Langevin Dynamics optimizer.
    Uses a smooth surrogate for the minimum distance objective.
    """
    n = len(centers_init)
    centers = centers_init.copy()
    rng = np.random.RandomState(42)
    
    best_centers = centers.copy()
    best_radius = _compute_radius(centers)
    
    # Schedule parameters
    for step in range(max_steps):
        # Linear annealing
        t = step / max_steps
        temp = start_temp * (1 - t) + end_temp * t
        lr = start_lr * (1 - t) + end_lr * t
        beta = 20.0 + 80.0 * t  # Increase sharpness over time
        
        # Compute smooth objective and gradient
        smooth_pair, smooth_wall, grad = _smooth_min_distance(centers, beta=beta)
        # We want to maximize the minimum distance, so minimize negative
        obj = -min(smooth_pair, smooth_wall)
        grad_neg = -grad
        
        # SGLD update: c_{t+1} = c_t - lr * grad + sqrt(2*lr*temp) * noise
        noise = rng.randn(n, 3)
        noise_scale = np.sqrt(2.0 * lr * temp)
        centers = centers - lr * grad_neg + noise_scale * noise
        
        # Keep centers within bounds
        centers = np.clip(centers, -0.4999, 0.4999)
        
        # Evaluate actual radius
        current_radius = _compute_radius(centers)
        if current_radius > best_radius:
            best_radius = current_radius
            best_centers = centers.copy()
    
    return best_centers, best_radius

def _optimize_from_init(centers_init, target_r, max_iter=500, tol=1e-12):
    """Optimize from a given initial configuration using L-BFGS-B."""
    def objective(x):
        c = x.reshape(-1, 3)
        e, g = _energy_and_grad(c, target_r)
        return e, g
    
    x0 = centers_init.flatten()
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    result = minimize(
        objective,
        x0,
        method='L-BFGS-B',
        jac=True,
        bounds=bounds,
        options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-8, 'disp': False}
    )
    
    return result.x.reshape(-1, 3), result.success, result.fun

def _grid_init(n):
    """Generate grid initial configuration."""
    m = int(np.ceil(n ** (1.0/3.0)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:n]

def _random_init(n, seed=None):
    """Generate random initial configuration inside the cube."""
    rng = np.random.RandomState(seed)
    return rng.uniform(-0.45, 0.45, size=(n, 3))

def _fine_grid_init(n):
    """Generate a fine grid with slight perturbations to break symmetry."""
    m = int(np.ceil(n ** (1.0/3.0)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    grid = grid[:n]
    # Add small deterministic perturbations
    rng = np.random.RandomState(123)
    grid = grid + rng.randn(n, 3) * 0.005
    grid = np.clip(grid, -0.49, 0.49)
    return grid

def construct():
    """Construct the optimal packing of N spheres in the unit cube."""
    start_time = time.time()
    deadline = start_time + 140.0  # 10 second safety margin
    
    best_centers = None
    best_radius = 0.0
    
    # Strategy 1: Grid initialization
    init_grid = _grid_init(N)
    r_grid = _compute_radius(init_grid)
    best_radius = r_grid
    best_centers = init_grid.copy()
    
    # Strategy 2: Fine grid with perturbations
    if time.time() < deadline - 60:
        init_fine = _fine_grid_init(N)
        r_fine = _compute_radius(init_fine)
        if r_fine > best_radius:
            best_radius = r_fine
            best_centers = init_fine.copy()
    
    # Strategy 3: SGLD optimization from best start
    if time.time() < deadline - 40:
        try:
            sgld_centers, sgld_radius = _sgld_optimize(
                best_centers,
                max_steps=3000,
                start_temp=0.008,
                end_temp=0.0005,
                start_lr=0.0008,
                end_lr=0.00005
            )
            if sgld_radius > best_radius:
                best_radius = sgld_radius
                best_centers = sgld_centers.copy()
        except Exception:
            pass
    
    # Strategy 4: L-BFGS-B polish from best SGLD result
    if time.time() < deadline - 20:
        try:
            target_r = best_radius * 1.005
            optimized, success, energy = _optimize_from_init(
                best_centers, target_r, max_iter=300, tol=1e-12
            )
            r_opt = _compute_radius(optimized)
            if r_opt > best_radius:
                best_radius = r_opt
                best_centers = optimized.copy()
        except Exception:
            pass
    
    # Strategy 5: Another SGLD round with different parameters
    if time.time() < deadline - 15:
        try:
            sgld_centers2, sgld_radius2 = _sgld_optimize(
                best_centers,
                max_steps=2000,
                start_temp=0.003,
                end_temp=0.0002,
                start_lr=0.0004,
                end_lr=0.00002
            )
            if sgld_radius2 > best_radius:
                best_radius = sgld_radius2
                best_centers = sgld_centers2.copy()
        except Exception:
            pass
    
    # Final L-BFGS-B polish
    if time.time() < deadline - 5:
        try:
            target_r = best_radius
            optimized, success, energy = _optimize_from_init(
                best_centers, target_r, max_iter=200, tol=1e-13
            )
            r_opt = _compute_radius(optimized)
            if r_opt > best_radius:
                best_radius = r_opt
                best_centers = optimized.copy()
        except Exception:
            pass
    
    # Final check and clip
    if best_centers is None:
        best_centers = _grid_init(N)
        
    best_centers = np.clip(best_centers, -0.4999, 0.4999)
    
    return best_centers
# EVOLVE-BLOCK-END
