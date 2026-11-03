"""Structural change: Stochastic density-field repulsion with gradient-based optimization."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

_DEADLINE = 120.0  # Safety margin below 150s kill


def _compute_radius(centers):
    """Return the largest common radius for given centers in [-0.5,0.5]^2."""
    if centers.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Boundary constraint
    boundary = np.min(np.minimum(
        0.5 - np.abs(centers[:, 0]),
        0.5 - np.abs(centers[:, 1])
    ))
    if boundary < 0:
        return 0.0
    # Pairwise distances
    diff = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    return min(boundary, min_pair / 2.0)


def _initial_hexagonal(n_points):
    """Generate a hexagonal lattice of approximately n_points within [-0.5,0.5]^2."""
    # Estimate spacing to fit ~n_points
    area = 1.0
    avg_area_per_point = area / n_points
    s = np.sqrt(avg_area_per_point / (np.sqrt(3) / 2))
    
    # Generate hex lattice points
    dx = s
    dy = s * np.sqrt(3) / 2
    points = []
    margin = 0.02
    x_min, x_max = -0.5 + margin, 0.5 - margin
    y_min, y_max = -0.5 + margin, 0.5 - margin
    
    row = 0
    y = y_min
    while y <= y_max:
        x_offset = (dx / 2) if (row % 2 == 1) else 0.0
        x = x_min + x_offset
        while x <= x_max:
            points.append((x, y))
            x += dx
        y += dy
        row += 1
    
    points = np.array(points)
    if len(points) < n_points:
        # Reduce spacing and retry
        s *= 0.9
        return _initial_hexagonal(n_points)
    
    # Trim to n_points: keep closest to center
    center_dist = np.sum(points ** 2, axis=1)
    idx = np.argsort(center_dist)[:n_points]
    return points[idx]


def _repulsive_energy_and_gradient(centers, power=4, soft_min=0.001):
    """
    Compute a smooth repulsive energy and its gradient.
    Energy = sum_{i<j} 1/d_ij^power + boundary_penalty
    Gradient computed analytically.
    """
    n = len(centers)
    energy = 0.0
    grad = np.zeros_like(centers)
    
    # Pairwise repulsion
    diff = centers[:, None, :] - centers[None, :, :]
    dists_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    dists = np.sqrt(dists_sq)
    
    # Avoid division by zero
    dists_safe = np.where(dists < soft_min, soft_min, dists)
    
    # Energy: sum 1/d^power
    inv_d_power = 1.0 / (dists_safe ** power)
    np.fill_diagonal(inv_d_power, 0.0)
    energy += np.sum(inv_d_power)
    
    # Gradient of 1/d^power w.r.t. position: -power/d^(power+2) * (c_i - c_j)
    # For pair (i,j): grad_i += -power/d^(power+2) * (c_i - c_j)
    #                grad_j += +power/d^(power+2) * (c_i - c_j)
    factor = power * inv_d_power / dists_safe  # This is power/d^(power+1)
    np.fill_diagonal(factor, 0.0)
    
    # grad_i = -sum_j factor[i,j] * (c_i - c_j)
    #        = -sum_j factor[i,j] * diff[i,j]
    grad -= np.sum(factor[:, :, None] * diff, axis=1)
    
    # Boundary penalty: push points away from walls
    # Penalty = sum over points of (0.5 - |c_x|)^2 for x near boundary, similarly for y
    # Actually, use a soft boundary: penalty = max(0, 0.5 - |c_x| - r_target)^2
    # But we don't know r_target yet. Use a fixed small margin.
    margin = 0.01
    x = centers[:, 0]
    y = centers[:, 1]
    
    # Boundary violation: if x < -0.5+margin or x > 0.5-margin
    bx = np.minimum(0.5 - margin - np.abs(x), 0)
    by = np.minimum(0.5 - margin - np.abs(y), 0)
    energy += np.sum(bx ** 2) + np.sum(by ** 2)
    
    # Gradient of boundary penalty
    # d/dx max(0, 0.5-margin-|x|)^2
    # If x > 0: d/dx (0.5-margin-x)^2 = -2(0.5-margin-x)
    # If x < 0: d/dx (0.5-margin+x)^2 = 2(0.5-margin+x)
    # So grad_x = -2 * sign(x) * max(0, 0.5-margin-|x|)
    grad_x_b = -2.0 * np.sign(x) * np.maximum(0, 0.5 - margin - np.abs(x))
    grad_y_b = -2.0 * np.sign(y) * np.maximum(0, 0.5 - margin - np.abs(y))
    grad[:, 0] += grad_x_b
    grad[:, 1] += grad_y_b
    
    return energy, grad


def _local_search_density(centers, max_iters=20000, initial_lr=0.001, lr_decay=0.999, 
                          anneal_temp=0.001, anneal_decay=0.999):
    """
    Gradient-based local search with simulated annealing perturbations.
    """
    centers = centers.copy()
    best_centers = centers.copy()
    best_radius = _compute_radius(centers)
    
    lr = initial_lr
    temp = anneal_temp
    
    for it in range(max_iters):
        if time.time() > _DEADLINE:
            break
        
        # Compute energy and gradient
        energy, grad = _repulsive_energy_and_gradient(centers, power=4)
        
        # Gradient step
        centers_new = centers - lr * grad
        
        # Simulated annealing perturbation
        if temp > 1e-6:
            noise = np.random.normal(0, temp, size=centers.shape)
            centers_new += noise
        
        # Clip to square with small margin
        centers_new[:, 0] = np.clip(centers_new[:, 0], -0.49, 0.49)
        centers_new[:, 1] = np.clip(centers_new[:, 1], -0.49, 0.49)
        
        # Compute radius of new configuration
        r_new = _compute_radius(centers_new)
        
        # Accept if better, or with probability exp((r_new - best_radius) / temp)
        if r_new >= best_radius:
            centers = centers_new
            best_centers = centers_new.copy()
            best_radius = r_new
        else:
            # Metropolis acceptance
            if temp > 1e-6:
                accept_prob = np.exp((r_new - best_radius) / temp)
                if np.random.random() < accept_prob:
                    centers = centers_new
        
        # Decay learning rate and temperature
        lr *= lr_decay
        temp *= anneal_decay
    
    return best_centers, best_radius


def construct():
    start_time = time.time()
    random.seed(42)
    np.random.seed(42)
    
    # Generate initial hexagonal lattice
    centers = _initial_hexagonal(N)
    
    # Run local search with density-field repulsion
    centers, r_best = _local_search_density(centers, max_iters=15000, 
                                             initial_lr=0.002, lr_decay=0.9995,
                                             anneal_temp=0.002, anneal_decay=0.9995)
    
    # Try a second restart with different perturbation if time allows
    if time.time() < _DEADLINE - 15:
        perturbed = centers + np.random.normal(0, 0.01, size=centers.shape)
        perturbed[:, 0] = np.clip(perturbed[:, 0], -0.48, 0.48)
        perturbed[:, 1] = np.clip(perturbed[:, 1], -0.48, 0.48)
        centers2, r2 = _local_search_density(perturbed, max_iters=10000,
                                              initial_lr=0.001, lr_decay=0.9995,
                                              anneal_temp=0.001, anneal_decay=0.9995)
        if r2 > r_best:
            centers = centers2
            r_best = r2
    
    # Final validation
    r_final = _compute_radius(centers)
    if r_final <= 0:
        # Fallback to square grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        centers = grid[:N]
    
    return centers
# EVOLVE-BLOCK-END
