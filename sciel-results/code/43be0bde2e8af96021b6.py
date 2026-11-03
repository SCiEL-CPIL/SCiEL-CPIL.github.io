"""Rewritten program for packing 200 equal circles in a unit square.
Uses a combination of force-directed spreading, gradient-based soft-min
optimization, and simulated annealing for multi-start local refinement."""

import numpy as np
import time
import math
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 200


def _compute_radius(centers):
    """Compute the minimum radius for a set of centers in [-0.5, 0.5]^2."""
    n = len(centers)
    if n < 2:
        return 0.5
    
    # Wall distances
    wall_dists = 0.5 - np.abs(centers)
    min_wall = wall_dists.min()
    
    # Pairwise distances
    dist_matrix = cdist(centers, centers)
    np.fill_diagonal(dist_matrix, np.inf)
    min_pair = dist_matrix.min()
    
    r = min(min_wall, min_pair / 2.0)
    return max(0.0, float(r))


def _soft_min_obj(x, k):
    """Soft-min objective: negative of soft-min of constraints."""
    centers = x.reshape(-1, 2)
    n = len(centers)
    
    # Wall constraints
    wall_dists = 0.5 - np.abs(centers)
    
    # Pairwise constraints (only nearby pairs for efficiency)
    # Use a threshold to avoid O(n^2) for large n
    dist_matrix = cdist(centers, centers)
    np.fill_diagonal(dist_matrix, np.inf)
    
    # Soft-min over all constraints
    all_constraints = np.concatenate([wall_dists.flatten(), dist_matrix / 2.0])
    
    # Soft-min: -log(sum(exp(-k * c))) / k
    # Clip to avoid overflow
    all_constraints = np.clip(all_constraints, -10, 10)
    soft_min_val = -np.log(np.sum(np.exp(-k * all_constraints))) / k
    
    return -soft_min_val


def _soft_min_grad(x, k):
    """Gradient of soft-min objective."""
    centers = x.reshape(-1, 2)
    n = len(centers)
    
    wall_dists = 0.5 - np.abs(centers)
    dist_matrix = cdist(centers, centers)
    np.fill_diagonal(dist_matrix, np.inf)
    
    all_constraints = np.concatenate([wall_dists.flatten(), dist_matrix / 2.0])
    all_constraints = np.clip(all_constraints, -10, 10)
    
    exp_vals = np.exp(-k * all_constraints)
    sum_exp = np.sum(exp_vals)
    
    # Gradient of -soft_min = -d/dx [-log(sum(exp(-k*c)))/k]
    # = (1/k) * d/dx log(sum(exp(-k*c)))
    # = (1/k) * sum_j (d c_j/dx * exp(-k*c_j)) / sum(exp(-k*c_j))
    # But we want -soft_min, so gradient is:
    # d(-soft_min)/dx = -(1/k) * sum_j (-k * dc_j/dx * exp(-k*c_j)) / sum_exp
    # = sum_j (dc_j/dx * exp(-k*c_j)) / sum_exp
    
    grad = np.zeros_like(x)
    
    # Wall contributions
    for i in range(n):
        for dim in range(2):
            idx = i * 2 + dim
            c_idx = i * 2 + dim  # index in wall_dists
            dc_dx = -np.sign(centers[i, dim]) if centers[i, dim] != 0 else 0.0
            grad[idx] += dc_dx * exp_vals[c_idx] / sum_exp
    
    # Pair contributions
    for i in range(n):
        for j in range(i + 1, n):
            c_idx = n * n + i * n + j  # index in dist_matrix
            c_idx_rev = n * n + j * n + i
            diff = centers[i] - centers[j]
            dist = np.linalg.norm(diff)
            if dist > 1e-12:
                dc_dx_i = diff / (2 * dist)
                dc_dx_j = -diff / (2 * dist)
                grad[i * 2:(i+1) * 2] += dc_dx_i * exp_vals[c_idx] / sum_exp
                grad[j * 2:(j+1) * 2] += dc_dx_j * exp_vals[c_idx_rev] / sum_exp
    
    return grad


def _generate_hex_layout(n_rows, d_scale=1.0):
    """Generate a hexagonal layout."""
    n_cols = math.ceil(N / n_rows)
    d = d_scale / (n_cols + 0.5)
    row_spacing = d * math.sqrt(3) / 2
    
    pts = []
    for row in range(n_rows):
        y = -0.5 + d / 2 + row * row_spacing
        if row % 2 == 0:
            n_in_row = n_cols
            x_start = -0.5 + d / 2
        else:
            n_in_row = n_cols - 1
            x_start = -0.5 + d
        
        for col in range(n_in_row):
            x = x_start + col * d
            pts.append([x, y])
    
    pts = np.array(pts)
    if len(pts) >= N:
        pts = pts[:N]
    else:
        return None
    
    # Clip to square
    pts = np.clip(pts, -0.5, 0.5)
    return pts


def _generate_grid_layout(m=None):
    """Generate a square grid layout."""
    if m is None:
        m = int(math.ceil(math.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    if len(grid) >= N:
        grid = grid[:N]
    return grid


def _force_directed_step(centers, steps=200, learning_rate=0.003):
    """Force-directed relaxation to spread points."""
    centers = centers.copy()
    n = len(centers)
    
    for step in range(steps):
        forces = np.zeros_like(centers)
        
        # Boundary forces (push away from walls)
        for i in range(n):
            for dim in range(2):
                if centers[i, dim] < -0.45:
                    forces[i, dim] += 0.05
                elif centers[i, dim] > 0.45:
                    forces[i, dim] -= 0.05
        
        # Pair repulsive forces
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=-1)
        np.fill_diagonal(dists, 1.0)
        
        cutoff = 0.15
        mask = dists < cutoff
        np.fill_diagonal(mask, False)
        
        w = np.zeros_like(dists)
        w[mask] = 1.0 / (dists[mask] ** 2)
        
        forces += np.einsum('ij,ijd->id', w, diff)
        
        # Centering force
        forces -= 0.01 * centers
        
        centers += learning_rate * forces
        centers = np.clip(centers, -0.5, 0.5)
    
    return centers


def _gradient_optimize(centers, max_iter=200):
    """Gradient-based optimization using L-BFGS-B on soft-min."""
    x0 = centers.flatten()
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    best_x = x0.copy()
    best_r = _compute_radius(centers)
    
    for k in [3.0, 5.0, 10.0, 20.0]:
        res = minimize(
            _soft_min_obj,
            x0,
            jac=_soft_min_grad,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50},
            args=(k,)
        )
        
        x_opt = np.clip(res.x, -0.5, 0.5)
        r_opt = _compute_radius(x_opt.reshape(-1, 2))
        
        if r_opt > best_r:
            best_r = r_opt
            best_x = x_opt.copy()
            x0 = x_opt
    
    return best_x.reshape(-1, 2), best_r


def _simulated_annealing(centers, start_temp=0.002, end_temp=0.0001, max_iter=500):
    """Simulated annealing for local refinement."""
    centers = centers.copy()
    current_r = _compute_radius(centers)
    best_centers = centers.copy()
    best_r = current_r
    
    for iteration in range(max_iter):
        temp = start_temp * (end_temp / start_temp) ** (iteration / max_iter)
        
        i = np.random.randint(N)
        orig = centers[i].copy()
        
        perturbation = np.random.randn(2) * temp
        new_center = orig + perturbation
        
        if new_center[0] < -0.5 or new_center[0] > 0.5:
            continue
        if new_center[1] < -0.5 or new_center[1] > 0.5:
            continue
        
        centers[i] = new_center
        new_r = _compute_radius(centers)
        
        if new_r > current_r:
            current_r = new_r
            if new_r > best_r:
                best_r = new_r
                best_centers = centers.copy()
        else:
            delta = new_r - current_r
            if np.random.rand() < math.exp(delta / (temp + 1e-10)):
                current_r = new_r
            else:
                centers[i] = orig
    
    return best_centers, best_r


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill
    
    best_centers = None
    best_radius = 0.0
    
    # Multi-start with diverse initializations
    initializations = []
    
    # Hexagonal layouts
    for n_rows in [12, 13, 14, 15, 16]:
        for d_scale in [0.9, 0.95, 1.0, 1.05, 1.1]:
            pts = _generate_hex_layout(n_rows, d_scale)
            if pts is not None:
                initializations.append(pts)
    
    # Square grid
    initializations.append(_generate_grid_layout())
    
    # Random initialization
    np.random.seed(42)
    initializations.append(np.random.uniform(-0.5, 0.5, size=(N, 2)))
    
    # Try each initialization
    for init in initializations:
        if time.time() > deadline - 20:
            break
        
        centers = init.copy()
        r = _compute_radius(centers)
        
        if r > best_radius:
            best_radius = r
            best_centers = centers.copy()
        
        # Force-directed relaxation
        if time.time() < deadline - 15:
            centers = _force_directed_step(centers, steps=100, learning_rate=0.002)
            r = _compute_radius(centers)
            if r > best_radius:
                best_radius = r
                best_centers = centers.copy()
        
        # Gradient optimization
        if time.time() < deadline - 10:
            centers_grad, r_grad = _gradient_optimize(centers, max_iter=100)
            if r_grad > best_radius:
                best_radius = r_grad
                best_centers = centers_grad.copy()
        
        # Simulated annealing polish
        if time.time() < deadline - 5:
            centers_sa, r_sa = _simulated_annealing(best_centers if best_centers is not None else centers, 
                                                      start_temp=0.001, end_temp=0.00005, max_iter=200)
            if r_sa > best_radius:
                best_radius = r_sa
                best_centers = centers_sa.copy()
    
    # Final gradient polish on best result
    if best_centers is not None and time.time() < deadline - 3:
        centers_final, r_final = _gradient_optimize(best_centers, max_iter=100)
        if r_final > best_radius:
            best_radius = r_final
            best_centers = centers_final.copy()
    
    # Fallback
    if best_centers is None:
        best_centers = _generate_grid_layout()
    
    return best_centers
