"""Structural change: Physics-based force-directed relaxation with simulated annealing."""
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


def _initial_grid(n_points):
    """Generate a grid of approximately n_points within [-0.5,0.5]^2."""
    # Estimate grid size
    m = int(np.ceil(np.sqrt(n_points)))
    # Adjust m to get closer to n_points
    while (m * (m + 1)) < n_points:
        m += 1
    
    # Create a grid of size m x (m+1) or similar
    # We'll generate a grid slightly larger than needed and trim
    cols = m
    rows = int(np.ceil(n_points / m))
    
    points = []
    # Center the grid
    x_coords = np.linspace(-0.45, 0.45, cols)
    y_coords = np.linspace(-0.45, 0.45, rows)
    
    for y in y_coords:
        for x in x_coords:
            points.append([x, y])
            
    points = np.array(points)
    
    # If we have too many, trim randomly or by distance from center
    if len(points) > n_points:
        # Keep points closest to center to maintain symmetry
        center_dist = np.sum(points ** 2, axis=1)
        idx = np.argsort(center_dist)[:n_points]
        points = points[idx]
        
    # If we have too few (shouldn't happen with ceil), pad with random
    if len(points) < n_points:
        extra = np.random.uniform(-0.45, 0.45, size=(n_points - len(points), 2))
        points = np.vstack([points, extra])
        
    return points[:n_points]


def _relax_md(centers, max_iters=2000, start_temp=0.01, end_temp=0.0001):
    """
    Molecular Dynamics relaxation.
    Forces:
    1. Repulsion between all pairs: F_rep = k_rep * (1/d^2) * unit_vec if d < 2*r_est
       We use a soft potential to avoid singularities.
    2. Attraction to center: F_attr = k_attr * pos
    3. Boundary repulsion: Push away from walls.
    
    We use a simulated annealing approach to accept moves that might temporarily
    decrease the radius if they help escape local minima.
    """
    centers = centers.copy()
    r_best = _compute_radius(centers)
    
    # Estimate a target spacing for force scaling
    # Area per circle ~ 1/200. Spacing ~ sqrt(1/200) ~ 0.07
    # Let's use a soft repulsion radius based on current min distance
    current_r = _compute_radius(centers)
    
    k_rep = 1.0
    k_attr = 0.01
    k_bound = 1.0
    
    for it in range(max_iters):
        if time.time() > _DEADLINE:
            break
            
        # Compute forces
        forces = np.zeros_like(centers)
        
        # Pairwise repulsion
        # Vectorized computation
        diff = centers[:, None, :] - centers[None, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, 1e9)
        
        # Only consider pairs that are "close" to save computation?
        # With N=200, N^2 is 40k, which is fast.
        # Force magnitude: k_rep / dist^2
        # Direction: diff / dist
        # F = sum_j k_rep * (c_i - c_j) / |c_i - c_j|^3
        
        # Avoid division by zero
        inv_d3 = 1.0 / (dists ** 3 + 1e-12)
        # Mask diagonal
        inv_d3[np.diag_indices_from(inv_d3)] = 0.0
        
        forces += np.sum(diff * inv_d3[:, :, np.newaxis], axis=1) * k_rep
        
        # Attraction to center
        forces -= centers * k_attr
        
        # Boundary forces
        # If x > 0.5 - r, push left. If x < -0.5 + r, push right.
        # We don't know r exactly, but we can use a margin.
        # Let's use a margin of 0.05 for safety.
        margin = 0.05
        boundary = 0.5 - margin
        
        # x boundaries
        forces[:, 0] += np.where(centers[:, 0] > boundary, -k_bound * (centers[:, 0] - boundary), 0)
        forces[:, 0] += np.where(centers[:, 0] < -boundary, -k_bound * (centers[:, 0] + boundary), 0)
        
        # y boundaries
        forces[:, 1] += np.where(centers[:, 1] > boundary, -k_bound * (centers[:, 1] - boundary), 0)
        forces[:, 1] += np.where(centers[:, 1] < -boundary, -k_bound * (centers[:, 1] + boundary), 0)
        
        # Simulated Annealing Step
        # Temperature decreases over time
        temp = start_temp * (end_temp / start_temp) ** (it / max_iters)
        
        # Propose new positions
        step_size = 0.01 * (1.0 + temp)
        new_centers = centers + forces * step_size + np.random.normal(0, temp * 0.01, size=centers.shape)
        
        # Clip to square (hard constraint)
        new_centers[:, 0] = np.clip(new_centers[:, 0], -0.5, 0.5)
        new_centers[:, 1] = np.clip(new_centers[:, 1], -0.5, 0.5)
        
        # Evaluate
        r_new = _compute_radius(new_centers)
        
        # Accept if better, or with probability exp((r_new - r_best) / temp)
        if r_new > r_best:
            centers = new_centers
            r_best = r_new
        else:
            delta = r_new - r_best
            if delta < 0:
                prob = np.exp(delta / (temp + 1e-12))
                if random.random() < prob:
                    centers = new_centers
            # If r_new == r_best, accept with 0.5 prob? No, just keep.
        
        # Optional: If we get stuck, do a small random kick
        if it % 500 == 0 and it > 0:
             # Check if we improved in last 500 steps
             # Simple heuristic: just kick
             kick = np.random.normal(0, 0.005, size=centers.shape)
             new_centers = centers + kick
             new_centers[:, 0] = np.clip(new_centers[:, 0], -0.5, 0.5)
             new_centers[:, 1] = np.clip(new_centers[:, 1], -0.5, 0.5)
             r_kick = _compute_radius(new_centers)
             if r_kick > r_best:
                 centers = new_centers
                 r_best = r_kick
                 
    return centers, r_best


def construct():
    start_time = time.time()
    
    # 1. Generate initial grid
    centers = _initial_grid(N)
    
    # 2. Initial MD Relaxation
    centers, r_best = _relax_md(centers, max_iters=3000)
    
    # 3. Multi-start: Try a few different initializations if time permits
    # The grid is deterministic, so we need to randomize the start for diversity
    # Or perturb the best result and re-relax
    
    # Let's try 3 more restarts with random perturbations of the best solution
    for i in range(3):
        if time.time() > _DEADLINE - 20:
            break
            
        # Perturb current best
        np.random.seed(42 + i)
        perturbed = centers + np.random.normal(0, 0.01, size=centers.shape)
        perturbed[:, 0] = np.clip(perturbed[:, 0], -0.49, 0.49)
        perturbed[:, 1] = np.clip(perturbed[:, 1], -0.49, 0.49)
        
        # Re-relax with lower starting temp to refine
        centers2, r2 = _relax_md(perturbed, max_iters=2000, start_temp=0.005, end_temp=0.00005)
        
        if r2 > r_best:
            centers = centers2
            r_best = r2
            
    # Final validation
    r_final = _compute_radius(centers)
    if r_final <= 0:
        # Fallback
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        centers = grid[:N]
        
    return centers
# EVOLVE-BLOCK-END
