"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_min_dist(centers):
    """
    Compute the minimum distance between any two sphere centers
    and the minimum distance from any center to the cube boundary.
    Returns the overall minimum distance (which is 2*r).
    """
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    # Mask diagonal
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Distance to boundaries
    dist_to_boundary = np.min(0.5 - np.abs(centers), axis=1)
    min_boundary_dist = np.min(dist_to_boundary)
    
    return min(min_pair_dist, min_boundary_dist)

def _repulsive_force_dynamics(centers, max_iter=20000, initial_step=0.01, final_step=0.0001):
    """
    Optimize sphere packing using repulsive force dynamics.
    Each sphere experiences a repulsive force from all others and an inward force from boundaries.
    """
    centers = centers.copy()
    best_centers = centers.copy()
    best_dist = _compute_min_dist(centers)
    n = centers.shape[0]
    
    for i in range(max_iter):
        # Adaptive step size
        progress = i / max_iter
        step = initial_step * (final_step / initial_step) ** progress
        
        # Calculate forces (Vectorized)
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        
        # Avoid division by zero
        dists_safe = np.where(dists > 1e-8, dists, np.inf)
        
        # Force magnitude: 1 / dist^2
        weights = np.zeros_like(dists)
        weights[dists_safe < np.inf] = 1.0 / (dists_safe[dists_safe < np.inf] ** 2)
        np.fill_diagonal(weights, 0)
        
        # forces[i] = sum_j (c_i - c_j) * w_ij
        forces = np.sum(weights[:, :, np.newaxis] * diff, axis=1)
        
        # Boundary forces
        # Push inward if close to boundary
        margin = 0.05
        for axis in range(3):
            low_mask = centers[:, axis] < (-0.5 + margin)
            if np.any(low_mask):
                forces[low_mask, axis] += ( (-0.5 + margin) - centers[low_mask, axis] ) * 10.0
                
            high_mask = centers[:, axis] > (0.5 - margin)
            if np.any(high_mask):
                forces[high_mask, axis] -= ( centers[high_mask, axis] - (0.5 - margin) ) * 10.0
        
        # Update positions
        new_centers = centers + forces * step
        new_centers = np.clip(new_centers, -0.5, 0.5)
        
        # Evaluate
        current_dist = _compute_min_dist(new_centers)
        
        if current_dist > best_dist:
            best_dist = current_dist
            best_centers = new_centers.copy()
            
        centers = new_centers
        
    return best_centers

def _grid_init():
    """
    Initialize with a grid-like structure.
    50 points.
    """
    x = np.linspace(-0.35, 0.35, 4)
    y = np.linspace(-0.35, 0.35, 4)
    z = np.linspace(-0.35, 0.35, 3)
    
    centers = np.array([[x[i], y[j], z[k]] for i in range(4) for j in range(4) for k in range(3)])
    centers = np.vstack([centers, [[0.0, 0.0, 0.0], [0.0, 0.0, 0.15]]])
    
    centers = np.clip(centers, -0.49, 0.49)
    return centers

def construct():
    best_centers = None
    best_radius = 0.0
    
    np.random.seed(42)
    
    start_time = time.time()
    time_limit = 110.0
    
    # 1. Grid initialization with Repulsive Force Dynamics
    centers_grid = _grid_init()
    optimized = _repulsive_force_dynamics(centers_grid, max_iter=15000, initial_step=0.02, final_step=0.0005)
    radius = _compute_min_dist(optimized) / 2.0
    if radius > best_radius:
        best_radius = radius
        best_centers = optimized.copy()
        
    if time.time() - start_time > time_limit:
        return best_centers

    # 2. Random initializations with Repulsive Force Dynamics
    n_random_starts = 20
    
    for _ in range(n_random_starts):
        if time.time() - start_time > time_limit:
            break
            
        centers_rand = np.random.uniform(-0.45, 0.45, (N, 3))
        
        optimized = _repulsive_force_dynamics(centers_rand, max_iter=5000, initial_step=0.01, final_step=0.0001)
        
        radius = _compute_min_dist(optimized) / 2.0
        if radius > best_radius:
            best_radius = radius
            best_centers = optimized.copy()
            
    return best_centers

# EVOLVE-BLOCK-END
