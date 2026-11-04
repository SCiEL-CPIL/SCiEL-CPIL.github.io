"""Rewritten program for 50 equal spheres in a unit cube.
Uses a multi-start local search with a force-based repulsion model to maximize the minimum distance.
"""
import numpy as np
import time

N = 50

# EVOLVE-BLOCK-START
def _compute_radius(centers):
    """Compute the valid packing radius for given centers.
    Returns -inf if invalid (out of bounds, coincident, non-finite).
    """
    if centers.shape != (N, 3):
        return -np.inf
    if not np.all(np.isfinite(centers)):
        return -np.inf
    # Check bounds: centers must be in [-0.5, 0.5]^3
    if np.any(centers < -0.5) or np.any(centers > 0.5):
        return -np.inf
    
    # Distance to walls
    wall_dists = np.min(np.stack([
        centers + 0.5,
        0.5 - centers
    ], axis=1), axis=1)  # shape (N, 3) -> min over 3 coords -> (N,)
    min_wall = np.min(wall_dists)
    
    if min_wall < 0:
        return -np.inf
    
    # Pairwise distances
    # Use broadcasting for pairwise distances
    # diff[i,j] = centers[i] - centers[j]
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    if min_pair < 0:
        return -np.inf
    
    return min(min_wall, min_pair / 2.0)


def _local_search(centers, time_budget=10.0):
    """
    Perform a force-directed local search to maximize the minimum distance.
    Uses a soft repulsive force that grows as spheres get closer.
    """
    centers = centers.copy()
    start_time = time.time()
    
    # Initial parameters
    step_size = 0.01
    decay = 0.999
    min_step = 1e-6
    
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    for iteration in range(20000):
        if time.time() - start_time > time_budget:
            break
        if step_size < min_step:
            break
            
        # Compute forces
        # Force is repulsive, stronger when close
        # We use a gradient of the objective: maximize min(dist/2, wall_dist)
        # Approximate by summing repulsive forces from all other spheres and walls
        
        forces = np.zeros_like(centers)
        
        # Pairwise repulsion
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Use a kernel that emphasizes close pairs
        # Force magnitude ~ 1/d^2 for close pairs, but capped
        eps = 1e-8
        inv_d2 = 1.0 / (dists * dists + eps)
        # Mask diagonal
        mask = ~np.eye(N, dtype=bool)
        inv_d2[~mask] = 0.0
        
        # Force direction: diff / dist
        # F_i = sum_j (diff_ij / dist_ij^3) * strength
        # To avoid singularities, we use a softened version
        force_scale = 1.0
        forces += np.sum(diff * inv_d2[:, :, np.newaxis] * force_scale, axis=1)
        
        # Wall forces: push away from walls
        # For each coordinate, if close to wall, push inward
        for k in range(3):
            # Distance to left wall: centers[:, k] + 0.5
            # Distance to right wall: 0.5 - centers[:, k]
            d_left = centers[:, k] + 0.5
            d_right = 0.5 - centers[:, k]
            
            # Force from left wall (push right if close)
            # If d_left < some threshold, push +x
            # Use a quadratic penalty gradient
            wall_strength = 1.0
            forces[:, k] += wall_strength * (1.0 / (d_left + eps) - 1.0 / (d_right + eps))
            # Actually, let's use a simpler approach: 
            # Force = -d/dx (1/d_left^2 + 1/d_right^2) 
            # d/dx (1/d_left^2) = -2/d_left^3
            # d/dx (1/d_right^2) = 2/d_right^3
            # So force = 2/d_left^3 - 2/d_right^3
            # Let's recompute properly
            
        # Recompute wall forces more carefully
        forces = np.zeros_like(centers)
        
        # Pairwise
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists_sq = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dists_sq, np.inf)
        dists = np.sqrt(dists_sq)
        
        # Soft repulsion: F ~ diff / dist^3
        # This gives F ~ 1/dist^2 magnitude
        eps = 1e-10
        inv_d3 = 1.0 / (dists_sq * np.sqrt(dists_sq + eps) + eps)
        np.fill_diagonal(inv_d3, 0.0)
        
        forces += np.sum(diff * inv_d3[:, :, np.newaxis], axis=1)
        
        # Wall forces: minimize sum of 1/d^2 for each wall
        # Gradient of 1/(x+0.5)^2 wrt x is -2/(x+0.5)^3
        # Gradient of 1/(0.5-x)^2 wrt x is 2/(0.5-x)^3
        # Total force in +x direction: 2/(0.5-x)^3 - 2/(x+0.5)^3
        for k in range(3):
            d_left = centers[:, k] + 0.5
            d_right = 0.5 - centers[:, k]
            # Avoid division by zero
            d_left = np.maximum(d_left, 1e-6)
            d_right = np.maximum(d_right, 1e-6)
            forces[:, k] += 2.0 / (d_right ** 3) - 2.0 / (d_left ** 3)
        
        # Normalize forces to prevent large steps
        force_norms = np.linalg.norm(forces, axis=1)
        max_norm = np.max(force_norms)
        if max_norm > 0:
            forces = forces / max_norm
        
        # Move centers
        centers += step_size * forces
        
        # Project back to cube (hard constraint)
        centers = np.clip(centers, -0.5 + 1e-9, 0.5 - 1e-9)
        
        # Check if improved
        current_radius = _compute_radius(centers)
        if current_radius > best_radius:
            best_radius = current_radius
            best_centers = centers.copy()
        
        # Decay step size
        step_size *= decay
    
    return best_centers


def construct():
    start_time = time.time()
    total_budget = 120.0  # Seconds
    safety_margin = 5.0
    effective_budget = total_budget - safety_margin
    
    # Start with a good initial configuration
    # Use a grid-based initialization with some randomness
    
    best_centers = None
    best_radius = -np.inf
    
    # Try multiple random restarts
    np.random.seed(42)
    
    # First, try a structured initialization
    # 50 spheres: try a 4x4x4 grid (64 points) and take 50
    # Or a better structured packing
    
    # Structured: 5 layers of 10 spheres each
    # Layer 1-5, each with 10 spheres in a 5x2 grid
    # This is a simple structured approach
    
    # Let's use a multi-start approach with different initializations
    
    initializations = []
    
    # Initialization 1: Grid
    m = 4  # 4x4x4 = 64 points
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    initializations.append(grid[:N])
    
    # Initialization 2: Random
    for _ in range(5):
        rand_init = np.random.uniform(-0.4, 0.4, size=(N, 3))
        initializations.append(rand_init)
    
    # Initialization 3: Layered
    # 5 layers, 10 per layer
    layer_y = np.linspace(-0.4, 0.4, 5)
    for y in layer_y:
        # 10 spheres in a row: 5x2 grid
        xs = np.linspace(-0.4, 0.4, 5)
        zs = np.linspace(-0.3, 0.3, 2)
        for x in xs:
            for z in zs:
                initializations.append(np.array([[x, y, z]]))
    
    # Flatten the last initialization
    if len(initializations[-1]) == 1:
        # Combine all layered inits into one
        layered = np.vstack([init for init in initializations[7:]])
        initializations = initializations[:7] + [layered]
    
    # Perform local search on each initialization
    for i, init in enumerate(initializations):
        if time.time() - start_time > effective_budget:
            break
        
        # Ensure valid shape
        if init.shape != (N, 3):
            continue
        
        # Run local search with remaining time budget
        remaining = effective_budget - (time.time() - start_time)
        if remaining < 1.0:
            break
        
        # Use a fraction of remaining time for each search
        search_time = min(remaining * 0.8, 10.0)
        
        result = _local_search(init, time_budget=search_time)
        
        radius = _compute_radius(result)
        if radius > best_radius:
            best_radius = radius
            best_centers = result.copy()
    
    # If we haven't found a valid result, use the grid
    if best_centers is None or best_radius < 0:
        best_centers = initializations[0].copy()
    
    return best_centers
# EVOLVE-BLOCK-END
