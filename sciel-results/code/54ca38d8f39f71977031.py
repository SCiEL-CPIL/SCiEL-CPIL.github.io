"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import math


def compute_radius(centers):
    """
    Compute the maximum radius for a given set of centers.
    r = min( min_{i<j} ||c_i - c_j|| / 2, min_{i,k} (0.5 - |c_ik|) )
    """
    # Calculate pairwise distances
    # centers shape: (N, 3)
    # diff shape: (N, N, 3)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    
    # Set diagonal to infinity so it doesn't affect min
    np.fill_diagonal(dists, np.inf)
    
    min_pair_dist = np.min(dists)
    
    # Calculate distance to boundaries
    # For each center i and coordinate k, distance to boundary is 0.5 - |c_ik|
    abs_centers = np.abs(centers)
    dists_to_boundary = 0.5 - abs_centers
    min_boundary_dist = np.min(dists_to_boundary)
    
    # The radius is half the minimum of the two constraints
    r = min(min_pair_dist, min_boundary_dist) / 2.0
    
    return r


def construct():
    """
    Use Vectorized Simulated Annealing to maximize the packing radius.
    """
    rng = np.random.default_rng(42)
    
    # Initial configuration: A grid-based start is usually better than pure random
    # 50 spheres. 4x4x4 = 64 slots. We take 50.
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid_points = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    
    # Randomly select 50 points from the grid to start
    indices = rng.choice(len(grid_points), size=N, replace=False)
    current_centers = grid_points[indices].copy()
    
    # Add small noise to break symmetry
    current_centers += rng.normal(0, 0.01, size=current_centers.shape)
    current_centers = np.clip(current_centers, -0.49, 0.49)
    
    current_radius = compute_radius(current_centers)
    best_centers = current_centers.copy()
    best_radius = current_radius
    
    # Simulated Annealing Parameters
    n_iterations = 8000
    t_start = 0.05
    t_end = 1e-6
    alpha = (t_end / t_start) ** (1.0 / n_iterations)
    
    temperature = t_start
    
    for i in range(n_iterations):
        # Generate perturbation
        # Step size should scale with temperature
        step_size = temperature * 0.5
        perturbations = rng.normal(0, step_size, size=(N, 3))
        
        new_centers = current_centers + perturbations
        # Keep within bounds (soft constraint or hard clip? Hard clip is safer for validity)
        new_centers = np.clip(new_centers, -0.49, 0.49)
        
        new_radius = compute_radius(new_centers)
        
        # Acceptance criterion
        if new_radius > current_radius:
            current_centers = new_centers
            current_radius = new_radius
            if current_radius > best_radius:
                best_radius = current_radius
                best_centers = current_centers.copy()
        else:
            # Metropolis acceptance
            delta = current_radius - new_radius
            if delta < 0:
                accept_prob = 1.0
            else:
                # Avoid division by zero or overflow
                if temperature < 1e-10:
                    accept_prob = 0.0
                else:
                    accept_prob = math.exp(-delta / temperature)
            
            if rng.random() < accept_prob:
                current_centers = new_centers
                current_radius = new_radius
        
        # Cool down
        temperature *= alpha
        
        # If temperature is very low, we can stop early or continue with very small steps
        if temperature < 1e-8:
            break

    return best_centers
# EVOLVE-BLOCK-END
