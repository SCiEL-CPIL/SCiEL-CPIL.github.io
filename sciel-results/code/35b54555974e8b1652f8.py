"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import random
import math


def construct():
    """
    Use a multi-start local optimization approach to maximize the minimum distance
    between sphere centers and from boundaries.
    
    The objective is to maximize r = min(min_{i<j} |c_i - c_j| / 2, min_i min_k (1/2 - |c_ik|))
    We can reformulate this as minimizing a penalty function that encourages:
    1. All pairwise distances >= 2r
    2. All coordinates within [-1/2, 1/2]
    
    We use a simple approach: start with a random or grid-based initial configuration,
    then perform iterative repulsion-based optimization.
    """
    # Set random seed for reproducibility
    rng = np.random.default_rng(42)
    
    # Start with a better initial guess: a perturbed grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    centers = grid[:N].copy()
    
    # Add small random perturbations to break symmetry
    centers += rng.normal(0, 0.01, size=centers.shape)
    
    # Clip to stay within bounds
    centers = np.clip(centers, -0.49, 0.49)
    
    # Iterative repulsion-based optimization
    # We'll use a simple gradient-free approach with random restarts
    
    best_centers = centers.copy()
    best_score = -np.inf
    
    def compute_score(centers):
        """Compute the radius for a given set of centers."""
        n = len(centers)
        min_dist = np.inf
        # Pairwise distances
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(centers[i] - centers[j])
                if d < min_dist:
                    min_dist = d
        # Distance to boundaries
        for i in range(n):
            for k in range(3):
                d = 0.5 - abs(centers[i, k])
                if d < min_dist:
                    min_dist = d
        return min_dist / 2.0
    
    # Use a simple local search with simulated annealing-like approach
    current_centers = centers.copy()
    current_score = compute_score(current_centers)
    
    # If initial grid is good, keep it
    if current_score > best_score:
        best_score = current_score
        best_centers = current_centers.copy()
    
    # Try a few different initial configurations and optimize each
    for trial in range(5):
        # Create a new initial configuration
        if trial == 0:
            # Use the grid-based start
            init_centers = centers.copy()
        else:
            # Use random starts with some structure
            init_centers = rng.uniform(-0.45, 0.45, size=(N, 3))
        
        # Perform local optimization using a simple repulsion method
        for iteration in range(200):
            # Compute forces on each center
            forces = np.zeros_like(init_centers)
            for i in range(N):
                for j in range(i + 1, N):
                    diff = init_centers[i] - init_centers[j]
                    dist = np.linalg.norm(diff)
                    if dist > 1e-10:
                        # Repulsive force, stronger when closer
                        force_mag = 1.0 / (dist * dist + 1e-6)
                        forces[i] += force_mag * diff / dist
                        forces[j] -= force_mag * diff / dist
                # Boundary forces: push back if too close to boundary
                for k in range(3):
                    if init_centers[i, k] > 0.45:
                        forces[i, k] -= (init_centers[i, k] - 0.45) * 10
                    elif init_centers[i, k] < -0.45:
                        forces[i, k] -= (init_centers[i, k] + 0.45) * 10
            
            # Update centers with a small step
            step_size = 0.01 / (1 + iteration * 0.01)
            init_centers += step_size * forces
            init_centers = np.clip(init_centers, -0.49, 0.49)
        
        # Evaluate the result
        trial_score = compute_score(init_centers)
        if trial_score > best_score:
            best_score = trial_score
            best_centers = init_centers.copy()
    
    # Final refinement: try to improve the best solution found
    # Use a more careful local optimization on the best solution
    final_centers = best_centers.copy()
    for iteration in range(100):
        forces = np.zeros_like(final_centers)
        for i in range(N):
            for j in range(i + 1, N):
                diff = final_centers[i] - final_centers[j]
                dist = np.linalg.norm(diff)
                if dist > 1e-10:
                    force_mag = 1.0 / (dist * dist + 1e-6)
                    forces[i] += force_mag * diff / dist
                    forces[j] -= force_mag * diff / dist
            for k in range(3):
                if final_centers[i, k] > 0.48:
                    forces[i, k] -= (final_centers[i, k] - 0.48) * 50
                elif final_centers[i, k] < -0.48:
                    forces[i, k] -= (final_centers[i, k] + 0.48) * 50
        
        step_size = 0.005 / (1 + iteration * 0.02)
        final_centers += step_size * forces
        final_centers = np.clip(final_centers, -0.49, 0.49)
    
    final_score = compute_score(final_centers)
    if final_score > best_score:
        best_centers = final_centers.copy()
    
    return best_centers
# EVOLVE-BLOCK-END
