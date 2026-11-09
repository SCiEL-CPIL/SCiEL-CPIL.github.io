"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import sys


def _compute_radius(centers):
    """Compute the maximum common radius for given centers."""
    # Distance to walls
    wall_dists = np.min(np.minimum(
        0.5 - np.abs(centers[:, 0]),
        0.5 - np.abs(centers[:, 1])
    ), axis=1)
    min_wall = np.min(wall_dists)
    
    # Pairwise distances
    if len(centers) > 1:
        dists = cdist(centers, centers)
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
    else:
        min_pair = np.inf
    
    return min(min_wall, min_pair)


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    np.random.seed(42)
    
    # Start with a good initial configuration
    # Use a hexagonal-like grid as starting point
    m = int(np.ceil(np.sqrt(N)))
    # Create a grid and then perturb
    base_positions = []
    for i in range(m):
        for j in range(m):
            if len(base_positions) < N:
                x = (i + 0.5) / m - 0.5
                y = (j + 0.5) / m - 0.5
                base_positions.append([x, y])
    
    centers = np.array(base_positions[:N], dtype=np.float64)
    
    # Add small random perturbations
    centers += np.random.uniform(-0.01, 0.01, size=centers.shape)
    # Clamp to valid region
    centers[:, 0] = np.clip(centers[:, 0], -0.49, 0.49)
    centers[:, 1] = np.clip(centers[:, 1], -0.49, 0.49)
    
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    # Define the objective function to maximize (negate for minimization)
    def objective(x):
        c = x.reshape(-1, 2)
        # Check validity
        if np.any(np.abs(c) > 0.5 + 1e-10):
            return 1e10
        r = _compute_radius(c)
        return -r
    
    # Use SLSQP with active-set style constraints
    # We'll use a multi-start approach with different initial conditions
    n_starts = 8
    attempts = 0
    
    for start_idx in range(n_starts):
        if time.time() > deadline:
            break
        
        # Different initial configurations
        if start_idx == 0:
            # Use the current best
            x0 = best_centers.flatten()
        elif start_idx == 1:
            # Hexagonal lattice
            hex_positions = []
            spacing = 1.0 / 8.0
            for row in range(9):
                for col in range(8):
                    if len(hex_positions) < N:
                        x = col * spacing - 0.5 + spacing/2
                        y = row * spacing - 0.5 + spacing/2
                        if row % 2 == 1:
                            x += spacing/2
                        if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                            hex_positions.append([x, y])
            if len(hex_positions) >= N:
                x0 = np.array(hex_positions[:N]).flatten()
                x0 += np.random.uniform(-0.02, 0.02, size=x0.shape)
                x0 = np.clip(x0, -0.49, 0.49)
            else:
                continue
        elif start_idx == 2:
            # Random initialization
            x0 = np.random.uniform(-0.45, 0.45, size=(N, 2)).flatten()
        else:
            # Perturbed best
            x0 = best_centers.flatten() + np.random.uniform(-0.05, 0.05, size=(N, 2)).flatten()
            x0 = np.clip(x0, -0.49, 0.49)
        
        attempts += 1
        
        # Run SLSQP optimization
        try:
            result = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=[(-0.5, 0.5)] * (2 * N),
                options={'maxiter': 200, 'ftol': 1e-12}
            )
            
            if result.success or result.fun < objective(x0):
                new_centers = result.x.reshape(-1, 2)
                # Clamp to ensure validity
                new_centers[:, 0] = np.clip(new_centers[:, 0], -0.5, 0.5)
                new_centers[:, 1] = np.clip(new_centers[:, 1], -0.5, 0.5)
                
                new_radius = _compute_radius(new_centers)
                if new_radius > best_radius:
                    best_radius = new_radius
                    best_centers = new_centers.copy()
        except Exception:
            pass
    
    # Local refinement: try to increase the minimum distance
    # Identify the critical pair (minimum distance) and try to separate them
    for refine_iter in range(50):
        if time.time() > deadline:
            break
        
        c = best_centers.copy()
        dists = cdist(c, c)
        np.fill_diagonal(dists, np.inf)
        min_idx = np.unravel_index(np.argmin(dists), dists.shape)
        i, j = min_idx
        min_dist = dists[i, j]
        
        # Try to move i and j apart
        direction = c[i] - c[j]
        norm = np.linalg.norm(direction)
        if norm < 1e-10:
            direction = np.array([1.0, 0.0])
            norm = 1.0
        direction /= norm
        
        # Move i in direction, j in opposite direction
        step = 0.001
        c[i] += direction * step
        c[j] -= direction * step
        
        # Clamp
        c[:, 0] = np.clip(c[:, 0], -0.5, 0.5)
        c[:, 1] = np.clip(c[:, 1], -0.5, 0.5)
        
        new_radius = _compute_radius(c)
        if new_radius > best_radius:
            best_radius = new_radius
            best_centers = c.copy()
        else:
            # Try smaller step
            c = best_centers.copy()
            step = 0.0001
            c[i] += direction * step
            c[j] -= direction * step
            c[:, 0] = np.clip(c[:, 0], -0.5, 0.5)
            c[:, 1] = np.clip(c[:, 1], -0.5, 0.5)
            new_radius = _compute_radius(c)
            if new_radius > best_radius:
                best_radius = new_radius
                best_centers = c.copy()
    
    # Final check: ensure all centers are valid
    best_centers[:, 0] = np.clip(best_centers[:, 0], -0.5, 0.5)
    best_centers[:, 1] = np.clip(best_centers[:, 1], -0.5, 0.5)
    
    return best_centers
# EVOLVE-BLOCK-END
