"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import itertools
from scipy.optimize import minimize

N = 100

# EVOLVE-BLOCK-START
import numpy as np


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # Initial grid configuration
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    centers = grid[:N].copy()
    
    best_score = _score(centers)
    best_centers = centers.copy()
    
    # Force-directed relaxation
    alpha = 0.01
    beta = 0.001
    gamma = 0.05
    
    while time.time() < deadline:
        # Compute forces
        forces = np.zeros_like(centers)
        
        # Pairwise repulsion
        for i in range(N):
            for j in range(i + 1, N):
                diff = centers[i] - centers[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-10:
                    # Random nudge
                    forces[i] += np.random.randn(3) * 0.001
                    forces[j] -= np.random.randn(3) * 0.001
                else:
                    # Soft repulsion: stronger when closer
                    force_mag = alpha / (dist ** 2 + beta)
                    forces[i] += force_mag * diff / dist
                    forces[j] -= force_mag * diff / dist
        
        # Wall repulsion
        for i in range(N):
            for k in range(3):
                # Left wall
                dist_left = centers[i, k] + 0.5
                if dist_left < gamma:
                    forces[i, k] += alpha / (dist_left ** 2 + beta)
                # Right wall
                dist_right = 0.5 - centers[i, k]
                if dist_right < gamma:
                    forces[i, k] -= alpha / (dist_right ** 2 + beta)
        
        # Update positions with damping
        centers += 0.1 * forces
        
        # Clamp to cube
        centers = np.clip(centers, -0.4999, 0.4999)
        
        # Check if improved
        current_score = _score(centers)
        if current_score > best_score:
            best_score = current_score
            best_centers = centers.copy()
            alpha *= 0.999  # Slowly decrease repulsion strength
        else:
            # Revert if worse
            centers = best_centers.copy()
            alpha *= 0.99
    
    return best_centers


def _score(centers):
    """Compute the minimum radius (minimum clearance)."""
    # Minimum pairwise distance / 2
    min_pair = np.inf
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(centers[i] - centers[j])
            if d < min_pair:
                min_pair = d
    min_pair /= 2.0
    
    # Minimum distance to walls
    min_wall = np.inf
    for i in range(N):
        for k in range(3):
            d = 0.5 - abs(centers[i, k])
            if d < min_wall:
                min_wall = d
    
    return min(min_pair, min_wall)
# EVOLVE-BLOCK-END
