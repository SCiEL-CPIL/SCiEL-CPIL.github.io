"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
import sys
from scipy.optimize import minimize

# Deadline guard: 145s safety margin
START_TIME = time.time()
DEADLINE = START_TIME + 145.0


def _compute_radius(centers):
    """Compute the packing radius for a set of centers."""
    # Minimum distance between any pair divided by 2
    # Use chunked computation to avoid memory issues
    n = len(centers)
    min_dist = float('inf')
    chunk_size = 100000
    
    for i in range(0, n, chunk_size):
        for j in range(i + chunk_size, n, chunk_size):
            block = centers[i:i+chunk_size]
            other = centers[j:j+chunk_size]
            # Compute pairwise distances
            diff = block[:, np.newaxis, :] - other[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            min_dist = min(min_dist, np.min(dists))
    
    # Also check within chunks
    for i in range(0, n, chunk_size):
        block = centers[i:i+chunk_size]
        diff = block[:, np.newaxis, :] - block[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        # Set diagonal to inf
        np.fill_diagonal(dists, float('inf'))
        min_dist = min(min_dist, np.min(dists))
    
    # Minimum distance to boundary
    norms = np.sqrt(np.sum(centers**2, axis=1))
    min_boundary = np.min(1.0 - norms)
    
    return min(min_dist / 2.0, min_boundary)


def _objective(x, centers_init=None):
    """Objective function to minimize (negative radius)."""
    centers = x.reshape(-1, 2)
    
    # Check for invalid configurations
    if not np.all(np.isfinite(centers)):
        return 1e10
    
    # Compute current radius
    r = _compute_radius(centers)
    
    # If any constraint is violated, return large penalty
    if r <= 0:
        return 1e10
    
    # Return negative radius (we minimize, so maximize r)
    return -r


def construct():
    global START_TIME, DEADLINE
    
    # Generate initial hexagonal lattice packing
    centers = _hex_initial(N)
    
    # Refine with optimization
    best_centers = centers.copy()
    best_radius = _compute_radius(centers)
    
    # Time-boxed optimization
    while time.time() < DEADLINE - 10:
        # Try a few random perturbations
        improved = False
        for _ in range(3):
            if time.time() >= DEADLINE - 10:
                break
            
            # Random perturbation
            perturbation = np.random.normal(0, 0.001, size=(N, 2))
            trial = best_centers + perturbation
            
            # Check validity
            if not np.all(np.isfinite(trial)):
                continue
            
            trial_radius = _compute_radius(trial)
            
            if trial_radius > best_radius:
                best_centers = trial.copy()
                best_radius = trial_radius
                improved = True
        
        # If no improvement, try larger perturbation
        if not improved:
            perturbation = np.random.normal(0, 0.005, size=(N, 2))
            trial = best_centers + perturbation
            if np.all(np.isfinite(trial)):
                trial_radius = _compute_radius(trial)
                if trial_radius > best_radius:
                    best_centers = trial.copy()
                    best_radius = trial_radius
    
    return best_centers


def _hex_initial(n):
    """Generate initial hexagonal lattice packing for n points."""
    points = []
    spacing = 0.05  # Initial spacing estimate
    
    # Generate hexagonal lattice points
    k = int(2.0 / spacing) + 5
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            if np.hypot(x, y) <= 1.0 - spacing / 2.0:
                points.append((x, y))
    
    points = np.array(points)
    
    # If we have too many, subsample; if too few, add more
    if len(points) > n:
        # Sort by distance from origin and take closest n
        dists = np.sqrt(np.sum(points**2, axis=1))
        idx = np.argsort(dists)[:n]
        points = points[idx]
    elif len(points) < n:
        # Add random points inside the circle
        while len(points) < n:
            theta = np.random.uniform(0, 2 * np.pi)
            r = np.random.uniform(0, 1.0 - spacing / 2.0)
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            points = np.vstack([points, [[x, y]]])
    
    return points[:n]
# EVOLVE-BLOCK-END

