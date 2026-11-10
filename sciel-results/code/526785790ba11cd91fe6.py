"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time


def _score(points):
    """Compute the common radius for a set of circle centers."""
    # Distance from center to origin
    dist_to_origin = np.linalg.norm(points, axis=1)
    # Minimum distance to boundary
    r_boundary = 1.0 - np.min(dist_to_origin)
    
    # Pairwise distances
    n = len(points)
    # Compute all pairwise distances efficiently
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    # Get minimum pairwise distance (excluding self-distances on diagonal)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    r_pair = min_pair_dist / 2.0
    
    return min(r_boundary, r_pair)


def _hexagonal_initial(n):
    """Generate initial hexagonal lattice configuration."""
    # Start with a reasonable spacing
    spacing = 2.0 / (np.sqrt(n) + 1.0)
    
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        mask = radius <= 1.0 - spacing / 2.0
        points = points[mask]
        if len(points) >= n:
            # Sort by distance from center and take first n
            dist = np.hypot(points[:, 0], points[:, 1])
            idx = np.argsort(dist, kind='stable')[:n]
            return points[idx]
        spacing *= 0.95


def construct():
    deadline = time.time() + 140.0  # Safety margin before 150s limit
    n = N
    
    # Start from hexagonal lattice
    points = _hexagonal_initial(n)
    current_score = _score(points)
    best_points = points.copy()
    best_score = current_score
    
    # Simulated annealing parameters
    T = 0.05  # Initial temperature
    T_min = 0.0001
    cooling = 0.9995
    max_iter = 100000
    
    rng = np.random.default_rng(42)
    
    for iteration in range(max_iter):
        if time.time() > deadline - 10.0:
            break
        
        # Generate a perturbation
        idx = rng.integers(0, n)
        delta_r = rng.normal(0, T * 0.1)
        delta_theta = rng.uniform(-np.pi, np.pi)
        
        # Move one circle
        new_points = points.copy()
        x, y = new_points[idx]
        r = np.hypot(x, y)
        theta = np.arctan2(y, x)
        
        new_r = max(0, r + delta_r)
        new_theta = theta + delta_theta * T
        
        new_points[idx, 0] = new_r * np.cos(new_theta)
        new_points[idx, 1] = new_r * np.sin(new_theta)
        
        # Check if still inside unit circle
        if np.linalg.norm(new_points[idx]) > 1.0:
            new_points[idx] = points[idx]
            new_score = current_score
        else:
            new_score = _score(new_points)
        
        # Accept or reject
        delta = new_score - current_score
        if delta > 0 or rng.random() < np.exp(delta / T):
            points = new_points
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_points = points.copy()
        
        T = max(T_min, T * cooling)
    
    return best_points
# EVOLVE-BLOCK-END
