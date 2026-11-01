"""Packing 50 equal circles in a unit circle using simulated annealing."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np


def _initial_positions(n, seed=42):
    """Generate initial positions using a hexagonal lattice pattern."""
    rng = np.random.default_rng(seed)
    
    # Hexagonal packing
    points = []
    spacing = 0.15
    for row in range(-15, 16):
        for col in range(-15, 16):
            x = col * spacing * 1.5
            y = row * spacing * np.sqrt(3) + (col % 2) * spacing * np.sqrt(3) / 2
            if x * x + y * y < 0.9:
                points.append([x, y])
    
    points = np.array(points)
    
    if len(points) < n:
        positions = rng.uniform(-0.9, 0.9, size=(n, 2))
        for i in range(n):
            while np.linalg.norm(positions[i]) > 0.9:
                positions[i] = rng.uniform(-0.9, 0.9, size=2)
        return positions
    
    dists = np.linalg.norm(points, axis=1)
    indices = np.argsort(dists)[:n]
    return points[indices].copy()


def _compute_radius(positions):
    """Compute the maximum radius for given positions using vectorized operations."""
    n = len(positions)
    if n == 0:
        return 0.0
    
    norms = np.linalg.norm(positions, axis=1)
    boundary_dist = 1.0 - norms
    min_boundary = np.min(boundary_dist)
    
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    mask = ~np.eye(n, dtype=bool)
    min_pair_dist = np.min(dists[mask])
    
    return min(min_boundary, min_pair_dist / 2.0)


def _simulated_annealing(positions, max_iter=20000, seed=42):
    """Improve packing using simulated annealing."""
    rng = np.random.default_rng(seed)
    n = len(positions)
    positions = positions.copy()
    
    current_radius = _compute_radius(positions)
    best_radius = current_radius
    best_positions = positions.copy()
    
    T = 0.05
    cooling = 0.9995
    
    for it in range(max_iter):
        idx = rng.integers(n)
        angle = rng.uniform(0, 2 * np.pi)
        step = T
        dx = step * np.cos(angle)
        dy = step * np.sin(angle)
        
        new_positions = positions.copy()
        new_positions[idx] += np.array([dx, dy])
        
        norm = np.linalg.norm(new_positions[idx])
        if norm > 0.99:
            new_positions[idx] = 0.99 * new_positions[idx] / norm
        
        new_radius = _compute_radius(new_positions)
        
        if new_radius > current_radius or rng.random() < np.exp((new_radius - current_radius) / T):
            positions = new_positions
            current_radius = new_radius
            if current_radius > best_radius:
                best_radius = current_radius
                best_positions = positions.copy()
        
        T *= cooling
        if T < 1e-8:
            break
            
    return best_positions, best_radius


def _gradient_descent(positions, max_iter=1000, seed=42):
    """Refine packing using local closest-pair push optimization."""
    n = len(positions)
    positions = positions.copy()
    
    for _ in range(max_iter):
        # Handle Boundary Violations (vectorized)
        norms = np.linalg.norm(positions, axis=1)
        mask = norms > 0.99
        if np.any(mask):
            positions[mask] = 0.99 * positions[mask] / norms[mask, np.newaxis]
            
        # Find closest pair
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        min_idx = np.unravel_index(np.argmin(dists), dists.shape)
        i, j = min_idx
        
        d_vec = positions[i] - positions[j]
        d_norm = np.linalg.norm(d_vec)
        
        if d_norm < 1e-10:
            d_vec = np.array([1.0, 0.0])
            d_norm = 1.0
            
        step = 0.005
        move = d_vec / d_norm * step
        
        new_i = positions[i] + move
        new_j = positions[j] - move
        
        if np.linalg.norm(new_i) < 1.0 and np.linalg.norm(new_j) < 1.0:
            positions[i] = new_i
            positions[j] = new_j
        else:
            if np.linalg.norm(new_i) < 1.0:
                positions[i] = new_i
            elif np.linalg.norm(new_j) < 1.0:
                positions[j] = new_j
            else:
                positions[i] = positions[i] * 0.999
                positions[j] = positions[j] * 0.999

    return positions, _compute_radius(positions)


def construct():
    """Construct centers of 50 equal circles inside the unit circle."""
    positions = _initial_positions(N, seed=42)
    
    # Run SA
    positions, radius = _simulated_annealing(positions, max_iter=20000, seed=42)
    
    # Run Local Optimization
    positions, radius = _gradient_descent(positions, max_iter=1000, seed=42)
    
    # Final Boundary Check
    norms = np.linalg.norm(positions, axis=1)
    mask = norms > 0.999
    if np.any(mask):
        positions[mask] = 0.999 * positions[mask] / norms[mask, np.newaxis]
        
    return positions
# EVOLVE-BLOCK-END
