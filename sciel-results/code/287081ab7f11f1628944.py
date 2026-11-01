"""Packing 50 equal circles in a unit circle using multistart local search with closest-pair repulsion."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np


def _compute_radius(positions):
    """Compute the maximum radius for given positions."""
    n = len(positions)
    if n == 0:
        return 0.0
    
    # Distance to boundary
    norms = np.linalg.norm(positions, axis=1)
    boundary_dist = 1.0 - norms
    
    # Minimum pairwise distance
    min_pair_dist = np.inf
    # Use vectorized computation for speed
    # Create distance matrix
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    # Mask diagonal
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    return min(np.min(boundary_dist), min_pair_dist / 2.0)


def _local_search(positions, max_iter=5000, seed=42):
    """
    Local search that iteratively pushes the closest pair of circles apart.
    """
    rng = np.random.default_rng(seed)
    n = len(positions)
    positions = positions.copy()
    
    best_radius = _compute_radius(positions)
    best_positions = positions.copy()
    
    # Pre-allocate arrays for efficiency if needed, but N is small
    # We will recompute distances each step to find the min pair
    
    step_size = 0.01
    for it in range(max_iter):
        # Compute distances
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Find index of minimum distance
        min_idx = np.unravel_index(np.argmin(dists), dists.shape)
        i, j = min_idx
        d_min = dists[i, j]
        
        # Check boundary violations
        norms = np.linalg.norm(positions, axis=1)
        max_norm = np.max(norms)
        
        # If the current configuration is valid and better, update best
        current_radius = _compute_radius(positions)
        if current_radius > best_radius:
            best_radius = current_radius
            best_positions = positions.copy()
        
        # Determine the "bottleneck"
        # If boundary is the limiting factor, push the most outward circle in.
        # If pair distance is the limiting factor, push the closest pair apart.
        
        # Simple heuristic: always push the closest pair apart, but also handle boundary.
        
        # Vector direction
        v = positions[i] - positions[j]
        norm_v = np.linalg.norm(v)
        if norm_v < 1e-10:
            v = rng.normal(size=2)
            norm_v = np.linalg.norm(v)
        
        v = v / norm_v
        
        # Push apart
        # Add a small random component to avoid oscillation
        noise = rng.normal(scale=0.01, size=2)
        v = v + noise
        v = v / np.linalg.norm(v)
        
        positions[i] += v * step_size
        positions[j] -= v * step_size
        
        # Handle boundaries: if any point is outside, push it in
        for k in range(n):
            norm_k = np.linalg.norm(positions[k])
            if norm_k > 1.0:
                # Push towards center
                positions[k] = positions[k] * (0.999 / norm_k)
                
        # Decay step size slightly? Or keep constant?
        # Constant step size with random noise is a form of stochastic local search.
        # Let's decay it slowly to allow fine-tuning.
        step_size *= 0.9999
        if step_size < 1e-6:
            step_size = 1e-6
            
    return best_positions, best_radius


def _initial_positions_random(n, seed):
    """Generate random initial positions inside the unit circle."""
    rng = np.random.default_rng(seed)
    positions = np.zeros((n, 2))
    for i in range(n):
        while True:
            x = rng.uniform(-1, 1)
            y = rng.uniform(-1, 1)
            if x*x + y*y <= 1.0:
                positions[i] = [x, y]
                break
    return positions


def _initial_positions_hexagonal(n, seed):
    """Generate initial positions using a hexagonal lattice, clipped to circle."""
    rng = np.random.default_rng(seed)
    positions = []
    spacing = 0.2
    for row in range(-10, 11):
        for col in range(-10, 11):
            x = col * spacing * 1.5
            y = row * spacing * np.sqrt(3) + (col % 2) * spacing * np.sqrt(3) / 2
            if x*x + y*y < 0.95:
                positions.append([x, y])
    
    if len(positions) < n:
        # Fill remaining with random
        while len(positions) < n:
            x = rng.uniform(-0.9, 0.9)
            y = rng.uniform(-0.9, 0.9)
            if x*x + y*y < 0.9:
                positions.append([x, y])
    
    positions = np.array(positions[:n])
    return positions


def construct():
    """Construct centers of 50 equal circles inside the unit circle."""
    best_global_radius = 0.0
    best_global_positions = np.zeros((N, 2))
    
    # Multistart strategy
    # Use a mix of random and hexagonal starts
    num_starts = 5
    
    for i in range(num_starts):
        seed = 12345 + i * 100
        if i % 2 == 0:
            # Hexagonal start
            pos = _initial_positions_hexagonal(N, seed)
        else:
            # Random start
            pos = _initial_positions_random(N, seed)
            
        # Run local search
        pos, radius = _local_search(pos, max_iter=3000, seed=seed)
        
        if radius > best_global_radius:
            best_global_radius = radius
            best_global_positions = pos.copy()
            
    # Final refinement: run a few more iterations on the best found
    # Or just return the best found
    
    # Ensure validity
    for i in range(N):
        norm = np.linalg.norm(best_global_positions[i])
        if norm > 0.999:
            best_global_positions[i] = 0.999 * best_global_positions[i] / norm
            
    return best_global_positions
# EVOLVE-BLOCK-END
