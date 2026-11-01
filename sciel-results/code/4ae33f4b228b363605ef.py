"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time


def _hex_lattice_init(n):
    """Generate initial positions on a hexagonal lattice, then select n closest to center."""
    # Generate a hexagonal grid of candidate points
    spacing = 0.2
    points = []
    k = 10
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            r = np.hypot(x, y)
            if r < 0.95:
                points.append((x, y))
    
    points = np.array(points)
    # Sort by distance from center, take n closest
    dists = np.hypot(points[:, 0], points[:, 1])
    idx = np.argsort(dists, kind='stable')[:n]
    return points[idx].copy()


def _score(positions):
    """Compute the packing radius (minimum of pairwise distances/2 and distance to boundary)."""
    n = len(positions)
    # Pairwise distances
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    # Distance to boundary
    radii = np.hypot(positions[:, 0], positions[:, 1])
    min_boundary = 1.0 - np.max(radii)
    
    return min(min_pair, min_boundary)


def _force_optimize(positions, max_time=100.0):
    """Optimize positions using force-based relaxation with time limit."""
    start_time = time.time()
    positions = positions.copy()
    n = len(positions)
    
    # Initial radius estimate
    current_r = _score(positions)
    best_r = current_r
    best_positions = positions.copy()
    
    # Adaptive step size
    step = 0.05
    lr = 0.995  # learning rate decay
    
    while time.time() - start_time < max_time:
        # Compute forces: repulsion between overlapping circles, attraction to boundary
        forces = np.zeros_like(positions)
        
        # Pairwise repulsion
        for i in range(n):
            for j in range(i + 1, n):
                diff = positions[i] - positions[j]
                dist = np.sqrt(np.sum(diff ** 2))
                if dist < 1e-10:
                    # Random small perturbation
                    forces[i] += np.random.randn(2) * 0.01
                    forces[j] -= np.random.randn(2) * 0.01
                    continue
                # Repulsive force when too close
                min_dist = 2.0 * current_r
                if dist < min_dist:
                    force_mag = (min_dist - dist) / max(dist, 1e-10)
                    forces[i] += force_mag * diff / dist
                    forces[j] -= force_mag * diff / dist
        
        # Boundary constraint: push inward if outside
        for i in range(n):
            r = np.hypot(positions[i, 0], positions[i, 1])
            max_r = 1.0 - current_r
            if r > max_r:
                force_mag = (r - max_r) / max(r, 1e-10)
                forces[i] -= force_mag * positions[i] / r
        
        # Apply forces with step size
        positions += step * forces
        
        # Keep inside container (hard constraint)
        radii = np.hypot(positions[:, 0], positions[:, 1])
        max_allowed = 1.0 - current_r * 0.99
        mask = radii > max_allowed
        if np.any(mask):
            positions[mask] *= max_allowed / radii[mask]
        
        # Update radius estimate
        new_r = _score(positions)
        if new_r > best_r:
            best_r = new_r
            best_positions = positions.copy()
        
        current_r = max(new_r, best_r * 0.99)
        
        # Decay step size
        step *= lr
        
        # If step gets too small, try a restart with larger step
        if step < 0.001:
            positions = best_positions + np.random.randn(n, 2) * 0.01
            step = 0.02
            current_r = best_r
    
    return best_positions, best_r


def construct():
    """Construct positions for N circles in the unit disk."""
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Initialize with hexagonal lattice
    positions = _hex_lattice_init(N)
    
    # Force-based optimization
    positions, best_r = _force_optimize(positions, max_time=100.0)
    
    # Check if we have time for a final polish
    if time.time() < deadline - 5.0:
        # Fine-tuning pass with smaller steps
        polish_positions = positions.copy()
        polish_r = best_r
        step = 0.005
        for _ in range(2000):
            if time.time() > deadline - 2.0:
                break
            forces = np.zeros_like(polish_positions)
            for i in range(N):
                for j in range(i + 1, N):
                    diff = polish_positions[i] - polish_positions[j]
                    dist = np.sqrt(np.sum(diff ** 2))
                    if dist < 1e-10:
                        forces[i] += np.random.randn(2) * 0.001
                        forces[j] -= np.random.randn(2) * 0.001
                        continue
                    min_dist = 2.0 * polish_r
                    if dist < min_dist:
                        force_mag = (min_dist - dist) / max(dist, 1e-10)
                        forces[i] += force_mag * diff / dist
                        forces[j] -= force_mag * diff / dist
            
            for i in range(N):
                r = np.hypot(polish_positions[i, 0], polish_positions[i, 1])
                max_r = 1.0 - polish_r
                if r > max_r:
                    force_mag = (r - max_r) / max(r, 1e-10)
                    forces[i] -= force_mag * polish_positions[i] / r
            
            polish_positions += step * forces
            
            radii = np.hypot(polish_positions[:, 0], polish_positions[:, 1])
            max_allowed = 1.0 - polish_r * 0.99
            mask = radii > max_allowed
            if np.any(mask):
                polish_positions[mask] *= max_allowed / radii[mask]
            
            new_r = _score(polish_positions)
            if new_r > polish_r:
                polish_r = new_r
                positions = polish_positions.copy()
            
            step *= 0.999
        
        best_r = polish_r
    
    # Final validation: ensure all circles are inside
    radii = np.hypot(positions[:, 0], positions[:, 1])
    max_allowed = 1.0 - _score(positions) * 0.999
    mask = radii > max_allowed
    if np.any(mask):
        positions[mask] *= max_allowed / radii[mask]
    
    return positions
# EVOLVE-BLOCK-END
