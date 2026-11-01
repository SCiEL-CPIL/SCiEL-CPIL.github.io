"""Packing 50 equal circles in a unit circle using force-directed relaxation."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np


def _initial_hexagonal(n, seed=42):
    """Generate initial positions using a hexagonal lattice pattern."""
    rng = np.random.default_rng(seed)
    points = []
    # Hexagonal lattice spacing
    # We want to fit ~50 points in unit circle.
    # Area of unit circle is pi. Area per point approx pi/50.
    # Hex cell area for side s is (3*sqrt(3)/2) * s^2.
    # Let's just generate a grid and filter.
    
    # Spacing approx 0.15
    spacing = 0.14
    for row in range(-10, 11):
        for col in range(-10, 11):
            x = col * spacing * 1.5
            y = row * spacing * np.sqrt(3) + (col % 2) * spacing * np.sqrt(3) / 2
            if x * x + y * y < 0.95:
                points.append([x, y])
    
    points = np.array(points)
    
    if len(points) < n:
        # Fallback: random
        positions = rng.uniform(-0.9, 0.9, size=(n, 2))
        for i in range(n):
            while np.linalg.norm(positions[i]) > 0.9:
                positions[i] = rng.uniform(-0.9, 0.9, size=2)
        return positions
    
    # Select n points, preferring those closer to center to ensure central coverage
    dists = np.linalg.norm(points, axis=1)
    indices = np.argsort(dists)[:n]
    return points[indices].copy()


def _relax(positions, target_radius, steps=2000, dt=0.01, damping=0.9):
    """
    Relax positions to avoid overlaps of circles with radius target_radius.
    Uses a soft-body force model.
    """
    n = len(positions)
    pos = positions.copy()
    
    # Pre-compute indices for pairwise distances to speed up inner loop?
    # For N=50, N^2/2 = 1225 pairs. 2000 steps * 1225 = 2.4M ops. Fast enough in Python with NumPy vectorization.
    
    for step in range(steps):
        forces = np.zeros_like(pos)
        
        # Pairwise repulsion
        # Vectorized pairwise distance calculation
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        
        # Avoid division by zero
        np.fill_diagonal(dists, np.inf)
        
        # We want to push apart if dist < 2 * target_radius
        min_dist = 2.0 * target_radius
        # Penalty force magnitude
        # F = k * (min_dist - dist) if dist < min_dist else 0
        # Using a smooth potential derivative: F ~ (min_dist - dist)^2 or similar
        # Let's use linear repulsion for simplicity and stability, with a strong constant.
        overlap = np.maximum(0, min_dist - dists)
        
        # Force direction
        # For each pair i, j, force on i is (pos_i - pos_j) / dist * overlap
        # We can sum these up.
        # diff[i, j] = pos[i] - pos[j]
        # force_dir = diff / dists
        # force_mag = overlap
        # F_i = sum_j (diff[i, j] / dists[i, j]) * overlap[i, j]
        
        # Handle dists == 0 (coincident points)
        safe_dists = np.where(dists > 1e-10, dists, 1.0)
        force_dirs = diff / safe_dists[:, :, np.newaxis]
        
        # Zero out diagonal
        np.fill_diagonal(force_dirs[:, 0, 0], 0) # This is tricky with 3D array, better to mask
        mask = (dists > 1e-10)
        force_dirs = force_dirs * mask[:, :, np.newaxis]
        
        # Sum forces
        forces += np.sum(force_dirs * overlap[:, :, np.newaxis], axis=1)
        
        # Boundary force
        norms = np.linalg.norm(pos, axis=1)
        # If norm > 1 - target_radius, push inward
        # Actually, the circle must be inside, so center must be within 1 - r.
        # Let's use 1 - target_radius as the boundary.
        boundary_limit = 1.0 - target_radius
        boundary_violation = np.maximum(0, norms - boundary_limit)
        
        # Force direction is inward: -pos / norm
        safe_norms = np.where(norms > 1e-10, norms, 1.0)
        boundary_dirs = -pos / safe_norms[:, np.newaxis]
        
        # Scale boundary force to be comparable to repulsion
        # Repulsion force magnitude is roughly overlap (max ~ target_radius)
        # Boundary force magnitude is violation
        forces += boundary_dirs * boundary_violation[:, np.newaxis] * 2.0 # Stronger boundary constraint
        
        # Update positions
        pos += dt * forces
        pos *= damping # Damping to stabilize
        
        # Hard constraint: clamp to boundary if necessary to prevent escape
        norms = np.linalg.norm(pos, axis=1)
        max_norm = 1.0 - target_radius
        for i in range(n):
            if norms[i] > max_norm:
                pos[i] = pos[i] * (max_norm / norms[i])
                
    return pos


def _compute_radius(positions):
    """Compute the maximum radius for given positions."""
    n = len(positions)
    if n == 0:
        return 0.0
    
    # Distance to boundary
    boundary_dist = 1.0 - np.linalg.norm(positions, axis=1)
    
    # Minimum pairwise distance
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    return min(np.min(boundary_dist), min_pair_dist / 2.0)


def construct():
    """Construct centers of 50 equal circles inside the unit circle."""
    # Step 1: Initial hexagonal lattice
    positions = _initial_hexagonal(N, seed=42)
    
    # Step 2: Relaxation with increasing target radius
    # Start with a small radius and increase it.
    # The best known is ~0.1258.
    # We will try to push towards 0.126.
    
    current_radius = 0.1
    best_positions = positions.copy()
    best_radius = _compute_radius(positions)
    
    # Iterate over target radii
    # We do a few passes to refine
    for pass_num in range(3):
        for target_r in np.arange(0.10, 0.126, 0.002):
            # Relax towards this target
            relaxed_pos = _relax(positions, target_r, steps=1000, dt=0.005, damping=0.85)
            
            # Check actual radius
            actual_r = _compute_radius(relaxed_pos)
            
            if actual_r > best_radius:
                best_radius = actual_r
                best_positions = relaxed_pos.copy()
                positions = relaxed_pos.copy() # Use this as base for next target
            else:
                # If we failed to achieve target_r, stick with previous good positions
                # But we might still want to use the relaxed pos if it's better than current 'positions'
                if actual_r > _compute_radius(positions):
                    positions = relaxed_pos.copy()
        
        # Fine tuning around best found radius
        if pass_num == 2:
             # Do a few fine steps around best_radius
             for delta in [0.0005, -0.0005, 0.001, -0.001]:
                 target_r = best_radius + delta
                 if target_r > 0.05:
                     relaxed_pos = _relax(positions, target_r, steps=500, dt=0.002, damping=0.8)
                     actual_r = _compute_radius(relaxed_pos)
                     if actual_r > best_radius:
                         best_radius = actual_r
                         best_positions = relaxed_pos.copy()
                         positions = relaxed_pos.copy()

    # Final safety check
    for i in range(N):
        norm = np.linalg.norm(best_positions[i])
        if norm > 1.0:
             best_positions[i] = best_positions[i] * (0.999 / norm)
             
    return best_positions
# EVOLVE-BLOCK-END
