"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time


def construct():
    deadline = time.time() + 115.0  # Safety margin before 120s limit
    
    # 1. Initialization: Hexagonal lattice
    # Approximate density: N circles in area pi. Area per circle ~ pi/N.
    # Spacing s ~ sqrt(pi/N).
    s = np.sqrt(np.pi / N) * 0.95  # Slightly tighter to allow expansion
    
    # Generate hex lattice points
    points = []
    k_max = int(2.0 / s) + 2
    for j in range(-k_max, k_max + 1):
        for i in range(-k_max, k_max + 1):
            x = i * s + (j % 2) * s * 0.5
            y = j * s * np.sqrt(3.0) / 2.0
            points.append((x, y))
            
    points = np.array(points)
    # Center the lattice
    points -= points.mean(axis=0)
    
    # Select N points closest to origin to form initial configuration
    dists = np.hypot(points[:, 0], points[:, 1])
    indices = np.argsort(dists)[:N]
    current_pts = points[indices].copy()
    
    # 2. Optimization: Physics-based relaxation with repulsive forces
    # We use a repulsive potential U = 1/d^2 between all pairs.
    # This drives the system to a state where distances are maximized (equalized).
    # We project points back into the unit circle after each step to enforce containment.
    
    def compute_repulsive_forces(pts):
        # Vectorized calculation of repulsive forces
        # diff[i, j] = pts[i] - pts[j]
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        
        # Mask to exclude self-interaction
        mask = ~np.eye(N, dtype=bool)
        # Avoid division by zero for self-interaction
        dist_sq_safe = np.where(mask, dist_sq, 1.0)
        dist_safe = np.sqrt(dist_sq_safe)
        
        # Force magnitude factor: d(1/d^2)/dx = -2/d^3. 
        # We want force in direction of diff (repulsion), so F = +2/d^3 * diff
        factor = 2.0 / (dist_safe**3)
        factor = np.where(mask, factor, 0.0)
        
        # Sum forces from all neighbors
        forces = np.sum(factor[:, :, np.newaxis] * diff, axis=1)
        return forces

    step_size = 0.005
    steps = 0
    max_steps = 10000
    
    while steps < max_steps and time.time() < deadline:
        forces = compute_repulsive_forces(current_pts)
        
        # Limit force magnitude to prevent instability
        force_mags = np.linalg.norm(forces, axis=1)
        max_force = 10.0
        scale = np.where(force_mags > max_force, max_force / force_mags, 1.0)
        forces *= scale[:, np.newaxis]
        
        current_pts += step_size * forces
        
        # Project points back into the unit circle (hard constraint)
        norms = np.linalg.norm(current_pts, axis=1)
        too_far = norms > 1.0
        if np.any(too_far):
            current_pts[too_far] /= norms[too_far, np.newaxis]
            
        steps += 1
        
    # Final cleanup: ensure strictly inside
    norms = np.linalg.norm(current_pts, axis=1)
    too_far = norms > 0.9999
    if np.any(too_far):
        current_pts[too_far] *= 0.9999 / norms[too_far, np.newaxis]
        
    return current_pts
# EVOLVE-BLOCK-END
