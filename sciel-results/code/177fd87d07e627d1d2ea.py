import numpy as np
import time

N = 50

def _hex_lattice_init():
    """Generate initial points using a hexagonal lattice."""
    points = []
    # Estimate spacing
    # Area of unit circle is pi. N circles of radius r.
    # r ~ sqrt(pi / N) / 2? No, packing density ~ 0.9.
    # N * pi * r^2 ~ 0.9 * pi => r ~ sqrt(0.9/N) ~ sqrt(0.9/50) ~ 0.134
    # Spacing ~ 2r ~ 0.27
    spacing = 0.25
    k = 5
    for i in range(-k, k + 1):
        for j in range(-k, k + 1):
            x = i * spacing + (j % 2) * spacing / 2
            y = j * spacing * np.sqrt(3) / 2
            if np.hypot(x, y) < 1.0:
                points.append([x, y])
    
    points = np.array(points)
    # Sort by radius to pick central ones?
    # Or just take first N.
    # Better: sort by distance to center, take N closest to ensure they are inside.
    radii = np.linalg.norm(points, axis=1)
    points = points[np.argsort(radii)]
    
    if len(points) < N:
        # Fallback to random
        return np.random.rand(N, 2) * 2 - 1
    else:
        return points[:N]

def _calc_min_radius(points):
    """Calculate the maximum radius for the given points."""
    n = len(points)
    min_pair_dist = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(points[i] - points[j])
            if d < min_pair_dist:
                min_pair_dist = d
    
    min_bound_dist = np.min(1.0 - np.linalg.norm(points, axis=1))
    return min(min_pair_dist / 2.0, min_bound_dist)

def _calculate_forces(points, r):
    """Calculate repulsive forces for a given radius r."""
    n = len(points)
    # Pair forces
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, 1.0)
    
    mask = dist < 2 * r
    np.fill_diagonal(mask, False)
    
    weights = np.where(mask, (2 * r - dist), 0.0)
    unit_vec = diff / dist[:, :, np.newaxis]
    force_pairs = np.sum(weights[:, :, np.newaxis] * unit_vec, axis=1)
    
    # Boundary forces
    d_bound = np.linalg.norm(points, axis=1)
    bound_mask = d_bound > 1 - r
    force_bound = np.zeros_like(points)
    if np.any(bound_mask):
        inv_d_bound = np.where(d_bound > 0, 1.0 / d_bound, 0.0)
        force_bound = (-points * inv_d_bound[:, np.newaxis]) * ((d_bound - (1 - r))[:, np.newaxis])
        force_bound *= bound_mask[:, np.newaxis]
        
    return force_pairs + force_bound

def _relax(points, r, alpha=0.1, max_steps=500, tol=1e-5):
    """Relax points to fit radius r."""
    for _ in range(max_steps):
        forces = _calculate_forces(points, r)
        force_norm = np.linalg.norm(forces, axis=1)
        if np.max(force_norm) < tol:
            break
        points += alpha * forces
        # Clip to unit circle to prevent divergence
        norms = np.linalg.norm(points, axis=1)
        mask = norms > 1.0
        if np.any(mask):
            points[mask] /= norms[mask, np.newaxis]
            
    return points

def construct():
    start_time = time.time()
    points = _hex_lattice_init()
    r = _calc_min_radius(points)
    
    delta_r = 0.001
    alpha = 0.1
    
    # Optimization loop
    while time.time() - start_time < 140:
        new_r = r + delta_r
        # Try to relax
        new_points = _relax(points.copy(), new_r, alpha=alpha, max_steps=100, tol=1e-5)
        
        # Check validity
        new_r_actual = _calc_min_radius(new_points)
        
        if new_r_actual >= new_r - 1e-6:
            # Success
            points = new_points
            r = new_r
            # Reduce delta_r slightly to refine
            delta_r *= 0.95
            # Reduce alpha slightly to converge better
            alpha *= 0.99
        else:
            # Failure
            delta_r *= 0.5
            if delta_r < 1e-7:
                break
            
    # Final polish
    # Run a few more steps with small alpha
    for _ in range(50):
        forces = _calculate_forces(points, r)
        points += 0.01 * forces
        norms = np.linalg.norm(points, axis=1)
        mask = norms > 1.0
        if np.any(mask):
            points[mask] /= norms[mask, np.newaxis]
            
    return points
