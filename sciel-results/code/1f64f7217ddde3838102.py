"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_radius(points):
    """Compute the packing radius for a set of points using vectorized operations."""
    if len(points) == 0:
        return 0.0
    n = len(points)
    
    # Compute all pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    
    # Minimum pair distance (excluding self)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Minimum distance to boundary
    center_dists = np.linalg.norm(points, axis=1)
    min_boundary_dist = 1.0 - np.max(center_dists)
    
    return min(min_pair_dist / 2.0, min_boundary_dist)


def _repulsion_forces(points, strength=1.0):
    """Compute repulsion forces between all pairs, vectorized."""
    n = len(points)
    if n == 0:
        return np.zeros((0, 2))
    
    # Compute all pairwise differences
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    
    # Avoid division by zero
    np.fill_diagonal(dists, 1.0)
    
    # Force magnitude: 1/dist^2
    force_mags = strength / (dists ** 2)
    np.fill_diagonal(force_mags, 0.0)
    
    # Force direction: diff / dist
    force_dirs = diff / dists[:, :, np.newaxis]
    
    # Sum forces
    forces = np.sum(force_mags[:, :, np.newaxis] * force_dirs, axis=1)
    
    return forces


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Initial hexagonal lattice
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    points = None
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        pts = np.array(rows)
        radius = np.hypot(pts[:, 0], pts[:, 1])
        pts, radius = pts[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
        if len(pts) >= N:
            points = pts[np.argsort(radius, kind="stable")[:N]].copy()
            break
        spacing *= 0.98
    
    if points is None:
        angles = np.random.uniform(0, 2 * np.pi, N)
        r = np.sqrt(np.random.uniform(0, 1, N))
        points = np.column_stack([r * np.cos(angles), r * np.sin(angles)])
    
    best_points = points.copy()
    best_radius = _compute_radius(points)
    
    # Phase 1: Iterative repulsion with cooling
    strength = 0.01
    lr = 0.005
    for iteration in range(8000):
        if time.time() > deadline:
            break
        
        # Compute repulsion forces
        forces = _repulsion_forces(points, strength)
        
        # Boundary force: push inward if too close to edge
        center_dists = np.linalg.norm(points, axis=1)
        boundary_forces = np.zeros_like(points)
        mask = center_dists > 0.9
        if np.any(mask):
            boundary_forces[mask] = -(center_dists[mask] - 0.9) * points[mask] / center_dists[mask, np.newaxis]
        
        # Update
        points = points + lr * (forces + boundary_forces)
        
        # Project inside unit circle
        norms = np.linalg.norm(points, axis=1)
        mask = norms > 0.999
        if np.any(mask):
            points[mask] *= 0.999 / norms[mask]
        
        new_radius = _compute_radius(points)
        if new_radius > best_radius:
            best_radius = new_radius
            best_points = points.copy()
        
        # Cool down
        strength *= 0.9995
        lr *= 0.9999
    
    return best_points
# EVOLVE-BLOCK-END

