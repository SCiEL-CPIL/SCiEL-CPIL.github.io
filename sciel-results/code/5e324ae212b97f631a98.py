"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_radius(points):
    """Compute the packing radius for a set of points."""
    if len(points) < 2:
        return np.inf
    # Minimum distance between any two points
    from scipy.spatial.distance import cdist
    dist_matrix = cdist(points, points)
    np.fill_diagonal(dist_matrix, np.inf)
    min_dist = np.min(dist_matrix)
    # Minimum distance to boundary
    radii = np.linalg.norm(points, axis=1)
    min_boundary = 1.0 - np.max(radii)
    return min(min_dist / 2.0, min_boundary)

def _hexagonal_lattice(n_target):
    """Generate a hexagonal lattice packing inside the unit disk."""
    # Use a hexagonal lattice with spacing d
    # For n circles, optimal spacing is roughly 2/sqrt(n)
    # Start with a reasonable estimate and adjust
    d = 2.0 / (np.sqrt(n_target) + 1.5)
    
    # Generate hexagonal lattice points
    # Hexagonal lattice: points at (i*d + (j%2)*d/2, j*d*sqrt(3)/2)
    # We need to generate enough points and then select those inside the unit disk
    
    max_range = int(2.0 / d) + 3
    points = []
    for j in range(-max_range, max_range + 1):
        for i in range(-max_range, max_range + 1):
            x = i * d + (j % 2) * d / 2.0
            y = j * d * np.sqrt(3.0) / 2.0
            if x*x + y*y <= 1.0:
                points.append([x, y])
    
    points = np.array(points)
    
    # Sort by distance from origin (keep center points)
    if len(points) >= n_target:
        dists = np.linalg.norm(points, axis=1)
        indices = np.argsort(dists)[:n_target]
        return points[indices]
    
    # If we don't have enough, reduce spacing
    return None

def _concentric_rings(n_target):
    """Generate a concentric ring packing inside the unit disk."""
    # Place circles in concentric rings
    # Center circle + rings
    points = [[0.0, 0.0]]
    
    ring_radius = 0.0
    ring_count = 1
    spacing = 2.0 / (np.sqrt(n_target) + 1.0)
    
    while len(points) < n_target:
        ring_radius += spacing
        if ring_radius >= 1.0 - spacing/2:
            break
        # Number of circles on this ring
        circumference = 2 * np.pi * ring_radius
        n_on_ring = max(1, int(circumference / spacing))
        # Ensure circles on ring don't overlap with each other
        angle_step = 2 * np.pi / n_on_ring
        for k in range(n_on_ring):
            angle = k * angle_step
            x = ring_radius * np.cos(angle)
            y = ring_radius * np.sin(angle)
            if x*x + y*y <= 1.0:
                points.append([x, y])
                if len(points) >= n_target:
                    break
    
    points = np.array(points[:n_target])
    return points

def _optimize_local(points, max_iters=100, tol=1e-8):
    """Local optimization using gradient-free methods to maximize minimum distance."""
    points = points.copy()
    current_radius = _compute_radius(points)
    
    for iteration in range(max_iters):
        improved = False
        # Try moving each point slightly
        for i in range(len(points)):
            orig = points[i].copy()
            best_pos = orig
            best_radius = current_radius
            
            # Try small perturbations
            for dx in [-0.01, 0.0, 0.01]:
                for dy in [-0.01, 0.0, 0.01]:
                    if dx == 0 and dy == 0:
                        continue
                    test_points = points.copy()
                    test_points[i] = orig + np.array([dx, dy])
                    # Check if still inside unit circle
                    if np.linalg.norm(test_points[i]) > 1.0:
                        continue
                    r = _compute_radius(test_points)
                    if r > best_radius + 1e-10:
                        best_radius = r
                        best_pos = test_points[i]
            
            if np.any(best_pos != orig):
                points[i] = best_pos
                current_radius = _compute_radius(points)
                improved = True
        
        if not improved:
            break
    
    return points, current_radius

def construct():
    """Construct centers of N equal circles inside the unit circle."""
    start_time = time.time()
    deadline = start_time + 140.0  # 140s deadline, 10s safety margin
    
    best_points = None
    best_radius = -np.inf
    
    # Strategy 1: Hexagonal lattice
    hex_points = _hexagonal_lattice(N)
    if hex_points is not None and len(hex_points) == N:
        r = _compute_radius(hex_points)
        if r > best_radius:
            best_radius = r
            best_points = hex_points.copy()
    
    # Strategy 2: Concentric rings
    ring_points = _concentric_rings(N)
    if len(ring_points) == N:
        r = _compute_radius(ring_points)
        if r > best_radius:
            best_radius = r
            best_points = ring_points.copy()
    
    # Strategy 3: Local optimization from best initial
    if best_points is not None:
        time_remaining = deadline - time.time()
        if time_remaining > 5.0:
            max_iters = int(time_remaining / 2.0)  # Rough estimate
            max_iters = min(max_iters, 50)
            optimized_points, opt_radius = _optimize_local(best_points, max_iters=max_iters)
            if opt_radius > best_radius:
                best_radius = opt_radius
                best_points = optimized_points
    
    # Final check: ensure all points are inside unit circle
    if best_points is not None:
        radii = np.linalg.norm(best_points, axis=1)
        # Clip points that are slightly outside
        mask = radii > 1.0
        if np.any(mask):
            best_points[mask] = best_points[mask] / radii[mask, None] * 0.999
    
    return best_points
# EVOLVE-BLOCK-END
