"""Rewritten program: use a multi-ring initialization plus local repulsion-based optimization."""
import numpy as np
import time

N = 300
DEADLINE = 140.0  # safety margin before 150s limit


def _compute_radius(points):
    """Compute the maximum common radius for given centers."""
    n = len(points)
    if n < 2:
        return 0.0
    # Min distance between any two centers
    min_dist = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(points[i] - points[j])
            if d < min_dist:
                min_dist = d
    # Min distance to boundary
    min_boundary = np.min(1.0 - np.linalg.norm(points, axis=1))
    return min(min_dist / 2.0, min_boundary)


def _compute_radius_fast(points):
    """Faster radius computation using broadcasting in chunks."""
    n = len(points)
    if n < 2:
        return 0.0
    # Compute all pairwise distances
    min_dist = np.inf
    chunk_size = 50
    for i in range(0, n, chunk_size):
        chunk = points[i:i+chunk_size]
        # Distance from chunk to all points
        diff = chunk[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        # Mask out self-distances
        rows = np.arange(i, min(i + chunk_size, n))
        for idx, row in enumerate(rows):
            dists[idx, row] = np.inf
        min_dist = min(min_dist, np.min(dists))
    min_boundary = np.min(1.0 - np.linalg.norm(points, axis=1))
    return min(min_dist / 2.0, min_boundary)


def _local_optimize(points, max_iters=200, dt=0.001):
    """Local optimization using repulsion forces."""
    n = len(points)
    points = points.copy()
    best_r = _compute_radius_fast(points)
    best_points = points.copy()
    
    for it in range(max_iters):
        if time.time() > DEADLINE:
            break
        
        # Compute forces
        forces = np.zeros_like(points)
        
        # Repulsion between circles
        for i in range(n):
            for j in range(i + 1, n):
                diff = points[i] - points[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-10:
                    # Random perturbation
                    forces[i] += np.random.randn(2) * 0.01
                    forces[j] -= np.random.randn(2) * 0.01
                else:
                    # Strong repulsion when close
                    strength = 1.0 / (dist * dist + 1e-6)
                    unit = diff / dist
                    forces[i] += unit * strength
                    forces[j] -= unit * strength
        
        # Attraction to boundary (push outward)
        for i in range(n):
            r = np.linalg.norm(points[i])
            if r > 1e-10:
                # Gentle push toward boundary
                forces[i] += (points[i] / r) * 0.1
        
        # Update positions
        points += dt * forces
        
        # Project inside unit circle
        radii = np.linalg.norm(points, axis=1)
        mask = radii > 0.999
        if np.any(mask):
            points[mask] *= 0.999 / radii[mask]
        
        # Compute current radius
        r = _compute_radius_fast(points)
        if r > best_r:
            best_r = r
            best_points = points.copy()
    
    return best_points, best_r


def _multi_ring_init(n_circles=300):
    """Initialize with multiple concentric rings."""
    points = []
    
    # Center point
    points.append([0.0, 0.0])
    
    # Rings
    ring_idx = 1
    remaining = n_circles - 1
    while remaining > 0:
        # Number of circles in this ring
        # Approximate: circumference / desired spacing
        ring_radius = 0.15 * ring_idx
        if ring_radius > 0.9:
            break
        
        # Max circles that fit in this ring
        max_in_ring = int(2 * np.pi * ring_radius / (0.15))
        if max_in_ring < 1:
            max_in_ring = 1
        
        num_in_ring = min(remaining, max_in_ring)
        if num_in_ring < 1:
            num_in_ring = 1
        
        angles = np.linspace(0, 2 * np.pi, num_in_ring, endpoint=False)
        for angle in angles:
            points.append([ring_radius * np.cos(angle), ring_radius * np.sin(angle)])
        
        remaining -= num_in_ring
        ring_idx += 1
    
    # If we still need more points, add them in outer regions
    while len(points) < n_circles:
        angle = np.random.uniform(0, 2 * np.pi)
        r = np.random.uniform(0.5, 0.95)
        points.append([r * np.cos(angle), r * np.sin(angle)])
    
    points = np.array(points[:n_circles])
    
    # Ensure all points are inside the unit circle
    radii = np.linalg.norm(points, axis=1)
    mask = radii > 0.999
    if np.any(mask):
        points[mask] *= 0.999 / radii[mask]
    
    return points


def construct():
    """Construct centers of N equal circles inside the unit circle."""
    start_time = time.time()
    
    # Generate initial configuration
    points = _multi_ring_init(N)
    
    # Local optimization with multiple restarts
    best_points = points.copy()
    best_r = _compute_radius_fast(points)
    
    # Try different initializations and optimize
    for restart in range(10):
        if time.time() > DEADLINE - 30:
            break
        
        if restart == 0:
            pts = points.copy()
        else:
            # Perturb the best solution found so far
            pts = best_points + np.random.randn(N, 2) * 0.05
            # Project inside unit circle
            radii = np.linalg.norm(pts, axis=1)
            mask = radii > 0.999
            if np.any(mask):
                pts[mask] *= 0.999 / radii[mask]
        
        optimized_pts, r = _local_optimize(pts, max_iters=150, dt=0.002)
        
        if r > best_r:
            best_r = r
            best_points = optimized_pts.copy()
    
    # Final check
    final_r = _compute_radius_fast(best_points)
    if final_r > best_r:
        best_r = final_r
    
    return best_points
