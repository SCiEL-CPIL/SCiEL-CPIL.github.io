"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time


def _score(points):
    """Compute the radius score for a set of points."""
    if len(points) != N:
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    # Check container constraint
    dist_to_center = np.sqrt(np.sum(points ** 2, axis=1))
    if np.any(dist_to_center >= 1.0):
        return 0.0
    # Compute pairwise distances
    # For 300 points, we can compute all pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    # Set diagonal to infinity
    np.fill_diagonal(dist, np.inf)
    min_pair_dist = np.min(dist)
    min_container_dist = 1.0 - np.max(dist_to_center)
    return min(min_pair_dist / 2.0, min_container_dist)


def _hex_init():
    """Initialize points on a hexagonal lattice within the unit disk."""
    # Generate hex lattice points
    spacing = 0.1
    points = []
    for j in range(-10, 11):
        for i in range(-10, 11):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3) / 2
            if x * x + y * y < 1.0:
                points.append([x, y])
    points = np.array(points)
    # Sort by distance from center, take closest N
    dist = np.sqrt(np.sum(points ** 2, axis=1))
    idx = np.argsort(dist)
    points = points[idx[:N]]
    return points


def _relax(points, max_iter=5000, tol=1e-10):
    """Force-based relaxation to maximize minimum distance."""
    points = points.copy()
    current_score = _score(points)
    
    for it in range(max_iter):
        if it % 100 == 0:
            elapsed = time.time() - _start_time
            if elapsed > 130:  # Safety margin
                break
        
        # Compute pairwise distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dist, np.inf)
        
        # Find minimum distance pair
        min_dist = np.min(dist)
        
        # Compute forces
        forces = np.zeros_like(points)
        
        # Repulsive forces between close pairs
        for i in range(N):
            for j in range(i + 1, N):
                d = dist[i, j]
                if d < min_dist * 2:  # Only consider close pairs
                    # Force proportional to 1/d^3, directed apart
                    if d > 1e-12:
                        f = (min_dist - d) / (d ** 3 + 1e-6)
                        if f > 0:
                            diff_ij = points[i] - points[j]
                            norm = np.linalg.norm(diff_ij)
                            if norm > 1e-12:
                                unit = diff_ij / norm
                                forces[i] += f * unit
                                forces[j] -= f * unit
        
        # Boundary forces: push points inward if they're near the boundary
        dist_to_center = np.sqrt(np.sum(points ** 2, axis=1))
        for i in range(N):
            if dist_to_center[i] > 0.9:
                # Push inward
                if dist_to_center[i] > 1e-12:
                    unit = points[i] / dist_to_center[i]
                    forces[i] -= (dist_to_center[i] - 0.9) * unit * 0.5
        
        # Update positions with adaptive step size
        step_size = 0.01
        new_points = points + step_size * forces
        
        # Clamp to unit disk
        new_dist = np.sqrt(np.sum(new_points ** 2, axis=1))
        mask = new_dist > 1.0
        if np.any(mask):
            new_points[mask] = new_points[mask] / new_dist[mask]
        
        # Check if improvement
        new_score = _score(new_points)
        if new_score > current_score + 1e-12:
            points = new_points
            current_score = new_score
        else:
            # Try smaller step
            step_size *= 0.5
            if step_size < 1e-6:
                break
    
    return points


def _polish(points, max_iter=1000):
    """High-precision polish pass with tighter tolerance."""
    points = points.copy()
    current_score = _score(points)
    
    for it in range(max_iter):
        if it % 50 == 0:
            elapsed = time.time() - _start_time
            if elapsed > 145:  # Safety margin
                break
        
        # Compute pairwise distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dist, np.inf)
        
        min_dist = np.min(dist)
        
        # Find all pairs at minimum distance
        min_pairs = []
        for i in range(N):
            for j in range(i + 1, N):
                if abs(dist[i, j] - min_dist) < 1e-8:
                    min_pairs.append((i, j))
        
        if not min_pairs:
            break
        
        # Compute forces from minimum-distance pairs
        forces = np.zeros_like(points)
        for i, j in min_pairs:
            diff_ij = points[i] - points[j]
            norm = np.linalg.norm(diff_ij)
            if norm > 1e-12:
                unit = diff_ij / norm
                forces[i] += unit * 0.001
                forces[j] -= unit * 0.001
        
        # Boundary forces
        dist_to_center = np.sqrt(np.sum(points ** 2, axis=1))
        for i in range(N):
            if dist_to_center[i] > 0.95:
                if dist_to_center[i] > 1e-12:
                    unit = points[i] / dist_to_center[i]
                    forces[i] -= (dist_to_center[i] - 0.95) * unit * 0.01
        
        # Update
        new_points = points + forces
        new_dist = np.sqrt(np.sum(new_points ** 2, axis=1))
        mask = new_dist > 1.0
        if np.any(mask):
            new_points[mask] = new_points[mask] / new_dist[mask]
        
        new_score = _score(new_points)
        if new_score > current_score + 1e-14:
            points = new_points
            current_score = new_score
        else:
            break
    
    return points


def construct():
    global _start_time
    _start_time = time.time()
    
    # Start with hexagonal lattice initialization
    points = _hex_init()
    current_score = _score(points)
    
    # Force-based relaxation
    points = _relax(points)
    current_score = _score(points)
    
    # Polish pass
    points = _polish(points)
    final_score = _score(points)
    
    # Ensure valid output
    if not np.all(np.isfinite(points)) or len(points) != N:
        # Fallback to hex init
        points = _hex_init()
    
    return points
# EVOLVE-BLOCK-END
