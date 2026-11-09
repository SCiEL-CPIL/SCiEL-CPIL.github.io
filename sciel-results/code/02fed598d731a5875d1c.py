"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_radius(points):
    """Compute the packing radius for a set of points."""
    if len(points) < 2:
        return np.inf
    from scipy.spatial.distance import cdist
    dist_matrix = cdist(points, points)
    np.fill_diagonal(dist_matrix, np.inf)
    min_dist = np.min(dist_matrix)
    radii = np.linalg.norm(points, axis=1)
    min_boundary = 1.0 - np.max(radii)
    return min(min_dist / 2.0, min_boundary)

def _hexagonal_lattice(n_target):
    """Generate a hexagonal lattice packing inside the unit disk."""
    d = 2.0 / (np.sqrt(n_target) + 1.5)
    max_range = int(2.0 / d) + 3
    points = []
    for j in range(-max_range, max_range + 1):
        for i in range(-max_range, max_range + 1):
            x = i * d + (j % 2) * d / 2.0
            y = j * d * np.sqrt(3.0) / 2.0
            if x*x + y*y <= 1.0:
                points.append([x, y])
    points = np.array(points)
    if len(points) >= n_target:
        dists = np.linalg.norm(points, axis=1)
        indices = np.argsort(dists)[:n_target]
        return points[indices]
    return None

def _langevin_dynamics(points, start_time, deadline, max_steps=50000):
    """Langevin dynamics to maximize minimum distance between circles."""
    points = points.copy()
    n = len(points)
    
    # Parameters
    dt = 0.001
    temperature = 0.01
    force_strength = 10.0
    boundary_strength = 5.0
    
    best_points = points.copy()
    best_radius = _compute_radius(points)
    
    # Velocity for Langevin
    velocity = np.zeros_like(points)
    
    for step in range(max_steps):
        # Check time
        if time.time() > deadline - 2.0:
            break
        
        # Compute forces
        forces = np.zeros_like(points)
        
        # Pairwise repulsion (only for close pairs to save time)
        from scipy.spatial.distance import cdist
        dist_matrix = cdist(points, points)
        np.fill_diagonal(dist_matrix, np.inf)
        
        # Find pairs that are too close
        threshold = 0.1
        close_pairs = np.argwhere(dist_matrix < threshold)
        
        for i, j in close_pairs:
            if i >= j:
                continue
            d = dist_matrix[i, j]
            if d < 1e-10:
                # Random small push
                forces[i] += np.random.randn(2) * 0.01
                forces[j] -= np.random.randn(2) * 0.01
            else:
                # Repulsive force: stronger when closer
                diff = points[i] - points[j]
                force_mag = force_strength / (d * d + 1e-6)
                forces[i] += diff / d * force_mag
                forces[j] -= diff / d * force_mag
        
        # Boundary force: push inward if too close to boundary
        radii = np.linalg.norm(points, axis=1)
        boundary_dist = 1.0 - radii
        for i in range(n):
            if boundary_dist[i] < 0.05:
                # Stronger force as we get closer to boundary
                r = radii[i]
                if r < 1e-10:
                    forces[i] += np.random.randn(2) * boundary_strength * 0.1
                else:
                    force_mag = boundary_strength / (boundary_dist[i] + 0.01)
                    forces[i] -= points[i] / r * force_mag
        
        # Langevin update: velocity = velocity * friction + force * dt + random_kick
        friction = 0.9
        velocity = velocity * friction + forces * dt
        # Add random thermal noise
        velocity += np.random.randn(n, 2) * temperature * np.sqrt(dt)
        
        # Update positions
        new_points = points + velocity * dt
        
        # Constrain to unit disk
        new_radii = np.linalg.norm(new_points, axis=1)
        mask = new_radii > 1.0
        if np.any(mask):
            new_points[mask] = new_points[mask] / new_radii[mask, None]
        
        points = new_points
        
        # Update best
        current_radius = _compute_radius(points)
        if current_radius > best_radius:
            best_radius = current_radius
            best_points = points.copy()
    
    return best_points, best_radius

def construct():
    """Construct centers of N equal circles inside the unit circle."""
    start_time = time.time()
    deadline = start_time + 140.0  # 140s deadline, 10s safety margin
    
    best_points = None
    best_radius = -np.inf
    
    # Strategy: Hexagonal lattice initialization + Langevin dynamics
    hex_points = _hexagonal_lattice(N)
    if hex_points is not None and len(hex_points) == N:
        r = _compute_radius(hex_points)
        best_radius = r
        best_points = hex_points.copy()
        
        # Run Langevin dynamics
        time_remaining = deadline - time.time()
        if time_remaining > 5.0:
            optimized_points, opt_radius = _langevin_dynamics(
                hex_points, start_time, deadline, max_steps=50000
            )
            if opt_radius > best_radius:
                best_radius = opt_radius
                best_points = optimized_points
    
    # Fallback: if hexagonal lattice failed, use random initialization
    if best_points is None:
        best_points = np.random.randn(N, 2) * 0.5
        best_radius = _compute_radius(best_points)
        
        time_remaining = deadline - time.time()
        if time_remaining > 5.0:
            optimized_points, opt_radius = _langevin_dynamics(
                best_points, start_time, deadline, max_steps=50000
            )
            if opt_radius > best_radius:
                best_radius = opt_radius
                best_points = optimized_points
    
    # Final check: ensure all points are inside unit circle
    radii = np.linalg.norm(best_points, axis=1)
    mask = radii > 1.0
    if np.any(mask):
        best_points[mask] = best_points[mask] / radii[mask, None] * 0.999
    
    return best_points
# EVOLVE-BLOCK-END
