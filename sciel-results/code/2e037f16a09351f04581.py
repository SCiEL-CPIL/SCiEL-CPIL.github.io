import numpy as np
import time

N = 50
SEED = 42

def construct():
    np.random.seed(SEED)
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    # Initialize positions on a hexagonal lattice for good starting configuration
    positions = _hex_lattice_init(N)
    
    # Track best configuration found
    best_score = 0.0
    best_positions = positions.copy()
    
    # Optimization parameters
    lr = 0.005  # Learning rate
    momentum = 0.9
    velocity = np.zeros_like(positions)
    
    # Main optimization loop
    while time.time() < deadline:
        # Compute forces
        forces = _compute_forces(positions)
        
        # Update with momentum
        velocity = momentum * velocity + lr * forces
        positions += velocity
        
        # Project positions back into unit disk (hard constraint)
        dists = np.linalg.norm(positions, axis=1)
        mask = dists > 1.0
        if np.any(mask):
            positions[mask] = positions[mask] / dists[mask][:, None]
        
        # Evaluate score
        score = _score(positions)
        
        # Update best
        if score > best_score:
            best_score = score
            best_positions = positions.copy()
        
        # Adaptive learning rate: reduce if score decreases
        if score < best_score * 0.99:
            lr *= 0.95
    
    return best_positions

def _hex_lattice_init(n):
    """Initialize positions on a hexagonal lattice cut to unit disk."""
    spacing = 2.0 / (np.sqrt(n) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        points = points[radius <= 1.0 - spacing / 2.0]
        if len(points) >= n:
            return points[np.argsort(radius, kind="stable")[:n]]
        spacing *= 0.98

def _score(positions):
    """Compute the minimum radius for given positions."""
    if len(positions) != N:
        return 0.0
    
    # Check for non-finite values
    if not np.all(np.isfinite(positions)):
        return 0.0
    
    # Distance between all pairs
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    
    # Minimum inter-circle distance
    min_pair_dist = np.min(dists)
    
    # Distance from container boundary
    radii = np.linalg.norm(positions, axis=1)
    min_boundary = np.min(1.0 - radii)
    
    # Score is half the minimum of these
    return 0.5 * min(min_pair_dist, min_boundary)

def _compute_forces(positions):
    """Compute repulsive forces between circles and from container boundary."""
    n = len(positions)
    forces = np.zeros_like(positions)
    
    # Inter-circle repulsive forces
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    
    # Short-range repulsive potential: F ~ 1/r^2 for r < threshold
    threshold = 0.5
    mask = dists < threshold
    for i in range(n):
        for j in range(n):
            if i != j and mask[i, j]:
                d = dists[i, j]
                if d > 1e-10:
                    force_mag = 1.0 / (d * d)
                    forces[i] += force_mag * diff[i, j] / d
    
    # Container boundary repulsion
    radii = np.linalg.norm(positions, axis=1)
    boundary_dist = 1.0 - radii
    for i in range(n):
        if boundary_dist[i] < 0.1:
            # Push inward
            if radii[i] > 1e-10:
                forces[i] -= 10.0 * positions[i] / radii[i]
    
    return forces
