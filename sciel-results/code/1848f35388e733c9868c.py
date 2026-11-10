"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import cKDTree

def _hexagonal_init(n):
    """Generate n points in a hexagonal lattice within unit circle."""
    # Start with a dense hexagonal grid and select n closest to center
    spacing = 0.05
    while True:
        k = int(1.0 / spacing) + 3
        rows = []
        for i in range(-k, k + 1):
            for j in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if x*x + y*y <= 1.0:
                    rows.append((x, y))
        points = np.array(rows)
        if len(points) >= n:
            # Select n points closest to center for better packing
            dists = np.hypot(points[:, 0], points[:, 1])
            idx = np.argsort(dists)[:n]
            return points[idx]
        spacing *= 0.97

def _compute_radius(points):
    """Compute the maximum common radius for the given points."""
    # Distance to boundary
    r_boundary = 1.0 - np.hypot(points[:, 0], points[:, 1])
    # Minimum pairwise distance / 2
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=2)
    min_pair = np.min(dists[:, 1]) / 2.0
    return min(r_boundary.min(), min_pair)

def _md_step(points, dt, alpha=1000.0, beta=5000.0):
    """
    One molecular dynamics step with:
    - Repulsive force between overlapping circles (inverse power law)
    - Confining force to keep circles inside unit circle
    """
    n = len(points)
    forces = np.zeros_like(points)
    
    # Pairwise repulsive forces
    tree = cKDTree(points)
    pairs = tree.query_pairs(r=0.2)  # Only consider nearby pairs
    
    for i, j in pairs:
        diff = points[i] - points[j]
        dist = np.linalg.norm(diff)
        if dist < 1e-10:
            # Add small random perturbation
            forces[i] += np.random.randn(2) * 0.001
            forces[j] -= np.random.randn(2) * 0.001
            continue
        if dist < 0.15:  # Overlap region
            # Strong repulsive force when overlapping
            force_mag = alpha * (0.15 - dist) ** 2
            forces[i] += force_mag * diff / dist
            forces[j] -= force_mag * diff / dist
    
    # Confining force: push points back inside unit circle
    r = np.hypot(points[:, 0], points[:, 1])
    for i in range(n):
        if r[i] > 0.95:
            # Inward force proportional to how far outside
            force_mag = beta * (r[i] - 0.95)
            forces[i] -= force_mag * points[i] / r[i]
    
    # Velocity Verlet-like update with damping
    points = points + dt * forces
    # Damping
    points *= 0.99
    
    # Project back inside unit circle
    r = np.hypot(points[:, 0], points[:, 1])
    mask = r > 1.0
    if np.any(mask):
        points[mask] = points[mask] / r[mask] * 0.999
    
    return points

def _local_optimize(points, max_iter=5000, time_limit=10.0):
    """Local optimization using molecular dynamics."""
    start_time = time.time()
    best_points = points.copy()
    best_radius = _compute_radius(points)
    
    dt = 0.01
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        points = _md_step(points, dt)
        r = _compute_radius(points)
        
        if r > best_radius:
            best_radius = r
            best_points = points.copy()
        
        # Anneal: reduce step size
        if it % 500 == 0:
            dt *= 0.95
    
    return best_points, best_radius

def _random_perturbation(points, sigma=0.01):
    """Apply random perturbation to escape local minima."""
    return points + np.random.randn(*points.shape) * sigma

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 20 second safety margin
    
    # Initialize from hexagonal lattice
    points = _hexagonal_init(N)
    best_points = points.copy()
    best_radius = _compute_radius(points)
    
    # Multi-restart local optimization
    n_restarts = 10
    for restart in range(n_restarts):
        if time.time() > deadline - 15.0:
            break
        
        # Perturb the current best
        if restart == 0:
            current = best_points.copy()
        else:
            current = _random_perturbation(best_points, sigma=0.02 * (1.0 + restart * 0.1))
        
        # Run local optimization
        current, radius = _local_optimize(current, max_iter=3000, time_limit=10.0)
        
        if radius > best_radius:
            best_radius = radius
            best_points = current.copy()
    
    # Final polish with finer steps
    if time.time() < deadline - 5.0:
        best_points, best_radius = _local_optimize(best_points, max_iter=2000, time_limit=10.0)
    
    return best_points
# EVOLVE-BLOCK-END
