"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist


def _compute_radius(points):
    """Compute the maximum common radius for given points."""
    n = len(points)
    if n == 0:
        return 0.0
    # Distance to boundary
    dist_boundary = 1.0 - np.linalg.norm(points, axis=1)
    min_boundary = np.min(dist_boundary)
    if min_boundary <= 0:
        return 0.0
    # Pairwise distances
    d = cdist(points, points)
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d) / 2.0
    return min(min_boundary, min_pair)


def _objective(x, N):
    """Objective function to minimize (negative radius)."""
    points = x.reshape(-1, 2)
    # Soft penalty for points outside the unit circle
    r = np.linalg.norm(points, axis=1)
    outside_penalty = np.sum(np.maximum(0, r - 1.0) ** 2)
    
    # Compute actual radius
    min_boundary = 1.0 - np.min(r)
    d = cdist(points, points)
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d) / 2.0
    radius = min(min_boundary, min_pair)
    
    # Return negative radius plus penalty
    return -radius + 100.0 * outside_penalty


def construct():
    """Construct centers of 300 equal circles in the unit disk."""
    # Start with a good initial configuration: concentric rings
    np.random.seed(42)
    
    # Use a spiral/ring-based initialization
    points = np.zeros((N, 2))
    
    # Place points in concentric rings
    ring_radii = np.linspace(0.05, 0.95, 20)
    per_ring = [int(N / len(ring_radii))] * len(ring_radii)
    # Adjust last ring
    per_ring[-1] = N - sum(per_ring[:-1])
    
    idx = 0
    for i, (r, count) in enumerate(zip(ring_radii, per_ring)):
        if count <= 0:
            continue
        angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
        # Add small random perturbation
        angles += np.random.uniform(-0.01, 0.01, count)
        points[idx:idx+count, 0] = r * np.cos(angles)
        points[idx:idx+count, 1] = r * np.sin(angles)
        idx += count
    
    # Ensure exactly N points
    if len(points) > N:
        points = points[:N]
    elif len(points) < N:
        extra = np.random.uniform(-0.9, 0.9, (N - len(points), 2))
        points = np.vstack([points, extra])
    
    # Optimize using L-BFGS-B
    best_points = points.copy()
    best_radius = _compute_radius(best_points)
    
    # Multiple restarts with different initializations
    for restart in range(5):
        if restart > 0:
            # Perturb the best solution found so far
            x0 = best_points.copy()
            x0 += np.random.normal(0, 0.01, x0.shape)
        else:
            x0 = points.copy()
        
        result = minimize(
            _objective,
            x0.flatten(),
            method='L-BFGS-B',
            args=(N,),
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        new_points = result.x.reshape(-1, 2)
        new_radius = _compute_radius(new_points)
        
        if new_radius > best_radius:
            best_radius = new_radius
            best_points = new_points.copy()
    
    # Final check: ensure all points are strictly inside
    r = np.linalg.norm(best_points, axis=1)
    mask = r < 1.0
    if not np.all(mask):
        # Project points back inside
        best_points[~mask] *= (0.99 / r[~mask, np.newaxis])
    
    return best_points
# EVOLVE-BLOCK-END
