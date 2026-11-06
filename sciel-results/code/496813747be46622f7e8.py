"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, rosenbrock


def _fibonacci_spiral(n):
    """Generate initial Fibonacci spiral points on unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _normalize(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    return pts / norms


def _covering_radius(pts, n_samples=20000):
    """Compute the covering radius using sampling.
    
    The covering radius is the maximum over all points y on the sphere
    of the angular distance from y to the nearest configuration point.
    We approximate this by sampling many points.
    """
    n = len(pts)
    # Generate random points on the sphere
    theta = np.random.uniform(0, 2 * np.pi, n_samples)
    cos_phi = np.random.uniform(-1, 1, n_samples)
    sin_phi = np.sqrt(1.0 - cos_phi * cos_phi)
    sample_pts = np.column_stack([
        sin_phi * np.cos(theta),
        sin_phi * np.sin(theta),
        cos_phi
    ])
    
    # For each sample point, find the nearest configuration point
    dot_s = sample_pts @ pts.T
    dot_s = np.clip(dot_s, -1.0, 1.0)
    min_dist = np.arccos(np.min(dot_s, axis=1))
    return np.max(min_dist)


def _find_worst_point(pts, n_samples=50000):
    """Find the point on the sphere farthest from the nearest configuration point.
    
    This is the 'pole of inaccessibility' - the center of the largest empty cap.
    """
    n = len(pts)
    # Generate random points on the sphere
    theta = np.random.uniform(0, 2 * np.pi, n_samples)
    cos_phi = np.random.uniform(-1, 1, n_samples)
    sin_phi = np.sqrt(1.0 - cos_phi * cos_phi)
    sample_pts = np.column_stack([
        sin_phi * np.cos(theta),
        sin_phi * np.sin(theta),
        cos_phi
    ])
    
    # For each sample point, find the nearest configuration point
    dot_s = sample_pts @ pts.T
    dot_s = np.clip(dot_s, -1.0, 1.0)
    min_dist = np.arccos(np.min(dot_s, axis=1))
    worst_idx = np.argmax(min_dist)
    return sample_pts[worst_idx], min_dist[worst_idx]


def _repulsion_optimize(pts, max_iters=500, time_limit=140.0):
    """Optimize points using repulsion forces to minimize covering radius.
    
    Strategy: 
    1. Find the worst-covered point (pole of inaccessibility)
    2. Move the nearest point toward that location
    3. Also apply repulsion to push nearby points apart
    """
    start_time = time.time()
    pts = _normalize(pts.copy())
    best_pts = pts.copy()
    best_radius = _covering_radius(pts, n_samples=10000)
    
    for it in range(max_iters):
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            break
        
        # Find the worst-covered point
        worst_pt, worst_dist = _find_worst_point(pts, n_samples=20000)
        
        # Find the nearest configuration point to the worst point
        dot_worst = pts @ worst_pt
        nearest_idx = np.argmax(dot_worst)
        
        # Move the nearest point toward the worst point
        # Use a small step size to avoid oscillation
        step_size = 0.02
        pts[nearest_idx] = pts[nearest_idx] + step_size * (worst_pt - pts[nearest_idx])
        pts[nearest_idx] = pts[nearest_idx] / np.linalg.norm(pts[nearest_idx])
        
        # Apply repulsion: for each point, push it away from its neighbors
        # This helps fill in gaps more evenly
        for i in range(N):
            # Find k nearest neighbors
            dot_i = pts @ pts[i]
            dot_i[i] = -2.0  # exclude self
            neighbor_idx = np.argsort(dot_i)[::-1][:5]  # 5 nearest neighbors
            
            # Compute repulsion force
            force = np.zeros(3)
            for j in neighbor_idx:
                if j == i:
                    continue
                # Push away from neighbor
                diff = pts[i] - pts[j]
                dist = np.linalg.norm(diff)
                if dist > 1e-10:
                    force += diff / dist**2  # inverse square repulsion
            
            # Apply force
            pts[i] = pts[i] + 0.005 * force
            pts[i] = pts[i] / np.linalg.norm(pts[i])
        
        # Check if we improved
        current_radius = _covering_radius(pts, n_samples=5000)
        if current_radius < best_radius:
            best_radius = current_radius
            best_pts = pts.copy()
    
    return best_pts


def construct():
    """Construct 130 points on the unit sphere minimizing the covering radius."""
    start_time = time.time()
    
    # Start with Fibonacci spiral
    pts = _fibonacci_spiral(N)
    
    # Apply repulsion-based optimization
    pts = _repulsion_optimize(pts, max_iters=300, time_limit=130.0)
    
    # Final normalization
    pts = _normalize(pts)
    
    return pts
# EVOLVE-BLOCK-END

