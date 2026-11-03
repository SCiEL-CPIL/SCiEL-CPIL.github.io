"""Program for covering sphere with 130 equal caps."""
import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130

def _normalize(v):
    """Normalize vector to unit length."""
    norm = np.linalg.norm(v)
    if norm < 1e-15:
        return v
    return v / norm

def _covering_radius(pts):
    """Compute the covering radius in degrees."""
    try:
        hull = ConvexHull(pts)
        max_radius = 0.0
        for simplex in hull.simplices:
            # Get the three points
            p1, p2, p3 = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
            # Compute circumcenter on the sphere
            # For three points on unit sphere, circumcenter is normalized cross product
            # Actually, we need to find point equidistant from all three
            # Use the fact that for unit vectors, circumcenter direction is:
            # (p1 x p2) x (p2 x p3) + ... or use linear algebra
            # Simpler: solve for u such that u·p1 = u·p2 = u·p3
            # This is equivalent to u perpendicular to (p1-p2) and (p2-p3)
            v1 = p2 - p1
            v2 = p3 - p1
            # Circumcenter direction is cross(v1, v2)
            u_dir = np.cross(v1, v2)
            u = _normalize(u_dir)
            # The circumradius is the angle between u and any of the points
            cos_r = np.dot(u, p1)
            cos_r = np.clip(cos_r, -1.0, 1.0)
            radius = np.degrees(np.arccos(cos_r))
            max_radius = max(max_radius, radius)
        return max_radius
    except Exception:
        return 90.0  # Default to hemisphere if hull fails

def _compute_gradients(pts, hull):
    """Compute gradient of covering radius for each point."""
    n = len(pts)
    grads = np.zeros((n, 3))
    max_radius = 0.0
    worst_simplex = None
    
    # Find worst facet and compute gradient
    for simplex in hull.simplices:
        p1, p2, p3 = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
        v1 = p2 - p1
        v2 = p3 - p1
        u_dir = np.cross(v1, v2)
        u = _normalize(u_dir)
        cos_r = np.clip(np.dot(u, p1), -1.0, 1.0)
        radius = np.degrees(np.arccos(cos_r))
        
        if radius > max_radius:
            max_radius = radius
            worst_simplex = simplex
            # Gradient: move each point towards circumcenter u
            # For p_i, gradient of -cos(theta) is -sin(theta) * (u - (u·p_i)*p_i) / |u - (u·p_i)*p_i|
            # Simpler: just move towards u
            for idx, p_idx in enumerate([simplex[0], simplex[1], simplex[2]]):
                p = pts[p_idx]
                # Direction towards u
                direction = u - np.dot(u, p) * p
                direction = _normalize(direction)
                grads[p_idx] += direction
    
    if worst_simplex is not None:
        # Scale gradient by max radius to have proper magnitude
        scale = max_radius
        grads *= scale
    return grads, max_radius

def _local_optimize(pts, max_time=60.0):
    """Local optimization using gradient descent with momentum."""
    start_time = time.time()
    pts = pts.copy()
    momentum = np.zeros_like(pts)
    best_radius = _covering_radius(pts)
    best_pts = pts.copy()
    
    step_size = 0.01
    decay = 0.999
    momentum_coef = 0.9
    
    for iteration in range(10000):
        # Check time
        if time.time() - start_time > max_time - 5.0:
            break
        
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
        
        grads, current_radius = _compute_gradients(pts, hull)
        
        # Update with momentum
        momentum = momentum_coef * momentum + grads
        pts = pts - step_size * momentum
        pts = np.array([_normalize(p) for p in pts])
        
        # Check for invalid configurations
        if not np.all(np.isfinite(pts)):
            break
        
        # Compute current radius
        current_radius = _covering_radius(pts)
        if current_radius < best_radius:
            best_radius = current_radius
            best_pts = pts.copy()
        
        # Decay step size
        step_size *= decay
        if step_size < 1e-6:
            break
    
    return best_pts, best_radius

def _fibonacci_spiral(n, seed=0):
    """Generate Fibonacci spiral points on unit sphere."""
    rng = np.random.RandomState(seed)
    # Add small perturbation for variety
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    # Small random perturbation
    pts += rng.normal(0, 0.01, size=(n, 3))
    pts = np.array([_normalize(p) for p in pts])
    return pts

def _random_initial(n, seed=0):
    """Generate random initial points on unit sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.normal(0, 1, size=(n, 3))
    pts = np.array([_normalize(p) for p in pts])
    return pts

def construct():
    """Main function: construct 130 points on unit sphere minimizing covering radius."""
    global_start = time.time()
    deadline = 150.0  # Total time limit
    
    best_pts = None
    best_radius = float('inf')
    
    # Strategy 1: Fibonacci spiral with different seeds
    for seed in range(5):
        if time.time() - global_start > deadline - 30:
            break
        pts = _fibonacci_spiral(N, seed)
        pts, radius = _local_optimize(pts, max_time=deadline - (time.time() - global_start) - 5)
        if radius < best_radius:
            best_radius = radius
            best_pts = pts.copy()
    
    # Strategy 2: Random initializations
    for seed in range(5):
        if time.time() - global_start > deadline - 30:
            break
        pts = _random_initial(N, seed)
        pts, radius = _local_optimize(pts, max_time=deadline - (time.time() - global_start) - 5)
        if radius < best_radius:
            best_radius = radius
            best_pts = pts.copy()
    
    # Final polish with more iterations if time allows
    if best_pts is not None and time.time() - global_start < deadline - 10:
        remaining_time = deadline - (time.time() - global_start) - 5
        if remaining_time > 5:
            best_pts, final_radius = _local_optimize(best_pts, max_time=remaining_time)
            if final_radius < best_radius:
                best_radius = final_radius
    
    # Ensure validity
    if best_pts is None:
        # Fallback: Fibonacci spiral
        best_pts = _fibonacci_spiral(N, 0)
    
    # Final normalization and validation
    best_pts = np.array([_normalize(p) for p in best_pts])
    
    # Check for coincident points and hemisphere issues
    # If invalid, use fallback
    if not np.all(np.isfinite(best_pts)) or len(set(map(tuple, best_pts.round(10)))) < N:
        best_pts = _fibonacci_spiral(N, 0)
    
    return best_pts

# EVOLVE-BLOCK-END
