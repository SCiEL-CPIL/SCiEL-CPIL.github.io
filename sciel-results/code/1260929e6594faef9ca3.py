"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time


def _normalize(pts):
    """Normalize points to unit sphere, handling 1D/2D."""
    pts = np.asarray(pts, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return pts / norms


def _covering_radius(pts):
    """
    Compute the covering radius of a set of points on the unit sphere.
    Uses the convex hull approach: the covering radius is the maximum circumradius
    of the spherical Voronoi cells, which equals the maximum over all facets of the
    convex hull of the angular distance from the facet center to the farthest vertex.
    
    For efficiency, we use a sampling-based approach: sample many points on the sphere
    and find the maximum distance to the nearest point. We also use the convex hull
    to get a tighter bound.
    """
    pts = _normalize(pts)
    n = len(pts)
    
    # Method 1: Sample-based estimate (fast, approximate)
    # Use a large number of sample points on the sphere
    n_samples = 200000
    # Generate random points on the sphere
    rng = np.random.RandomState(42)
    samples = rng.randn(n_samples, 3)
    samples /= np.linalg.norm(samples, axis=1, keepdims=True)
    
    # Compute distances from each sample to each point
    # Use dot product: cos(theta) = dot product
    dots = samples @ pts.T  # (n_samples, n)
    dots = np.clip(dots, -1.0, 1.0)
    max_dots = np.max(dots, axis=1)  # (n_samples,)
    min_angles = np.arccos(max_dots)  # (n_samples,)
    sample_radius = np.max(min_angles)
    
    # Method 2: Use convex hull for tighter estimate
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(pts)
        # For each facet, compute the circumradius
        max_circumradius = 0.0
        for simplex in hull.simplices:
            face_pts = pts[simplex]
            # Find the point on the sphere farthest from this face
            # The farthest point from a plane through the origin in direction n
            # is in the direction of the face normal
            # Compute face normal
            v1 = face_pts[1] - face_pts[0]
            v2 = face_pts[2] - face_pts[0]
            normal = np.cross(v1, v2)
            norm = np.linalg.norm(normal)
            if norm < 1e-10:
                continue
            normal /= norm
            # The angular distance from the face center to the farthest vertex
            # is the circumradius of the spherical triangle
            # Compute distances from normal to each vertex
            dots_face = np.dot(face_pts, normal)
            dots_face = np.clip(dots_face, -1.0, 1.0)
            angles = np.arccos(dots_face)
            max_angle = np.max(angles)
            max_circumradius = max(max_circumradius, max_angle)
        
        # The covering radius is the maximum of the sample-based and hull-based estimates
        # But actually, the hull-based one should be more accurate
        # Use the maximum of both to be safe
        return max(sample_radius, max_circumradius)
    except Exception:
        # If hull fails, use sample-based
        return sample_radius


def _covering_radius_fast(pts):
    """
    Fast covering radius computation using only sampling.
    """
    pts = _normalize(pts)
    n = len(pts)
    
    # Use a large number of sample points on the sphere
    n_samples = 100000
    rng = np.random.RandomState(42)
    samples = rng.randn(n_samples, 3)
    samples /= np.linalg.norm(samples, axis=1, keepdims=True)
    
    # Compute distances from each sample to each point
    dots = samples @ pts.T  # (n_samples, n)
    dots = np.clip(dots, -1.0, 1.0)
    max_dots = np.max(dots, axis=1)  # (n_samples,)
    min_angles = np.arccos(max_dots)  # (n_samples,)
    sample_radius = np.max(min_angles)
    
    return sample_radius


def _local_optimize(pts, max_iter=5000, lr=0.01, time_limit=100.0):
    """
    Local optimization using gradient descent on the sphere.
    Objective: minimize the covering radius.
    We approximate the gradient by moving points away from their nearest neighbors'
    Voronoi cell boundaries.
    """
    start_time = time.time()
    pts = _normalize(pts).copy()
    n = len(pts)
    
    best_pts = pts.copy()
    best_radius = _covering_radius_fast(pts)
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Compute the covering radius and identify the worst point
        # For each point, compute the distance to its nearest neighbor
        dots = pts @ pts.T
        dots = np.clip(dots, -1.0, 1.0)
        np.fill_diagonal(dots, -1.0)
        nearest_dots = np.max(dots, axis=1)
        nearest_angles = np.arccos(nearest_dots)
        
        # Find the point with the largest nearest-neighbor distance
        worst_idx = np.argmax(nearest_angles)
        
        # Compute the gradient: move points away from each other
        # Use a simple repulsive force model
        forces = np.zeros_like(pts)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                # Force is proportional to 1/distance^2, pushing i away from j
                diff = pts[i] - pts[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-10:
                    continue
                # Spherical repulsion
                force_mag = 1.0 / (dist * dist + 0.1)
                forces[i] += force_mag * diff / dist
        
        # Also add a force to move the worst point toward the center of its Voronoi cell
        # Approximate by moving it toward the average of its nearest neighbors
        nearest_j = np.argmax(dots[worst_idx])
        forces[worst_idx] -= 0.5 * (pts[worst_idx] - pts[nearest_j])
        
        # Update points
        new_pts = pts + lr * forces
        new_pts = _normalize(new_pts)
        
        # Check if the new configuration is better
        new_radius = _covering_radius_fast(new_pts)
        if new_radius < best_radius:
            best_radius = new_radius
            best_pts = new_pts.copy()
            pts = new_pts
        else:
            # Reduce learning rate
            lr *= 0.95
    
    return best_pts, best_radius


def construct():
    """
    Construct 50 points on the unit sphere to minimize the covering radius.
    """
    start_time = time.time()
    time_limit = 140.0  # Leave 10s safety margin
    
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Local optimization
    pts, radius = _local_optimize(pts, max_iter=5000, lr=0.01, time_limit=time_limit - (time.time() - start_time))
    
    # Final normalization
    pts = _normalize(pts)
    
    return pts
# EVOLVE-BLOCK-END
