"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.spatial import ConvexHull, Delaunay
from scipy.optimize import minimize

N = 130

# EVOLVE-BLOCK-START

def _covering_radius(points):
    """Compute the covering radius in degrees for a set of points on the unit sphere."""
    if len(points) < 3:
        return 180.0
    # Normalize
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    points = points / norms
    
    # Check if all points are in a closed hemisphere
    # If so, the covering radius is at least 90 degrees (actually could be up to 180)
    # Simple check: if there exists a vector v such that v dot p_i >= 0 for all i
    # This is a linear programming problem, but we can do a quick check
    # Actually, for 130 points, they should not be in a hemisphere if well-distributed
    
    try:
        hull = ConvexHull(points)
    except Exception:
        return 180.0
    
    # The covering radius is the maximum over all facets of the circumradius
    # For a spherical cap defined by a facet, the center of the cap is the
    # normalized normal vector of the facet, and the radius is the angular distance
    # from that center to the farthest vertex of the facet.
    
    # Get the facets (simplices) of the convex hull
    # Each facet is a set of indices
    max_circumradius = 0.0
    
    for simplex in hull.simplices:
        # Get the points defining this facet
        pts = points[simplex]
        n = len(pts)
        
        if n == 2:
            # Edge case - shouldn't happen for 3D convex hull with >= 4 points
            continue
        
        # Compute the normal to the facet
        # For a triangle (n=3):
        if n == 3:
            v1 = pts[1] - pts[0]
            v2 = pts[2] - pts[0]
            normal = np.cross(v1, v2)
        elif n == 4:
            # Tetrahedron - use the first 3 points
            v1 = pts[1] - pts[0]
            v2 = pts[2] - pts[0]
            normal = np.cross(v1, v2)
        else:
            # Use first 3 points
            v1 = pts[1] - pts[0]
            v2 = pts[2] - pts[0]
            normal = np.cross(v1, v2)
        
        norm_normal = np.linalg.norm(normal)
        if norm_normal < 1e-10:
            continue
        normal = normal / norm_normal
        
        # The center of the spherical cap is the normal vector (pointing outward)
        # The circumradius is the max angular distance from normal to any vertex
        cos_angles = np.dot(pts, normal)
        # Clamp for numerical stability
        cos_angles = np.clip(cos_angles, -1.0, 1.0)
        angles = np.arccos(cos_angles)
        
        circumradius = np.max(angles)
        max_circumradius = max(max_circumradius, circumradius)
    
    return np.degrees(max_circumradius)


def _soft_covering_radius(points, temperature=0.05):
    """Compute a smooth surrogate for the covering radius using soft-max."""
    if len(points) < 3:
        return 180.0
    
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    points = points / norms
    
    try:
        hull = ConvexHull(points)
    except Exception:
        return 180.0
    
    max_logsumexp = -np.inf
    
    for simplex in hull.simplices:
        pts = points[simplex]
        n = len(pts)
        
        if n < 3:
            continue
        
        v1 = pts[1] - pts[0]
        v2 = pts[2] - pts[0]
        normal = np.cross(v1, v2)
        
        norm_normal = np.linalg.norm(normal)
        if norm_normal < 1e-10:
            continue
        normal = normal / norm_normal
        
        cos_angles = np.dot(pts, normal)
        cos_angles = np.clip(cos_angles, -1.0, 1.0)
        angles = np.arccos(cos_angles)
        
        # Soft max of angles
        max_angle = np.max(angles)
        shifted = angles - max_angle
        logsumexp = max_angle + np.log(np.sum(np.exp(shifted / temperature)) * temperature)
        
        max_logsumexp = max(max_logsumexp, logsumexp)
    
    return np.degrees(max_logsumexp)


def _gradient_soft_covering(points, temperature=0.05):
    """Compute gradient of the soft covering radius with respect to points."""
    M = len(points)
    grad = np.zeros_like(points)
    
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    points = points / norms
    
    try:
        hull = ConvexHull(points)
    except Exception:
        return grad
    
    for simplex in hull.simplices:
        pts = points[simplex]
        n = len(pts)
        
        if n < 3:
            continue
        
        v1 = pts[1] - pts[0]
        v2 = pts[2] - pts[0]
        normal = np.cross(v1, v2)
        
        norm_normal = np.linalg.norm(normal)
        if norm_normal < 1e-10:
            continue
        normal = normal / norm_normal
        
        cos_angles = np.dot(pts, normal)
        cos_angles = np.clip(cos_angles, -1.0, 1.0)
        angles = np.arccos(cos_angles)
        
        # Soft max weights
        max_angle = np.max(angles)
        shifted = angles - max_angle
        weights = np.exp(shifted / temperature)
        weights = weights / np.sum(weights)
        
        # The gradient of arccos(cos_angle) w.r.t. point p_i is:
        # d/dp_i arccos(p_i . normal) = -normal / sin(angle_i)
        # But we need to be careful: the normal itself depends on the points.
        # For simplicity, we'll use the chain rule approximation.
        
        for k in range(n):
            idx = simplex[k]
            sin_angle = np.sin(angles[k])
            if sin_angle < 1e-10:
                sin_angle = 1e-10
            # Gradient contribution from this facet
            grad[idx] += weights[k] * (-normal / sin_angle)
    
    # Project gradient onto tangent space
    for i in range(M):
        grad[i] -= np.dot(grad[i], points[i]) * points[i]
    
    return grad


def _project_to_sphere(points):
    """Project points back to the unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return points / norms


def _refine_local(points, max_iter=500, tol=1e-10):
    """Local refinement using gradient descent on the sphere."""
    points = _project_to_sphere(points.copy())
    
    best_points = points.copy()
    best_radius = _covering_radius(points)
    
    temperature = 0.1
    lr = 0.01
    
    for it in range(max_iter):
        grad = _gradient_soft_covering(points, temperature)
        
        # Gradient descent step
        points_new = points - lr * grad
        points_new = _project_to_sphere(points_new)
        
        # Check for improvement
        radius_new = _covering_radius(points_new)
        
        if radius_new < best_radius - tol:
            best_radius = radius_new
            best_points = points_new.copy()
            points = points_new
            # Decrease learning rate slowly
            lr = max(0.001, lr * 0.99)
        else:
            # Try smaller step
            lr *= 0.5
            if lr < 1e-6:
                break
    
    return best_points, best_radius


def _perturb(points, scale=0.1):
    """Apply random perturbations to points."""
    noise = np.random.randn(*points.shape) * scale
    points_new = points + noise
    return _project_to_sphere(points_new)


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Initial Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Initial refinement
    points, best_radius = _refine_local(points, max_iter=200)
    
    # Multi-start local search
    n_restarts = 10
    for restart in range(n_restarts):
        if time.time() > deadline - 30.0:
            break
        
        # Perturb the current best
        scale = 0.1 * (1 - restart / n_restarts) + 0.01
        perturbed = _perturb(best_points if 'best_points' in dir() else points, scale)
        
        # Refine
        refined, radius = _refine_local(perturbed, max_iter=300)
        
        if radius < best_radius:
            best_radius = radius
            points = refined
    
    # Final high-precision polish
    if time.time() < deadline - 10.0:
        points, best_radius = _refine_local(points, max_iter=1000, tol=1e-12)
    
    # Ensure valid output
    points = _project_to_sphere(points)
    
    return points

# EVOLVE-BLOCK-END
