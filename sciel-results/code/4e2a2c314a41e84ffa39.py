"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.spatial import ConvexHull

N = 50
SEED_TIME_LIMIT = 140.0  # seconds, safety margin before 150s kill

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
import time


def _normalize(v):
    """Normalize a set of 3D vectors to the unit sphere."""
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return v / norms


def _compute_covering_radius(points):
    """
    Compute the covering radius in degrees.
    The covering radius is the maximum angular distance from any point on the
    sphere to its nearest configuration point. This equals the maximum
    circumradius of the facets of the convex hull (for points in general
    position on the sphere, the farthest point from the set lies on a facet).
    
    For a triangle (a, b, c) on the unit sphere, the circumradius (angular)
    is computed via the spherical excess or via the formula:
    cos(theta) = (a·b + b·c + c·a - 1) / (2 * (a·b * b·c * c·a - 1))
    Actually, a cleaner approach: the circumcenter of a spherical triangle
    is in the plane spanned by a, b, c. The angular circumradius R satisfies:
    The distance from the circumcenter to each vertex is R.
    
    A robust method: for a spherical triangle with sides a, b, c (angular),
    the circumradius R satisfies:
    sin(R) = sin(a) * sin(b) * sin(c) / (2 * area)  ... not quite.
    
    Let's use the formula: for a triangle with vertices on the unit sphere,
    the circumradius R (angular) is given by:
    cos(R) = (cos(a) + cos(b) + cos(c) - 1) / (2 * (cos(a)*cos(b) + cos(b)*cos(c) + cos(c)*cos(a) - 1))
    where a, b, c are the side lengths (angular distances between pairs).
    
    Actually, the most robust approach is to find the point on the sphere
    equidistant from a, b, c. This is the spherical circumcenter.
    It lies in the direction of the cross-product-based normal to the plane
    containing a, b, c (if the triangle doesn't contain the origin in its
    convex hull in R^3, which it won't for small caps).
    
    The circumcenter direction is proportional to:
    a * |b|^2 * |c|^2 * ... 
    Actually, for points on the unit sphere, the circumcenter is in the
    direction of the vector that is equidistant from all three.
    
    Let's use: the circumcenter u satisfies u·a = u·b = u·c = cos(R).
    u is in the plane of a, b, c. We can write u = alpha*a + beta*b + gamma*c
    and solve the linear system.
    
    A simpler formula: 
    Let A = a, B = b, C = c.
    The circumcenter is proportional to:
    (A * |B×C|^2 + B * |C×A|^2 + C * |A×B|^2)  ... no, that's for circumcenter in 3D.
    
    For the unit sphere, the spherical circumcenter direction is:
    n = (A × B) × (B × C)  ... not quite.
    
    Let's use the standard approach: solve for u in span(A, B, C) such that
    u·A = u·B = u·C.
    This gives:
    u·(A-B) = 0, u·(B-C) = 0
    So u is perpendicular to (A-B) and (B-C), hence u is in the direction of
    (A-B) × (B-C).
    
    Wait, that's the normal to the plane containing A, B, C. But the circumcenter
    on the sphere is not necessarily in that direction unless the triangle is
    "balanced". 
    
    Actually, for three points on the unit sphere, the set of points equidistant
    from all three is a line (the intersection of two planes: u·(A-B)=0 and
    u·(B-C)=0). This line passes through the origin and is in the direction
    of (A-B) × (B-C). So the circumcenter on the sphere is ±((A-B)×(B-C))/|...|.
    
    But we need the one that is on the same side as the triangle (the cap center).
    The correct sign is the one where u·A > 0 (for small caps).
    
    Then R = arccos(u·A).
    """
    hull = ConvexHull(points)
    max_radius = 0.0
    for simplex in hull.simplices:
        a = points[simplex[0]]
        b = points[simplex[1]]
        c = points[simplex[2]]
        
        # Compute circumcenter direction
        d1 = a - b
        d2 = b - c
        n = np.cross(d1, d2)
        n_norm = np.linalg.norm(n)
        
        if n_norm < 1e-15:
            # Degenerate triangle, skip
            continue
        
        n = n / n_norm
        
        # Choose sign: the circumcenter should be on the same side as the
        # facet's outward normal, which for a convex hull facet is the
        # direction away from the origin. Since points are on the unit sphere,
        # the facet center (a+b+c)/3 points outward.
        # The circumcenter should satisfy u·(a+b+c) > 0
        if np.dot(n, a + b + c) < 0:
            n = -n
        
        # Angular circumradius
        cos_R = np.clip(np.dot(n, a), -1.0, 1.0)
        R = np.arccos(cos_R)
        
        if R > max_radius:
            max_radius = R
    
    return np.degrees(max_radius)


def _spherical_circumcenter(a, b, c):
    """
    Return the spherical circumcenter of triangle (a, b, c) on the unit sphere,
    and its angular circumradius.
    """
    d1 = a - b
    d2 = b - c
    n = np.cross(d1, d2)
    n_norm = np.linalg.norm(n)
    
    if n_norm < 1e-15:
        return None, 0.0
    
    n = n / n_norm
    
    # Choose sign so that circumcenter is on the "outside" of the triangle
    # (same hemisphere as the facet)
    if np.dot(n, a + b + c) < 0:
        n = -n
    
    cos_R = np.clip(np.dot(n, a), -1.0, 1.0)
    R = np.arccos(cos_R)
    
    return n, R


def _worst_facet(points):
    """
    Find the convex hull facet with the largest spherical circumradius.
    Returns (index_of_worst_facet, circumcenter, circumradius, simplex_indices)
    """
    hull = ConvexHull(points)
    max_R = 0.0
    worst = None
    worst_center = None
    worst_simplex = None
    
    for i, simplex in enumerate(hull.simplices):
        a = points[simplex[0]]
        b = points[simplex[1]]
        c = points[simplex[2]]
        
        d1 = a - b
        d2 = b - c
        n = np.cross(d1, d2)
        n_norm = np.linalg.norm(n)
        
        if n_norm < 1e-15:
            continue
        
        n = n / n_norm
        if np.dot(n, a + b + c) < 0:
            n = -n
        
        cos_R = np.clip(np.dot(n, a), -1.0, 1.0)
        R = np.arccos(cos_R)
        
        if R > max_R:
            max_R = R
            worst = i
            worst_center = n
            worst_simplex = simplex
    
    return worst, worst_center, max_R, worst_simplex


def _move_toward_circumcenter(points, simplex, center, step):
    """
    Move the three vertices of the given simplex toward the circumcenter
    along great circles by the given step (in radians).
    """
    new_points = points.copy()
    for idx in simplex:
        p = points[idx]
        # Move p toward center by step radians along the great circle
        # The direction from p to center on the sphere
        # We use spherical linear interpolation (slerp)
        cos_angle = np.clip(np.dot(p, center), -1.0, 1.0)
        angle = np.arccos(cos_angle)
        
        if angle < 1e-15:
            continue
        
        # slerp: p' = (sin((1-t)*angle)/sin(angle))*p + (sin(t*angle)/sin(angle))*center
        t = step / angle
        t = np.clip(t, 0.0, 1.0)
        
        sin_angle = np.sin(angle)
        if sin_angle < 1e-15:
            continue
        
        coeff_p = np.sin((1 - t) * angle) / sin_angle
        coeff_c = np.sin(t * angle) / sin_angle
        
        new_points[idx] = coeff_p * p + coeff_c * center
        new_points[idx] = new_points[idx] / np.linalg.norm(new_points[idx])
    
    return new_points


def construct():
    start_time = time.time()
    
    # Initial configuration: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    points = _normalize(points)
    
    best_points = points.copy()
    best_radius = _compute_covering_radius(points)
    
    # Iterative worst-facet circumcenter projection
    step = 0.3  # initial step size (radians)
    min_step = 1e-6
    max_iterations = 2000
    
    for iteration in range(max_iterations):
        if time.time() - start_time > SEED_TIME_LIMIT:
            break
        
        worst_idx, center, R, simplex = _worst_facet(points)
        
        if worst_idx is None:
            break
        
        # Update best
        if R < best_radius:
            best_radius = R
            best_points = points.copy()
        
        # If the worst facet is very small, we're close to optimal
        if R < 0.001:
            break
        
        # Move vertices toward circumcenter
        new_points = _move_toward_circumcenter(points, simplex, center, step)
        
        # Check if this improves the covering radius
        new_radius = _compute_covering_radius(new_points)
        
        if new_radius < R - 1e-10:
            points = new_points
            # Slightly reduce step for convergence
            step = max(step * 0.95, min_step)
        else:
            # Reduce step and try again
            step = max(step * 0.5, min_step)
            if step <= min_step:
                break
    
    # Final polish: use the best points found
    # Do a final high-precision check
    if time.time() - start_time < SEED_TIME_LIMIT - 5:
        final_radius = _compute_covering_radius(best_points)
        if final_radius < best_radius:
            best_radius = final_radius
    
    return best_points
# EVOLVE-BLOCK-END
