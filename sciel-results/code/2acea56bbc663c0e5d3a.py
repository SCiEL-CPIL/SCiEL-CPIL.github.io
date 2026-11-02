"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

# Deadline guard: keep a safety margin before the 150s kill
_START_TIME = time.time()
_DEADLINE = 125.0  # seconds, leaves 25s safety margin

def _elapsed():
    return time.time() - _START_TIME

def _check_deadline():
    return _elapsed() > _DEADLINE

def _normalize(points):
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return points / norms

def _covering_radius(points):
    """Compute the covering radius in radians using convex hull facets."""
    points = _normalize(points)
    # Add origin to ensure convex hull is well-defined
    # Actually, for points on the sphere, the convex hull facets correspond to
    # the spherical Voronoi cells. The covering radius is the max over facets
    # of the angular distance from the facet center to the farthest vertex.
    # 
    # A more robust approach: the covering radius equals the maximum over all
    # points y on the sphere of the angular distance to the nearest point.
    # This is achieved at the circumcenters of the facets of the convex hull
    # of the points (when viewed from the origin).
    #
    # For each facet (triangle of 3 points), the circumcenter in 3D is the
    # point equidistant from all 3 vertices. The angular distance from this
    # circumcenter to any of the 3 vertices is the circumradius of that facet.
    # The covering radius is the maximum of these circumradii.
    
    try:
        hull = ConvexHull(points)
    except Exception:
        # Fallback: brute force on a fine grid
        return _covering_radius_grid(points)
    
    max_circumradius = 0.0
    for simplex in hull.simplices:
        # simplex contains indices of 3 points forming a triangle
        p1, p2, p3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
        
        # The circumcenter of the triangle on the unit sphere is the normalized
        # cross product... no, it's the point on the sphere equidistant from all 3.
        # For 3 points on the unit sphere, the circumcenter (in 3D) lies along
        # the line from the origin through the centroid of the triangle, but only
        # if the triangle is "small". More precisely, the circumcenter C satisfies
        # |C - p1| = |C - p2| = |C - p3| and C is on the unit sphere.
        #
        # Actually, the covering radius for a facet is the angular distance from
        # the circumcenter of the spherical triangle to its vertices. The circumcenter
        # of a spherical triangle can be found as follows:
        # The plane containing the 3 points has equation n · x = d where n is the
        # normal. The circumcenter in 3D (not on the sphere) is the intersection of
        # the plane with the line from the origin through the circumcenter of the
        # planar triangle.
        #
        # Simpler: the circumradius of the spherical triangle is the angular distance
        # from the circumcenter to a vertex. For a triangle with side lengths a, b, c
        # (angular distances), the circumradius R satisfies:
        # sin(R) = a * b * c / (4 * area)  where area is the spherical excess... 
        # This is getting complex. Let me use a different approach.
        #
        # The circumcenter in 3D: solve for point C on the unit sphere such that
        # C·p1 = C·p2 = C·p3 (equal angular distances means equal dot products).
        # So C is in the null space of [p1 - p2; p2 - p3] intersected with the unit sphere.
        
        # Solve (p1 - p2) · C = 0 and (p2 - p3) · C = 0
        A = np.array([p1 - p2, p2 - p3])
        # Find null space vector
        _, _, Vt = np.linalg.svd(A)
        c = Vt[-1]
        c = c / np.linalg.norm(c)
        
        # The angular distance from c to p1
        dot_val = np.clip(np.dot(c, p1), -1.0, 1.0)
        circumradius = np.arccos(dot_val)
        
        if circumradius > max_circumradius:
            max_circumradius = circumradius
    
    return max_circumradius

def _covering_radius_grid(points):
    """Fallback: estimate covering radius on a fine icosahedral grid."""
    points = _normalize(points)
    # Use a moderately fine grid
    # Generate points on a grid
    n = 60
    max_dist = 0.0
    for i in range(n + 1):
        theta = np.pi * i / n
        sin_theta = np.sin(theta)
        for j in range(int(2 * np.pi * sin_theta * n / np.pi) + 1):
            if sin_theta < 1e-10:
                phi = 0.0
            else:
                phi = 2 * np.pi * j / (int(2 * np.pi * sin_theta * n / np.pi) + 1)
            y = np.array([sin_theta * np.cos(phi), sin_theta * np.sin(phi), np.cos(theta)])
            # Find nearest point
            dots = points @ y
            max_dot = np.max(dots)
            angular_dist = np.arccos(np.clip(max_dot, -1.0, 1.0))
            if angular_dist > max_dist:
                max_dist = angular_dist
    return max_dist

def _cvt_step(points, n_iter=50):
    """One round of CVT: move each point to the centroid of its Voronoi cell (facet)."""
    points = _normalize(points)
    for _ in range(n_iter):
        if _check_deadline():
            break
        try:
            hull = ConvexHull(points)
        except Exception:
            break
        
        # For each point, find the facets that include it, and compute the
        # centroid of the corresponding spherical Voronoi cell.
        # The Voronoi cell of point i is the set of facets adjacent to i.
        # The centroid of the cell is the average of the circumcenters of those facets.
        
        # Build mapping: point index -> list of simplex indices containing it
        point_to_simplex = [[] for _ in range(len(points))]
        for si, simplex in enumerate(hull.simplices):
            for idx in simplex:
                point_to_simplex[idx].append(si)
        
        new_points = np.zeros_like(points)
        for i in range(len(points)):
            if len(point_to_simplex[i]) == 0:
                new_points[i] = points[i]
                continue
            
            # Compute circumcenters of all facets containing point i
            centers = []
            for si in point_to_simplex[i]:
                simplex = hull.simplices[si]
                p1, p2, p3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
                A = np.array([p1 - p2, p2 - p3])
                _, _, Vt = np.linalg.svd(A)
                c = Vt[-1]
                c = c / np.linalg.norm(c)
                centers.append(c)
            
            # Average the centers and normalize
            avg_center = np.mean(centers, axis=0)
            norm = np.linalg.norm(avg_center)
            if norm > 1e-15:
                new_points[i] = avg_center / norm
            else:
                new_points[i] = points[i]
        
        new_points = _normalize(new_points)
        points = new_points
    
    return points

def _gradient_descent(points, n_iter=200, lr=0.01):
    """Gradient descent on the smoothed covering radius objective."""
    points = _normalize(points)
    
    for it in range(n_iter):
        if _check_deadline():
            break
        
        # Compute the covering radius and identify the "worst" facet
        try:
            hull = ConvexHull(points)
        except Exception:
            break
        
        # For each facet, compute the circumcenter and circumradius
        facet_centers = []
        facet_radii = []
        for simplex in hull.simplices:
            p1, p2, p3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
            A = np.array([p1 - p2, p2 - p3])
            _, _, Vt = np.linalg.svd(A)
            c = Vt[-1]
            c = c / np.linalg.norm(c)
            dot_val = np.clip(np.dot(c, p1), -1.0, 1.0)
            radius = np.arccos(dot_val)
            facet_centers.append(c)
            facet_radii.append(radius)
        
        facet_radii = np.array(facet_radii)
        facet_centers = np.array(facet_centers)
        
        # Use softmax weighting to create a smooth surrogate
        # Weight each facet by exp(radius / tau)
        tau = 0.01
        weights = np.exp(facet_radii / tau - np.max(facet_radii) / tau)
        weights /= weights.sum()
        
        # The "worst" point is a weighted average of the facet centers
        # The gradient of the covering radius w.r.t. each point is approximately
        # the negative of the weighted sum of the vectors from the point to the
        # facet centers that contain it.
        
        # For each point, accumulate the gradient
        grad = np.zeros_like(points)
        for si, simplex in enumerate(hull.simplices):
            w = weights[si]
            c = facet_centers[si]
            for idx in simplex:
                # The point should move toward the circumcenter
                grad[idx] += w * (c - points[idx])
        
        # Normalize gradient and take a step
        grad_norm = np.linalg.norm(grad, axis=1, keepdims=True)
        grad_norm[grad_norm < 1e-15] = 1.0
        grad = grad / grad_norm
        
        # Project gradient to the tangent space (remove radial component)
        for i in range(len(points)):
            radial = np.dot(grad[i], points[i])
            grad[i] -= radial * points[i]
        
        points = points + lr * grad
        points = _normalize(points)
    
    return points

def construct():
    global _START_TIME, _DEADLINE
    _START_TIME = time.time()
    _DEADLINE = 125.0
    
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    points = _normalize(points)
    
    best_points = points.copy()
    best_radius = _covering_radius(points)
    
    # Phase 1: CVT iterations
    if not _check_deadline():
        points = _cvt_step(points, n_iter=100)
        radius = _covering_radius(points)
        if radius < best_radius:
            best_points = points.copy()
            best_radius = radius
    
    # Phase 2: Gradient descent on smoothed objective
    if not _check_deadline():
        points = _gradient_descent(points, n_iter=300, lr=0.05)
        radius = _covering_radius(points)
        if radius < best_radius:
            best_points = points.copy()
            best_radius = radius
    
    # Phase 3: More CVT
    if not _check_deadline():
        points = _cvt_step(points, n_iter=100)
        radius = _covering_radius(points)
        if radius < best_radius:
            best_points = points.copy()
            best_radius = radius
    
    # Phase 4: Final gradient polish
    if not _check_deadline():
        points = _gradient_descent(points, n_iter=500, lr=0.02)
        radius = _covering_radius(points)
        if radius < best_radius:
            best_points = points.copy()
            best_radius = radius
    
    # Phase 5: One more CVT pass
    if not _check_deadline():
        points = _cvt_step(points, n_iter=50)
        radius = _covering_radius(points)
        if radius < best_radius:
            best_points = points.copy()
            best_radius = radius
    
    return _normalize(best_points)
# EVOLVE-BLOCK-END
