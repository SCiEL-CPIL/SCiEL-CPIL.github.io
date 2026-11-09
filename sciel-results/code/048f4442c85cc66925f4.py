"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull, Delaunay
from scipy.optimize import minimize
from scipy.spatial.distance import cdist


def _normalize(points):
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return points / norms


def _covering_radius(points):
    """Compute the covering radius in radians using the convex hull method."""
    # Ensure points are on the unit sphere
    points = _normalize(points)
    
    # Compute convex hull to get facets
    try:
        hull = ConvexHull(points)
    except Exception:
        # Fallback: use Delaunay triangulation on the sphere
        try:
            tri = Delaunay(points)
            # For each simplex (tetrahedron in 3D), compute circumradius
            # But this is complex; use a grid-based fallback
            return _covering_radius_grid(points)
        except Exception:
            return _covering_radius_grid(points)
    
    max_radius = 0.0
    # Hull simplices are the facets
    for simplex in hull.simplices:
        facet_points = points[simplex]
        # Compute the spherical circumcenter of this facet
        # For a set of points on the sphere, the circumcenter is the direction
        # that is equidistant (in angle) from all points in the facet
        # We can find this by solving for the point c on the sphere that
        # minimizes the maximum angular distance to the facet points
        # Actually, the circumcenter of a spherical polygon is the point on the sphere
        # that is equidistant from all vertices of the polygon (if it exists)
        
        # For a triangle (3 points), the circumcenter can be found as follows:
        # The circumcenter c satisfies |c - p_i| = |c - p_j| for all i, j
        # This means c is in the direction of the cross product of differences
        
        # More robustly: find the point c on the unit sphere that minimizes
        # the maximum angular distance to the facet points
        # This is equivalent to finding the center of the smallest spherical cap
        # covering the facet points
        
        # For small facets (triangles), we can use the following:
        # The circumcenter of a spherical triangle is the intersection of the sphere
        # with the line through the origin perpendicular to the plane containing
        # the three points (if they are not collinear)
        
        if len(simplex) == 3:
            p0, p1, p2 = facet_points
            # The circumcenter direction is perpendicular to the plane of the triangle
            # and points away from the origin (or towards, depending on orientation)
            v1 = p1 - p0
            v2 = p2 - p0
            normal = np.cross(v1, v2)
            norm_normal = np.linalg.norm(normal)
            if norm_normal < 1e-10:
                # Degenerate triangle, use midpoint
                c = (p0 + p1 + p2) / 3.0
                c = c / np.linalg.norm(c)
            else:
                c = normal / norm_normal
                # Ensure c is on the same side as the facet
                # Check dot product with the centroid
                centroid = (p0 + p1 + p2) / 3.0
                if np.dot(c, centroid) < 0:
                    c = -c
            
            # Compute the angular distance from c to each vertex
            dists = np.arccos(np.clip(np.dot(facet_points, c), -1.0, 1.0))
            radius = np.max(dists)
        elif len(simplex) == 4:
            # Tetrahedron facet - this shouldn't happen on a convex hull of points on a sphere
            # unless we're in 3D. Actually, ConvexHull in 3D gives tetrahedra as simplices.
            # For points on a sphere, the facets should be triangles.
            # Let's handle this by finding the point on the sphere that minimizes
            # the maximum distance to the 4 points
            # Use optimization
            def obj(x):
                x = x / np.linalg.norm(x)
                dists = np.arccos(np.clip(np.dot(facet_points, x), -1.0, 1.0))
                return np.max(dists)
            
            # Start from the centroid direction
            x0 = np.sum(facet_points, axis=0)
            x0 = x0 / np.linalg.norm(x0)
            res = minimize(obj, x0, method='Nelder-Mead', options={'maxiter': 100})
            c = res.x / np.linalg.norm(res.x)
            dists = np.arccos(np.clip(np.dot(facet_points, c), -1.0, 1.0))
            radius = np.max(dists)
        else:
            # Fallback
            c = np.mean(facet_points, axis=0)
            c = c / np.linalg.norm(c)
            dists = np.arccos(np.clip(np.dot(facet_points, c), -1.0, 1.0))
            radius = np.max(dists)
        
        max_radius = max(max_radius, radius)
    
    return max_radius


def _covering_radius_grid(points, n_grid=100):
    """Fallback method using a grid on the sphere."""
    points = _normalize(points)
    # Create a grid on the sphere
    theta = np.linspace(0, np.pi, n_grid)
    phi = np.linspace(0, 2 * np.pi, n_grid)
    theta_grid, phi_grid = np.meshgrid(theta, phi)
    x = np.sin(theta_grid) * np.cos(phi_grid)
    y = np.sin(theta_grid) * np.sin(phi_grid)
    z = np.cos(theta_grid)
    grid_points = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    grid_points = _normalize(grid_points)
    
    # Compute distances from each grid point to the nearest configuration point
    dists = cdist(grid_points, points, metric='cosine')
    min_dists = np.min(dists, axis=1)
    return np.max(min_dists)


def _covering_radius_degrees(points):
    """Compute the covering radius in degrees."""
    return np.degrees(_covering_radius(points))


def _improve_facet(points, facet_indices, max_iter=100, tol=1e-6):
    """Improve the covering radius by moving the vertices of the worst facet."""
    points = points.copy()
    facet_points = points[facet_indices]
    
    # Find the circumcenter of the facet
    if len(facet_indices) == 3:
        p0, p1, p2 = facet_points
        v1 = p1 - p0
        v2 = p2 - p0
        normal = np.cross(v1, v2)
        norm_normal = np.linalg.norm(normal)
        if norm_normal < 1e-10:
            return points
        c = normal / norm_normal
        centroid = (p0 + p1 + p2) / 3.0
        if np.dot(c, centroid) < 0:
            c = -c
    else:
        # For larger facets, use the centroid direction
        c = np.mean(facet_points, axis=0)
        c = c / np.linalg.norm(c)
    
    # Move the facet vertices towards the circumcenter
    for idx in facet_indices:
        # Project the movement onto the tangent space
        p = points[idx]
        # The direction from p to c on the sphere
        # Use the tangent vector
        tangent = c - np.dot(c, p) * p
        tangent_norm = np.linalg.norm(tangent)
        if tangent_norm < 1e-10:
            continue
        tangent = tangent / tangent_norm
        # Small step towards the circumcenter
        step_size = 0.1
        new_p = p + step_size * tangent
        new_p = new_p / np.linalg.norm(new_p)
        points[idx] = new_p
    
    return points


def construct():
    """Construct 50 points on the unit sphere to minimize the covering radius."""
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin
    
    # Start with a good initial configuration
    # Use the golden angle spiral as a starting point
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    points = _normalize(points)
    
    best_points = points.copy()
    best_radius = _covering_radius(points)
    
    # Iteratively improve by finding the worst facet and moving its vertices
    iteration = 0
    while time.time() < deadline:
        iteration += 1
        
        # Compute the convex hull to find the worst facet
        try:
            hull = ConvexHull(points)
        except Exception:
            break
        
        # Find the facet with the largest circumradius
        max_radius = 0.0
        worst_facet = None
        for simplex in hull.simplices:
            facet_points = points[simplex]
            if len(simplex) == 3:
                p0, p1, p2 = facet_points
                v1 = p1 - p0
                v2 = p2 - p0
                normal = np.cross(v1, v2)
                norm_normal = np.linalg.norm(normal)
                if norm_normal < 1e-10:
                    continue
                c = normal / norm_normal
                centroid = (p0 + p1 + p2) / 3.0
                if np.dot(c, centroid) < 0:
                    c = -c
                dists = np.arccos(np.clip(np.dot(facet_points, c), -1.0, 1.0))
                radius = np.max(dists)
            else:
                c = np.mean(facet_points, axis=0)
                c = c / np.linalg.norm(c)
                dists = np.arccos(np.clip(np.dot(facet_points, c), -1.0, 1.0))
                radius = np.max(dists)
            
            if radius > max_radius:
                max_radius = radius
                worst_facet = simplex
        
        if worst_facet is None:
            break
        
        # Check if we've converged
        if max_radius < best_radius * (1 - 1e-8):
            best_radius = max_radius
            best_points = points.copy()
        
        # Improve the worst facet
        new_points = _improve_facet(points, worst_facet)
        
        # Check if the new configuration is better
        new_radius = _covering_radius(new_points)
        if new_radius < best_radius:
            best_radius = new_radius
            best_points = new_points.copy()
            points = new_points
        else:
            # Try a smaller step or different direction
            # Reduce the step size by modifying the improvement
            points = _improve_facet(points, worst_facet)
            # If still not better, break to avoid cycling
            if _covering_radius(points) >= best_radius * (1 + 1e-6):
                # Try random perturbation
                if np.random.random() < 0.1:
                    idx = np.random.randint(N)
                    p = points[idx]
                    # Random tangent direction
                    random_dir = np.random.randn(3)
                    tangent = random_dir - np.dot(random_dir, p) * p
                    tangent = tangent / np.linalg.norm(tangent)
                    new_p = p + 0.05 * tangent
                    new_p = new_p / np.linalg.norm(new_p)
                    test_points = points.copy()
                    test_points[idx] = new_p
                    test_radius = _covering_radius(test_points)
                    if test_radius < best_radius:
                        best_radius = test_radius
                        best_points = test_points.copy()
                        points = test_points
    
    return _normalize(best_points)
# EVOLVE-BLOCK-END
