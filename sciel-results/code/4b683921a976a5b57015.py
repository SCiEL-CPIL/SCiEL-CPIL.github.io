"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay

N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay
import time


def _normalize(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms


def _covering_radius(pts):
    """Compute the covering radius using convex hull facets.
    For each facet, compute the circumcenter and check if it's on the sphere.
    The covering radius is the max angular distance from any point to nearest config point.
    We use the convex hull approach: the worst point is the circumcenter of the largest facet.
    """
    # For covering radius on sphere, we need to find the maximum distance
    # from any point on the sphere to the nearest of the N points.
    # This equals the maximum circumradius of the spherical Voronoi cells,
    # which corresponds to the circumcenters of the convex hull facets.
    
    # Get convex hull
    try:
        hull = ConvexHull(pts)
    except Exception:
        # Fallback: use grid-based estimation
        return _covering_radius_grid(pts)
    
    # For each facet, compute the circumcenter
    max_angle = 0.0
    for simplex in hull.simplices:
        # Get the facet points
        facet_pts = pts[simplex]
        
        # Compute circumcenter of the facet
        # For a triangle on the sphere, the circumcenter is the point equidistant from all 3 vertices
        # We can find it by solving the system of equations
        
        if len(facet_pts) == 3:
            a, b, c = facet_pts[0], facet_pts[1], facet_pts[2]
            
            # The circumcenter is perpendicular to the plane of the triangle
            # and equidistant from all three points
            # n = cross(b-a, c-a) gives normal to the plane
            n = np.cross(b - a, c - a)
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-15:
                continue
            n = n / n_norm
            
            # Circumcenter = a + t * n where t is chosen so that |a + t*n - a| = |a + t*n - b|
            # |t*n| = |a - b + t*n|
            # t^2 = |a-b|^2 + 2t*n·(a-b) + t^2
            # 0 = |a-b|^2 + 2t*n·(a-b)
            # t = -|a-b|^2 / (2*n·(a-b))
            
            d1 = b - a
            d2 = c - a
            t = -(np.dot(d1, d1)) / (2.0 * np.dot(n, d1))
            
            circum = a + t * n
            circum_norm = np.linalg.norm(circum)
            if circum_norm < 1e-15:
                continue
            circum = circum / circum_norm
            
            # Check if circumcenter is on the correct side (outside the convex hull)
            # The circumcenter should be on the opposite side of the facet from the origin
            # Actually, for points on the unit sphere, the circumcenter of a facet
            # that is a face of the convex hull should be "outside" (away from origin)
            
            # The angle from circumcenter to nearest point
            dists = np.arccos(np.clip(np.abs(np.dot(circum, pts)), -1.0, 1.0))
            min_dist = np.min(dists)
            max_angle = max(max_angle, min_dist)
    
    return np.degrees(max_angle)


def _covering_radius_grid(pts, n_grid=360):
    """Fallback: estimate covering radius using a grid of points on the sphere."""
    # Use a Fibonacci sphere for the grid
    i = np.arange(n_grid) + 0.5
    z = 1.0 - 2.0 * i / n_grid
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    grid_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # For each grid point, find nearest config point
    # Use dot product for speed
    dot = grid_pts @ pts.T
    min_dots = np.max(dot, axis=1)
    angles = np.arccos(np.clip(min_dots, -1.0, 1.0))
    return np.degrees(np.max(angles))


def _covering_radius_refined(pts, n_grid=720):
    """Refined covering radius with finer grid."""
    return _covering_radius_grid(pts, n_grid)


def _initial_fibonacci():
    """Generate initial Fibonacci spiral points."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _refine_facets(pts, max_iter=200, time_limit=None, start_time=None):
    """Refine points by moving them toward the circumcenters of the worst facets."""
    if start_time is None:
        start_time = time.time()
    if time_limit is None:
        time_limit = 140.0
    
    current_radius = _covering_radius(pts)
    best_pts = pts.copy()
    best_radius = current_radius
    
    for iteration in range(max_iter):
        if time.time() - start_time > time_limit - 5.0:
            break
        
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
        
        # Find the worst facet (largest circumradius)
        worst_circum = None
        worst_angle = 0.0
        worst_facet = None
        
        for simplex in hull.simplices:
            facet_pts = pts[simplex]
            if len(facet_pts) != 3:
                continue
            
            a, b, c = facet_pts[0], facet_pts[1], facet_pts[2]
            n = np.cross(b - a, c - a)
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-15:
                continue
            n = n / n_norm
            
            d1 = b - a
            t = -(np.dot(d1, d1)) / (2.0 * np.dot(n, d1))
            
            circum = a + t * n
            circum_norm = np.linalg.norm(circum)
            if circum_norm < 1e-15:
                continue
            circum = circum / circum_norm
            
            dists = np.arccos(np.clip(np.abs(np.dot(circum, pts)), -1.0, 1.0))
            min_dist = np.min(dists)
            
            if min_dist > worst_angle:
                worst_angle = min_dist
                worst_circum = circum
                worst_facet = simplex
        
        if worst_circum is None:
            break
        
        # Move the points in the worst facet toward the circumcenter
        # Use a small step size
        step = 0.01
        
        new_pts = pts.copy()
        for idx in worst_facet:
            # Move point toward circumcenter
            direction = worst_circum - pts[idx]
            direction_norm = np.linalg.norm(direction)
            if direction_norm < 1e-15:
                continue
            direction = direction / direction_norm
            new_pts[idx] = pts[idx] + step * direction
            new_pts[idx] = new_pts[idx] / np.linalg.norm(new_pts[idx])
        
        new_radius = _covering_radius(new_pts)
        
        if new_radius < best_radius:
            best_radius = new_radius
            best_pts = new_pts.copy()
            pts = new_pts
        else:
            # Try smaller step
            step = 0.005
            new_pts = pts.copy()
            for idx in worst_facet:
                direction = worst_circum - pts[idx]
                direction_norm = np.linalg.norm(direction)
                if direction_norm < 1e-15:
                    continue
                direction = direction / direction_norm
                new_pts[idx] = pts[idx] + step * direction
                new_pts[idx] = new_pts[idx] / np.linalg.norm(new_pts[idx])
            
            new_radius = _covering_radius(new_pts)
            if new_radius < best_radius:
                best_radius = new_radius
                best_pts = new_pts.copy()
                pts = new_pts
            else:
                # No improvement, stop
                break
    
    return best_pts, best_radius


def _gradient_descent(pts, max_iter=500, time_limit=None, start_time=None):
    """Simple gradient descent on the covering radius."""
    if start_time is None:
        start_time = time.time()
    if time_limit is None:
        time_limit = 140.0
    
    best_pts = pts.copy()
    best_radius = _covering_radius(pts)
    
    for iteration in range(max_iter):
        if time.time() - start_time > time_limit - 5.0:
            break
        
        # Compute gradient of covering radius w.r.t. each point
        # The covering radius is determined by the worst facet
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
        
        # Find worst facet
        worst_circum = None
        worst_angle = 0.0
        worst_facet = None
        
        for simplex in hull.simplices:
            facet_pts = pts[simplex]
            if len(facet_pts) != 3:
                continue
            
            a, b, c = facet_pts[0], facet_pts[1], facet_pts[2]
            n = np.cross(b - a, c - a)
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-15:
                continue
            n = n / n_norm
            
            d1 = b - a
            t = -(np.dot(d1, d1)) / (2.0 * np.dot(n, d1))
            
            circum = a + t * n
            circum_norm = np.linalg.norm(circum)
            if circum_norm < 1e-15:
                continue
            circum = circum / circum_norm
            
            dists = np.arccos(np.clip(np.abs(np.dot(circum, pts)), -1.0, 1.0))
            min_dist = np.min(dists)
            
            if min_dist > worst_angle:
                worst_angle = min_dist
                worst_circum = circum
                worst_facet = simplex
        
        if worst_circum is None:
            break
        
        # Compute gradient: move points in worst facet toward circumcenter
        # and points not in worst facet slightly away
        lr = 0.005
        new_pts = pts.copy()
        
        for idx in worst_facet:
            direction = worst_circum - pts[idx]
            direction_norm = np.linalg.norm(direction)
            if direction_norm > 1e-15:
                direction = direction / direction_norm
                new_pts[idx] = pts[idx] + lr * direction
        
        # Normalize
        for i in range(N):
            norm = np.linalg.norm(new_pts[i])
            if norm > 1e-15:
                new_pts[i] = new_pts[i] / norm
        
        new_radius = _covering_radius(new_pts)
        
        if new_radius < best_radius:
            best_radius = new_radius
            best_pts = new_pts.copy()
            pts = new_pts
        else:
            break
    
    return best_pts, best_radius


def construct():
    start_time = time.time()
    time_limit = 145.0  # Leave 5s margin
    
    # Start with Fibonacci spiral
    pts = _initial_fibonacci()
    best_pts = pts.copy()
    best_radius = _covering_radius(pts)
    
    # Multi-start with different perturbations
    np.random.seed(42)
    
    for restart in range(10):
        if time.time() - start_time > time_limit - 10.0:
            break
        
        # Generate a perturbed starting point
        if restart == 0:
            current = pts.copy()
        else:
            current = pts + 0.1 * np.random.randn(N, 3)
            current = _normalize(current)
        
        # Refine
        refined, radius = _refine_facets(current, max_iter=100, time_limit=time_limit - 5.0, start_time=start_time)
        
        if radius < best_radius:
            best_radius = radius
            best_pts = refined.copy()
        
        # Further refine with gradient descent
        refined2, radius2 = _gradient_descent(refined, max_iter=200, time_limit=time_limit - 5.0, start_time=start_time)
        
        if radius2 < best_radius:
            best_radius = radius2
            best_pts = refined2.copy()
    
    # Final polish
    if time.time() - start_time < time_limit - 10.0:
        polished, radius_polished = _refine_facets(best_pts, max_iter=100, time_limit=time_limit - 5.0, start_time=start_time)
        if radius_polished < best_radius:
            best_radius = radius_polished
            best_pts = polished.copy()
    
    return best_pts
# EVOLVE-BLOCK-END
