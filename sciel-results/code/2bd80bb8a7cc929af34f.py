"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import sys
from scipy.spatial import ConvexHull, Delaunay
from scipy.optimize import minimize
import warnings

N = 130

# EVOLVE-BLOCK-START

def _covering_radius(points):
    """Compute the covering radius in degrees for points on the unit sphere.
    
    The covering radius is the maximum angular distance from any point on the sphere
    to its nearest chosen point. This is equivalent to half the largest angular edge
    length of the convex hull facets, or more precisely, the circumradius of the
    largest face of the Delaunay triangulation on the sphere.
    
    For points on a sphere, the covering radius equals the maximum over all Delaunay
    triangles (or edges for the boundary) of the circumradius of that triangle/edge.
    """
    points = np.asarray(points)
    # Normalize to unit sphere
    norms = np.linalg.norm(points, axis=1)
    points = points / norms[:, None]
    
    # Use Delaunay triangulation in 3D
    try:
        tri = Delaunay(points)
    except Exception:
        # Fallback: use convex hull
        try:
            hull = ConvexHull(points)
            # Get the faces
            faces = hull.simplices
            # For each face (triangle), compute the circumradius
            max_circ = 0.0
            for face in faces:
                p1, p2, p3 = points[face[0]], points[face[1]], points[face[2]]
                # Compute circumradius of triangle on unit sphere
                # The circumcenter of a triangle on the unit sphere can be found
                # by intersecting the plane of the triangle with the sphere
                # The circumradius is the angle from the circumcenter to any vertex
                # 
                # For a triangle with vertices a, b, c on the unit sphere:
                # The circumradius R satisfies: sin(R) = |a x b| / (2 * |a - b|) ... no
                # Actually, for a triangle on a sphere, the circumradius is the angle
                # from the center of the circumscribed circle to a vertex.
                # 
                # A simpler approach: the circumradius of a spherical triangle is
                # related to the side lengths. For a triangle with sides a, b, c
                # (angular distances), the circumradius R satisfies:
                # tan(R) = (abc) / (4 * area) where area is the spherical excess
                # 
                # Let's use a different approach: compute the circumcenter directly.
                # The circumcenter of a triangle on the unit sphere is the intersection
                # of the sphere with the plane perpendicular to the triangle's plane
                # that passes through the origin.
                
                # Compute the normal to the triangle
                n = np.cross(p2 - p1, p3 - p1)
                n_norm = np.linalg.norm(n)
                if n_norm < 1e-12:
                    continue
                n = n / n_norm
                
                # The circumcenter c satisfies: |c| = 1, c . n = d, and
                # |c - p1|^2 = |c - p2|^2 = |c - p3|^2
                # From |c - p1|^2 = |c - p2|^2:
                # |c|^2 - 2c.p1 + |p1|^2 = |c|^2 - 2c.p2 + |p2|^2
                # -2c.p1 = -2c.p2 (since |p1|=|p2|=1)
                # c.(p1 - p2) = 0
                # Similarly c.(p1 - p3) = 0
                # So c is parallel to n.
                # c = ±n
                # The circumradius is the angle from c to p1.
                # We need to choose the sign such that c is on the same side as the
                # "interior" of the triangle.
                
                # The circumradius R is given by:
                # cos(R) = c . p1
                # We want the smaller R, so we pick the sign that gives c . p1 > 0
                # if possible, but actually we want the cap that contains the triangle.
                
                # The angular circumradius is:
                # R = arccos(|n . p1|) ... no, that's not right.
                
                # Let's compute it properly.
                # The circumcenter is c = n or c = -n.
                # The circumradius is the angular distance from c to any vertex.
                # For c = n: R = arccos(n . p1)
                # For c = -n: R = arccos(-n . p1)
                # We want the minimum of these two (the smaller cap).
                
                r1 = np.arccos(np.clip(n @ p1, -1, 1))
                r2 = np.arccos(np.clip(-n @ p1, -1, 1))
                r = min(r1, r2)
                max_circ = max(max_circ, r)
            
            return np.degrees(max_circ)
        except Exception:
            # Last resort: brute force on a grid
            return _covering_radius_brute(points)
    
    # Get the Delaunay simplices (tetrahedra in 3D)
    # But we want the spherical Delaunay, which corresponds to the convex hull faces
    # Let's use the convex hull approach instead for correctness
    try:
        hull = ConvexHull(points)
        faces = hull.simplices
    except Exception:
        return _covering_radius_brute(points)
    
    max_circ = 0.0
    for face in faces:
        p1, p2, p3 = points[face[0]], points[face[1]], points[face[2]]
        n = np.cross(p2 - p1, p3 - p1)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-12:
            continue
        n = n / n_norm
        
        r1 = np.arccos(np.clip(n @ p1, -1, 1))
        r2 = np.arccos(np.clip(-n @ p1, -1, 1))
        r = min(r1, r2)
        max_circ = max(max_circ, r)
    
    return np.degrees(max_circ)


def _covering_radius_brute(points, n_theta=180, n_phi=360):
    """Brute force covering radius computation on a grid."""
    points = np.asarray(points)
    norms = np.linalg.norm(points, axis=1)
    points = points / norms[:, None]
    
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    TH, PH = np.meshgrid(theta, phi, indexing='ij')
    
    # Grid points on sphere
    x = np.sin(TH) * np.cos(PH)
    y = np.sin(TH) * np.sin(PH)
    z = np.cos(TH)
    grid = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    
    # Compute distances from each grid point to each site
    # This can be memory intensive, so do it in chunks
    max_dist = 0.0
    chunk_size = 10000
    for start in range(0, len(grid), chunk_size):
        end = min(start + chunk_size, len(grid))
        g = grid[start:end]
        # Dot products
        dots = g @ points.T
        dots = np.clip(dots, -1, 1)
        # Max dot product gives min angular distance
        max_dots = np.max(dots, axis=1)
        min_angs = np.arccos(max_dots)
        max_dist = max(max_dist, np.max(min_angs))
    
    return np.degrees(max_dist)


def _gradient_circumradius(points, face, n):
    """Compute the gradient of the circumradius with respect to the face vertices.
    
    This is an approximation for the optimization.
    """
    p1, p2, p3 = points[face[0]], points[face[1]], points[face[2]]
    grad = np.zeros_like(points)
    
    # The circumradius R = arccos(c . p1) where c is the circumcenter
    # For simplicity, we use a finite difference approach in the optimization
    # instead of analytic gradients
    return grad


def _optimize_covering(points, max_time=100):
    """Optimize the covering using a gradient-based approach on the sphere.
    
    Uses a simplified approach: repeatedly find the worst facet and move
    its vertices to reduce the circumradius.
    """
    points = np.array(points, dtype=float)
    start_time = time.time()
    best_points = points.copy()
    best_radius = _covering_radius(points)
    
    # Local optimization: move points to reduce the max circumradius
    # Use a simple iterative approach
    lr = 0.01
    for it in range(500):
        if time.time() - start_time > max_time:
            break
        
        # Compute covering radius and find worst facet
        norms = np.linalg.norm(points, axis=1)
        points = points / norms[:, None]
        
        try:
            hull = ConvexHull(points)
            faces = hull.simplices
        except Exception:
            break
        
        # Find the facet with largest circumradius
        max_circ = 0.0
        worst_face = None
        for face in faces:
            p1, p2, p3 = points[face[0]], points[face[1]], points[face[2]]
            n = np.cross(p2 - p1, p3 - p1)
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-12:
                continue
            n = n / n_norm
            
            r1 = np.arccos(np.clip(n @ p1, -1, 1))
            r2 = np.arccos(np.clip(-n @ p1, -1, 1))
            r = min(r1, r2)
            if r > max_circ:
                max_circ = r
                worst_face = face
        
        if worst_face is None:
            break
        
        # Compute gradient direction: move each vertex of the worst face
        # in the direction that reduces the circumradius
        p1, p2, p3 = points[worst_face[0]], points[worst_face[1]], points[worst_face[2]]
        n = np.cross(p2 - p1, p3 - p1)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-12:
            break
        n = n / n_norm
        
        # The circumcenter is c = n or -n, choose the one that gives the cap
        # containing the triangle interior
        c = n if (n @ p1) > 0 else -n
        
        # Move vertices away from the circumcenter to reduce the circumradius
        # Actually, we want to move them toward the circumcenter? No, we want
        # to make the triangle smaller, so move vertices toward each other
        # or toward the center of the cap.
        
        # A better approach: move each vertex in the direction that reduces
        # the angular distance to the circumcenter
        for idx in worst_face:
            p = points[idx]
            # Project the circumcenter onto the tangent plane at p
            # The direction to move is the component of (c - p) tangent to the sphere
            d = c - p
            # Remove the radial component
            d = d - (d @ p) * p
            d_norm = np.linalg.norm(d)
            if d_norm > 1e-12:
                d = d / d_norm
                points[idx] = points[idx] + lr * d
        
        # Renormalize
        norms = np.linalg.norm(points, axis=1)
        points = points / norms[:, None]
        
        # Check improvement
        new_radius = _covering_radius(points)
        if new_radius < best_radius:
            best_radius = new_radius
            best_points = points.copy()
        else:
            # Revert and reduce learning rate
            points = best_points.copy()
            lr *= 0.9
    
    return best_points


def construct():
    """Construct 130 points on the unit sphere minimizing the covering radius."""
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Normalize
    norms = np.linalg.norm(points, axis=1)
    points = points / norms[:, None]
    
    best_points = points.copy()
    best_radius = _covering_radius(points)
    
    # Try local optimization
    if time.time() < deadline - 20:
        optimized = _optimize_covering(points, max_time=60)
        opt_radius = _covering_radius(optimized)
        if opt_radius < best_radius:
            best_points = optimized
            best_radius = opt_radius
    
    # Try a few random restarts if time permits
    rng = np.random.default_rng(42)
    for restart in range(5):
        if time.time() > deadline - 30:
            break
        
        # Generate random points
        rand_points = rng.standard_normal((N, 3))
        norms = np.linalg.norm(rand_points, axis=1)
        rand_points = rand_points / norms[:, None]
        
        # Optimize
        opt = _optimize_covering(rand_points, max_time=20)
        rad = _covering_radius(opt)
        if rad < best_radius:
            best_points = opt
            best_radius = rad
    
    # Final normalization
    norms = np.linalg.norm(best_points, axis=1)
    best_points = best_points / norms[:, None]
    
    return best_points

# EVOLVE-BLOCK-END
