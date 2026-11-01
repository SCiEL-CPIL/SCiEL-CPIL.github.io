"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, LinearConstraint

N = 50
DEADLINE = 115.0  # safety margin before 120s kill


def _normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-15:
        return np.zeros(3)
    return v / n


def _covering_radius(pts):
    """Compute covering radius in degrees using convex hull facet circumradii.
    
    The covering radius is the max over all facets of the convex hull of the
    circumradius of that facet (the angular radius of the smallest cap containing
    the facet vertices, which corresponds to the circumcenter if it's on the
    correct side).
    """
    try:
        hull = ConvexHull(pts)
    except Exception:
        return 180.0
    
    max_ang = 0.0
    for simplex in hull.simplices:
        # Get the facet vertices
        verts = pts[simplex]  # shape (k, 3), k=3 typically
        
        # Compute circumcenter of the spherical triangle
        # For a triangle on the unit sphere, the circumcenter is the normalized
        # cross-product-based solution.
        # Actually, for covering radius, we need the point on the sphere farthest
        # from all three vertices but still in the "cap" defined by the facet.
        # The relevant point is the circumcenter of the triangle if it lies on
        # the same side as the hull's outward normal.
        
        a, b, c = verts[0], verts[1], verts[2]
        
        # Circumcenter of spherical triangle:
        # Solve for point x such that x·a = x·b = x·c = cos(r), |x|=1
        # This is equivalent to finding the point equidistant (angularly) from a, b, c.
        
        # Method: The circumcenter direction is proportional to:
        # a × b × c + ... no, let's use the linear system.
        # x · a = x · b = x · c
        # (a - b) · x = 0
        # (a - c) · x = 0
        # So x is parallel to (a-b) × (a-c)
        
        v1 = a - b
        v2 = a - c
        x = np.cross(v1, v2)
        x = _normalize(x)
        
        # Check if x is on the outward side (away from origin relative to facet)
        # The facet's outward normal for the convex hull
        # For a point on the sphere, the outward direction from the facet is
        # the direction from the centroid of the facet away from the origin.
        centroid = (a + b + c) / 3.0
        outward = _normalize(centroid)
        
        # The circumcenter should be on the outward side
        if np.dot(x, outward) < 0:
            x = -x
        
        # Angular distance from circumcenter to vertex a
        dot_val = np.clip(np.dot(x, a), -1.0, 1.0)
        ang = np.degrees(np.arccos(dot_val))
        
        if ang > max_ang:
            max_ang = ang
    
    return max_ang


def _covering_radius_grid(pts, n_grid=200):
    """Fallback covering radius using a grid on the sphere."""
    # Use Fibonacci grid for better coverage
    i = np.arange(n_grid) + 0.5
    z = 1.0 - 2.0 * i / n_grid
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    grid = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Compute angular distances from each grid point to nearest configuration point
    # dot products
    dots = grid @ pts.T  # (n_grid, N)
    max_dots = np.max(dots, axis=1)
    min_ang = np.degrees(np.arccos(np.clip(max_dots, -1.0, 1.0)))
    return np.max(min_ang)


def _fibonacci_spiral(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _local_optimize(pts, time_limit=10.0):
    """Local optimization using gradient-free approach on the sphere.
    
    We optimize the covering radius by moving points in the direction that
    reduces the worst-case angular distance.
    """
    start_time = time.time()
    pts = pts.copy()
    best_pts = pts.copy()
    best_cr = _covering_radius(pts)
    
    step_size = 0.05
    n_iter = 0
    
    while time.time() - start_time < time_limit:
        # Find the worst facet (largest circumradius)
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
        
        # Find the facet with largest circumradius
        worst_facet = None
        worst_ang = 0.0
        for simplex in hull.simplices:
            verts = pts[simplex]
            a, b, c = verts[0], verts[1], verts[2]
            v1 = a - b
            v2 = a - c
            x = np.cross(v1, v2)
            x = _normalize(x)
            centroid = (a + b + c) / 3.0
            outward = _normalize(centroid)
            if np.dot(x, outward) < 0:
                x = -x
            dot_val = np.clip(np.dot(x, a), -1.0, 1.0)
            ang = np.degrees(np.arccos(dot_val))
            if ang > worst_ang:
                worst_ang = ang
                worst_facet = simplex
        
        if worst_facet is None:
            break
        
        # Move the vertices of the worst facet toward the circumcenter
        # This reduces the circumradius of that facet
        verts = pts[worst_facet].copy()
        a, b, c = verts[0], verts[1], verts[2]
        v1 = a - b
        v2 = a - c
        x = np.cross(v1, v2)
        x = _normalize(x)
        centroid = (a + b + c) / 3.0
        outward = _normalize(centroid)
        if np.dot(x, outward) < 0:
            x = -x
        
        # Move each vertex slightly toward x
        for idx in worst_facet:
            pts[idx] = _normalize(pts[idx] + step_size * 0.3 * x)
        
        # Also try to move other points to improve overall coverage
        # Compute the current covering radius
        cr = _covering_radius(pts)
        n_iter += 1
        
        if cr < best_cr - 1e-9:
            best_cr = cr
            best_pts = pts.copy()
            # Try smaller step for next iteration
            step_size = max(0.001, step_size * 0.95)
        else:
            # Revert and try different perturbation
            pts = best_pts.copy()
            step_size = max(0.001, step_size * 0.7)
            
            # Random perturbation to escape local minimum
            for idx in range(N):
                random_dir = np.random.randn(3)
                random_dir = _normalize(random_dir)
                # Make it tangential to the sphere
                tangent = random_dir - np.dot(random_dir, pts[idx]) * pts[idx]
                tangent = _normalize(tangent)
                pts[idx] = _normalize(pts[idx] + step_size * 0.1 * tangent)
            
            cr = _covering_radius(pts)
            if cr < best_cr:
                best_cr = cr
                best_pts = pts.copy()
    
    return best_pts, best_cr


def _restart_with_perturbation(base_pts, strength=0.1):
    """Create a perturbed version of base_pts."""
    pts = base_pts.copy()
    for idx in range(N):
        random_dir = np.random.randn(3)
        random_dir = _normalize(random_dir)
        tangent = random_dir - np.dot(random_dir, pts[idx]) * pts[idx]
        tangent = _normalize(tangent)
        pts[idx] = _normalize(pts[idx] + strength * tangent)
    return pts


def construct():
    global DEADLINE
    t_start = time.time()
    
    # Start with Fibonacci spiral
    pts = _fibonacci_spiral(N)
    best_pts = pts.copy()
    best_cr = _covering_radius(pts)
    
    # Local optimization from initial point
    if time.time() - t_start < DEADLINE - 15:
        best_pts, best_cr = _local_optimize(best_pts, time_limit=min(20.0, DEADLINE - (time.time() - t_start) - 5))
    
    # Multiple restarts with perturbation
    n_restarts = 0
    while time.time() - t_start < DEADLINE - 10:
        # Perturb the best solution
        strength = 0.05 + 0.05 * np.random.random()
        perturbed = _restart_with_perturbation(best_pts, strength)
        
        # Local optimize the perturbed version
        remaining = DEADLINE - (time.time() - t_start) - 5
        if remaining < 5:
            break
        time_limit = min(10.0, remaining)
        new_pts, new_cr = _local_optimize(perturbed, time_limit=time_limit)
        
        if new_cr < best_cr:
            best_cr = new_cr
            best_pts = new_pts.copy()
        
        n_restarts += 1
    
    # Final check: make sure points are on the unit sphere and valid
    best_pts = np.array([_normalize(p) for p in best_pts])
    
    # Verify no coincident points
    for i in range(N):
        for j in range(i+1, N):
            if np.linalg.norm(best_pts[i] - best_pts[j]) < 1e-6:
                # Separate them
                best_pts[j] = _normalize(best_pts[j] + 1e-4 * np.random.randn(3))
    
    return best_pts
# EVOLVE-BLOCK-END
