"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull, Delaunay
from scipy.spatial.transform import Rotation
import time


def _covering_radius(pts):
    """
    Compute the covering radius (in degrees) of a set of points on the unit sphere.
    Uses the convex hull: the covering radius is the maximum circumradius of the
    spherical facets. For a planar triangle with vertices on the unit sphere,
    the spherical circumradius is the angle from the sphere center to the
    circumcenter projected onto the sphere... actually, for a facet (triangle) of
    the convex hull of points on the unit sphere, the covering radius contribution
    is the angular distance from the facet's center (normalized) to the farthest vertex.
    
    More precisely: for each facet (a triangle with vertices v1, v2, v3 on the unit sphere),
    the point on the sphere farthest from all three vertices is along the direction of
    the normalized sum (for acute triangles) or the normalized edge midpoint (for obtuse).
    The covering radius is the max over all facets of the angular distance from the
    facet's "center" to its farthest vertex.
    
    Actually, the standard approach: the covering radius equals the maximum over all
    facets of the angular radius of the smallest spherical cap centered at a point on
    the sphere that contains the facet's vertices. For a planar triangle on the unit sphere,
    this is determined by the circumcenter of the triangle (in 3D, projected to the sphere).
    
    For a triangle with vertices a, b, c (unit vectors), the circumcenter direction is
    the normalized vector: a + b + c (if the triangle is acute in the spherical sense),
    otherwise it's the midpoint of the longest edge.
    
    The covering radius is max over facets of max(angular_dist(center, a), angular_dist(center, b), angular_dist(center, c)).
    """
    pts = np.asarray(pts, dtype=np.float64)
    # Normalize
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    pts = pts / norms
    
    # Check for coincident points
    for i in range(len(pts)):
        for j in range(i+1, len(pts)):
            if np.linalg.norm(pts[i] - pts[j]) < 1e-10:
                return 180.0  # Invalid: coincident points
    
    # Check if all points are in a closed hemisphere
    # This is a necessary condition for a valid covering; if they are, the covering radius is >= 90 degrees
    # We check by trying to find a plane through the origin that has all points on one side
    # Simple heuristic: check if the origin is inside the convex hull
    try:
        hull = ConvexHull(pts)
    except Exception:
        return 180.0
    
    # Check if origin is inside the hull
    # The origin is inside if for all facets, the origin is on the same side as the interior
    # A simpler check: if all points lie in a hemisphere, then there exists a unit vector u
    # such that u . p >= 0 for all p. We can check this by looking at the convex hull.
    # Actually, for the covering problem, we just need the max circumradius.
    
    # Get the triangular facets
    simplices = hull.simplices  # shape (F, 3)
    
    max_radius = 0.0
    for simplex in simplices:
        v1, v2, v3 = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
        
        # Compute the circumcenter of the triangle on the sphere
        # The circumcenter is the point on the sphere equidistant (angularly) from v1, v2, v3
        # This is the normalized cross product of (v1-v2) and (v1-v3) ... no.
        # The circumcenter direction is perpendicular to the plane of the triangle.
        # The plane normal is cross(v2-v1, v3-v1)
        normal = np.cross(v2 - v1, v3 - v1)
        normal_norm = np.linalg.norm(normal)
        if normal_norm < 1e-12:
            # Degenerate triangle
            # Use the farthest pair
            d12 = np.arccos(np.clip(np.dot(v1, v2), -1, 1))
            d13 = np.arccos(np.clip(np.dot(v1, v3), -1, 1))
            d23 = np.arccos(np.clip(np.dot(v2, v3), -1, 1))
            max_d = max(d12, d13, d23)
            max_radius = max(max_radius, max_d / 2.0)
            continue
        
        # The circumcenter is along the normal direction, but we need to find the specific
        # point on the sphere that is equidistant from all three vertices.
        # For points on the unit sphere, the circumcenter of the spherical triangle is
        # the normalized vector that is a linear combination of v1, v2, v3 with equal
        # angular distances.
        
        # A better approach: the circumcenter of the planar triangle (in 3D) projected
        # to the sphere. The planar circumcenter of triangle v1,v2,v3 (as 3D points)
        # is the point equidistant from all three. Since all are on the unit sphere,
        # the planar circumcenter is the origin! That's not helpful.
        
        # Instead, we need the spherical circumcenter. For a spherical triangle,
        # the circumcenter is the point where the perpendicular bisectors of the sides meet.
        # In 3D, this is the normalized vector:
        # c = v1 + v2 + v3 (for acute triangles)
        # but this only works if the triangle is "acute" in the right sense.
        
        # More robust: solve for c such that |c - v1| = |c - v2| = |c - v3| and |c| = 1
        # This means c . v1 = c . v2 = c . v3 = cos(theta) for some theta.
        # So c is orthogonal to (v1-v2) and (v1-v3), i.e., c is parallel to cross(v1-v2, v1-v3)
        
        c = np.cross(v1 - v2, v1 - v3)
        c_norm = np.linalg.norm(c)
        if c_norm < 1e-12:
            # Degenerate
            d12 = np.arccos(np.clip(np.dot(v1, v2), -1, 1))
            d13 = np.arccos(np.clip(np.dot(v1, v3), -1, 1))
            d23 = np.arccos(np.clip(np.dot(v2, v3), -1, 1))
            max_d = max(d12, d13, d23)
            max_radius = max(max_radius, max_d / 2.0)
            continue
        c = c / c_norm
        
        # Check if c is on the correct side (same side as the triangle interior)
        # The normal of the facet from ConvexHull points outward
        # We need c to be on the sphere side of the facet
        # For now, just use c and -c, pick the one closer to the vertices
        d_c = np.arccos(np.clip(np.dot(c, v1), -1, 1))
        d_neg_c = np.arccos(np.clip(np.dot(-c, v1), -1, 1))
        if d_neg_c < d_c:
            c = -c
            d_c = d_neg_c
        
        # The covering radius for this facet is the max angular distance from c to each vertex
        # But we need to verify that c is actually the point on the sphere farthest from all vertices
        # within this facet's region. For a convex spherical triangle, the farthest point
        # from the vertices is either the circumcenter (if acute) or the midpoint of the
        # longest edge (if obtuse).
        
        # Check if the triangle is obtuse
        d12 = np.arccos(np.clip(np.dot(v1, v2), -1, 1))
        d13 = np.arccos(np.clip(np.dot(v1, v3), -1, 1))
        d23 = np.arccos(np.clip(np.dot(v2, v3), -1, 1))
        
        # The largest angle determines if it's obtuse
        # If the circumcenter is outside the triangle, the farthest point is the midpoint of the longest edge
        # A simple check: if any side is > 90 degrees, the triangle is obtuse
        # More precisely, check if the circumcenter lies inside the spherical triangle
        
        # For simplicity, compute the max distance from c to the vertices
        # and also from the midpoints of the longest edge
        max_dist_c = max(d_c,
                         np.arccos(np.clip(np.dot(c, v2), -1, 1)),
                         np.arccos(np.clip(np.dot(c, v3), -1, 1)))
        
        # Check midpoints of edges
        # The midpoint of the longest edge
        if d12 >= d13 and d12 >= d23:
            mid = (v1 + v2) / np.linalg.norm(v1 + v2) if np.linalg.norm(v1 + v2) > 1e-10 else v1
            max_dist_mid = max(np.arccos(np.clip(np.dot(mid, v1), -1, 1)),
                               np.arccos(np.clip(np.dot(mid, v2), -1, 1)),
                               np.arccos(np.clip(np.dot(mid, v3), -1, 1)))
        elif d13 >= d12 and d13 >= d23:
            mid = (v1 + v3) / np.linalg.norm(v1 + v3) if np.linalg.norm(v1 + v3) > 1e-10 else v1
            max_dist_mid = max(np.arccos(np.clip(np.dot(mid, v1), -1, 1)),
                               np.arccos(np.clip(np.dot(mid, v2), -1, 1)),
                               np.arccos(np.clip(np.dot(mid, v3), -1, 1)))
        else:
            mid = (v2 + v3) / np.linalg.norm(v2 + v3) if np.linalg.norm(v2 + v3) > 1e-10 else v2
            max_dist_mid = max(np.arccos(np.clip(np.dot(mid, v1), -1, 1)),
                               np.arccos(np.clip(np.dot(mid, v2), -1, 1)),
                               np.arccos(np.clip(np.dot(mid, v3), -1, 1)))
        
        facet_radius = max(max_dist_c, max_dist_mid)
        max_radius = max(max_radius, facet_radius)
    
    return np.degrees(max_radius)


def _smooth_covering_radius(pts, grid_pts, beta=50.0):
    """
    Smooth surrogate for the covering radius using a dense grid.
    For each grid point, compute the min angular distance to any configuration point.
    Then use a soft-max (log-sum-exp) to get a smooth approximation of the max.
    """
    pts = np.asarray(pts, dtype=np.float64)
    grid_pts = np.asarray(grid_pts, dtype=np.float64)
    
    # Compute all pairwise dot products: (G, N)
    dots = grid_pts @ pts.T  # shape (G, N)
    # Clip to [-1, 1]
    dots = np.clip(dots, -1.0, 1.0)
    # Angular distances in radians
    dists = np.arccos(dots)  # shape (G, N)
    
    # Min distance for each grid point
    min_dists = np.min(dists, axis=1)  # shape (G,)
    
    # Soft max of min_dists
    # Use log-sum-exp for numerical stability
    beta_inv = 1.0 / beta
    max_val = np.max(min_dists)
    exp_vals = np.exp((min_dists - max_val) * beta)
    soft_max = max_val + beta_inv * np.log(np.sum(exp_vals))
    
    return np.degrees(soft_max)


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Generate a dense grid for the smooth surrogate
    # Use a Fibonacci sphere with many points
    G = 20000
    i = np.arange(G) + 0.5
    z = 1.0 - 2.0 * i / G
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    grid_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Initial configuration: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    best_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_radius = _covering_radius(best_pts)
    
    # Try multiple restarts with different initial configurations
    for restart in range(10):
        if time.time() > deadline - 30.0:
            break
        
        if restart == 0:
            pts = best_pts.copy()
        else:
            # Perturb the best solution
            rng = np.random.default_rng(42 + restart)
            pts = best_pts.copy()
            noise = rng.normal(0, 0.01 * (1 + restart*0.5), size=pts.shape)
            pts = pts + noise
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            pts = pts / norms
        
        # Optimize using L-BFGS-B with the smooth surrogate
        def objective(x):
            if time.time() > deadline - 10.0:
                return 180.0
            p = x.reshape(-1, 3)
            # Normalize
            norms = np.linalg.norm(p, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            p = p / norms
            return _smooth_covering_radius(p, grid_pts, beta=50.0)
        
        def gradient(x):
            if time.time() > deadline - 10.0:
                return np.zeros_like(x)
            p = x.reshape(-1, 3)
            norms = np.linalg.norm(p, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            p = p / norms
            
            # Numerical gradient
            eps = 1e-7
            grad = np.zeros_like(x)
            for i in range(len(x)):
                x_plus = x.copy()
                x_plus[i] += eps
                f_plus = objective(x_plus)
                
                x_minus = x.copy()
                x_minus[i] -= eps
                f_minus = objective(x_minus)
                
                grad[i] = (f_plus - f_minus) / (2 * eps)
            
            return grad
        
        x0 = pts.flatten()
        
        try:
            result = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 500}
            )
            
            if result.success or result.fun < best_radius:
                new_pts = result.x.reshape(-1, 3)
                norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                new_pts = new_pts / norms
                
                new_radius = _covering_radius(new_pts)
                if new_radius < best_radius:
                    best_pts = new_pts
                    best_radius = new_radius
                    
        except Exception:
            pass
    
    # Final polish: use a finer grid and more iterations
    if time.time() < deadline - 20.0:
        G_fine = 50000
        i = np.arange(G_fine) + 0.5
        z = 1.0 - 2.0 * i / G_fine
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        grid_pts_fine = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        
        def objective_fine(x):
            if time.time() > deadline - 5.0:
                return 180.0
            p = x.reshape(-1, 3)
            norms = np.linalg.norm(p, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            p = p / norms
            return _smooth_covering_radius(p, grid_pts_fine, beta=100.0)
        
        x0 = best_pts.flatten()
        
        try:
            result = minimize(
                objective_fine,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 300}
            )
            
            if result.success or result.fun < best_radius:
                new_pts = result.x.reshape(-1, 3)
                norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                new_pts = new_pts / norms
                
                new_radius = _covering_radius(new_pts)
                if new_radius < best_radius:
                    best_pts = new_pts
                    best_radius = new_radius
                    
        except Exception:
            pass
    
    # Ensure the result is valid
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    best_pts = best_pts / norms
    
    return best_pts
# EVOLVE-BLOCK-END
