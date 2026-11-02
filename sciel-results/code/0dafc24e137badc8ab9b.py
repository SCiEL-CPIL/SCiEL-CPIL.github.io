"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return pts


def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms


def _covering_radius_deg(pts):
    """Compute covering radius in degrees using convex hull facets."""
    from scipy.spatial import ConvexHull
    pts = _normalize(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        return 180.0
    
    # For each facet, compute the circumradius of the spherical cap
    # The covering radius is the max over facets of the angular distance
    # from the facet center to the farthest vertex (in angular terms)
    
    # A more accurate approach: for each facet, find the point on the sphere
    # that is farthest from all points in the facet (the circumcenter of the
    # spherical polygon), and compute the angular distance.
    # 
    # Actually, the covering radius equals the maximum over all facets of the
    # angular radius of the empty spherical cap centered at the facet's circumcenter.
    # For a planar facet with vertices v1,...,vk, the circumcenter in 3D is
    # the point equidistant from all vertices. The angular radius is the angle
    # between the circumcenter (normalized) and any vertex.
    
    max_radius = 0.0
    for simplex in hull.simplices:
        verts = pts[simplex]
        # Find circumcenter of the spherical polygon (triangle for 3D hull)
        # For a triangle on the sphere, the circumcenter is the normalized
        # cross-product-based solution, or we can use the fact that for small
        # caps it's approximately the normalized average.
        # More precisely: the circumcenter c satisfies |c - vi| = const for all i
        # which means c·vi = const for all i (since |c|=|vi|=1, |c-vi|^2 = 2-2c·vi)
        # So c·v1 = c·v2 = c·v3
        # (v1-v2)·c = 0 and (v1-v3)·c = 0
        # c is perpendicular to (v1-v2) and (v1-v3)
        c = np.cross(verts[0] - verts[1], verts[0] - verts[2])
        c_norm = np.linalg.norm(c)
        if c_norm < 1e-12:
            # Degenerate, use centroid
            c = verts.mean(axis=0)
            c_norm = np.linalg.norm(c)
            if c_norm < 1e-12:
                continue
            c = c / c_norm
        else:
            c = c / c_norm
        
        # Ensure c is on the correct side (same side as the interior)
        # The facet normal from hull should point outward; c should be on the
        # same side as the interior, so c·facet_normal < 0
        # Actually, let's just compute the angular distance
        # The angular radius is the angle between c and any vertex
        cos_angle = np.clip(np.dot(c, verts), -1.0, 1.0)
        angle = np.arccos(cos_angle)
        # The covering radius contribution is the maximum angular distance
        # from c to any vertex, but actually it's the angular distance from
        # the farthest point on the sphere to the nearest config point.
        # For an empty cap centered at c with angular radius r, the covering
        # radius is at least r. But we need the actual covering radius.
        # 
        # The correct approach: the covering radius is the maximum over all
        # points y on the sphere of min_i angle(y, p_i). This maximum is
        # achieved at a circumcenter of a facet of the convex hull of the
        # configuration points. For each facet, the relevant point is the
        # circumcenter of the spherical polygon, and the angular distance
        # to the nearest configuration point is the angular radius.
        
        # For a triangle facet, the circumcenter c (normalized) has angular
        # distance to each vertex. The covering radius contribution is this
        # angular distance (since c is equidistant to all three vertices,
        # and the nearest config point to c is one of these vertices or another).
        # Actually, we need min over ALL config points, not just the facet vertices.
        
        # For efficiency, let's just compute the angular distance from c to
        # all config points and take the min
        cos_all = np.clip(pts @ c, -1.0, 1.0)
        min_angle = np.arccos(cos_all).max()  # Wait, we want the min distance
        # Actually, the distance from c to the nearest point is min_i angle(c, p_i)
        # But angle(c, p_i) = arccos(c·p_i), so min angle = arccos(max c·p_i)
        max_dot = np.max(pts @ c)
        min_angle = np.arccos(np.clip(max_dot, -1.0, 1.0))
        
        if min_angle > max_radius:
            max_radius = min_angle
    
    return np.degrees(max_radius)


def _soft_covering_radius(pts, grid_pts, temperature=0.1):
    """Soft-max surrogate for the covering radius."""
    # Compute all pairwise angular distances
    # For efficiency, use dot products
    dots = grid_pts @ pts.T  # shape (M, N)
    angles = np.arccos(np.clip(dots, -1.0, 1.0))  # shape (M, N)
    min_angles = angles.min(axis=1)  # shape (M,)
    
    # Soft-max
    max_val = min_angles.max()
    shifted = (min_angles - max_val) / temperature
    log_sum_exp = np.log(np.sum(np.exp(shifted))) + max_val / temperature
    return log_sum_exp * temperature


def _params_to_points(params):
    """Convert parameters to points on the sphere."""
    # Use a parametrization that ensures points are on the sphere
    # We'll use the fact that we can represent each point by 2 parameters
    # (e.g., spherical coordinates) but that has singularities.
    # Instead, we'll work with the 3D coordinates and normalize.
    pts = params.reshape(N, 3)
    return _normalize(pts)


def _points_to_params(pts):
    return pts.flatten()


def _objective(params, grid_pts, temperature=0.1):
    pts = _params_to_points(params)
    return _soft_covering_radius(pts, grid_pts, temperature)


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Create a dense grid of sample points on the sphere
    # Use a Fibonacci sphere with many points for the grid
    n_grid = 20000
    grid_pts = _fibonacci_sphere(n_grid)
    
    # Initial configuration: Fibonacci spiral
    best_pts = _fibonacci_sphere(N)
    best_score = _covering_radius_deg(best_pts)
    
    # Try multiple restarts with different initial configurations
    rng = np.random.default_rng(42)
    
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        if restart == 0:
            init_params = _points_to_params(best_pts)
        else:
            # Perturbed initial configuration
            noise = rng.normal(0, 0.1, size=(N, 3))
            init_pts = _normalize(best_pts + noise)
            init_params = _points_to_params(init_pts)
        
        # Optimize using L-BFGS-B
        try:
            result = minimize(
                _objective,
                init_params,
                args=(grid_pts, 0.05),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if result.success or result.fun < best_score:
                opt_pts = _params_to_points(result.x)
                score = _covering_radius_deg(opt_pts)
                if score < best_score:
                    best_score = score
                    best_pts = opt_pts
        except Exception:
            pass
    
    # Final polish: targeted gradient descent on worst facets
    if time.time() < deadline - 10:
        for _ in range(20):
            if time.time() > deadline:
                break
            pts = _normalize(best_pts)
            # Find the worst facet
            from scipy.spatial import ConvexHull
            try:
                hull = ConvexHull(pts)
            except Exception:
                break
            
            worst_simplex = None
            worst_radius = 0.0
            for simplex in hull.simplices:
                verts = pts[simplex]
                c = np.cross(verts[0] - verts[1], verts[0] - verts[2])
                c_norm = np.linalg.norm(c)
                if c_norm < 1e-12:
                    c = verts.mean(axis=0)
                    c_norm = np.linalg.norm(c)
                    if c_norm < 1e-12:
                        continue
                    c = c / c_norm
                else:
                    c = c / c_norm
                
                max_dot = np.max(pts @ c)
                min_angle = np.arccos(np.clip(max_dot, -1.0, 1.0))
                
                if min_angle > worst_radius:
                    worst_radius = min_angle
                    worst_simplex = simplex
            
            if worst_simplex is None:
                break
            
            # Move the points in the worst simplex slightly toward the circumcenter
            c = np.cross(pts[worst_simplex[0]] - pts[worst_simplex[1]], 
                         pts[worst_simplex[0]] - pts[worst_simplex[2]])
            c_norm = np.linalg.norm(c)
            if c_norm < 1e-12:
                break
            c = c / c_norm
            
            # Move each point in the simplex toward c
            step_size = 0.01
            for idx in worst_simplex:
                best_pts[idx] = best_pts[idx] + step_size * c
            best_pts = _normalize(best_pts)
    
    return _normalize(best_pts)
# EVOLVE-BLOCK-END
