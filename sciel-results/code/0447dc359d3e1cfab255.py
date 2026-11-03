"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull

_START_TIME = time.time()
_DEADLINE = _START_TIME + 145.0  # safety margin before 150s kill

def _to_sphere(pts):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return pts / norms

def _compute_hull_and_radii(pts):
    """
    Compute convex hull and circumradius for each facet.
    Returns:
        hull: ConvexHull object
        radii: array of circumradii for each facet
        max_radius: maximum circumradius
        critical_indices: indices of facets with circumradius close to max
    """
    try:
        hull = ConvexHull(pts)
    except Exception:
        return None, None, np.pi, []
    
    radii = np.zeros(len(hull.simplices))
    max_radius = 0.0
    
    for idx, simplex in enumerate(hull.simplices):
        a = pts[simplex[0]]
        b = pts[simplex[1]]
        c = pts[simplex[2]]
        
        # Vectors
        v1 = b - a
        v2 = c - a
        
        # Normal vector
        cross = np.cross(v1, v2)
        norm_cross = np.linalg.norm(cross)
        
        if norm_cross < 1e-12:
            radii[idx] = 0.0
            continue
            
        # Circumcenter direction
        n = cross / norm_cross
        
        # Check which side of the plane the origin is on relative to the triangle normal.
        centroid = (a + b + c) / 3.0
        if np.dot(n, centroid) > 0:
            C = n
        else:
            C = -n
            
        # C is already normalized
        dot_val = np.clip(np.dot(C, a), -1.0, 1.0)
        r = np.arccos(dot_val)
        radii[idx] = r
        
        if r > max_radius:
            max_radius = r
            
    # Find critical indices
    threshold = max_radius - 1e-6
    critical_indices = np.where(radii >= threshold)[0]
    
    return hull, radii, max_radius, critical_indices

def _worst_facet_projection(pts, max_iter=200, tol=1e-10):
    """
    Iteratively identify the convex hull facet with the largest spherical circumradius 
    and project the three defining vertices along their negative spherical gradients 
    to shrink that specific facet.
    """
    pts = _to_sphere(pts.copy())
    
    for _ in range(max_iter):
        if time.time() > _DEADLINE:
            break
            
        hull, radii, max_r, critical_indices = _compute_hull_and_radii(pts)
        if hull is None or len(critical_indices) == 0:
            break
            
        # Select the single worst facet
        worst_idx = critical_indices[0]
        simplex = hull.simplices[worst_idx]
        
        a_idx, b_idx, c_idx = simplex
        
        a = pts[a_idx]
        b = pts[b_idx]
        c = pts[c_idx]
        
        # Compute circumcenter C of the spherical triangle
        v1 = b - a
        v2 = c - a
        cross = np.cross(v1, v2)
        norm_cross = np.linalg.norm(cross)
        
        if norm_cross < 1e-12:
            break
            
        n = cross / norm_cross
        centroid = (a + b + c) / 3.0
        if np.dot(n, centroid) < 0:
            n = -n
            
        C = n
        
        # The objective is to minimize the distance from the circumcenter C to the vertices.
        # We want to move the vertices towards C.
        # On the sphere, the gradient of the distance from a point P to C is (C - P) projected onto the tangent plane at P.
        # Since C is on the sphere, the vector C - P is not necessarily tangent.
        # The tangent component is (C - P) - ((C-P).P)P.
        # Since P is on the unit sphere, (C-P).P = C.P - 1.
        # Let d = C.P.
        # grad_dist = (C - P) - (d - 1)P = C - P - dP + P = C - dP.
        # Wait, C - dP is the projection of C onto the tangent plane at P?
        # No, the gradient of f(P) = arccos(C.P) is -C / sqrt(1-(C.P)^2).
        # This vector is tangent to the sphere at P because C.P * (-C/sqrt(...)).P = -C.P^2/sqrt(...) != 0?
        # Actually, the gradient of the distance on the sphere is the projection of the Euclidean gradient.
        # Euclidean gradient of distance to C is (P-C)/||P-C||.
        # Project onto tangent plane at P: (P-C) - ((P-C).P)P.
        # (P-C).P = 1 - C.P.
        # So grad = (P-C) - (1-C.P)P = P - C - P + (C.P)P = (C.P)P - C.
        # This points from C towards P? No, it points in the direction of increasing distance.
        # We want to decrease distance, so we move in the direction -grad = C - (C.P)P.
        # This is the projection of C onto the tangent plane at P.
        
        # Let's use this direction.
        dir_a = C - np.dot(C, a) * a
        dir_b = C - np.dot(C, b) * b
        dir_c = C - np.dot(C, c) * c
        
        # Normalize directions
        norm_a = np.linalg.norm(dir_a)
        norm_b = np.linalg.norm(dir_b)
        norm_c = np.linalg.norm(dir_c)
        
        if norm_a < 1e-12 or norm_b < 1e-12 or norm_c < 1e-12:
            break
            
        dir_a /= norm_a
        dir_b /= norm_b
        dir_c /= norm_c
        
        # Step size
        # We want to shrink the facet. A small step is safer.
        # The radius is max_r. We can try a step proportional to max_r.
        step_size = 0.01
        
        new_a = _to_sphere(a + step_size * dir_a)
        new_b = _to_sphere(b + step_size * dir_b)
        new_c = _to_sphere(c + step_size * dir_c)
        
        new_pts = pts.copy()
        new_pts[a_idx] = new_a
        new_pts[b_idx] = new_b
        new_pts[c_idx] = new_c
        
        # Check if this move improves the global max radius
        _, _, new_max_r, _ = _compute_hull_and_radii(new_pts)
        
        if new_max_r < max_r - tol:
            pts = new_pts
            max_r = new_max_r
        else:
            # If it doesn't improve, try a smaller step or break
            # For simplicity, break or try half step
            step_size *= 0.5
            if step_size < 1e-6:
                break
            # Retry with smaller step
            new_a = _to_sphere(a + step_size * dir_a)
            new_b = _to_sphere(b + step_size * dir_b)
            new_c = _to_sphere(c + step_size * dir_c)
            
            new_pts = pts.copy()
            new_pts[a_idx] = new_a
            new_pts[b_idx] = new_b
            new_pts[c_idx] = new_c
            
            _, _, new_max_r2, _ = _compute_hull_and_radii(new_pts)
            if new_max_r2 < max_r - tol:
                pts = new_pts
                max_r = new_max_r2
            else:
                # Stuck, try to perturb slightly to escape local min? 
                # Or just break.
                break
                
    return pts

def construct():
    """
    Construct 50 points on the unit sphere minimizing the covering radius.
    """
    # Start with a good initial configuration (Fibonacci spiral)
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    pts = _to_sphere(pts)
    
    best_pts = pts.copy()
    _, _, best_obj, _ = _compute_hull_and_radii(pts)
    
    # Phase 1: Worst Facet Projection
    remaining = _DEADLINE - time.time()
    if remaining > 5:
        pts_proj = _worst_facet_projection(pts, max_iter=100)
        _, _, obj_proj, _ = _compute_hull_and_radii(pts_proj)
        if obj_proj < best_obj:
            best_obj = obj_proj
            best_pts = pts_proj.copy()
            
    # Phase 2: Refine with more iterations
    remaining = _DEADLINE - time.time()
    if remaining > 5:
        pts_proj2 = _worst_facet_projection(best_pts, max_iter=100)
        _, _, obj_proj2, _ = _compute_hull_and_radii(pts_proj2)
        if obj_proj2 < best_obj:
            best_obj = obj_proj2
            best_pts = pts_proj2.copy()
            
    # Final Validation and Normalization
    best_pts = _to_sphere(best_pts)
    
    # Check for coincident points
    dots = best_pts @ best_pts.T
    np.fill_diagonal(dots, 0)
    if np.max(np.abs(dots)) > 0.999999:
        # Perturb slightly
        best_pts = _to_sphere(best_pts + np.random.randn(N, 3) * 0.01)
        
    return best_pts
# EVOLVE-BLOCK-END
