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
        # The circumcenter of the triangle on the sphere lies along the normal to the plane of the triangle.
        # We need to determine the sign.
        n = cross / norm_cross
        
        # Check which side of the plane the origin is on relative to the triangle normal.
        # Actually, simpler: The circumcenter C satisfies |C-a|^2 = |C-b|^2 = |C-c|^2.
        # Let C = k * n.
        # |k*n - a|^2 = k^2 - 2k(n.a) + |a|^2 = k^2 - 2k(n.a) + 1
        # |k*n - b|^2 = k^2 - 2k(n.b) + 1
        # So k(n.a) = k(n.b) => n.a = n.b (if k != 0). This is always true for the normal.
        # Wait, the circumcenter is the intersection of the normal line with the plane? No.
        # The circumcenter of a spherical triangle is the point on the sphere equidistant to the vertices.
        # It lies on the line passing through the origin and the circumcenter of the planar triangle?
        # No, the circumcenter of the spherical triangle is the intersection of the sphere with the line
        # passing through the origin and the circumcenter of the planar triangle formed by a,b,c?
        # Actually, the vector C is proportional to (a+b+c) only if the triangle is isosceles? No.
        
        # Correct method:
        # The circumcenter C on the sphere is the normalized vector of the circumcenter of the planar triangle?
        # No. The circumcenter of the planar triangle (a,b,c) is P_planar.
        # The vector from origin to P_planar is NOT necessarily the circumcenter on the sphere.
        # However, the circumcenter on the sphere C is in the plane of a,b,c? No, C is on the sphere.
        # The plane of a,b,c does not pass through the origin (usually).
        
        # Let's use the property: C is equidistant to a, b, c.
        # C . a = C . b = C . c = cos(r)
        # C is a unit vector.
        # C is in the span of a, b, c? Yes, because the set of points equidistant to a and b is a plane through the origin perpendicular to a-b.
        # So C is the intersection of the sphere with the line defined by the cross products?
        # C is parallel to (b-a) x (c-a) ? No, that's the normal to the plane.
        # The circumcenter C is the intersection of the sphere and the line passing through the origin and the circumcenter of the planar triangle?
        # Let's verify.
        # If C is the circumcenter on the sphere, then C is in the plane defined by a,b,c? No.
        # C is equidistant to a,b,c.
        # The locus of points equidistant to a and b is a plane through the origin with normal (a-b).
        # The locus of points equidistant to a and c is a plane through the origin with normal (a-c).
        # The intersection of these two planes is a line through the origin.
        # This line is perpendicular to (a-b) and (a-c), so it is parallel to (a-b) x (a-c).
        # So C is parallel to the normal of the plane containing a,b,c.
        # Yes! C is parallel to n = (a-b) x (a-c).
        
        # So C = ± n / ||n||.
        # Which sign?
        # The circumcenter must be on the same side of the great circle arcs as the triangle?
        # Actually, for a convex hull facet, the interior is on the side of the normal pointing towards the origin?
        # The normal n computed as (b-a)x(c-a) points outwards or inwards?
        # For a convex hull, the normal should point outwards (away from the center of mass/origin).
        # So if n . a > 0, it points outwards.
        # The circumcenter C is the point on the sphere farthest from the facet?
        # No, the circumcenter is the center of the empty cap.
        # The cap is the set of points y such that y is farther from a,b,c than any other point?
        # No, the Voronoi cell of the facet is the set of points closest to the facet's vertices?
        # No, the covering radius is the max distance to the nearest point.
        # The farthest points from the set {a,b,c} within the facet's Voronoi cell are the circumcenters of the triangle a,b,c.
        
        # So C is the circumcenter of the spherical triangle a,b,c.
        # C is parallel to n.
        # We need to choose the sign such that C is the "center" of the triangle.
        # If the triangle is small, C is close to the centroid.
        # Centroid G = (a+b+c)/3.
        # If n . G > 0, then C = n/||n||, else C = -n/||n||.
        
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

def _spherical_lloyd_iteration(pts, max_iter=50, tol=1e-8):
    """
    Perform Spherical Lloyd iterations.
    Moves each point to the centroid of its Voronoi cell (approximated by convex hull facets).
    This is a global optimization step that tends to equalize the covering radius.
    """
    pts = _to_sphere(pts.copy())
    
    for _ in range(max_iter):
        hull, radii, max_r, _ = _compute_hull_and_radii(pts)
        if hull is None:
            break
            
        new_pts = pts.copy()
        
        # For each point, find the facets incident to it.
        # The Voronoi cell of a point is the union of the facets incident to it.
        # The centroid of the Voronoi cell is the weighted average of the circumcenters of the incident facets?
        # Or the centroid of the polygon formed by the circumcenters?
        # A simpler approximation: Move each point towards the average of the circumcenters of the facets it belongs to.
        
        # Build incidence: point index -> list of facet indices
        incidence = [[] for _ in range(N)]
        for idx, simplex in enumerate(hull.simplices):
            for p_idx in simplex:
                incidence[p_idx].append(idx)
                
        # Compute circumcenters for all facets
        circumcenters = np.zeros((len(hull.simplices), 3))
        for idx, simplex in enumerate(hull.simplices):
            a = pts[simplex[0]]
            b = pts[simplex[1]]
            c = pts[simplex[2]]
            v1 = b - a
            v2 = c - a
            n = np.cross(v1, v2)
            nn = np.linalg.norm(n)
            if nn < 1e-12:
                continue
            n = n / nn
            centroid = (a + b + c) / 3.0
            if np.dot(n, centroid) < 0:
                n = -n
            circumcenters[idx] = n
            
        # Update points
        for i in range(N):
            if not incidence[i]:
                continue
            # Average the circumcenters of incident facets
            # Weight by the area of the facet? Or just average?
            # Averaging circumcenters is a standard heuristic for spherical Lloyd.
            avg_cc = np.mean(circumcenters[incidence[i]], axis=0)
            norm_avg = np.linalg.norm(avg_cc)
            if norm_avg < 1e-12:
                continue
            avg_cc = avg_cc / norm_avg
            
            # Move towards the centroid
            # Damped step
            new_pts[i] = pts[i] + 0.5 * (avg_cc - pts[i])
            
        new_pts = _to_sphere(new_pts)
        
        # Check convergence
        if np.linalg.norm(new_pts - pts) < tol:
            break
        pts = new_pts
        
    return pts

def _steepest_descent_critical(pts, max_iter=100, tol=1e-9):
    """
    Steepest descent on the critical facets.
    Moves the vertices of the critical facets towards their circumcenters.
    """
    pts = _to_sphere(pts.copy())
    
    for _ in range(max_iter):
        hull, radii, max_r, critical_indices = _compute_hull_and_radii(pts)
        if hull is None or len(critical_indices) == 0:
            break
            
        # Compute gradient
        grad = np.zeros_like(pts)
        
        for idx in critical_indices:
            simplex = hull.simplices[idx]
            a = pts[simplex[0]]
            b = pts[simplex[1]]
            c = pts[simplex[2]]
            
            v1 = b - a
            v2 = c - a
            n = np.cross(v1, v2)
            nn = np.linalg.norm(n)
            if nn < 1e-12:
                continue
            n = n / nn
            centroid = (a + b + c) / 3.0
            if np.dot(n, centroid) < 0:
                n = -n
            
            # C is the circumcenter
            C = n
            
            # Gradient of the radius w.r.t. a, b, c
            # r = arccos(C . a)
            # dr/da = -C / sqrt(1 - (C.a)^2)
            # But C depends on a, b, c. This is complex.
            # Heuristic: Move a, b, c towards C.
            
            grad[simplex[0]] += (C - a)
            grad[simplex[1]] += (C - b)
            grad[simplex[2]] += (C - c)
            
        # Normalize gradient
        grad_norms = np.linalg.norm(grad, axis=1, keepdims=True)
        grad_norms[grad_norms == 0] = 1.0
        grad = grad / grad_norms
        
        # Step
        step_size = 0.05
        new_pts = _to_sphere(pts + step_size * grad)
        
        # Check improvement
        _, _, new_max_r, _ = _compute_hull_and_radii(new_pts)
        
        if new_max_r < max_r:
            pts = new_pts
            if max_r - new_max_r < tol:
                break
        else:
            # Reduce step size
            step_size *= 0.5
            if step_size < 1e-6:
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
    
    deadline = _START_TIME + 145.0
    
    # Phase 1: Spherical Lloyd (Global/Coarse)
    remaining = deadline - time.time()
    if remaining > 10:
        pts_lloyd = _spherical_lloyd_iteration(pts, max_iter=20)
        _, _, obj_lloyd, _ = _compute_hull_and_radii(pts_lloyd)
        if obj_lloyd < best_obj:
            best_obj = obj_lloyd
            best_pts = pts_lloyd.copy()
            
    # Phase 2: Steepest Descent on Critical Facets (Fine)
    remaining = deadline - time.time()
    if remaining > 10:
        pts_sd = _steepest_descent_critical(best_pts, max_iter=50)
        _, _, obj_sd, _ = _compute_hull_and_radii(pts_sd)
        if obj_sd < best_obj:
            best_obj = obj_sd
            best_pts = pts_sd.copy()
            
    # Phase 3: Repeat Lloyd + SD for refinement
    remaining = deadline - time.time()
    if remaining > 10:
        pts_lloyd2 = _spherical_lloyd_iteration(best_pts, max_iter=10)
        pts_sd2 = _steepest_descent_critical(pts_lloyd2, max_iter=50)
        _, _, obj_final, _ = _compute_hull_and_radii(pts_sd2)
        if obj_final < best_obj:
            best_obj = obj_final
            best_pts = pts_sd2.copy()
            
    # Final Polish
    remaining = deadline - time.time()
    if remaining > 5:
        pts_polish = _steepest_descent_critical(best_pts, max_iter=20)
        _, _, obj_polish, _ = _compute_hull_and_radii(pts_polish)
        if obj_polish < best_obj:
            best_obj = obj_polish
            best_pts = pts_polish.copy()
            
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
