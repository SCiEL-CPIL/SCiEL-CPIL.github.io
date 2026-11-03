"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
import time


def _normalize(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return pts / norms


def _spherical_circumradius(p1, p2, p3):
    """Compute the spherical circumradius of a triangle on the unit sphere.
    
    Returns the angular distance from the circumcenter to each vertex.
    """
    # Vectors
    a = p1 - p2
    b = p3 - p2
    
    # Cross product gives normal direction
    n = np.cross(a, b)
    n_norm = np.linalg.norm(n)
    if n_norm < 1e-12:
        return 0.0
    
    # The circumcenter lies along the normal from the centroid
    # For a spherical triangle, the circumcenter is the point equidistant from all three vertices
    # We can find it by solving the system
    
    # Alternative: use the formula for spherical circumradius
    # The spherical circumradius R satisfies:
    # sin(R) = (a*b*c) / (4 * Area) where a,b,c are side lengths and Area is spherical excess
    
    # Side lengths (angular distances)
    def ang_dist(u, v):
        d = np.dot(u, v)
        d = np.clip(d, -1.0, 1.0)
        return np.arccos(d)
    
    side_a = ang_dist(p2, p3)
    side_b = ang_dist(p1, p3)
    side_c = ang_dist(p1, p2)
    
    # Spherical area using L'Huilier's formula
    s = (side_a + side_b + side_c) / 2.0
    if s <= 0 or s >= np.pi:
        return 0.0
    
    # Check for valid triangle
    if side_a >= np.pi or side_b >= np.pi or side_c >= np.pi:
        return 0.0
    
    # L'Huilier's formula for spherical excess
    try:
        tan_E_over_4 = np.sqrt(
            np.tan(s/2) * np.tan((s-side_a)/2) * np.tan((s-side_b)/2) * np.tan((s-side_c)/2)
        )
        E = 4 * np.arctan(tan_E_over_4)
    except:
        return 0.0
    
    if E <= 0:
        return 0.0
    
    # Spherical circumradius
    # sin(R) = (sin(a) * sin(b) * sin(c)) / (4 * sin(E/2) * ... ) 
    # Actually, for a spherical triangle, the circumradius R satisfies:
    # tan(R) = (tan(a/2) * tan(b/2) * tan(c/2)) / (cos((a+b+c)/2) + cos((a-b+c)/2) + cos((a+b-c)/2) + cos((a-b-c)/2))
    # This is complex. Let's use a simpler approach.
    
    # The circumcenter C satisfies |C - p1| = |C - p2| = |C - p3| on the sphere
    # C = normalize(alpha * p1 + beta * p2 + gamma * p3)
    # where alpha, beta, gamma are barycentric-like coordinates
    
    # Simpler: the circumradius is the angle from circumcenter to any vertex
    # We can find the circumcenter by intersecting the perpendicular bisectors
    
    # For a triangle on the unit sphere, the circumcenter is:
    # C = normalize(p1 * sin(2*B) + p2 * sin(2*A) + p3 * sin(2*C_angle))
    # where A, B, C_angle are the angles at the vertices
    
    # Angles at vertices using spherical law of cosines
    # cos(A) = (cos(a) - cos(b)*cos(c)) / (sin(b)*sin(c))
    def sphere_angle(opposite, adj1, adj2):
        cos_a = np.cos(opposite)
        cos_b = np.cos(adj1)
        cos_c = np.cos(adj2)
        sin_b = np.sin(adj1)
        sin_c = np.sin(adj2)
        if abs(sin_b * sin_c) < 1e-12:
            return 0.0
        cos_A = (cos_a - cos_b * cos_c) / (sin_b * sin_c)
        cos_A = np.clip(cos_A, -1.0, 1.0)
        return np.arccos(cos_A)
    
    A = sphere_angle(side_a, side_b, side_c)  # angle at p1, opposite side a
    B = sphere_angle(side_b, side_a, side_c)  # angle at p2, opposite side b  
    C = sphere_angle(side_c, side_a, side_b)  # angle at p3, opposite side c
    
    # Circumcenter
    C_center = (np.sin(2*B) * p1 + np.sin(2*A) * p2 + np.sin(2*C) * p3)
    C_center_norm = np.linalg.norm(C_center)
    if C_center_norm < 1e-12:
        return 0.0
    C_center = C_center / C_center_norm
    
    # Circumradius is angular distance from circumcenter to any vertex
    R = ang_dist(C_center, p1)
    return R


def _compute_covering_radius_and_facets(pts):
    """Compute the convex hull, find facets, and compute circumradii.
    
    Returns:
        covering_radius: the maximum circumradius in radians
        facets: list of facet indices (triangles)
        circumradii: array of circumradii for each facet
        circumcenters: array of circumcenters for each facet
    """
    pts = _normalize(pts)
    
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.pi, [], np.array([]), np.array([])
    
    facets = hull.simplices
    n_facets = len(facets)
    
    circumradii = np.zeros(n_facets)
    circumcenters = np.zeros((n_facets, 3))
    
    for i, facet in enumerate(facets):
        p1 = pts[facet[0]]
        p2 = pts[facet[1]]
        p3 = pts[facet[2]]
        
        R = _spherical_circumradius(p1, p2, p3)
        circumradii[i] = R
        
        # Recompute circumcenter for this facet
        a = p1 - p2
        b = p3 - p2
        n = np.cross(a, b)
        n_norm = np.linalg.norm(n)
        if n_norm > 1e-12:
            # Side lengths
            def ang_dist(u, v):
                d = np.dot(u, v)
                d = np.clip(d, -1.0, 1.0)
                return np.arccos(d)
            
            side_a = ang_dist(p2, p3)
            side_b = ang_dist(p1, p3)
            side_c = ang_dist(p1, p2)
            
            def sphere_angle(opposite, adj1, adj2):
                cos_a = np.cos(opposite)
                cos_b = np.cos(adj1)
                cos_c = np.cos(adj2)
                sin_b = np.sin(adj1)
                sin_c = np.sin(adj2)
                if abs(sin_b * sin_c) < 1e-12:
                    return 0.0
                cos_A = (cos_a - cos_b * cos_c) / (sin_b * sin_c)
                cos_A = np.clip(cos_A, -1.0, 1.0)
                return np.arccos(cos_A)
            
            A = sphere_angle(side_a, side_b, side_c)
            B = sphere_angle(side_b, side_a, side_c)
            C = sphere_angle(side_c, side_a, side_b)
            
            C_center = (np.sin(2*B) * p1 + np.sin(2*A) * p2 + np.sin(2*C) * p3)
            C_center_norm = np.linalg.norm(C_center)
            if C_center_norm > 1e-12:
                circumcenters[i] = C_center / C_center_norm
    
    covering_radius = np.max(circumradii) if n_facets > 0 else np.pi
    
    return covering_radius, facets, circumradii, circumcenters


def _gradient_step(pts, facets, circumradii, circumcenters, step_size):
    """Compute a gradient-based step to reduce the maximum circumradius.
    
    The idea: for facets with large circumradius, move their vertices away from
    the circumcenter to increase the radius (pushing points apart), and for facets
    with small circumradius, we don't need to do much. But we want to equalize.
    
    Actually, to minimize the max circumradius, we should move points on large-radius
    facets inward toward the circumcenter... but that's on the sphere.
    
    Better approach: The covering radius is determined by the largest facet circumradius.
    To reduce it, we need to move the vertices of the largest facets so that the
    circumradius decreases. This means moving them closer together (toward the 
    circumcenter of that facet).
    
    But we also want to avoid making other facets worse. So we use a combined objective:
    minimize max(circumradii) + lambda * variance(circumradii)
    """
    n = len(pts)
    grad = np.zeros_like(pts)
    
    if len(facets) == 0:
        return grad
    
    max_R = np.max(circumradii)
    mean_R = np.mean(circumradii)
    var_R = np.var(circumradii)
    
    # For each facet, compute the contribution to the gradient
    for i, facet in enumerate(facets):
        R = circumradii[i]
        C_center = circumcenters[i]
        
        if np.linalg.norm(C_center) < 1e-12:
            continue
        
        # Weight: larger for facets with larger circumradius (they dominate the max)
        # Also penalize variance to encourage equalization
        weight = R / max_R if max_R > 1e-12 else 1.0
        weight += 0.1 * (R - mean_R)  # variance penalty
        
        # The circumcenter is the point on the sphere farthest from the facet vertices
        # To reduce the circumradius, move vertices toward the circumcenter
        # The direction from each vertex to the circumcenter, projected tangent to the sphere
        for j, idx in enumerate(facet):
            p = pts[idx]
            # Direction from p to C_center, projected to tangent space at p
            d = C_center - p
            d = d - np.dot(d, p) * p  # project to tangent
            d_norm = np.linalg.norm(d)
            if d_norm > 1e-12:
                d = d / d_norm
            
            # Move toward circumcenter to reduce circumradius
            # But scale by weight
            grad[idx] += weight * d
    
    return grad


def construct():
    """Construct 130 points on the unit sphere minimizing the covering radius."""
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Initial configuration: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Initial evaluation
    best_pts = pts.copy()
    best_radius, best_facets, best_circumradii, best_circumcenters = _compute_covering_radius_and_facets(pts)
    
    # Gradient-based optimization
    step_size = 0.01
    patience = 0
    max_patience = 50
    
    while time.time() < deadline:
        # Compute gradient
        grad = _gradient_step(pts, best_facets, best_circumradii, best_circumcenters, step_size)
        
        # Normalize gradient to unit sphere tangent
        # Apply gradient step
        new_pts = pts - step_size * grad
        new_pts = _normalize(new_pts)
        
        # Evaluate new configuration
        new_radius, new_facets, new_circumradii, new_circumcenters = _compute_covering_radius_and_facets(new_pts)
        
        if new_radius < best_radius:
            best_radius = new_radius
            best_pts = new_pts.copy()
            best_facets = new_facets
            best_circumradii = new_circumradii
            best_circumcenters = new_circumcenters
            pts = new_pts.copy()
            patience = 0
            # Decrease step size slightly on improvement
            step_size *= 0.99
        else:
            patience += 1
            # Backtrack: revert step
            step_size *= 0.8
            if patience > max_patience:
                break
        
        # Check time
        if time.time() > deadline - 5.0:
            break
    
    return _normalize(best_pts)

# EVOLVE-BLOCK-END
