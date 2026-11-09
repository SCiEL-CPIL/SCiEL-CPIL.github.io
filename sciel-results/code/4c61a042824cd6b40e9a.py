"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
import time


def _normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-15:
        return v
    return v / n


def _covering_radius_degrees(pts):
    """Compute the covering radius in degrees."""
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.pi  # fallback: worst case
    
    # Get unique vertices
    verts = pts[hull.vertices]
    
    # For each facet, compute the circumradius (angular distance from facet center to farthest vertex)
    max_radius = 0.0
    
    for simplex in hull.simplices:
        # Get the vertices of this facet
        face_pts = pts[simplex]
        
        # Compute the circumcenter of this spherical triangle/face
        # For a spherical simplex, the circumcenter is the point equidistant from all vertices
        # We can find it by intersecting the perpendicular bisectors, but a simpler approach:
        # The circumcenter of a spherical triangle is the pole of the great circle through the vertices
        # Actually, for a facet of the convex hull, the covering radius contribution is the
        # angular distance from the facet's "center" to its vertices.
        
        # A simpler approach: the covering radius for a facet is the maximum angular distance
        # from any point in the facet's Voronoi region. The worst case is the circumcenter of the facet.
        
        # For a triangle (3 vertices), we can compute the circumcenter on the sphere
        if len(simplex) == 3:
            a, b, c = face_pts
            # Compute the circumcenter using vector operations
            # The circumcenter is proportional to the cross product of (b-a) and (c-a) rotated...
            # Actually, let's use a different approach:
            # The circumcenter of a spherical triangle is the normalized vector:
            # (a * sin(2*A) + b * sin(2*B) + c * sin(2*C)) where A, B, C are the angles
            # This is complex. Let's use a numerical approach instead.
            
            # Alternative: the circumradius of a spherical triangle can be computed from its angles
            # But we need the actual covering radius, which is the max distance from any point
            # in the spherical Voronoi cell to the nearest site.
            
            # Simpler: sample points on the facet and find max distance to nearest site
            # But this is slow. Let's use the property that for the convex hull,
            # the covering radius is the max over all facets of the circumradius.
            
            # For a planar triangle, circumradius = abc / (4 * area)
            # For a spherical triangle, we can use:
            # R = arccos((cos(a) + cos(b) + cos(c) - 1) / (2 * cos(A) * cos(B) * cos(C)))... no
            
            # Let's just compute it numerically: the circumcenter is the point on the sphere
            # equidistant from all three vertices. We can find it by solving:
            # |x - a|^2 = |x - b|^2 = |x - c|^2 with |x| = 1
            # This gives: x . (a - b) = x . (a - c) = 0
            # So x is perpendicular to (a-b) and (a-c), i.e., x is parallel to (a-b) x (a-c)
            
            n1 = np.cross(b - a, c - a)
            if np.linalg.norm(n1) < 1e-15:
                # Degenerate
                continue
            circumcenter = _normalize(n1)
            
            # Check if this is the correct pole (there are two poles)
            # The correct one should be on the same side as the facet normal
            facet_normal = _normalize(np.cross(b - a, c - a))
            if np.dot(circumcenter, facet_normal) < 0:
                circumcenter = -circumcenter
            
            # Compute the angular distance from circumcenter to each vertex
            for v in face_pts:
                cos_angle = np.clip(np.dot(circumcenter, v), -1.0, 1.0)
                angle = np.arccos(cos_angle)
                max_radius = max(max_radius, angle)
        
        elif len(simplex) == 4:
            # Tetrahedron: the circumcenter is the point equidistant from all 4 vertices
            # Solve: x . (v_i - v_0) = 0 for i = 1, 2, 3, with |x| = 1
            a, b, c, d = face_pts
            # x is perpendicular to (b-a), (c-a), (d-a)
            # So x is parallel to (b-a) x (c-a) if they are independent, but we need
            # x perpendicular to all three. This means x is the null space of the matrix
            # with rows (b-a), (c-a), (d-a). Since we're in R^3, this is only possible
            # if the three vectors are linearly dependent, which they are (all from a).
            # Actually, for 4 points in R^3, the circumcenter exists if they're not coplanar.
            # The circumcenter satisfies: |x - v_i|^2 = const for all i
            # |x|^2 - 2 x . v_i + |v_i|^2 = const
            # Since |x| = 1 and |v_i| = 1: 1 - 2 x . v_i + 1 = const
            # So x . v_i = (2 - const) / 2 = constant for all i
            # This means x . (v_i - v_j) = 0 for all i, j
            # So x is perpendicular to all differences, i.e., x is in the null space of
            # the matrix with rows (v_i - v_0) for i = 1, 2, 3
            M = np.array([b - a, c - a, d - a])
            # Find null space
            try:
                u, s, vh = np.linalg.svd(M)
                x = vh[-1]
                x = _normalize(x)
                
                # Check sign
                centroid = _normalize(a + b + c + d)
                if np.dot(x, centroid) < 0:
                    x = -x
                
                for v in face_pts:
                    cos_angle = np.clip(np.dot(x, v), -1.0, 1.0)
                    angle = np.arccos(cos_angle)
                    max_radius = max(max_radius, angle)
            except:
                continue
    
    return np.degrees(max_radius)


def _objective(pts):
    """Objective function: covering radius in degrees."""
    return _covering_radius_degrees(pts)


def _gradient_step(pts, step_size=0.01):
    """Perform one gradient descent step on the sphere."""
    # Compute the covering radius and identify the worst facet
    try:
        hull = ConvexHull(pts)
    except Exception:
        return pts
    
    max_radius = 0.0
    worst_simplex = None
    
    for simplex in hull.simplices:
        face_pts = pts[simplex]
        
        if len(simplex) == 3:
            a, b, c = face_pts
            n1 = np.cross(b - a, c - a)
            if np.linalg.norm(n1) < 1e-15:
                continue
            circumcenter = _normalize(n1)
            facet_normal = _normalize(np.cross(b - a, c - a))
            if np.dot(circumcenter, facet_normal) < 0:
                circumcenter = -circumcenter
            
            max_face_radius = 0.0
            for v in face_pts:
                cos_angle = np.clip(np.dot(circumcenter, v), -1.0, 1.0)
                angle = np.arccos(cos_angle)
                max_face_radius = max(max_face_radius, angle)
            
            if max_face_radius > max_radius:
                max_radius = max_face_radius
                worst_simplex = simplex
        
        elif len(simplex) == 4:
            a, b, c, d = face_pts
            M = np.array([b - a, c - a, d - a])
            try:
                u, s, vh = np.linalg.svd(M)
                x = _normalize(vh[-1])
                centroid = _normalize(a + b + c + d)
                if np.dot(x, centroid) < 0:
                    x = -x
                
                max_face_radius = 0.0
                for v in face_pts:
                    cos_angle = np.clip(np.dot(x, v), -1.0, 1.0)
                    angle = np.arccos(cos_angle)
                    max_face_radius = max(max_face_radius, angle)
                
                if max_face_radius > max_radius:
                    max_radius = max_face_radius
                    worst_simplex = simplex
            except:
                continue
    
    if worst_simplex is None:
        return pts
    
    # Move the vertices of the worst facet toward the circumcenter
    face_pts = pts[worst_simplex]
    
    # Compute the circumcenter again
    if len(worst_simplex) == 3:
        a, b, c = face_pts
        n1 = np.cross(b - a, c - a)
        circumcenter = _normalize(n1)
        facet_normal = _normalize(np.cross(b - a, c - a))
        if np.dot(circumcenter, facet_normal) < 0:
            circumcenter = -circumcenter
    else:
        a, b, c, d = face_pts
        M = np.array([b - a, c - a, d - a])
        try:
            u, s, vh = np.linalg.svd(M)
            circumcenter = _normalize(vh[-1])
            centroid = _normalize(a + b + c + d)
            if np.dot(circumcenter, centroid) < 0:
                circumcenter = -circumcenter
        except:
            return pts
    
    # Move each vertex toward the circumcenter
    new_pts = pts.copy()
    for idx in worst_simplex:
        v = new_pts[idx]
        # Move toward circumcenter
        direction = circumcenter - v
        direction = _normalize(direction)
        new_pts[idx] = _normalize(v + step_size * direction)
    
    return new_pts


def construct():
    # Golden-angle (Fibonacci) spiral points on the unit sphere: a good but improvable start.
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Optimize using gradient descent with multiple restarts
    best_pts = pts.copy()
    best_score = _objective(pts)
    
    start_time = time.time()
    deadline = start_time + 130.0  # Leave 20s margin
    
    # Phase 1: Gradient descent from the initial point
    step_size = 0.05
    for iteration in range(10000):
        if time.time() > deadline:
            break
        
        new_pts = _gradient_step(pts, step_size)
        new_score = _objective(new_pts)
        
        if new_score < best_score:
            best_score = new_score
            best_pts = new_pts.copy()
            pts = new_pts
        else:
            # Reduce step size
            step_size *= 0.9
            if step_size < 1e-6:
                break
    
    # Phase 2: Try multiple random restarts
    rng = np.random.default_rng(42)
    for restart in range(20):
        if time.time() > deadline:
            break
        
        # Random start
        pts_restart = rng.standard_normal((N, 3))
        pts_restart = np.array([_normalize(v) for v in pts_restart])
        
        # Optimize
        step_size = 0.05
        local_best = pts_restart.copy()
        local_best_score = _objective(pts_restart)
        
        for iteration in range(5000):
            if time.time() > deadline:
                break
            
            new_pts = _gradient_step(pts_restart, step_size)
            new_score = _objective(new_pts)
            
            if new_score < local_best_score:
                local_best_score = new_score
                local_best = new_pts.copy()
                pts_restart = new_pts
            else:
                step_size *= 0.9
                if step_size < 1e-6:
                    break
        
        if local_best_score < best_score:
            best_score = local_best_score
            best_pts = local_best.copy()
    
    return best_pts
# EVOLVE-BLOCK-END
