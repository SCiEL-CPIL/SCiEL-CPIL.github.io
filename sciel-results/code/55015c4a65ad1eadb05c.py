"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

N = 12

# EVOLVE-BLOCK-START
def construct():
    """
    Place 12 disjoint unit regular hexagons inside a regular hexagon,
    minimizing the outer side length.
    
    Returns: (inner, center, side, angle)
    - inner: list of [x, y, angle_degrees] for each inner hexagon
    - center: (x, y) of outer hexagon center
    - side: outer hexagon side length
    - angle: outer hexagon rotation angle in degrees
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # A regular hexagon with side length s has vertices at:
    # (s*cos(theta + k*60°), s*sin(theta + k*60°)) for k=0..5
    # where theta is the rotation angle.
    
    # The distance from center to each vertex (circumradius) = s
    # The distance from center to each edge midpoint (inradius) = s * sqrt(3)/2
    
    def hex_vertices(cx, cy, side, angle_deg):
        """Return 6 vertices of a regular hexagon."""
        angle_rad = np.radians(angle_deg)
        angles = angle_rad + np.arange(6) * np.pi / 3
        return np.column_stack([cx + side * np.cos(angles),
                                 cy + side * np.sin(angles)])
    
    def hex_edge_distances(point, vertices):
        """
        Compute the minimum distance from a point to the boundary of a hexagon.
        Returns signed distance (negative if inside, positive if outside).
        For a convex polygon, we can compute distance to each edge and take min.
        """
        n = len(vertices)
        min_dist = np.inf
        for i in range(n):
            v1 = vertices[i]
            v2 = vertices[(i + 1) % n]
            # Distance from point to line segment v1-v2
            d = _point_to_segment_distance(point, v1, v2)
            min_dist = min(min_dist, d)
        
        # Determine if point is inside the hexagon using winding number or cross products
        inside = _point_in_convex_polygon(point, vertices)
        return -min_dist if inside else min_dist
    
    def _point_to_segment_distance(p, a, b):
        """Distance from point p to line segment ab."""
        ab = b - a
        ap = p - a
        ab_len_sq = np.dot(ab, ab)
        if ab_len_sq == 0:
            return np.linalg.norm(p - a)
        t = np.dot(ap, ab) / ab_len_sq
        t = np.clip(t, 0, 1)
        closest = a + t * ab
        return np.linalg.norm(p - closest)
    
    def _point_in_convex_polygon(p, vertices):
        """Check if point p is inside convex polygon defined by vertices."""
        n = len(vertices)
        sign = 0
        for i in range(n):
            v1 = vertices[i]
            v2 = vertices[(i + 1) % n]
            cross = np.cross(v2 - v1, p - v1)
            if cross > 1e-10:
                if sign < 0:
                    return False
                sign = 1
            elif cross < -1e-10:
                if sign > 0:
                    return False
                sign = -1
        return True
    
    def hex_hex_overlap_distance(h1_verts, h2_verts):
        """
        Compute a measure of separation between two hexagons.
        Returns positive if separated, negative if overlapping.
        We use the minimum of distances from each vertex of one to the other's boundary,
        and vice versa, combined with checking if any vertex is inside the other.
        """
        # Check if any vertex of h1 is inside h2
        for v in h1_verts:
            if _point_in_convex_polygon(v, h2_verts):
                return -1.0  # Overlapping
        
        # Check if any vertex of h2 is inside h1
        for v in h2_verts:
            if _point_in_convex_polygon(v, h1_verts):
                return -1.0  # Overlapping
        
        # Compute minimum distance between the two hexagons
        min_dist = np.inf
        for v1 in h1_verts:
            for v2 in h2_verts:
                d = np.linalg.norm(v1 - v2)
                min_dist = min(min_dist, d)
        
        # Also check edge-edge distances
        n1 = len(h1_verts)
        n2 = len(h2_verts)
        for i in range(n1):
            e1_start = h1_verts[i]
            e1_end = h1_verts[(i + 1) % n1]
            for j in range(n2):
                e2_start = h2_verts[j]
                e2_end = h2_verts[(j + 1) % n2]
                d = _segment_segment_distance(e1_start, e1_end, e2_start, e2_end)
                min_dist = min(min_dist, d)
        
        return min_dist
    
    def _segment_segment_distance(p1, p2, p3, p4):
        """Minimum distance between two line segments."""
        # Simple approach: check all vertex-to-segment distances
        d1 = _point_to_segment_distance(p1, p3, p4)
        d2 = _point_to_segment_distance(p2, p3, p4)
        d3 = _point_to_segment_distance(p3, p1, p2)
        d4 = _point_to_segment_distance(p4, p1, p2)
        return min(d1, d2, d3, d4)
    
    def inner_in_outer(inner_verts, outer_verts):
        """Check if all vertices of inner hex are inside outer hex."""
        for v in inner_verts:
            if not _point_in_convex_polygon(v, outer_verts):
                return False
        return True
    
    def objective(params):
        """
        Objective function: minimize outer side length while ensuring
        all inner hexagons are disjoint and inside the outer hexagon.
        
        params: 3*N + 2 = 38 parameters
        - For each of N inner hexagons: (x, y, angle_deg) -> 3*N
        - Outer hexagon: (side, angle_deg) -> 2
        """
        if time.time() > deadline:
            return 1e6
        
        # Parse parameters
        inner_params = params[:3 * N].reshape(N, 3)
        outer_side = params[3 * N]
        outer_angle = params[3 * N + 1]
        
        # Ensure outer side is positive
        if outer_side <= 0:
            return 1e6
        
        # Outer hexagon centered at origin
        outer_verts = hex_vertices(0.0, 0.0, outer_side, outer_angle)
        
        # Compute all inner hexagon vertices
        inner_verts_list = []
        for i in range(N):
            x, y, ang = inner_params[i]
            verts = hex_vertices(x, y, 1.0, ang)
            inner_verts_list.append(verts)
        
        # Penalty: outer hexagon must contain all inner hexagons
        penalty = 0.0
        for i in range(N):
            if not inner_in_outer(inner_verts_list[i], outer_verts):
                # Compute how far outside
                for v in inner_verts_list[i]:
                    if not _point_in_convex_polygon(v, outer_verts):
                        # Distance to outer boundary
                        min_d = np.inf
                        for j in range(6):
                            v1 = outer_verts[j]
                            v2 = outer_verts[(j + 1) % 6]
                            d = _point_to_segment_distance(v, v1, v2)
                            min_d = min(min_d, d)
                        penalty += min_d ** 2
        
        # Penalty: inner hexagons must be disjoint
        for i in range(N):
            for j in range(i + 1, N):
                sep = hex_hex_overlap_distance(inner_verts_list[i], inner_verts_list[j])
                if sep < 0:
                    penalty += (-sep) ** 2
        
        # We want to minimize outer_side, but penalize violations
        # Use a large penalty for constraint violations
        return outer_side + 100.0 * penalty
    
    # Initial guess: arrange hexagons in a roughly circular pattern
    # Unit hexagons have circumradius 1, so we need spacing of about 2 between centers
    # For 12 hexagons, place them in a ring
    
    best_result = None
    best_score = np.inf
    
    # Try multiple random initializations
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Random initial positions in a disk of radius ~5
        if restart == 0:
            # Structured initial guess: ring of 12 hexagons
            inner_params = np.zeros((N, 3))
            for i in range(N):
                angle = 2 * np.pi * i / N
                r = 3.5
                inner_params[i, 0] = r * np.cos(angle)
                inner_params[i, 1] = r * np.sin(angle)
                inner_params[i, 2] = 0.0
            outer_side = 6.0
            outer_angle = 0.0
        else:
            # Random initialization
            inner_params = np.zeros((N, 3))
            for i in range(N):
                angle = 2 * np.pi * i / N + np.random.uniform(-0.3, 0.3)
                r = np.random.uniform(2.0, 5.0)
                inner_params[i, 0] = r * np.cos(angle)
                inner_params[i, 1] = r * np.sin(angle)
                inner_params[i, 2] = np.random.uniform(0, 60)
            outer_side = np.random.uniform(5.0, 8.0)
            outer_angle = np.random.uniform(0, 60)
        
        params0 = np.concatenate([inner_params.flatten(), [outer_side, outer_angle]])
        
        # Bounds
        bounds = []
        for i in range(N):
            bounds.append((-10, 10))  # x
            bounds.append((-10, 10))  # y
            bounds.append((0, 60))    # angle
        bounds.append((2.0, 10.0))    # outer side
        bounds.append((0, 60))        # outer angle
        
        try:
            result = minimize(objective, params0, method='L-BFGS-B', bounds=bounds,
                             options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-8})
            
            if result.success or result.fun < best_score:
                score = result.fun
                if score < best_score:
                    best_score = score
                    best_result = result.x
        except Exception as e:
            pass
    
    # If no good result, use the structured arrangement
    if best_result is None:
        # Fallback: structured arrangement with estimated outer size
        inner_params = np.zeros((N, 3))
        for i in range(N):
            angle = 2 * np.pi * i / N
            r = 3.5
            inner_params[i, 0] = r * np.cos(angle)
            inner_params[i, 1] = r * np.sin(angle)
            inner_params[i, 2] = 0.0
        outer_side = 6.5
        outer_angle = 0.0
        best_result = np.concatenate([inner_params.flatten(), [outer_side, outer_angle]])
    
    # Parse best result
    inner_params = best_result[:3 * N].reshape(N, 3)
    outer_side = best_result[3 * N]
    outer_angle = best_result[3 * N + 1]
    
    # Build inner list
    inner = []
    for i in range(N):
        x, y, ang = inner_params[i]
        inner.append([float(x), float(y), float(ang)])
    
    return inner, (0.0, 0.0), float(outer_side), float(outer_angle)
# EVOLVE-BLOCK-END
