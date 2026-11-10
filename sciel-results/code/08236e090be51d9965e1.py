import numpy as np
from scipy.optimize import minimize

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    """
    n = 12
    
    # Initial guess: A grid-like arrangement
    # We use a hexagonal grid layout for a better start
    # Rows of 4, 4, 4 hexagons
    # Vertical spacing between rows in hex grid is sqrt(3)
    # Horizontal spacing is 1.5 * 2? No, center to center is sqrt(3) for neighbors.
    # Let's just use a rectangular grid and let the optimizer fix it.
    
    # Spread out initially to avoid initial overlaps
    xs = np.array([0, -2, 2, -1, 1, -1, 1, -3, 3, -3, 3, 0])
    ys = np.array([0, 0, 0, 1.5, 1.5, -1.5, -1.5, 2.5, 2.5, -2.5, -2.5, -3])
    
    # Angles all 0
    angles = np.zeros(n)
    
    # Outer hexagon centered at 0,0, angle 0
    # Variable s
    s_init = 6.0
    
    # Variables: x1..x12, y1..y12, s
    x0 = np.concatenate([xs, ys, [s_init]])
    
    def get_hex_vertices(cx, cy, angle, side=1.0):
        a = np.deg2rad(angle)
        angles_rad = a + np.deg2rad(np.arange(6) * 60)
        return np.array([cx + side * np.cos(angles_rad), cy + side * np.sin(angles_rad)]).T
    
    def get_outer_vertices(s, cx=0, cy=0, angle=0):
        a = np.deg2rad(angle)
        angles_rad = a + np.deg2rad(np.arange(6) * 60)
        return np.array([cx + s * np.cos(angles_rad), cy + s * np.sin(angles_rad)]).T

    def penalty(params):
        xs_opt = params[:n]
        ys_opt = params[n:2*n]
        s_opt = params[2*n]
        
        # Ensure s is positive
        if s_opt <= 0:
            return 1e10
            
        # Calculate vertices for all inner hexagons
        inner_verts = [get_hex_vertices(xs_opt[i], ys_opt[i], 0) for i in range(n)]
        
        # Calculate outer vertices
        outer_verts = get_outer_vertices(s_opt)
        
        pen = 0.0
        
        # 1. Non-overlap constraint
        # For each pair, check min distance between edges
        # Approximation: Check distance between all vertices of one and all edges of the other?
        # Or use SAT.
        # For speed, we can just check if any vertex of hex i is inside hex j.
        # If so, penalty.
        # This is not sufficient (edges can cross without vertices inside), but for hexagons
        # starting from non-overlapping, it's a good local constraint.
        # Better: Use scipy.spatial.distance? No, we need polygon distance.
        
        # Let's use a simpler constraint: Center distance >= sqrt(3) - epsilon?
        # As established, this is not sufficient for all directions.
        # However, if we assume the optimizer keeps them roughly aligned, it might work.
        # Let's try a vertex-inside check.
        
        for i in range(n):
            for j in range(i+1, n):
                # Check if any vertex of i is in j
                for v in inner_verts[i]:
                    if point_in_polygon(v, inner_verts[j]):
                        pen += 100.0
                # Check if any vertex of j is in i
                for v in inner_verts[j]:
                    if point_in_polygon(v, inner_verts[i]):
                        pen += 100.0
        
        # 2. Containment constraint
        # All inner vertices must be inside outer hexagon
        for i in range(n):
            for v in inner_verts[i]:
                if not point_in_polygon(v, outer_verts):
                    # Penalty proportional to distance outside
                    # Simple: just add a large penalty
                    pen += 100.0
        
        # Objective: Minimize s + Penalty
        return s_opt + pen

    def point_in_polygon(point, polygon):
        # Ray casting algorithm
        x, y = point
        inside = False
        n_poly = len(polygon)
        j = n_poly - 1
        for i in range(n_poly):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside

    # Optimize
    # Use Nelder-Mead for robustness with non-smooth penalties
    res = minimize(penalty, x0, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-4, 'fatol': 1e-4})
    
    xs_opt = res.x[:n]
    ys_opt = res.x[n:2*n]
    s_opt = res.x[2*n]
    
    # Final validation
    inner_hex_data = np.zeros((n, 3))
    for i in range(n):
        inner_hex_data[i] = [xs_opt[i], ys_opt[i], 0]
    outer_hex_data = np.array([0, 0, 0])
    
    # Check validity strictly
    if not is_valid_packing(inner_hex_data, outer_hex_data, s_opt):
        # Fallback to a known valid packing (tighter than seed)
        # Let's construct a valid packing manually
        # A 3x3 grid of hexagons (9) + 3 more?
        # Let's just use the seed's configuration but with s=6.
        # Or better, use a configuration that is definitely valid.
        # The seed is valid. We can just return the seed if optimization fails.
        # But we want to beat it.
        # Let's try to return a slightly optimized seed.
        # Actually, if the optimizer fails, we can just return the seed.
        # Score will be 1.0.
        # But we expect it to succeed.
        
        # Let's try one more simple configuration:
        # 12 hexagons in a 2x6 strip?
        # No, let's just return the seed if invalid.
        return _get_seed_packing()
        
    return inner_hex_data, outer_hex_data, s_opt

def _get_seed_packing():
    inner_hex_data = np.array(
        [
            [0, 0, 0],
            [-2.5, 0, 0],
            [2.5, 0, 0],
            [-1.25, 2.17, 0],
            [1.25, 2.17, 0],
            [-1.25, -2.17, 0],
            [1.25, -2.17, 0],
            [-3.75, 2.17, 0],
            [3.75, 2.17, 0],
            [-3.75, -2.17, 0],
            [3.75, -2.17, 0],
            [0, -4, 0],
        ]
    )
    outer_hex_data = np.array([0, 0, 0])
    outer_hex_side_length = 8
    return inner_hex_data, outer_hex_data, outer_hex_side_length

def is_valid_packing(inner, outer, s):
    # Implement strict checks as per contract
    # 1. Finite values
    if not np.all(np.isfinite(inner)) or not np.all(np.isfinite(outer)) or not np.isfinite(s):
        return False
    if s < np.sqrt(12):
        return False
        
    n = len(inner)
    
    # Get vertices
    def get_verts(cx, cy, angle, side):
        a = np.deg2rad(angle)
        angles_rad = a + np.deg2rad(np.arange(6) * 60)
        return np.array([cx + side * np.cos(angles_rad), cy + side * np.sin(angles_rad)]).T

    inner_verts = [get_verts(inner[i,0], inner[i,1], inner[i,2], 1.0) for i in range(n)]
    outer_verts = get_verts(outer[0], outer[1], outer[2], s)
    
    # 2. Outer vertices distinct
    for i in range(6):
        for j in range(i+1, 6):
            if np.linalg.norm(outer_verts[i] - outer_verts[j]) < 1e-8:
                return False

    # 3. Edge length check
    # Shift centers by minus outer center
    shift = np.array([outer[0], outer[1]])
    for i in range(n):
        c = np.array([inner[i,0], inner[i,1]]) - shift
        v = get_verts(c[0], c[1], inner[i,2], 1.0)
        for k in range(6):
            p1 = v[k]
            p2 = v[(k+1)%6]
            if abs(np.linalg.norm(p2-p1) - 1.0) > 1e-8:
                return False

    # 4. Non-overlap (SAT)
    for i in range(n):
        for j in range(i+1, n):
            if not sat_non_overlap(inner_verts[i], inner_verts[j]):
                return False

    # 5. Containment
    for i in range(n):
        for v in inner_verts[i]:
            if not point_in_polygon_strict(v, outer_verts):
                return False

    return True

def sat_non_overlap(poly1, poly2):
    # Separating Axis Theorem
    # Get all edge normals
    def get_normals(poly):
        normals = []
        for i in range(len(poly)):
            p1 = poly[i]
            p2 = poly[(i+1)%len(poly)]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            normal = normal / np.linalg.norm(normal)
            normals.append(normal)
        return normals

    n1 = get_normals(poly1)
    n2 = get_normals(poly2)
    
    for n in n1 + n2:
        proj1 = np.dot(poly1, n)
        proj2 = np.dot(poly2, n)
        
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        # If intervals do not overlap, they are separated
        if max1 < min2 - 1e-6 or max2 < min1 - 1e-6:
            return True
            
    return False

def point_in_polygon_strict(point, polygon):
    # Cross product method for convex polygon
    # For every edge (p1, p2), cross(p2-p1, point-p1) must have same sign
    sign = None
    for i in range(len(polygon)):
        p1 = polygon[i]
        p2 = polygon[(i+1)%len(polygon)]
        cross = np.cross(p2 - p1, point - p1)
        if sign is None:
            sign = np.sign(cross)
        elif np.sign(cross) != sign and abs(cross) > 1e-8:
            # Allow for boundary
            if cross < -1e-6:
                return False
    return True
