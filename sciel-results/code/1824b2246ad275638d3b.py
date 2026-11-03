import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

N = 12

def hex_vertices(cx, cy, side, angle_deg=0.0):
    """Get vertices of a regular hexagon centered at (cx, cy) with given side and rotation."""
    angle_rad = np.radians(angle_deg)
    vertices = []
    for i in range(6):
        theta = angle_rad + i * np.pi / 3
        x = cx + side * np.cos(theta)
        y = cy + side * np.sin(theta)
        vertices.append((x, y))
    return np.array(vertices)

def point_in_hexagon(px, py, cx, cy, side, angle_deg=0.0):
    """Check if point (px, py) is inside hexagon centered at (cx, cy)."""
    # Transform to hexagon's local frame
    dx = px - cx
    dy = py - cy
    angle_rad = np.radians(angle_deg)
    # Rotate point by -angle to align hexagon
    cos_a = np.cos(-angle_rad)
    sin_a = np.sin(-angle_rad)
    lx = dx * cos_a - dy * sin_a
    ly = dx * sin_a + dy * cos_a
    
    # For a regular hexagon with flat top (angle=0), vertices at angles 0, 60, 120, 180, 240, 300
    # The hexagon can be described by: |x| <= side, |y| <= side*sqrt(3)/2, |x| + |y|*2/sqrt(3) <= side
    # Actually, for a pointy-top hexagon (vertex at top), let me use the standard form
    # For angle=0, vertices are at 0, 60, 120, 180, 240, 300 degrees
    # This is a "flat-bottom" hexagon
    # The six half-planes defining the hexagon:
    # For each edge, check if point is on the correct side
    
    vertices = hex_vertices(0, 0, side, 0.0)
    for i in range(6):
        v1 = vertices[i]
        v2 = vertices[(i + 1) % 6]
        # Edge from v1 to v2
        ex = v2[0] - v1[0]
        ey = v2[1] - v1[1]
        # Normal pointing inward
        nx = -ey
        ny = ex
        # Check if point is on the inward side
        val = (lx - v1[0]) * nx + (ly - v1[1]) * ny
        if val < -1e-9:
            return False
    return True

def polygons_disjoint(v1, v2, tol=1e-7):
    """Check if two polygons are disjoint (no overlap). They can touch."""
    # Check if any vertex of v1 is inside v2 or vice versa
    for p in v1:
        if point_in_hexagon(p[0], p[1], 0, 0, 1.0, 0.0):
            # Need to check if p is inside v2's hexagon
            pass
    # Simpler: use shapely-like approach or check edge intersections
    # For now, use a simple vertex-in-polygon test
    def point_in_polygon(px, py, poly):
        n = len(poly)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = poly[i]
            xj, yj = poly[j]
            if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside
    
    # Check if any vertex of v1 is strictly inside v2
    for p in v1:
        if point_in_polygon(p[0], p[1], v2):
            return False
    for p in v2:
        if point_in_polygon(p[0], p[1], v1):
            return False
    
    # Check for edge intersections
    def segments_intersect(p1, p2, p3, p4):
        def ccw(a, b, c):
            return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])
        return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)
    
    for i in range(6):
        for j in range(6):
            if segments_intersect(v1[i], v1[(i+1)%6], v2[j], v2[(j+1)%6]):
                return False
    return True

def compute_min_outer_side(inner_hexes):
    """
    inner_hexes: list of (cx, cy, angle_deg) for N unit hexagons
    Returns: minimal side length of a regular hexagon that contains all inner hexagons
    """
    # Collect all vertices of all inner hexagons
    all_points = []
    for cx, cy, ang in inner_hexes:
        verts = hex_vertices(cx, cy, 1.0, ang)
        all_points.append(verts)
    all_points = np.vstack(all_points)
    
    # Find the minimal enclosing regular hexagon
    # A regular hexagon with center (hx, hy), side s, rotation theta
    # contains a point (px, py) if the point is inside the hexagon
    
    def hexagon_contains_all(hx, hy, s, theta_deg):
        theta_rad = np.radians(theta_deg)
        for px, py in all_points:
            # Check if point is inside hexagon at (hx, hy) with side s and rotation theta
            dx = px - hx
            dy = py - hy
            cos_t = np.cos(-theta_rad)
            sin_t = np.sin(-theta_rad)
            lx = dx * cos_t - dy * sin_t
            ly = dx * sin_t + dy * cos_t
            
            # For a regular hexagon with side s, flat-bottom orientation (vertices at 0, 60, ...)
            # The hexagon is defined by: |lx| <= s, |ly| <= s*sqrt(3)/2, and the diagonal constraints
            # More precisely, the six half-planes:
            # For i in range(6): the constraint from edge i
            # Edge i goes from vertex i to vertex i+1
            # Vertex i is at angle i*60 deg, distance s from center
            # Let me use the standard: for a hexagon with "pointy top" (vertex at 90 deg), 
            # but we have flat bottom (vertex at 0 deg)
            
            # For flat-bottom hexagon (vertices at 0, 60, 120, 180, 240, 300):
            # The constraints are:
            # |lx| <= s
            # |ly| <= s * sqrt(3) / 2
            # |lx| + |ly| * 2/sqrt(3) <= s  (for the diagonal edges)
            # Actually, let me just check against the 6 edges
            
            for k in range(6):
                a1 = np.radians(k * 60)
                a2 = np.radians((k + 1) * 60)
                v1x = s * np.cos(a1)
                v1y = s * np.sin(a1)
                v2x = s * np.cos(a2)
                v2y = s * np.sin(a2)
                ex = v2x - v1x
                ey = v2y - v1y
                nx = -ey
                ny = ex
                # Normalize normal
                norm = np.sqrt(nx*nx + ny*ny)
                nx /= norm
                ny /= norm
                # Check if point is on inward side
                val = (lx - v1x) * nx + (ly - v1y) * ny
                if val < -1e-9:
                    return False
        return True
    
    def objective(params):
        hx, hy, s, theta = params
        if s < 0:
            return 1e10
        if not hexagon_contains_all(hx, hy, s, theta):
            # Penalty: how much do we need to expand?
            # Compute max violation
            theta_rad = np.radians(theta)
            max_violation = 0
            for px, py in all_points:
                dx = px - hx
                dy = py - hy
                cos_t = np.cos(-theta_rad)
                sin_t = np.sin(-theta_rad)
                lx = dx * cos_t - dy * sin_t
                ly = dx * sin_t + dy * cos_t
                for k in range(6):
                    a1 = np.radians(k * 60)
                    a2 = np.radians((k + 1) * 60)
                    v1x = s * np.cos(a1)
                    v1y = s * np.sin(a1)
                    v2x = s * np.cos(a2)
                    v2y = s * np.sin(a2)
                    ex = v2x - v1x
                    ey = v2y - v1y
                    nx = -ey
                    ny = ex
                    norm = np.sqrt(nx*nx + ny*ny)
                    nx /= norm
                    ny /= norm
                    val = (lx - v1x) * nx + (ly - v1y) * ny
                    if val < 0:
                        max_violation = max(max_violation, -val)
            return s + max_violation
        return s
    
    # Initial guess
    # Center of all points
    cx_mean = np.mean(all_points[:, 0])
    cy_mean = np.mean(all_points[:, 1])
    # Rough side length: max distance from center
    dists = np.sqrt((all_points[:, 0] - cx_mean)**2 + (all_points[:, 1] - cy_mean)**2)
    s_guess = np.max(dists) * 1.5
    
    best_result = None
    best_score = 1e10
    
    # Try multiple random starts
    start_time = time.time()
    for trial in range(50):
        if time.time() - start_time > 100:
            break
        hx = cx_mean + np.random.uniform(-2, 2)
        hy = cy_mean + np.random.uniform(-2, 2)
        s = s_guess + np.random.uniform(-1, 1)
        theta = np.random.uniform(0, 60)
        
        result = minimize(objective, [hx, hy, s, theta], method='Nelder-Mead', 
                         options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4})
        if result.fun < best_score:
            best_score = result.fun
            best_result = result
    
    return best_score

def construct():
    start_time = time.time()
    deadline = 140  # Leave 10s margin
    
    # Initial honeycomb-like configuration
    # Place hexagons in a honeycomb pattern
    # Hexagon spacing: for adjacent hexagons, center distance = sqrt(3) for side-by-side
    # Or 2 for vertex-to-vertex
    
    # Let's try a compact arrangement:
    # Row 0: 4 hexagons
    # Row 1: 4 hexagons (offset)
    # Row 2: 4 hexagons
    # Total: 12
    
    # For flat-bottom hexagons, horizontal spacing = 2*side = 2
    # Vertical spacing = sqrt(3)*side = sqrt(3)
    # Offset rows shift by 1 horizontally
    
    def make_honeycomb():
        hexes = []
        # 3 rows of 4
        for row in range(3):
            for col in range(4):
                x = col * 2.0
                y = row * np.sqrt(3)
                if row % 2 == 1:
                    x += 1.0
                # Center the configuration
                x -= 3.0
                y -= np.sqrt(3)
                hexes.append([x, y, 0.0])
        return hexes
    
    inner = make_honeycomb()
    
    # Check if disjoint
    def is_valid(inner_hexes):
        verts_list = []
        for cx, cy, ang in inner_hexes:
            verts_list.append(hex_vertices(cx, cy, 1.0, ang))
        for i in range(len(verts_list)):
            for j in range(i+1, len(verts_list)):
                if not polygons_disjoint(verts_list[i], verts_list[j]):
                    return False
        return True
    
    if not is_valid(inner):
        # Adjust spacing
        scale = 1.01
        for h in inner:
            h[0] *= scale
            h[1] *= scale
        if not is_valid(inner):
            scale = 1.02
            for h in inner:
                h[0] *= scale
                h[1] *= scale
    
    # Compute initial outer side
    outer_side = compute_min_outer_side(inner)
    best_side = outer_side
    best_inner = [h[:] for h in inner]
    
    # Try to optimize positions
    if time.time() - start_time < deadline - 30:
        # Use differential evolution on positions
        def pack_objective(params):
            if time.time() - start_time > deadline - 10:
                return 1e10
            # params: 36 values (12 x, y, angle)
            hexes = []
            for i in range(12):
                x = params[i*3]
                y = params[i*3+1]
                ang = params[i*3+2]
                hexes.append([x, y, ang])
            
            # Check disjointness
            verts_list = []
            for cx, cy, ang in hexes:
                verts_list.append(hex_vertices(cx, cy, 1.0, ang))
            for i in range(12):
                for j in range(i+1, 12):
                    if not polygons_disjoint(verts_list[i], verts_list[j]):
                        return 1e10
            
            # Compute outer side
            side = compute_min_outer_side(hexes)
            return side
        
        # Bounds
        bounds = [(-5, 5)] * 36
        # Angles in [0, 60)
        for i in range(12):
            bounds[i*3+2] = (0, 60)
        
        try:
            result = differential_evolution(pack_objective, bounds, maxiter=50, 
                                           popsize=20, tol=1e-4, seed=42,
                                           polish=True)
            if result.fun < best_side:
                best_side = result.fun
                for i in range(12):
                    best_inner[i] = [result.x[i*3], result.x[i*3+1], result.x[i*3+2]]
        except:
            pass
    
    # Final computation of outer hexagon parameters
    if time.time() - start_time > deadline - 5:
        # Just return what we have
        pass
    
    # Compute the actual outer hexagon center and rotation
    all_points = []
    for cx, cy, ang in best_inner:
        verts = hex_vertices(cx, cy, 1.0, ang)
        all_points.append(verts)
    all_points = np.vstack(all_points)
    
    cx_mean = np.mean(all_points[:, 0])
    cy_mean = np.mean(all_points[:, 1])
    
    # Find optimal rotation and center for the outer hexagon
    def outer_obj(params):
        hx, hy, s, theta = params
        if s < 0:
            return 1e10
        theta_rad = np.radians(theta)
        for px, py in all_points:
            dx = px - hx
            dy = py - hy
            cos_t = np.cos(-theta_rad)
            sin_t = np.sin(-theta_rad)
            lx = dx * cos_t - dy * sin_t
            ly = dx * sin_t + dy * cos_t
            for k in range(6):
                a1 = np.radians(k * 60)
                a2 = np.radians((k + 1) * 60)
                v1x = s * np.cos(a1)
                v1y = s * np.sin(a1)
                v2x = s * np.cos(a2)
                v2y = s * np.sin(a2)
                ex = v2x - v1x
                ey = v2y - v1y
                nx = -ey
                ny = ex
                norm = np.sqrt(nx*nx + ny*ny)
                if norm < 1e-10:
                    continue
                nx /= norm
                ny /= norm
                val = (lx - v1x) * nx + (ly - v1y) * ny
                if val < -1e-9:
                    return s + abs(val) + 0.1
        return s
    
    best_outer = None
    best_outer_score = 1e10
    for trial in range(30):
        if time.time() - start_time > deadline - 2:
            break
        hx = cx_mean + np.random.uniform(-1, 1)
        hy = cy_mean + np.random.uniform(-1, 1)
        s = best_side + np.random.uniform(-0.5, 0.5)
        theta = np.random.uniform(0, 60)
        
        res = minimize(outer_obj, [hx, hy, s, theta], method='Nelder-Mead',
                      options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-4})
        if res.fun < best_outer_score:
            best_outer_score = res.fun
            best_outer = res
    
    if best_outer is not None:
        hx, hy, s, theta = best_outer.x
    else:
        hx, hy = cx_mean, cy_mean
        s = best_side
        theta = 0.0
    
    return best_inner, (hx, hy), s, theta

