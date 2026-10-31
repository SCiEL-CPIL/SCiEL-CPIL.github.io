import numpy as np
from scipy.optimize import minimize
import time

N = 11
START_TIME = time.time()
TIME_LIMIT = 120.0
SAFETY_MARGIN = 5.0

def get_hex_vertices(x, y, s, angle_deg):
    """
    Get vertices of a regular hexagon.
    x, y: center
    s: side length
    angle_deg: rotation in degrees
    """
    angles = np.deg2rad(angle_deg) + np.deg2rad(60 * np.arange(6))
    # Standard hexagon vertices for side length s centered at origin
    # Vertices are at distance s from center (circumradius = s for regular hexagon)
    # Wait, for a regular hexagon, side length = circumradius.
    # So distance from center to vertex is s.
    vertices = np.column_stack([s * np.cos(angles), s * np.sin(angles)])
    vertices[:, 0] += x
    vertices[:, 1] += y
    return vertices

def check_collision(hex1, hex2, eps=1e-9):
    """
    Check if two hexagons (defined by center, side, angle) collide.
    hex1: (x1, y1, s1, a1)
    hex2: (x2, y2, s2, a2)
    Returns True if they overlap (distance between centers < min_dist).
    For unit hexagons, minimum distance between centers for non-overlapping is 2 * inradius?
    No, for regular hexagons of side 1:
    Inradius (apothem) = sqrt(3)/2 * s.
    Circumradius = s.
    If aligned, min distance between centers is 2 * inradius = sqrt(3).
    If rotated, the distance required depends on orientation.
    Simplest robust check: compute minimum distance between the two convex polygons.
    If min distance > eps, no collision.
    """
    x1, y1, s1, a1 = hex1
    x2, y2, s2, a2 = hex2
    
    v1 = get_hex_vertices(x1, y1, s1, a1)
    v2 = get_hex_vertices(x2, y2, s2, a2)
    
    # Use scipy.spatial to check distance between two convex hulls?
    # Or simpler: since they are convex, we can check if any edge of v1 intersects any edge of v2,
    # or if one is inside the other.
    # Given N=11, we can afford a slightly more expensive check if needed, but let's try a quick bounding box first.
    
    # Bounding box check
    min_x1, max_x1 = np.min(v1[:, 0]), np.max(v1[:, 0])
    min_y1, max_y1 = np.min(v1[:, 1]), np.max(v1[:, 1])
    min_x2, max_x2 = np.min(v2[:, 0]), np.max(v2[:, 0])
    min_y2, max_y2 = np.min(v2[:, 1]), np.max(v2[:, 1])
    
    if max_x1 <= min_x2 + eps or max_x2 <= min_x1 + eps or max_y1 <= min_y2 + eps or max_y2 <= min_y1 + eps:
        return False
    
    # For regular hexagons of same size, if distance between centers > 2*inradius, they don't collide.
    # inradius = sqrt(3)/2 * s.
    # dist = sqrt((x1-x2)^2 + (y1-y2)^2)
    # However, rotation matters. The minimum distance between two regular hexagons of side s
    # is achieved when they are aligned, and is sqrt(3)*s.
    # If rotated, the "effective" radius can be larger (up to s).
    # So a safe check is dist > 2*s. But that's conservative.
    # Let's use a more precise check: distance between two convex polygons.
    
    # We can use scipy.spatial.ConvexHull and distance? No, scipy doesn't have direct convex-convex distance.
    # We can check if any vertex of v1 is inside v2 or vice versa, or if edges intersect.
    # Given small N, let's just check vertex-in-polygon and edge intersections.
    
    def point_in_poly(pt, poly):
        # Ray casting algorithm
        x, y = pt
        inside = False
        n = len(poly)
        j = n - 1
        for i in range(n):
            xi, yi = poly[i]
            xj, yj = poly[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside

    def segments_intersect(p1, p2, p3, p4):
        # Check if segment p1-p2 intersects p3-p4
        def ccw(a, b, c):
            return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])
        return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)

    # Check if any vertex of v1 is in v2
    for pt in v1:
        if point_in_poly(pt, v2):
            return True
    # Check if any vertex of v2 is in v1
    for pt in v2:
        if point_in_poly(pt, v1):
            return True
    # Check edge intersections
    for i in range(6):
        if segments_intersect(v1[i], v1[(i+1)%6], v2[i], v2[(i+1)%6]):
            return True
            
    return False

def check_outer_containment(inner_hexes, outer_center, outer_side, outer_angle):
    """
    Check if all inner hexagons are inside the outer hexagon.
    """
    outer_verts = get_hex_vertices(outer_center[0], outer_center[1], outer_side, outer_angle)
    
    def point_in_poly(pt, poly):
        x, y = pt
        inside = False
        n = len(poly)
        j = n - 1
        for i in range(n):
            xi, yi = poly[i]
            xj, yj = poly[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside

    for hex_def in inner_hexes:
        x, y, s, a = hex_def
        verts = get_hex_vertices(x, y, s, a)
        for pt in verts:
            if not point_in_poly(pt, outer_verts):
                return False
    return True

def objective(params):
    """
    params: [x1, y1, a1, x2, y2, a2, ..., xN, yN, aN, outer_x, outer_y, outer_side, outer_angle]
    Returns outer_side + penalty if invalid.
    """
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return 1e10
        
    n_hex = N
    inner_hexes = []
    for i in range(n_hex):
        x = params[i*3]
        y = params[i*3+1]
        a = params[i*3+2]
        inner_hexes.append((x, y, 1.0, a)) # Side length fixed at 1.0
        
    outer_center = (params[n_hex*3], params[n_hex*3+1])
    outer_side = params[n_hex*3+2]
    outer_angle = params[n_hex*3+3]
    
    # Ensure outer_side is positive
    if outer_side < 0:
        return 1e10
        
    # Check collisions
    for i in range(n_hex):
        for j in range(i+1, n_hex):
            if check_collision(inner_hexes[i], inner_hexes[j]):
                return 1e10 + outer_side # Penalize
                
    # Check containment
    if not check_outer_containment(inner_hexes, outer_center, outer_side, outer_angle):
        return 1e10 + outer_side
        
    return outer_side

def construct():
    global START_TIME
    START_TIME = time.time()
    
    # Initial configuration: Hexagonal flower pattern
    # Center at (0,0)
    # 6 neighbors at distance sqrt(3) in directions 0, 60, 120, 180, 240, 300
    # But wait, if we place them at distance sqrt(3), they touch.
    # To minimize outer hexagon, we want them as tight as possible.
    # A "flower" of 7 hexagons (1 center, 6 around) fits in a larger hexagon.
    # For N=11, we can add 4 more.
    # Let's start with a compact cluster.
    
    # Coordinates for 7 hexagons (side 1)
    # Center: (0,0)
    # Neighbors: (sqrt(3), 0), (sqrt(3)/2, 1.5), (-sqrt(3)/2, 1.5), (-sqrt(3), 0), (-sqrt(3)/2, -1.5), (sqrt(3)/2, -1.5)
    # Note: y-offset is 3/2 * s? No.
    # Inradius = sqrt(3)/2. Distance between centers of touching hexes = 2 * inradius = sqrt(3).
    # So x-offset = sqrt(3), y-offset = 0.
    # Next ring?
    
    # Let's just place them in a grid-like structure initially and let optimizer fix it.
    # Or better, use a known good packing for 11 hexagons.
    # Actually, let's just initialize with a random-ish but valid cluster.
    
    sqrt3 = np.sqrt(3)
    init_pos = []
    # 1 center
    init_pos.append((0, 0))
    # 6 around
    for i in range(6):
        angle = i * np.pi / 3
        init_pos.append((sqrt3 * np.cos(angle), sqrt3 * np.sin(angle)))
    # 4 more... let's put them in the gaps or extend the ring.
    # To keep it simple and valid, let's start with a line or a small grid and let optimizer work.
    # A line is easy to initialize but hard to optimize to a compact shape.
    # Let's try a 3x4 grid of hexagons?
    # Or just use the 7-hex flower and add 4 more at slightly larger radius.
    
    # Let's use a simpler initialization: all in a line, then optimize.
    # But a line is far from optimal.
    # Let's try to initialize with a "honeycomb" patch.
    
    # For N=11, a good shape is a hexagon of side 2 (which has 7 hexes) plus 4 more.
    # Or a hexagon of side 2 is 1+6=7.
    # Side 3 is 1+6+12=19.
    # So 11 is between.
    
    # Let's initialize with a random valid configuration.
    # We can generate random positions and check validity, or just use a heuristic.
    # Given the time limit, a good initial guess is crucial.
    
    # Let's try to place them in a circle.
    # Radius R such that 11 hexes fit.
    # Area of 11 hexes = 11 * (3*sqrt(3)/2) * 1^2 ≈ 28.58.
    # Area of outer hex with side S = (3*sqrt(3)/2) * S^2.
    # So S^2 ≈ 28.58 / (3*sqrt(3)/2) ≈ 11. So S ≈ 3.3.
    # So outer side should be around 3.3 to 3.5.
    
    # Let's initialize inner hexes in a spiral or grid.
    # Grid:
    # Row 0: 3 hexes
    # Row 1: 4 hexes
    # Row 2: 4 hexes
    # Total 11.
    
    # Hex grid coordinates (axial):
    # (0,0), (1,0), (2,0)
    # (0,1), (1,1), (2,1), (3,1)
    # (0,2), (1,2), (2,2), (3,2)
    
    # Convert axial to cartesian:
    # x = sqrt(3) * (q + r/2)
    # y = 3/2 * r
    
    init_cart = []
    qrs = [
        (0,0), (1,0), (2,0),
        (0,1), (1,1), (2,1), (3,1),
        (0,2), (1,2), (2,2), (3,2)
    ]
    for q, r in qrs:
        x = sqrt3 * (q + r/2)
        y = 1.5 * r
        init_cart.append((x, y))
        
    # Center the cluster
    xs = [p[0] for p in init_cart]
    ys = [p[1] for p in init_cart]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    init_cart = [(x - cx, y - cy) for x, y in init_cart]
    
    # Initial outer hexagon
    # Estimate side length
    max_dist = max(np.sqrt(x**2 + y**2) for x, y in init_cart)
    # Max distance from center to vertex of inner hex is 1.
    # So outer radius must be at least max_dist + 1.
    # Outer hex circumradius = S.
    # So S >= max_dist + 1.
    init_side = max_dist + 1.5 # Add some margin
    
    # Build initial params
    params = []
    for x, y in init_cart:
        params.extend([x, y, 0.0]) # angle 0
    params.extend([0.0, 0.0, init_side, 0.0]) # outer center, side, angle
    
    # Optimization
    best_params = np.array(params)
    best_score = objective(best_params)
    
    # Nelder-Mead
    # We need to be careful with the time limit.
    # Nelder-Mead can be slow.
    # Let's run it with a timeout.
    
    try:
        res = minimize(objective, best_params, method='Nelder-Mead', 
                       options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-4, 'adaptive': True})
        
        if res.success or res.fun < best_score:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                best_params = res.x
                best_score = res.fun
    except Exception:
        pass
    
    # If the optimized solution is invalid (penalty applied), fall back to initial if it was valid?
    # Or just return the best found.
    
    # Extract result
    n_hex = N
    inner = []
    for i in range(n_hex):
        x = best_params[i*3]
        y = best_params[i*3+1]
        a = best_params[i*3+2]
        inner.append([x, y, a])
        
    outer_center = (best_params[n_hex*3], best_params[n_hex*3+1])
    outer_side = best_params[n_hex*3+2]
    outer_angle = best_params[n_hex*3+3]
    
    # Final validity check
    inner_hexes = [(x, y, 1.0, a) for x, y, a in inner]
    valid = True
    for i in range(n_hex):
        for j in range(i+1, n_hex):
            if check_collision(inner_hexes[i], inner_hexes[j]):
                valid = False
                break
        if not valid: break
        
    if valid:
        valid = check_outer_containment(inner_hexes, outer_center, outer_side, outer_angle)
        
    if not valid:
        # Fallback to a very safe, large hexagon
        # Just return the initial setup with a large outer side
        # Or the seed program's logic?
        # Let's just return a valid, albeit suboptimal, configuration.
        # We can use the initial cart positions and a large side.
        inner = [[x, y, 0.0] for x, y in init_cart]
        outer_center = (0.0, 0.0)
        outer_side = 10.0
        outer_angle = 0.0
        
    return inner, outer_center, outer_side, outer_angle
