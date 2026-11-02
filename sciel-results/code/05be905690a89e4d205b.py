import numpy as np
from scipy.optimize import minimize
import time

N = 12
DEADLINE = 140.0  # safety margin before 150s kill

def hexagon_vertices(cx, cy, side, angle_deg):
    """Get 6 vertices of a regular hexagon."""
    angle_rad = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = angle_rad + i * np.pi / 3
        x = cx + side * np.cos(theta)
        y = cy + side * np.sin(theta)
        verts.append([x, y])
    return np.array(verts)

def min_dist_polygons(v1, v2):
    """Minimum distance between two convex polygons (hexagons).
    Returns 0 if they overlap or touch, positive if separated."""
    # Check if any vertex of v1 is inside v2 or vice versa
    # For simplicity, compute min distance between all edge pairs
    min_d = np.inf
    n1, n2 = len(v1), len(v2)
    for i in range(n1):
        p1 = v1[i]
        p2 = v1[(i+1) % n1]
        for j in range(n2):
            q1 = v2[j]
            q2 = v2[(j+1) % n2]
            # Point-to-segment distance from p1 to segment q1-q2
            d1 = point_segment_dist(p1, q1, q2)
            d2 = point_segment_dist(p2, q1, q2)
            d3 = point_segment_dist(q1, p1, p2)
            d4 = point_segment_dist(q2, p1, p2)
            min_d = min(min_d, d1, d2, d3, d4)
    return min_d

def point_segment_dist(p, a, b):
    """Distance from point p to segment ab."""
    ab = b - a
    ap = p - a
    t = np.dot(ap, ab) / (np.dot(ab, ab) + 1e-12)
    t = np.clip(t, 0, 1)
    closest = a + t * ab
    return np.linalg.norm(p - closest)

def get_all_vertices(inner, outer_center, outer_side, outer_angle):
    """Get vertices of all inner hexagons and outer hexagon."""
    inner_verts = []
    for row in inner:
        x, y, ang = row[0], row[1], row[2]
        v = hexagon_vertices(x, y, 1.0, ang)
        inner_verts.append(v)
    outer_v = hexagon_vertices(outer_center[0], outer_center[1], outer_side, outer_angle)
    return inner_verts, outer_v

def objective(params, inner_count=12):
    """Objective: minimize outer side length with collision constraints.
    params layout: [x1,y1,a1, x2,y2,a2, ..., x12,y2,a12, cx,cy, side, oang]
    Total: 3*12 + 4 = 40 params
    """
    deadline = time.time() + 0.5
    if time.time() > deadline:
        return 1e6
    
    inner = []
    for i in range(inner_count):
        x = params[3*i]
        y = params[3*i+1]
        a = params[3*i+2]
        inner.append([x, y, a])
    
    cx = params[36]
    cy = params[37]
    side = params[38]
    oang = params[39]
    
    # Ensure side is positive
    if side < 1.0:
        return 1e6
    
    inner_verts, outer_v = get_all_vertices(inner, (cx, cy), side, oang)
    
    # Penalty for inner hexagons going outside outer hexagon
    penalty = 0.0
    for iv in inner_verts:
        for v in iv:
            # Check if vertex v is outside outer hexagon
            # Use signed distance: for a regular hexagon centered at (cx,cy) with side s
            # and angle oang, a point is inside if it satisfies all 6 half-plane constraints
            # Simpler: check if point is within the convex hull
            if not point_in_polygon(v, outer_v):
                # Distance to nearest edge of outer hex
                d = min_dist_point_polygon(v, outer_v)
                penalty += d * d
    
    # Penalty for inner hexagons overlapping each other
    for i in range(inner_count):
        for j in range(i+1, inner_count):
            d = min_dist_polygons(inner_verts[i], inner_verts[j])
            if d < 0.1:
                penalty += (0.1 - d) ** 2
    
    return side + penalty

def point_in_polygon(p, poly):
    """Ray casting algorithm to check if point p is inside polygon poly."""
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > p[1]) != (yj > p[1])) and (p[0] < (xj - xi) * (p[1] - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside

def min_dist_point_polygon(p, poly):
    """Minimum distance from point p to polygon edges."""
    min_d = np.inf
    n = len(poly)
    for i in range(n):
        d = point_segment_dist(p, poly[i], poly[(i+1) % n])
        min_d = min(min_d, d)
    return min_d

def make_honeycomb_init(scale=1.0):
    """Create a honeycomb arrangement of 12 hexagons."""
    inner = []
    # Honeycomb: rows of hexagons offset by half
    # For 12 hexagons: 3 rows of 4, or 2 rows of 6, etc.
    # Let's use a 3-4-3-2 arrangement or similar compact layout
    # Actually, let's use a proper honeycomb cluster
    
    # Hexagonal lattice spacing: for unit hexagons, center-to-center distance is sqrt(3)
    dx = np.sqrt(3) * scale
    dy = 1.5 * scale
    
    # 3 rows: 4, 4, 4
    positions = []
    for row in range(3):
        for col in range(4):
            x = col * dx + (row % 2) * dx / 2
            y = row * dy
            positions.append([x, y, 0.0])
    
    # Center them
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    for p in positions:
        p[0] -= cx
        p[1] -= cy
    
    return positions

def make_ring_init():
    """Create a ring of hexagons around a center."""
    inner = []
    # 6 in a ring + 6 more
    r1 = np.sqrt(3)
    for i in range(6):
        ang = i * np.pi / 3
        x = r1 * np.cos(ang)
        y = r1 * np.sin(ang)
        inner.append([x, y, 0.0])
    # Add 6 more in a larger ring
    r2 = 2 * np.sqrt(3)
    for i in range(6):
        ang = i * np.pi / 3 + np.pi / 6
        x = r2 * np.cos(ang)
        y = r2 * np.sin(ang)
        inner.append([x, y, 0.0])
    return inner

def make_grid_init():
    """Create a grid-like arrangement."""
    inner = []
    dx = np.sqrt(3)
    dy = 1.5
    # 4x3 grid
    for row in range(3):
        for col in range(4):
            x = col * dx + (row % 2) * dx / 2
            y = row * dy
            inner.append([x, y, 0.0])
    # Center
    xs = [p[0] for p in inner]
    ys = [p[1] for p in inner]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    for p in inner:
        p[0] -= cx
        p[1] -= cy
    return inner

def estimate_outer_size(inner):
    """Estimate a good outer side length from inner positions."""
    xs = [p[0] for p in inner]
    ys = [p[1] for p in inner]
    # Add margin for hexagon radius
    margin = 1.0
    max_x = max(abs(x) for x in xs) + margin
    max_y = max(abs(y) for y in ys) + margin
    # For a regular hexagon with side s, the "radius" to vertex is s
    # The width across flats is s * sqrt(3)
    # We need the hexagon to contain all points
    # Approximate: side length should be at least max(max_x, max_y / (sqrt(3)/2))
    side = max(max_x, max_y * 2 / np.sqrt(3)) + 0.5
    return side

def optimize_from_init(inner_init, outer_angle=0.0, max_iter=200):
    """Optimize from a given initial configuration."""
    n = len(inner_init)
    # Initial params
    params = []
    for p in inner_init:
        params.extend([p[0], p[1], p[2]])
    # Outer: center at origin, estimated side
    side_est = estimate_outer_size(inner_init)
    params.extend([0.0, 0.0, side_est, outer_angle])
    
    bounds = []
    for i in range(n):
        bounds.extend([(-10, 10), (-10, 10), (-180, 180)])
    bounds.extend([(-5, 5), (-5, 5), (1.0, 20.0), (-180, 180)])
    
    try:
        result = minimize(objective, params, method='L-BFGS-B', bounds=bounds,
                         options={'maxiter': max_iter, 'ftol': 1e-8, 'gtol': 1e-8})
        return result
    except Exception:
        return None

def construct():
    start_time = time.time()
    
    best_score = 1e10
    best_result = None
    
    # Multiple initializations
    inits = [
        ("honeycomb", make_honeycomb_init(1.0)),
        ("honeycomb_tight", make_honeycomb_init(0.95)),
        ("ring", make_ring_init()),
        ("grid", make_grid_init()),
    ]
    
    # Try different outer angles
    outer_angles = [0.0, 30.0, 15.0]
    
    for name, init in inits:
        if time.time() - start_time > DEADLINE - 10:
            break
        for oang in outer_angles:
            if time.time() - start_time > DEADLINE - 10:
                break
            result = optimize_from_init(init, oang, max_iter=100)
            if result is not None and result.success:
                score = result.fun
                if score < best_score:
                    best_score = score
                    best_result = result
                    # Extract configuration
                    params = result.x
                    inner = []
                    for i in range(N):
                        inner.append([params[3*i], params[3*i+1], params[3*i+2]])
                    cx, cy = params[36], params[37]
                    side = params[38]
                    oang_out = params[39]
    
    # If optimization didn't give good results, fall back to a reasonable static config
    if best_result is None or best_score > 15.0:
        # Use honeycomb with estimated outer size
        inner = make_honeycomb_init(1.0)
        side = estimate_outer_size(inner)
        return inner, (0.0, 0.0), side, 0.0
    
    params = best_result.x
    inner = []
    for i in range(N):
        inner.append([params[3*i], params[3*i+1], params[3*i+2]])
    cx, cy = params[36], params[37]
    side = params[38]
    oang = params[39]
    
    return inner, (cx, cy), side, oang
