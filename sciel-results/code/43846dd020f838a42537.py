"""Improved packing: SLSQP with multi-restart for 12 unit hexagons in a hexagon."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 12
DEADLINE = 145.0  # safety margin before 150s kill

def hexagon_vertices(cx, cy, s, angle_deg):
    """Return 6 vertices of a regular hexagon with side s, center (cx,cy), rotation angle_deg."""
    a = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * math.pi / 3.0
        verts.append((cx + s * math.cos(theta), cy + s * math.sin(theta)))
    return verts

def point_in_polygon(pt, poly):
    """Ray casting point-in-polygon test."""
    x, y = pt
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside

def polygon_collision(p1, p2, eps=1e-10):
    """Check if two convex polygons intersect (including touching)."""
    # Check if any vertex of p1 is inside p2 or vice versa
    for v in p1:
        if point_in_polygon(v, p2):
            return True
    for v in p2:
        if point_in_polygon(v, p1):
            return True
    # Check edge intersections
    for i in range(len(p1)):
        for j in range(len(p2)):
            if segments_intersect(p1[i], p1[(i+1)%len(p1)], p2[j], p2[(j+1)%len(p2)], eps):
                return True
    return False

def segments_intersect(p1, p2, p3, p4, eps=1e-10):
    """Check if segment p1-p2 intersects segment p3-p4."""
    def ccw(a, b, c):
        return (c[1]-a[1])*(b[0]-a[0]) - (b[1]-a[1])*(c[0]-a[0])
    d1 = ccw(p3, p4, p1)
    d2 = ccw(p3, p4, p2)
    d3 = ccw(p1, p2, p3)
    d4 = ccw(p1, p2, p4)
    if ((d1 > eps and d2 < -eps) or (d1 < -eps and d2 > eps)) and \
       ((d3 > eps and d4 < -eps) or (d3 < -eps and d4 > eps)):
        return True
    return False

def get_all_vertices(inner, outer_center, outer_side, outer_angle):
    """Get all vertices of all hexagons."""
    verts_list = []
    for i in range(N):
        x, y, ang = inner[i]
        verts = hexagon_vertices(x, y, 1.0, ang)
        verts_list.append(verts)
    return verts_list

def outer_hexagon(center, side, angle_deg):
    """Get vertices of outer hexagon."""
    cx, cy = center
    return hexagon_vertices(cx, cy, side, angle_deg)

def check_validity(inner, center, side, angle_deg):
    """Check if packing is valid: no overlaps, all inside outer hexagon."""
    inner_verts = get_all_vertices(inner, center, side, angle_deg)
    outer_verts = outer_hexagon(center, side, angle_deg)
    
    # Check all inner hexagons are inside outer
    for i in range(N):
        for v in inner_verts[i]:
            if not point_in_polygon(v, outer_verts):
                return False
    
    # Check no overlaps between inner hexagons
    for i in range(N):
        for j in range(i+1, N):
            if polygon_collision(inner_verts[i], inner_verts[j]):
                return False
    return True

def objective(params, center, angle_deg):
    """Objective: maximize minimum distance from any inner vertex to outer boundary,
    subject to non-overlap. We minimize negative of that, or use a penalty."""
    x = params[:2*N]
    y = params[2*N:4*N]
    inner = [[x[i], y[i], 0.0] for i in range(N)]
    
    # Compute the required outer side length: find max distance from center to any vertex
    # The outer hexagon must contain all vertices.
    # For a regular hexagon with side s centered at center, the inscribed circle radius is s*sqrt(3)/2
    # and circumscribed circle radius is s.
    # A point at distance d from center is inside if d <= s (for the circumradius direction)
    # but the hexagon is not a circle. We need the actual containment.
    
    # Simpler approach: compute the minimum side length needed.
    # For a regular hexagon with center c and side s, a point p is inside iff
    # for all 6 edges, the point is on the inside.
    # The distance from center to each edge (apothem) = s * sqrt(3)/2
    # So the hexagon contains all points within apothem distance in each of the 6 normal directions.
    
    # The minimum s such that all points are inside:
    # For each point p, the required s is max over the 6 directions of (dot(p-c, n_i) / (sqrt(3)/2))
    # where n_i are the unit normals to the 6 edges.
    
    cx, cy = center
    min_s = 0.0
    for i in range(N):
        for v in hexagon_vertices(x[i], y[i], 1.0, 0.0):
            dx, dy = v[0] - cx, v[1] - cy
            # 6 edge normals for a hexagon with angle 0: directions at 0, 60, 120, 180, 240, 300 degrees
            for k in range(6):
                theta = math.radians(60.0 * k)
                nx, ny = math.cos(theta), math.sin(theta)
                proj = dx * nx + dy * ny
                # apothem = s * sqrt(3)/2, so s >= proj / (sqrt(3)/2) = 2*proj/sqrt(3)
                if proj > 0:
                    s_req = 2.0 * proj / math.sqrt(3.0)
                    if s_req > min_s:
                        min_s = s_req
    return min_s

def constraint_nonoverlap(params, center, angle_deg):
    """Return list of constraint values (>=0 means satisfied)."""
    x = params[:2*N]
    y = params[2*N:4*N]
    inner_verts = get_all_vertices([[x[i], y[i], 0.0] for i in range(N)], center, 100.0, 0.0)
    
    # For non-overlap, we need distance between hexagons > 0.
    # Approximate using center-to-center distance: two unit hexagons don't overlap if center dist >= 2
    # (this is conservative; actual minimum is sqrt(3) for edge-to-edge)
    # Use a tighter bound: two unit regular hexagons with same orientation don't overlap if
    # center distance >= 2 (circumradius*2)
    # Actually for hexagons, the minimum center distance for no overlap is 2 (vertex to vertex)
    # but can be less if rotated. For same orientation, it's 2.
    
    cons = []
    for i in range(N):
        for j in range(i+1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = math.sqrt(dx*dx + dy*dy)
            # Minimum distance for non-overlap of two unit hexagons (same orientation) is 2.0
            # But to be safe, use 1.9 as constraint (allow slight touching)
            cons.append(dist - 2.0)
    return cons

def solve_packing():
    """Multi-restart SLSQP to find optimal packing."""
    best_side = float('inf')
    best_params = None
    start_time = time.time()
    
    # Initial guess: hexagonal lattice arrangement
    # Place 12 hexagons in a hexagonal pattern
    base_positions = []
    # Center hexagon
    base_positions.append((0.0, 0.0))
    # Ring of 6
    for i in range(6):
        theta = math.radians(60.0 * i)
        base_positions.append((2.0 * math.cos(theta), 2.0 * math.sin(theta)))
    # Ring of 5 more (partial second ring)
    for i in range(5):
        theta = math.radians(60.0 * i + 30.0)
        base_positions.append((4.0 * math.cos(theta), 4.0 * math.sin(theta)))
    
    n_restarts = 30
    rng = np.random.default_rng(42)
    
    for restart in range(n_restarts):
        if time.time() - start_time > DEADLINE - 5:
            break
        
        if restart == 0:
            x0 = np.array([p[0] for p in base_positions] + [p[1] for p in base_positions])
        else:
            # Perturb
            scale = 0.5 * restart
            x0 = np.array([p[0] + rng.normal(0, scale) for p in base_positions] + 
                          [p[1] + rng.normal(0, scale) for p in base_positions])
        
        # Use SLSQP to minimize the required outer side length
        # Parameters: x[0..11], y[0..11]
        center = (0.0, 0.0)
        angle_deg = 0.0
        
        # Objective: minimize required outer side
        def obj(p):
            return objective(p, center, angle_deg)
        
        # Constraints: non-overlap (distance >= 2 between centers for same-orientation hexagons)
        def cons_func(p):
            return constraint_nonoverlap(p, center, angle_deg)
        
        # Split into individual constraints for SLSQP
        cons = []
        for i in range(N):
            for j in range(i+1, N):
                idx_i, idx_j = i, j
                def make_con(idx_i, idx_j):
                    def con(p):
                        dx = p[idx_i] - p[idx_i + N]
                        dy = p[idx_i + N] - p[idx_j + N]
                        return math.sqrt(dx*dx + dy*dy) - 2.0
                    return con
                cons.append({'type': 'ineq', 'fun': make_con(idx_i, idx_j)})
        
        try:
            result = minimize(obj, x0, method='SLSQP', constraints=cons, 
                             options={'maxiter': 200, 'ftol': 1e-8})
            if result.success or result.fun < best_side:
                if result.fun < best_side:
                    best_side = result.fun
                    best_params = result.x.copy()
        except Exception:
            continue
    
    # Build final answer
    if best_params is not None:
        x = best_params[:2*N]
        y = best_params[2*N:4*N]
        inner = [[x[i], y[i], 0.0] for i in range(N)]
        # Center the packing
        cx = sum(p[0] for p in inner) / N
        cy = sum(p[1] for p in inner) / N
        for p in inner:
            p[0] -= cx
            p[1] -= cy
        return inner, (0.0, 0.0), best_side, 0.0
    else:
        # Fallback: initial arrangement
        inner = [[p[0], p[1], 0.0] for p in base_positions]
        return inner, (0.0, 0.0), 10.0, 0.0

def construct():
    start_time = time.time()
    try:
        inner, center, side, angle = solve_packing()
        return inner, center, side, angle
    except Exception:
        # Fallback: simple arrangement
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
