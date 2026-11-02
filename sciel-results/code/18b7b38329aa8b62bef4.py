"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 120.0  # 120s safety margin within 150s limit

def _get_hex_vertices(cx, cy, angle_deg=0.0):
    """Get vertices of a unit regular hexagon centered at (cx, cy) with rotation."""
    angle_rad = math.radians(angle_deg)
    verts = []
    for k in range(6):
        a = angle_rad + math.pi * k / 3
        verts.append((cx + math.cos(a), cy + math.sin(a)))
    return verts

def _min_enclosing_hex_side_lp(inner_hexes, angle_deg):
    """Use LP to find optimal center for given outer hexagon angle.
    Returns (cx, cy, side_length)."""
    from scipy.optimize import linprog
    
    all_vertices = []
    for ix, iy, iangle in inner_hexes:
        verts = _get_hex_vertices(ix, iy, iangle)
        all_vertices.extend(verts)
    
    n_pts = len(all_vertices)
    n = 6  # 6 constraints per direction (max and min)
    
    # Variables: cx, cy, t (inradius)
    # Minimize t
    c = [0.0, 0.0, 1.0]
    
    A_ub = []
    b_ub = []
    
    for k in range(6):
        a = math.radians(angle_deg + 30 + 60 * k)
        nx = math.cos(a)
        ny = math.sin(a)
        
        # For each vertex: (vx - cx)*nx + (vy - cy)*ny <= t
        # => -nx*cx - ny*cy + nx*vx + ny*vy <= t
        # => -nx*cx - ny*cy - t <= -nx*vx - ny*vy
        for vx, vy in all_vertices:
            A_ub.append([-nx, -ny, -1.0])
            b_ub.append(-nx * vx - ny * vy)
        
        # For each vertex: -(vx - cx)*nx - (vy - cy)*ny <= t
        # => nx*cx + ny*cy - nx*vx - ny*vy <= t
        # => nx*cx + ny*cy - t <= nx*vx + ny*vy
        for vx, vy in all_vertices:
            A_ub.append([nx, ny, -1.0])
            b_ub.append(nx * vx + ny * vy)
    
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(None, None), (None, None), (0, None)], method='highs')
    
    if result.success:
        cx, cy, t = result.x
        side = t / (math.sqrt(3) / 2)
        return cx, cy, side
    
    # Fallback: use centroid
    cx = sum(v[0] for v in all_vertices) / n_pts
    cy = sum(v[1] for v in all_vertices) / n_pts
    required_inradius = 0.0
    for k in range(6):
        a = math.radians(angle_deg + 30 + 60 * k)
        nx = math.cos(a)
        ny = math.sin(a)
        max_proj = -float('inf')
        min_proj = float('inf')
        for vx, vy in all_vertices:
            proj = (vx - cx) * nx + (vy - cy) * ny
            max_proj = max(max_proj, proj)
            min_proj = min(min_proj, proj)
        required_inradius = max(required_inradius, max(max_proj, -min_proj))
    side = required_inradius / (math.sqrt(3) / 2)
    return cx, cy, side

def _min_enclosing_hex_side(inner_hexes):
    """Find minimum side length of a regular hexagon that contains all inner hexagons.
    Returns (center_x, center_y, side_length, angle_deg)."""
    best_side = float('inf')
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    # Try different angles for the outer hexagon (every 10 degrees for better coverage)
    for angle_deg in [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]:
        cx, cy, side = _min_enclosing_hex_side_lp(inner_hexes, angle_deg)
        if side < best_side:
            best_side = side
            best_center = (cx, cy)
            best_angle = angle_deg
    
    return best_center[0], best_center[1], best_side, best_angle

def _check_disjoint(inner_hexes):
    """Check if all inner hexagons have disjoint interiors using polygon intersection."""
    n = len(inner_hexes)
    polygons = [_get_hex_vertices(x, y, a) for x, y, a in inner_hexes]
    
    for i in range(n):
        for j in range(i + 1, n):
            if _polygons_overlap(polygons[i], polygons[j]):
                return False
    return True

def _polygons_overlap(poly1, poly2, eps=1e-9):
    """Check if two convex polygons have overlapping interiors."""
    # Check if any vertex of poly1 is inside poly2 (strictly)
    for v in poly1:
        if _point_in_convex_polygon(v, poly2, eps):
            return True
    # Check if any vertex of poly2 is inside poly1 (strictly)
    for v in poly2:
        if _point_in_convex_polygon(v, poly1, eps):
            return True
    # Check if any edges intersect
    for i in range(len(poly1)):
        for j in range(len(poly2)):
            if _segments_intersect(poly1[i], poly1[(i+1)%len(poly1)], 
                                   poly2[j], poly2[(j+1)%len(poly2)], eps):
                return True
    return False

def _point_in_convex_polygon(point, polygon, eps=1e-9):
    """Check if point is strictly inside a convex polygon."""
    px, py = point
    n = len(polygon)
    for i in range(n):
        ax, ay = polygon[i]
        bx, by = polygon[(i+1) % n]
        # Cross product to check if point is on the right side of edge
        cross = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
        if cross > eps:
            return False
    return True

def _segments_intersect(p1, p2, p3, p4, eps=1e-9):
    """Check if two line segments intersect (including endpoints)."""
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    
    d1 = cross(p3, p4, p1)
    d2 = cross(p3, p4, p2)
    d3 = cross(p1, p2, p3)
    d4 = cross(p1, p2, p4)
    
    if ((d1 > eps and d2 < -eps) or (d1 < -eps and d2 > eps)) and \
       ((d3 > eps and d4 < -eps) or (d3 < -eps and d4 > eps)):
        return True
    
    # Check collinear cases
    if abs(d1) <= eps and _on_segment(p3, p1, p4, eps):
        return True
    if abs(d2) <= eps and _on_segment(p3, p2, p4, eps):
        return True
    if abs(d3) <= eps and _on_segment(p1, p3, p2, eps):
        return True
    if abs(d4) <= eps and _on_segment(p1, p4, p2, eps):
        return True
    
    return False

def _on_segment(p, q, r, eps=1e-9):
    """Check if point q lies on segment pr."""
    if (min(p[0], r[0]) - eps <= q[0] <= max(p[0], r[0]) + eps and
        min(p[1], r[1]) - eps <= q[1] <= max(p[1], r[1]) + eps and
        abs((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])) <= eps):
        return True
    return False

def _nelder_mead_polish(inner_hexes, max_iter=200, tol=1e-6):
    """Polish inner hexagon positions using Nelder-Mead to minimize enclosing hexagon side."""
    from scipy.optimize import minimize
    
    def objective(params):
        # params: flattened [x1, y1, a1, x2, y2, a2, ...]
        new_inner = []
        for i in range(N):
            x = params[i * 3]
            y = params[i * 3 + 1]
            a = params[i * 3 + 2]
            new_inner.append([x, y, a])
        
        # Check disjointness - if violated, return large penalty
        if not _check_disjoint(new_inner):
            # Compute a better penalty based on minimum separation
            min_sep = float('inf')
            for i in range(N):
                for j in range(i + 1, N):
                    dx = new_inner[i][0] - new_inner[j][0]
                    dy = new_inner[i][1] - new_inner[j][1]
                    dist = math.sqrt(dx*dx + dy*dy)
                    min_sep = min(min_sep, dist)
            return 1000.0 + max(0, (math.sqrt(3) - 0.001 - min_sep)) * 1000
        
        cx, cy, side, angle = _min_enclosing_hex_side(new_inner)
        return side
    
    # Initial params
    x0 = []
    for h in inner_hexes:
        x0.extend([h[0], h[1], h[2]])
    
    result = minimize(objective, x0, method='Nelder-Mead', 
                      options={'maxiter': max_iter, 'xatol': tol, 'fatol': tol, 'adaptive': True})
    
    if result.success and result.fun < 1000.0:
        polished = []
        for i in range(N):
            polished.append([result.x[i*3], result.x[i*3+1], result.x[i*3+2]])
        return polished
    
    return inner_hexes

def construct():
    global _START_TIME, _DEADLINE
    
    # Check deadline
    if time.time() > _DEADLINE - 5.0:
        # Return a safe fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    # Generate 11 points in a hexagonal cluster arrangement
    # Central hexagon + ring of 6 + 4 more in second ring
    # Use triangular lattice with spacing sqrt(3) for edge-to-edge contact
    sqrt3 = math.sqrt(3)
    # Scale factor to ensure strict disjointness (small gap between hexagons)
    scale = 1.001
    u = (sqrt3 * scale, 0.0)
    v = (sqrt3 / 2 * scale, 1.5 * scale)
    
    # Generate candidate points in a spiral/grid around the origin
    # We want the 11 closest points to the origin in the triangular lattice
    # Lattice points are (i, j) -> i*u + j*v
    # Let's generate points with |i|, |j| <= 3 and sort by distance
    
    candidates = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            x = i * u[0] + j * v[0]
            y = i * u[1] + j * v[1]
            dist_sq = x * x + y * y
            candidates.append((dist_sq, x, y))
    
    # Sort by distance
    candidates.sort(key=lambda t: t[0])
    
    # Take the first N unique points
    positions = []
    for _, x, y in candidates:
        positions.append((x, y))
        if len(positions) == N:
            break
    
    # Create inner hexagons with all angles = 30 (Pointy-Top)
    # This aligns the lattice vectors (0 and 60 deg) with the flat-to-flat normals,
    # requiring a minimum separation of sqrt(3) instead of 2.
    inner = [[pos[0], pos[1], 30.0] for pos in positions]
    
    # Compute initial minimum enclosing hexagon
    if time.time() > _DEADLINE - 10.0:
        # Quick estimate
        xs = [h[0] for h in inner]
        ys = [h[1] for h in inner]
        cx = (max(xs) + min(xs)) / 2
        cy = (max(ys) + min(ys)) / 2
        max_dist = 0
        for h in inner:
            d = math.sqrt((h[0] - cx)**2 + (h[1] - cy)**2)
            if d > max_dist:
                max_dist = d
        # Add circumradius 1 to max_dist for a rough bound
        side = (max_dist + 1.0) / (math.sqrt(3) / 2) + 0.1
        return inner, (cx, cy), side, 0.0
    
    cx, cy, side, angle = _min_enclosing_hex_side(inner)
    best_side = side
    best_inner = [h[:] for h in inner]
    best_center = (cx, cy)
    best_angle = angle
    
    # Try Nelder-Mead polish if time permits
    if time.time() < _DEADLINE - 30.0:
        polished = _nelder_mead_polish(inner, max_iter=300, tol=1e-7)
        if polished is not inner:
            pcx, pcy, pside, pangle = _min_enclosing_hex_side(polished)
            if pside < best_side - 1e-6:
                best_side = pside
                best_inner = polished
                best_center = (pcx, pcy)
                best_angle = pangle
    
    # Final high-precision LP polish for outer hexagon
    if time.time() < _DEADLINE - 15.0:
        # Try finer angles around the best angle
        for da in [-10, -8, -6, -4, -2, -1, -0.5, 0, 0.5, 1, 2, 4, 6, 8, 10]:
            test_angle = (best_angle + da) % 60
            tcx, tcy, tside = _min_enclosing_hex_side_lp(best_inner, test_angle)
            if tside < best_side - 1e-10:
                best_side = tside
                best_center = (tcx, tcy)
                best_angle = test_angle
        
        # One more Nelder-Mead polish with tighter tolerance if time permits
        if time.time() < _DEADLINE - 20.0:
            polished2 = _nelder_mead_polish(best_inner, max_iter=200, tol=1e-8)
            if polished2 is not best_inner:
                pcx2, pcy2, pside2, pangle2 = _min_enclosing_hex_side(polished2)
                if pside2 < best_side - 1e-8:
                    best_side = pside2
                    best_inner = polished2
                    best_center = (pcx2, pcy2)
                    best_angle = pangle2
    
    return best_inner, (best_center[0], best_center[1]), best_side, best_angle
# EVOLVE-BLOCK-END
