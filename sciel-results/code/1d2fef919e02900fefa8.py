"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import numpy as np
from scipy.optimize import linprog, minimize

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 115.0  # safety margin within 150s limit

def _get_hex_vertices(cx, cy, angle_deg=0.0):
    """Get vertices of a unit regular hexagon centered at (cx, cy) with rotation."""
    angle_rad = math.radians(angle_deg)
    verts = []
    for k in range(6):
        a = angle_rad + math.pi * k / 3
        verts.append((cx + math.cos(a), cy + math.sin(a)))
    return verts

def _support_function_diff(p1, a1, p2, a2, direction):
    """Compute h1(d) - h2(d) where h is the support function of a unit hexagon
    centered at p with rotation a, in direction d (unit vector).
    
    The support function of a regular hexagon with inradius 1 (side length 1) is:
    h(d) = max over vertices v of (v . d)
    For a unit hexagon (side=1), the vertices are at distance 1 from center.
    Actually, for side length s, the circumradius is s and inradius is s*sqrt(3)/2.
    Wait: for a regular hexagon with side length 1, the vertices are at distance 1 from center.
    So circumradius = 1, inradius = sqrt(3)/2.
    
    The support function in direction d for a hexagon with vertices at angles 
    a + k*pi/3 is:
    h(d) = max_k cos(angle(d) - a - k*pi/3)
    """
    d_angle = math.atan2(direction[1], direction[0])
    a1_rad = math.radians(a1)
    a2_rad = math.radians(a2)
    
    max1 = -float('inf')
    max2 = -float('inf')
    for k in range(6):
        angle1 = a1_rad + k * math.pi / 3
        angle2 = a2_rad + k * math.pi / 3
        max1 = max(max1, math.cos(d_angle - angle1))
        max2 = max(max2, math.cos(d_angle - angle2))
    
    return max1 - max2

def _check_overlap(p1, a1, p2, a2, eps=1e-9):
    """Check if two unit hexagons overlap using support function test.
    Two convex sets A and B are disjoint iff there exists a direction d such that
    h_A(d) + h_B(-d) < 0 (separating axis theorem).
    For our case, we check if the minimum over all relevant directions of
    (h_A(d) + h_B(-d) - |p1-p2| projected) > 0.
    
    Actually, simpler: two hexagons centered at p1, p2 with rotations a1, a2 overlap
    iff for ALL directions d, h1(d) + h2(-d) >= (p2-p1).d
    where h1 is support of hex1 centered at origin, h2 of hex2 centered at origin.
    
    The critical directions to check are the normals to edges of both hexagons (12 directions).
    """
    p1 = np.array(p1)
    p2 = np.array(p2)
    diff = p2 - p1
    
    # Edge normals for hex1 (6 directions) and hex2 (6 directions)
    a1_rad = math.radians(a1)
    a2_rad = math.radians(a2)
    
    min_margin = float('inf')
    
    # Check 12 critical directions (normals to edges of both hexagons)
    for k in range(6):
        # Normal to edge k of hex1
        ang1 = a1_rad + math.pi / 6 + k * math.pi / 3
        d1 = np.array([math.cos(ang1), math.sin(ang1)])
        
        # Support of hex1 in direction d1 (centered at origin)
        s1 = max(math.cos(math.atan2(d1[1], d1[0]) - (a1_rad + j * math.pi / 3)) for j in range(6))
        # Support of hex2 in direction -d1 (centered at origin)
        s2 = max(math.cos(math.atan2(-d1[1], -d1[0]) - (a2_rad + j * math.pi / 3)) for j in range(6))
        
        margin = s1 + s2 - np.dot(diff, d1)
        if margin < min_margin:
            min_margin = margin
    
    for k in range(6):
        # Normal to edge k of hex2
        ang2 = a2_rad + math.pi / 6 + k * math.pi / 3
        d2 = np.array([math.cos(ang2), math.sin(ang2)])
        
        s1 = max(math.cos(math.atan2(d2[1], d2[0]) - (a1_rad + j * math.pi / 3)) for j in range(6))
        s2 = max(math.cos(math.atan2(-d2[1], -d2[0]) - (a2_rad + j * math.pi / 3)) for j in range(6))
        
        margin = s1 + s2 - np.dot(diff, d2)
        if margin < min_margin:
            min_margin = margin
    
    # If min_margin > 0, they are disjoint (separated)
    # If min_margin <= 0, they overlap or touch
    return min_margin <= eps

def _min_clearance(p_new, a_new, placed):
    """Compute minimum clearance between new hexagon and all placed hexagons.
    Returns the minimum margin (positive = separated, negative = overlapping).
    """
    p_new = np.array(p_new)
    min_margin = float('inf')
    
    for (px, py, pa) in placed:
        diff = np.array([px, py]) - p_new
        a_new_rad = math.radians(a_new)
        a_placed_rad = math.radians(pa)
        
        # Check 12 critical directions
        for k in range(6):
            ang1 = a_new_rad + math.pi / 6 + k * math.pi / 3
            d = np.array([math.cos(ang1), math.sin(ang1)])
            s1 = max(math.cos(math.atan2(d[1], d[0]) - (a_new_rad + j * math.pi / 3)) for j in range(6))
            s2 = max(math.cos(math.atan2(-d[1], -d[0]) - (a_placed_rad + j * math.pi / 3)) for j in range(6))
            margin = s1 + s2 - np.dot(diff, d)
            min_margin = min(min_margin, margin)
        
        for k in range(6):
            ang2 = a_placed_rad + math.pi / 6 + k * math.pi / 3
            d = np.array([math.cos(ang2), math.sin(ang2)])
            s1 = max(math.cos(math.atan2(d[1], d[0]) - (a_new_rad + j * math.pi / 3)) for j in range(6))
            s2 = max(math.cos(math.atan2(-d[1], -d[0]) - (a_placed_rad + j * math.pi / 3)) for j in range(6))
            margin = s1 + s2 - np.dot(diff, d)
            min_margin = min(min_margin, margin)
    
    return min_margin

def _min_enclosing_hex_side_lp(inner_hexes, angle_deg):
    """For a fixed outer hexagon angle, find optimal center and side length using LP."""
    all_vertices = []
    for ix, iy, iangle in inner_hexes:
        verts = _get_hex_vertices(ix, iy, iangle)
        all_vertices.extend(verts)
    
    n_pts = len(all_vertices)
    if n_pts == 0:
        return 0.0, 0.0, 0.0, angle_deg
    
    A_ub = []
    b_ub = []
    
    for k in range(6):
        a = math.radians(angle_deg + 30 + 60 * k)
        nx = math.cos(a)
        ny = math.sin(a)
        for vx, vy in all_vertices:
            A_ub.append([-nx, -ny, -1.0])
            b_ub.append(-vx * nx - vy * ny)
    
    A_ub.append([0.0, 0.0, -1.0])
    b_ub.append(0.0)
    
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    c = np.array([0.0, 0.0, 1.0])
    
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, method='highs')
    
    if result.success:
        cx_opt = result.x[0]
        cy_opt = result.x[1]
        r_opt = result.x[2]
    else:
        cx_mean = sum(v[0] for v in all_vertices) / n_pts
        cy_mean = sum(v[1] for v in all_vertices) / n_pts
        r_opt = 0.0
        for k in range(6):
            a = math.radians(angle_deg + 30 + 60 * k)
            nx = math.cos(a)
            ny = math.sin(a)
            for vx, vy in all_vertices:
                proj = (vx - cx_mean) * nx + (vy - cy_mean) * ny
                r_opt = max(r_opt, proj)
        cx_opt = cx_mean
        cy_opt = cy_mean
    
    side = r_opt / (math.sqrt(3) / 2)
    return cx_opt, cy_opt, side, angle_deg

def _min_enclosing_hex_side(inner_hexes):
    """Find minimum side length of a regular hexagon that contains all inner hexagons."""
    best_side = float('inf')
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    # Coarse sweep
    for angle_deg in np.arange(0, 60, 2.0):
        if time.time() > _DEADLINE - 10.0:
            break
        cx, cy, side, ang = _min_enclosing_hex_side_lp(inner_hexes, float(angle_deg))
        if side < best_side:
            best_side = side
            best_center = (cx, cy)
            best_angle = float(angle_deg)
    
    # Fine sweep
    if best_side < float('inf'):
        for angle_deg in np.arange(max(0, best_angle - 5), min(60, best_angle + 5), 0.5):
            if time.time() > _DEADLINE - 10.0:
                break
            cx, cy, side, ang = _min_enclosing_hex_side_lp(inner_hexes, float(angle_deg))
            if side < best_side:
                best_side = side
                best_center = (cx, cy)
                best_angle = float(angle_deg)
    
    # Even finer
    if best_side < float('inf'):
        for angle_deg in np.arange(max(0, best_angle - 1), min(60, best_angle + 1), 0.1):
            if time.time() > _DEADLINE - 10.0:
                break
            cx, cy, side, ang = _min_enclosing_hex_side_lp(inner_hexes, float(angle_deg))
            if side < best_side:
                best_side = side
                best_center = (cx, cy)
                best_angle = float(angle_deg)
    
    return best_center[0], best_center[1], best_side, best_angle

def _greedy_max_min_placement():
    """Greedy construction: place each hexagon to maximize minimum clearance."""
    sqrt3 = math.sqrt(3)
    
    # Start with center hexagon
    placed = [(0.0, 0.0, 30.0)]
    
    for idx in range(1, N):
        if time.time() > _DEADLINE - 30.0:
            # Fallback: place on lattice
            # Generate lattice positions
            u = (sqrt3 * 1.001, 0.0)
            v = (sqrt3 / 2 * 1.001, 1.5 * 1.001)
            candidates = []
            for i in range(-4, 5):
                for j in range(-4, 5):
                    x = i * u[0] + j * v[0]
                    y = i * u[1] + j * v[1]
                    candidates.append((x, y, 30.0))
            # Sort by distance from origin, skip already placed
            candidates.sort(key=lambda c: c[0]**2 + c[1]**2)
            for c in candidates:
                if not any(abs(c[0]-p[0]) < 0.01 and abs(c[1]-p[1]) < 0.01 for p in placed):
                    placed.append(c)
                    if len(placed) == N:
                        break
            return placed
        
        # Try to place next hexagon to maximize min clearance
        best_pos = None
        best_clearance = -float('inf')
        
        # Search in a grid around the cluster
        # The cluster is roughly within radius 3*sqrt(3)
        search_radius = 4.0
        step = 0.3
        
        for dx in np.arange(-search_radius, search_radius + step, step):
            for dy in np.arange(-search_radius, search_radius + step, step):
                if time.time() > _DEADLINE - 25.0:
                    break
                # Try angle 30 (aligned with lattice)
                pos = (dx, dy, 30.0)
                
                # Quick check: minimum distance to any placed hexagon
                min_dist = min(math.sqrt((dx - px)**2 + (dy - py)**2) for px, py, _ in placed)
                if min_dist < sqrt3 * 0.95:  # Too close, skip
                    continue
                
                clearance = _min_clearance(pos, 30.0, placed)
                if clearance > best_clearance:
                    best_clearance = clearance
                    best_pos = pos
        
        if best_pos is None:
            # Fallback: place on lattice
            u = (sqrt3 * 1.001, 0.0)
            v = (sqrt3 / 2 * 1.001, 1.5 * 1.001)
            candidates = []
            for i in range(-5, 6):
                for j in range(-5, 6):
                    x = i * u[0] + j * v[0]
                    y = i * u[1] + j * v[1]
                    candidates.append((x, y, 30.0))
            candidates.sort(key=lambda c: c[0]**2 + c[1]**2)
            for c in candidates:
                if not any(abs(c[0]-p[0]) < 0.01 and abs(c[1]-p[1]) < 0.01 for p in placed):
                    placed.append(c)
                    if len(placed) == N:
                        break
            return placed
        
        placed.append(best_pos)
    
    return placed

def _lattice_placement():
    """Place hexagons on a triangular lattice (known good configuration)."""
    sqrt3 = math.sqrt(3)
    scale = 1.001  # Small gap to ensure strict disjointness
    u = (sqrt3 * scale, 0.0)
    v = (sqrt3 / 2 * scale, 1.5 * scale)
    
    candidates = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            x = i * u[0] + j * v[0]
            y = i * u[1] + j * v[1]
            dist_sq = x * x + y * y
            candidates.append((dist_sq, x, y))
    
    candidates.sort(key=lambda t: t[0])
    
    positions = []
    for _, x, y in candidates:
        positions.append((x, y, 30.0))
        if len(positions) == N:
            break
    
    return positions

def construct():
    global _START_TIME, _DEADLINE
    
    # Check deadline
    if time.time() > _DEADLINE - 5.0:
        inner = _lattice_placement()
        return [[h[0], h[1], h[2]] for h in inner], (0.0, 0.0), 10.0, 0.0
    
    # Try greedy max-min placement
    try:
        inner_greedy = _greedy_max_min_placement()
        if len(inner_greedy) >= N:
            # Verify no overlaps
            valid = True
            for i in range(len(inner_greedy)):
                for j in range(i + 1, len(inner_greedy)):
                    if _check_overlap(inner_greedy[i][:2], inner_greedy[i][2], 
                                      inner_greedy[j][:2], inner_greedy[j][2], eps=1e-6):
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                # Compute enclosing hexagon
                cx, cy, side, angle = _min_enclosing_hex_side(inner_greedy)
                return [[h[0], h[1], h[2]] for h in inner_greedy], (cx, cy), side, angle
    except Exception:
        pass
    
    # Fallback: lattice placement
    inner_lattice = _lattice_placement()
    
    if time.time() > _DEADLINE - 15.0:
        # Quick estimate
        xs = [h[0] for h in inner_lattice]
        ys = [h[1] for h in inner_lattice]
        cx = (max(xs) + min(xs)) / 2
        cy = (max(ys) + min(ys)) / 2
        max_dist = max(math.sqrt((h[0] - cx)**2 + (h[1] - cy)**2) for h in inner_lattice)
        side = (max_dist + 1.0) / (math.sqrt(3) / 2) + 0.1
        return [[h[0], h[1], h[2]] for h in inner_lattice], (cx, cy), side, 0.0
    
    cx, cy, side, angle = _min_enclosing_hex_side(inner_lattice)
    return [[h[0], h[1], h[2]] for h in inner_lattice], (cx, cy), side, angle
# EVOLVE-BLOCK-END
