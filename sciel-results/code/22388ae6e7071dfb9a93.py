"""Direct hexagon packing: 11 unit hexagons in a hexagonal lattice pattern."""
import math
import time
import itertools

N = 11


def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon with side length `side`,
    centered at (cx, cy), rotated by angle_deg degrees."""
    verts = []
    for k in range(6):
        a = math.radians(angle_deg + 60 * k)
        verts.append((cx + side * math.cos(a), cy + side * math.sin(a)))
    return verts


def _point_in_hex(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside hexagon centered at (cx,cy) with side `side` rotated by angle_deg."""
    dx = px - cx
    dy = py - cy
    a = -math.radians(angle_deg)
    lx = dx * math.cos(a) - dy * math.sin(a)
    ly = dx * math.sin(a) + dy * math.cos(a)
    apothem = side * math.sqrt(3) / 2
    for k in range(6):
        a_norm = math.radians(30 + 60 * k)
        nx = math.cos(a_norm)
        ny = math.sin(a_norm)
        if lx * nx + ly * ny > apothem + 1e-9:
            return False
    return True


def _segments_intersect(p1, p2, p3, p4):
    """Check if segment p1-p2 properly intersects segment p3-p4."""
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    d1 = cross(p3, p4, p1)
    d2 = cross(p3, p4, p2)
    d3 = cross(p1, p2, p3)
    d4 = cross(p1, p2, p4)
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    return False


def _hexagons_disjoint(hex1, hex2, eps=1e-9):
    """Check if two hexagons have disjoint interiors."""
    cx1, cy1, s1, a1 = hex1
    cx2, cy2, s2, a2 = hex2
    r1 = s1
    r2 = s2
    dx = cx2 - cx1
    dy = cy2 - cy1
    if dx * dx + dy * dy > (r1 + r2) ** 2 + 1e-12:
        return True
    verts1 = _hex_vertices(cx1, cy1, s1, a1)
    for vx, vy in verts1:
        if _point_in_hex(vx, vy, cx2, cy2, s2, a2):
            return False
    verts2 = _hex_vertices(cx2, cy2, s2, a2)
    for vx, vy in verts2:
        if _point_in_hex(vx, vy, cx1, cy1, s1, a1):
            return False
    for i in range(6):
        v1a = verts1[i]
        v1b = verts1[(i + 1) % 6]
        for j in range(6):
            v2a = verts2[j]
            v2b = verts2[(j + 1) % 6]
            if _segments_intersect(v1a, v1b, v2a, v2b):
                return False
    return True


def _all_disjoint(inner_hexes, eps=1e-9):
    """Check if all pairs of hexagons are disjoint."""
    n = len(inner_hexes)
    for i in range(n):
        for j in range(i + 1, n):
            if not _hexagons_disjoint(inner_hexes[i], inner_hexes[j], eps):
                return False
    return True


def _inner_in_outer(inner_hex, outer_hex):
    """Check if inner hexagon is fully inside outer hexagon."""
    cx, cy, s, a = inner_hex
    ox, oy, oside, oangle = outer_hex
    verts = _hex_vertices(cx, cy, s, a)
    for vx, vy in verts:
        if not _point_in_hex(vx, vy, ox, oy, oside, oangle):
            return False
    return True


def _min_outer_side(hexes, ox, oy, oangle=0.0):
    """Find minimum side length of outer hexagon centered at (ox, oy) with rotation oangle
    that contains all hexagons in `hexes`."""
    max_val = 0.0
    a = -math.radians(oangle)
    cos_a = math.cos(a)
    sin_a = math.sin(a)
    for cx, cy, s, ang in hexes:
        verts = _hex_vertices(cx, cy, s, ang)
        for vx, vy in verts:
            dx = vx - ox
            dy = vy - oy
            lx = dx * cos_a - dy * sin_a
            ly = dx * sin_a + dy * cos_a
            for k in range(6):
                theta = math.radians(30 + 60 * k)
                proj = lx * math.cos(theta) + ly * math.sin(theta)
                if proj > max_val:
                    max_val = proj
    return max_val * 2.0 / math.sqrt(3)


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s limit

    sqrt3 = math.sqrt(3)

    # Flat-top hexagon mapping using axial coordinates:
    # For a unit hexagon (side=1) with flat-top orientation:
    # x = 1.5 * q
    # y = sqrt(3) * (r + q/2)
    def axial_to_xy(q, r):
        x = 1.5 * q
        y = sqrt3 * (r + q / 2.0)
        return (x, y)

    # We need to place 11 hexagons. Start with a hexagonal lattice of 7 (center + 6 neighbors),
    # then add 4 more from ring 2.
    
    # Center + ring 1 (6 hexagons)
    pos_axial = [(0, 0)]
    ring1 = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
    pos_axial.extend(ring1)
    
    # Ring 2 positions (distance 2 from center) - 12 hexagons
    ring2 = [(2, 0), (2, -1), (2, -2), (1, -2), (0, -2), (-1, -2),
             (-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2), (-1, 2),
             (0, 2), (1, 2)]
    
    best_side = float('inf')
    best_positions = None
    best_center = (0.0, 0.0)
    best_angle = 0.0

    # Pre-compute all ring2 combinations
    all_combos = list(itertools.combinations(range(len(ring2)), 4))
    
    # Phase 1: Quick scan with coarse grid to find promising configurations
    for combo in all_combos:
        if time.time() > deadline - 30.0:  # Leave 30s for phase 2
            break
            
        selected_r2 = [ring2[i] for i in combo]
        full_axial = pos_axial + selected_r2
        
        # Convert to Cartesian
        full_xy = [axial_to_xy(q, r) for q, r in full_axial]
        
        # Compute centroid
        cx_mean = sum(x for x, y in full_xy) / 11.0
        cy_mean = sum(y for x, y in full_xy) / 11.0
        
        inner_hexes = [(x, y, 1.0, 0.0) for x, y in full_xy]
        
        # Try several outer hexagon orientations
        for oang in [0.0, 30.0, 15.0, -15.0, 45.0, -45.0, 60.0, -60.0, 75.0, -75.0]:
            side = _min_outer_side(inner_hexes, cx_mean, cy_mean, oang)
            if side < best_side:
                best_side = side
                best_positions = full_xy
                best_center = (cx_mean, cy_mean)
                best_angle = oang
    
    # Phase 2: Fine-tune the best configuration with local optimization
    if best_positions is not None and time.time() < deadline:
        inner_hexes = [(x, y, 1.0, 0.0) for x, y in best_positions]
        
        # Multi-scale local search with decreasing step sizes
        cur_cx, cur_cy, cur_ang, cur_side = best_center[0], best_center[1], best_angle, best_side
        
        for step_size in [0.05, 0.02, 0.01, 0.005, 0.001, 0.0002]:
            improved = True
            while improved and time.time() < deadline:
                improved = False
                
                # Try perturbations in center and angle
                for dx in [-step_size, -step_size/2, 0, step_size/2, step_size]:
                    for dy in [-step_size, -step_size/2, 0, step_size/2, step_size]:
                        for da in [-step_size*10, -step_size*5, 0, step_size*5, step_size*10]:
                            ncx = cur_cx + dx
                            ncy = cur_cy + dy
                            nang = cur_ang + da
                            s = _min_outer_side(inner_hexes, ncx, ncy, nang)
                            if s < cur_side - 1e-12:
                                cur_cx, cur_cy, cur_ang, cur_side = ncx, ncy, nang, s
                                improved = True
                                if s < best_side:
                                    best_side = s
                                    best_center = (ncx, ncy)
                                    best_angle = nang

    if best_positions is None:
        # Fallback: just use the 7-hexagon cluster
        best_positions = [axial_to_xy(q, r) for q, r in pos_axial]
        best_side = _min_outer_side([(x, y, 1.0, 0.0) for x, y in best_positions], 0, 0, 0)
        best_center = (0, 0)
        best_angle = 0.0

    # Final polish: high-precision optimization of outer hexagon parameters
    if best_positions is not None and time.time() < deadline:
        inner_hexes = [(x, y, 1.0, 0.0) for x, y in best_positions]
        
        # Use gradient-free local search with decreasing step sizes
        cur_cx, cur_cy = best_center
        cur_ang = best_angle
        cur_side = best_side
        
        for step_size in [0.05, 0.01, 0.002, 0.0005]:
            improved = True
            while improved and time.time() < deadline:
                improved = False
                # Try small perturbations
                for dx in [-step_size, step_size]:
                    for dy in [-step_size, step_size]:
                        for da in [-step_size * 5, step_size * 5]:  # angle in degrees
                            ncx = cur_cx + dx
                            ncy = cur_cy + dy
                            nang = cur_ang + da
                            s = _min_outer_side(inner_hexes, ncx, ncy, nang)
                            if s < cur_side - 1e-12:
                                cur_cx, cur_cy, cur_ang, cur_side = ncx, ncy, nang, s
                                improved = True
                # Also try axis-aligned moves
                for dx, dy, da in [(-step_size, 0, 0), (step_size, 0, 0),
                                   (0, -step_size, 0), (0, step_size, 0),
                                   (0, 0, -step_size * 5), (0, 0, step_size * 5)]:
                    ncx = cur_cx + dx
                    ncy = cur_cy + dy
                    nang = cur_ang + da
                    s = _min_outer_side(inner_hexes, ncx, ncy, nang)
                    if s < cur_side - 1e-12:
                        cur_cx, cur_cy, cur_ang, cur_side = ncx, ncy, nang, s
                        improved = True
        
        if cur_side < best_side:
            best_side = cur_side
            best_center = (cur_cx, cur_cy)
            best_angle = cur_ang

    # Build final inner hexagons
    inner_hexes = [(x, y, 1.0, 0.0) for x, y in best_positions]

    # Verify disjointness - with proper axial spacing, this should always pass
    # The minimum distance between centers of adjacent hexagons is sqrt(3) ≈ 1.732
    # which is > 2*1.0 = 2.0? No, wait: for flat-top hexagons with side 1,
    # the distance between centers of adjacent hexagons should be sqrt(3) ≈ 1.732
    # But the hexagons have "radius" 1 (center to vertex), so they would overlap!
    
    # Actually, let me reconsider. For flat-top hexagons with side length 1:
    # - The distance from center to vertex is 1
    # - The distance from center to edge midpoint (apothem) is sqrt(3)/2 ≈ 0.866
    # - For two hexagons to not overlap, their centers must be far enough apart
    
    # In a hexagonal lattice with flat-top hexagons of side 1:
    # Adjacent hexagons share an edge. The distance between centers is 2 * apothem = sqrt(3) ≈ 1.732
    # But since the hexagons extend to radius 1 from center, and the shared edge is at apothem distance,
    # the hexagons actually share an edge and have disjoint interiors.
    
    # Let me verify: if two flat-top hexagons of side 1 are placed with centers distance sqrt(3) apart
    # horizontally, they share an edge. Their interiors are disjoint. Good.
    
    if not _all_disjoint(inner_hexes):
        # If there's an issue, try scaling up positions slightly
        # This shouldn't happen with correct axial mapping
        for scale in [1.001, 1.01, 1.05]:
            scaled_hexes = [(x * scale, y * scale, 1.0, 0.0) for x, y in best_positions]
            if _all_disjoint(scaled_hexes):
                inner_hexes = scaled_hexes
                cx_mean = sum(x for x, y, s, a in inner_hexes) / 11.0
                cy_mean = sum(y for x, y, s, a in inner_hexes) / 11.0
                best_side = _min_outer_side(inner_hexes, cx_mean, cy_mean, 0.0)
                best_center = (cx_mean, cy_mean)
                best_angle = 0.0
                break

    # Verify all inner hexagons are inside outer with tight tolerance
    outer_hex = (best_center[0], best_center[1], best_side, best_angle)
    valid = True
    for h in inner_hexes:
        if not _inner_in_outer(h, outer_hex):
            valid = False
            break

    if not valid:
        # Binary search for minimum valid side length
        lo, hi = best_side, best_side * 1.1
        for _ in range(50):
            mid = (lo + hi) / 2
            test_outer = (best_center[0], best_center[1], mid, best_angle)
            ok = all(_inner_in_outer(h, test_outer) for h in inner_hexes)
            if ok:
                hi = mid
            else:
                lo = mid
        best_side = hi
        outer_hex = (best_center[0], best_center[1], best_side, best_angle)

    # Format output: inner rows (x, y, angle_degrees)
    inner = [[h[0], h[1], h[3]] for h in inner_hexes]

    return inner, (best_center[0], best_center[1]), best_side, best_angle
