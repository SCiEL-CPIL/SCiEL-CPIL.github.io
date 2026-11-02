import numpy as np
import time
import math

N = 12

def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon centered at (cx,cy) with given side and rotation."""
    angle = math.radians(angle_deg)
    verts = []
    for i in range(6):
        a = angle + i * math.pi / 3
        verts.append((cx + side * math.cos(a), cy + side * math.sin(a)))
    return verts

def _polygon_area_signed(pts):
    n = len(pts)
    a = 0.0
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        a += x1 * y2 - x2 * y1
    return a / 2.0

def _polygons_overlap(p1, p2, eps=1e-9):
    """Check if two convex polygons overlap (interior intersection).
    Returns True if they overlap (not just touching)."""
    # Check if any vertex of p1 is strictly inside p2
    for v in p1:
        if _point_in_polygon_strict(v, p2, eps):
            return True
    for v in p2:
        if _point_in_polygon_strict(v, p1, eps):
            return True
    # Check edge intersections
    for i in range(len(p1)):
        p1a = p1[i]
        p1b = p1[(i + 1) % len(p1)]
        for j in range(len(p2)):
            p2a = p2[j]
            p2b = p2[(j + 1) % len(p2)]
            if _segments_intersect_strict(p1a, p1b, p2a, p2b, eps):
                return True
    return False

def _point_in_polygon_strict(pt, poly, eps=1e-9):
    """Check if point is strictly inside polygon (not on boundary)."""
    px, py = pt
    n = len(poly)
    for i in range(n):
        ax, ay = poly[i]
        bx, by = poly[(i + 1) % n]
        cross = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
        if cross < -eps:
            return False
    # Check not on boundary
    for i in range(n):
        ax, ay = poly[i]
        bx, by = poly[(i + 1) % n]
        dx, dy = bx - ax, by - ay
        if abs(dx) > abs(dy):
            if min(ax, bx) - eps > px or px > max(ax, bx) + eps:
                continue
            if abs(dx) < eps:
                continue
            y_at_x = ay + (px - ax) * dy / dx
            if abs(y_at_x - py) < eps:
                return False
        else:
            if min(ay, by) - eps > py or py > max(ay, by) + eps:
                continue
            if abs(dy) < eps:
                continue
            x_at_y = ax + (py - ay) * dx / dy
            if abs(x_at_y - px) < eps:
                return False
    return True

def _segments_intersect_strict(p1, p2, p3, p4, eps=1e-9):
    """Check if two line segments intersect at an interior point of both."""
    def orient(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    
    d1 = orient(p3, p4, p1)
    d2 = orient(p3, p4, p2)
    d3 = orient(p1, p2, p3)
    d4 = orient(p1, p2, p4)
    
    if ((d1 > eps and d2 < -eps) or (d1 < -eps and d2 > eps)) and \
       ((d3 > eps and d4 < -eps) or (d3 < -eps and d4 > eps)):
        return True
    return False

def _outer_contains_inner(outer_pts, inner_pts, eps=1e-9):
    """Check if all inner vertices are inside or on the outer polygon."""
    for v in inner_pts:
        if not _point_in_polygon_non_strict(v, outer_pts, eps):
            return False
    return True

def _point_in_polygon_non_strict(pt, poly, eps=1e-9):
    """Check if point is inside or on boundary of polygon."""
    px, py = pt
    n = len(poly)
    for i in range(n):
        ax, ay = poly[i]
        bx, by = poly[(i + 1) % n]
        cross = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
        if cross < -eps:
            return False
    return True

def _max_radius_from_center(cx, cy, side, angle_deg):
    """Maximum distance from center to any vertex."""
    return side

def _support_function_hex(cx, cy, side, angle_deg, ux, uy):
    """Support function of hexagon in direction (ux, uy)."""
    angle = math.radians(angle_deg)
    max_val = 0.0
    for i in range(6):
        a = angle + i * math.pi / 3
        vx = cx + side * math.cos(a)
        vy = cy + side * math.sin(a)
        val = vx * ux + vy * uy
        if val > max_val:
            max_val = val
    return max_val

def _separation_distance(hex1, hex2):
    """Minimum distance between two hexagons (0 if touching/overlapping)."""
    cx1, cy1, a1 = hex1
    cx2, cy2, a2 = hex2
    p1 = _hex_vertices(cx1, cy1, 1.0, a1)
    p2 = _hex_vertices(cx2, cy2, 1.0, a2)
    
    min_dist_sq = float('inf')
    for i in range(6):
        for j in range(6):
            dx = p1[i][0] - p2[j][0]
            dy = p1[i][1] - p2[j][1]
            d2 = dx * dx + dy * dy
            if d2 < min_dist_sq:
                min_dist_sq = d2
    
    # Also check edge-edge distances
    for i in range(6):
        p1a = p1[i]
        p1b = p1[(i + 1) % 6]
        for j in range(6):
            p2a = p2[j]
            p2b = p2[(j + 1) % 6]
            d2 = _seg_seg_dist_sq(p1a, p1b, p2a, p2b)
            if d2 < min_dist_sq:
                min_dist_sq = d2
    
    return math.sqrt(min_dist_sq) if min_dist_sq < float('inf') else 0.0

def _seg_seg_dist_sq(p1, p2, p3, p4):
    """Minimum squared distance between two segments."""
    def point_seg_dist_sq(p, a, b):
        ax, ay = a
        bx, by = b
        px, py = p
        dx, dy = bx - ax, by - ay
        if dx == 0 and dy == 0:
            return (px - ax)**2 + (py - ay)**2
        t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
        t = max(0, min(1, t))
        cx = ax + t * dx
        cy = ay + t * dy
        return (px - cx)**2 + (py - cy)**2
    
    # Check if segments intersect
    if _segments_intersect_strict(p1, p2, p3, p4, 1e-12):
        return 0.0
    
    return min(point_seg_dist_sq(p1, p3, p4),
               point_seg_dist_sq(p2, p3, p4),
               point_seg_dist_sq(p3, p1, p2),
               point_seg_dist_sq(p4, p1, p2))

def _compute_required_outer_side(inner_hexes, outer_center, outer_angle_deg, tol=1e-6):
    """Compute minimum outer side length needed to contain all inner hexagons."""
    ox, oy = outer_center
    oa = math.radians(outer_angle_deg)
    
    # Outer hexagon vertices for a unit side, then scale
    outer_unit = []
    for i in range(6):
        a = oa + i * math.pi / 3
        outer_unit.append((math.cos(a), math.sin(a)))
    
    # We need to find minimum s such that all inner vertices are inside s * outer_unit + (ox, oy)
    # Using support functions: for each direction, max support of inner <= s * support of outer_unit
    
    # Sample many directions
    best_s = 0.0
    num_dirs = 72
    for k in range(num_dirs):
        theta = 2 * math.pi * k / num_dirs
        ux, uy = math.cos(theta), math.sin(theta)
        
        # Max support of all inner hexagons in this direction
        max_inner = 0.0
        for hx, hy, ha in inner_hexes:
            hangle = math.radians(ha)
            for i in range(6):
                a = hangle + i * math.pi / 3
                vx = hx + math.cos(a)
                vy = hy + math.sin(a)
                val = vx * ux + vy * uy
                if val > max_inner:
                    max_inner = val
        
        # Support of outer unit hexagon in this direction (centered at origin)
        max_outer = 0.0
        for i in range(6):
            a = oa + i * math.pi / 3
            val = math.cos(a) * ux + math.sin(a) * uy
            if val > max_outer:
                max_outer = val
        
        if max_outer > 1e-12:
            s_needed = max_inner / max_outer
            if s_needed > best_s:
                best_s = s_needed
        else:
            # Degenerate direction, use radius
            max_rad = 0.0
            for hx, hy, ha in inner_hexes:
                hangle = math.radians(ha)
                for i in range(6):
                    a = hangle + i * math.pi / 3
                    vx = hx + math.cos(a)
                    vy = hy + math.sin(a)
                    rad = math.sqrt(vx * vx + vy * vy)
                    if rad > max_rad:
                        max_rad = rad
            if max_rad > best_s:
                best_s = max_rad
    
    return best_s

def _validate(inner, outer_center, outer_side, outer_angle_deg, eps=1e-6):
    """Validate that all inner hexagons are inside the outer hexagon and don't overlap each other."""
    ox, oy = outer_center
    outer_pts = _hex_vertices(ox, oy, outer_side, outer_angle_deg)
    
    # Check containment
    for hx, hy, ha in inner:
        inner_pts = _hex_vertices(hx, hy, 1.0, ha)
        if not _outer_contains_inner(outer_pts, inner_pts, eps):
            return False
    
    # Check non-overlap
    for i in range(len(inner)):
        for j in range(i + 1, len(inner)):
            p1 = _hex_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2])
            p2 = _hex_vertices(inner[j][0], inner[j][1], 1.0, inner[j][2])
            if _polygons_overlap(p1, p2, eps):
                return False
    
    return True

def _build_initial_config():
    """Build a reasonable initial configuration of 12 unit hexagons."""
    # Use a honeycomb-like arrangement
    # Hexagonal lattice with spacing 2*sqrt(3) for non-overlapping unit hexagons
    # Actually for regular hexagons of side 1, the minimum center distance for non-overlap
    # depends on relative orientation. For same orientation, it's 2*sqrt(3) ≈ 3.464
    
    config = []
    # Central hexagon
    config.append((0.0, 0.0, 0.0))
    
    # 6 hexagons around center at distance 2*sqrt(3)
    d = 2.0 * math.sqrt(3)
    for i in range(6):
        a = i * math.pi / 3
        config.append((d * math.cos(a), d * math.sin(a), 0.0))
    
    # 5 more hexagons in second ring
    d2 = 4.0 * math.sqrt(3)
    for i in range(5):
        a = i * 2 * math.pi / 5 + math.pi / 5
        config.append((d2 * math.cos(a), d2 * math.sin(a), 0.0))
    
    return config

def _simulate_annealing(config, max_iters=5000, t_start=2.0, t_end=0.01, deadline=None):
    """Simulated annealing to optimize the configuration."""
    current = list(config)
    current_side = _compute_required_outer_side(current, (0.0, 0.0), 0.0)
    best = list(current)
    best_side = current_side
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    # First, try to find a valid configuration
    # Compute outer hexagon and check validity
    for it in range(max_iters):
        if deadline and time.time() > deadline - 5:
            break
        
        T = t_start * (t_end / t_start) ** (it / max_iters)
        
        # Generate perturbation
        new_config = [list(c) for c in current]
        
        # Perturb a random hexagon
        idx = int(np.random.randint(len(new_config)))
        new_config[idx][0] += np.random.normal(0, T * 0.5)
        new_config[idx][1] += np.random.normal(0, T * 0.5)
        if np.random.random() < 0.3:
            new_config[idx][2] += np.random.normal(0, T * 10)
        
        # Compute new objective
        new_side = _compute_required_outer_side(new_config, (0.0, 0.0), 0.0)
        
        # Check if configuration is valid (non-overlapping)
        valid = True
        for i in range(len(new_config)):
            for j in range(i + 1, len(new_config)):
                p1 = _hex_vertices(new_config[i][0], new_config[i][1], 1.0, new_config[i][2])
                p2 = _hex_vertices(new_config[j][0], new_config[j][1], 1.0, new_config[j][2])
                if _polygons_overlap(p1, p2, 1e-8):
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            continue
        
        # Accept or reject
        delta = new_side - current_side
        if delta < 0 or np.random.random() < math.exp(-delta / max(T, 1e-10)):
            current = new_config
            current_side = new_side
            
            if current_side < best_side:
                best = [list(c) for c in current]
                best_side = current_side
    
    return best, best_side

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Build initial configuration
    config = _build_initial_config()
    
    # Compute initial outer side
    initial_side = _compute_required_outer_side(config, (0.0, 0.0), 0.0)
    
    # Verify initial configuration is valid
    if not _validate(config, (0.0, 0.0), initial_side, 0.0):
        # Try to fix by spreading out more
        for spread in [1.2, 1.5, 2.0, 3.0]:
            config_spread = [(x * spread, y * spread, a) for x, y, a in config]
            side_spread = _compute_required_outer_side(config_spread, (0.0, 0.0), 0.0)
            if _validate(config_spread, (0.0, 0.0), side_spread, 0.0):
                config = config_spread
                initial_side = side_spread
                break
    
    best_config = [list(c) for c in config]
    best_side = initial_side
    
    # Run simulated annealing
    if time.time() < deadline - 10:
        sa_config, sa_side = _simulate_annealing(config, max_iters=3000, t_start=1.5, t_end=0.05, deadline=deadline)
        if sa_side < best_side:
            best_config = sa_config
            best_side = sa_side
    
    # Try different outer angles
    if time.time() < deadline - 5:
        for angle in [0, 30, 60, 90, 120, 150]:
            side = _compute_required_outer_side(best_config, (0.0, 0.0), float(angle))
            if side < best_side:
                best_side = side
    
    # Final validation
    final_valid = _validate(best_config, (0.0, 0.0), best_side, 0.0)
    if not final_valid:
        # Fallback to a guaranteed valid configuration
        # Place hexagons in a line with large spacing
        fallback = [(10.0 * k, 0.0, 0.0) for k in range(N)]
        fb_side = _compute_required_outer_side(fallback, (0.0, 0.0), 0.0)
        if _validate(fallback, (0.0, 0.0), fb_side, 0.0):
            return fallback, (0.0, 0.0), fb_side, 0.0
        else:
            # Very conservative fallback
            fallback2 = [(20.0 * k, 0.0, 0.0) for k in range(N)]
            return fallback2, (0.0, 0.0), 80.0, 0.0
    
    return best_config, (0.0, 0.0), best_side, 0.0
