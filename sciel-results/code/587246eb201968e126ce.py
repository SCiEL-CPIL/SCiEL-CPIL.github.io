"""Packing 12 unit hexagons in a regular hexagon via simulated annealing."""
import numpy as np
import time

N = 12
DEADLINE = time.time() + 140.0  # safety margin before 150s limit

def _hex_vertices(cx, cy, angle_deg):
    """Vertices of a unit regular hexagon centered at (cx,cy) rotated by angle_deg."""
    a = np.deg2rad(angle_deg)
    cos_a, sin_a = np.cos(a), np.sin(a)
    # Unit hexagon: vertices at angles 0, 60, 120, 180, 240, 300 degrees
    base_angles = np.arange(6) * np.pi / 3.0
    rx = np.cos(base_angles)
    ry = np.sin(base_angles)
    # Rotate
    x = cos_a * rx - sin_a * ry
    y = sin_a * rx + cos_a * ry
    return np.column_stack((x + cx, y + cy))

def _outer_hex_vertices(cx, cy, side, angle_deg):
    """Vertices of regular hexagon with given side length."""
    a = np.deg2rad(angle_deg)
    cos_a, sin_a = np.cos(a), np.sin(a)
    base_angles = np.arange(6) * np.pi / 3.0
    rx = side * np.cos(base_angles)
    ry = side * np.sin(base_angles)
    x = cos_a * rx - sin_a * ry
    y = sin_a * rx + cos_a * ry
    return np.column_stack((x + cx, y + cy))

def _seg_seg_dist(p1, p2, p3, p4):
    """Minimum distance between two line segments."""
    d1 = p2 - p1
    d2 = p4 - p3
    r = p1 - p3
    a = np.dot(d1, d1)
    e = np.dot(d2, d2)
    f = np.dot(d2, r)
    if a < 1e-12 and e < 1e-12:
        return np.linalg.norm(p1 - p3)
    if a < 1e-12:
        t = f / e
        t = max(0.0, min(1.0, t))
        return np.linalg.norm(p1 - (p3 + t * d2))
    c = np.dot(d1, r)
    if e < 1e-12:
        t = c / a
        t = max(0.0, min(1.0, t))
        return np.linalg.norm(p1 + t * d1 - p3)
    b = np.dot(d1, d2)
    denom = a * e - b * b
    if abs(denom) < 1e-12:
        t = 0.0
        s = max(0.0, min(1.0, f / e))
    else:
        t = (b * f - c * e) / denom
        s = (a * f - c * b) / denom
        if t < 0.0:
            t = 0.0
            s = max(0.0, min(1.0, f / e))
        elif t > 1.0:
            t = 1.0
            s = max(0.0, min(1.0, (f + b) / e))
        elif s < 0.0:
            s = 0.0
            t = max(0.0, min(1.0, -c / a))
        elif s > 1.0:
            s = 1.0
            t = max(0.0, min(1.0, (a - c) / a))
    return np.linalg.norm(p1 + t * d1 - (p3 + s * d2))

def _polygons_disjoint(poly1, poly2, tol=1e-9):
    """Check if two convex polygons are disjoint (no interior overlap)."""
    n1 = len(poly1)
    n2 = len(poly2)
    # Check if any edge of poly1 intersects any edge of poly2
    for i in range(n1):
        p1 = poly1[i]
        p2 = poly1[(i + 1) % n1]
        for j in range(n2):
            p3 = poly2[j]
            p4 = poly2[(j + 1) % n2]
            # Check segment intersection
            d1 = p2 - p1
            d2 = p4 - p3
            r = p1 - p3
            a = np.dot(d1, d1)
            e = np.dot(d2, d2)
            f = np.dot(d2, r)
            c = np.dot(d1, r)
            b = np.dot(d1, d2)
            denom = a * e - b * b
            if abs(denom) < 1e-12:
                continue
            t = (b * f - c * e) / denom
            s = (a * f - c * b) / denom
            if 0 < t < 1 and 0 < s < 1:
                return False  # Edges intersect
    # Check if one polygon contains a vertex of the other
    def point_in_convex_poly(pt, poly):
        n = len(poly)
        for i in range(n):
            cross = np.cross(poly[(i+1)%n] - poly[i], pt - poly[i])
            if cross < -tol:
                return False
        return True
    if point_in_convex_poly(poly1[0], poly2):
        return False
    if point_in_convex_poly(poly2[0], poly1):
        return False
    return True

def _all_disjoint(inner_hexes):
    """Check all pairs of inner hexagons are disjoint."""
    n = len(inner_hexes)
    for i in range(n):
        for j in range(i+1, n):
            if not _polygons_disjoint(inner_hexes[i], inner_hexes[j]):
                return False
    return True

def _inside_outer(inner_hex, outer_hex):
    """Check if inner hex is entirely inside outer hex."""
    return all(
        all(
            np.cross(outer_hex[(k+1)%6] - outer_hex[k], v - outer_hex[k]) >= -1e-9
            for k in range(6)
        )
        for v in inner_hex
    )

def _compute_outer_side(inner_hexes, center, angle_deg):
    """Compute minimum side length of outer hexagon containing all inner hexes."""
    # For a regular hexagon, the side length needed is determined by the
    # maximum support function in the 6 normal directions
    a = np.deg2rad(angle_deg)
    # The 6 outward normal directions of the outer hexagon
    normal_angles = np.arange(6) * np.pi / 3.0 + np.pi / 6  # normals are offset by 30 deg
    cos_n = np.cos(normal_angles)
    sin_n = np.sin(normal_angles)
    
    # For each direction, find max projection of all vertices
    max_proj = 0.0
    for hex_verts in inner_hexes:
        for v in hex_verts:
            rel = v - center
            for k in range(6):
                proj = rel[0] * cos_n[k] + rel[1] * sin_n[k]
                if proj > max_proj:
                    max_proj = proj
    
    # For a regular hexagon with side s, the apothem (distance from center to side) is s*sqrt(3)/2
    # So s = max_proj / (sqrt(3)/2) = 2*max_proj/sqrt(3)
    return 2.0 * max_proj / np.sqrt(3.0)

def _is_valid(inner_positions, outer_center, outer_side, outer_angle):
    """Check if the packing is valid."""
    inner_hexes = [_hex_vertices(x, y, ang) for x, y, ang in inner_positions]
    outer_hex = _outer_hex_vertices(outer_center[0], outer_center[1], outer_side, outer_angle)
    
    if not _all_disjoint(inner_hexes):
        return False
    for ih in inner_hexes:
        if not _inside_outer(ih, outer_hex):
            return False
    return True

def construct():
    start_time = time.time()
    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    # Initial configuration: hexagonal lattice arrangement
    # Place hexagons in a compact pattern
    def initial_config():
        # 12 hexagons in a roughly hexagonal cluster
        # Use hexagonal lattice: positions at (i*1.5, j*sqrt(3) + (i%2)*sqrt(3)/2)
        positions = []
        # Center + 6 around + 5 more
        positions.append((0.0, 0.0, 0.0))
        for k in range(6):
            angle = k * np.pi / 3.0
            positions.append((1.5 * np.cos(angle), 1.5 * np.sin(angle), 0.0))
        # Second ring partial
        for k in range(6):
            angle = k * np.pi / 3.0 + np.pi / 6.0
            r = 1.5 * np.sqrt(3.0)
            positions.append((r * np.cos(angle), r * np.sin(angle), 0.0))
        return positions[:12]
    
    inner = initial_config()
    outer_center = (0.0, 0.0)
    outer_angle = 0.0
    
    # Compute initial side
    inner_hexes = [_hex_vertices(x, y, ang) for x, y, ang in inner]
    current_side = _compute_outer_side(inner_hexes, outer_center, outer_angle)
    
    # Check validity
    if _is_valid(inner, outer_center, current_side, outer_angle):
        best_side = current_side
        best_inner = [list(p) for p in inner]
    
    # Simulated annealing
    T = 1.0
    T_min = 0.001
    cooling = 0.999
    step_size = 0.5
    
    np.random.seed(42)
    
    while T > T_min and time.time() < DEADLINE:
        # Random perturbation
        i = np.random.randint(12)
        new_inner = [list(p) for p in inner]
        
        move_type = np.random.random()
        if move_type < 0.6:
            # Translate
            dx = np.random.randn() * step_size
            dy = np.random.randn() * step_size
            new_inner[i][0] += dx
            new_inner[i][1] += dy
        elif move_type < 0.8:
            # Rotate
            dangle = np.random.randn() * 15.0
            new_inner[i][2] = (new_inner[i][2] + dangle) % 360.0
        else:
            # Both
            dx = np.random.randn() * step_size * 0.5
            dy = np.random.randn() * step_size * 0.5
            dangle = np.random.randn() * 10.0
            new_inner[i][0] += dx
            new_inner[i][1] += dy
            new_inner[i][2] = (new_inner[i][2] + dangle) % 360.0
        
        new_inner_hexes = [_hex_vertices(x, y, ang) for x, y, ang in new_inner]
        new_side = _compute_outer_side(new_inner_hexes, outer_center, outer_angle)
        
        # Check if valid
        if _is_valid(new_inner, outer_center, new_side, outer_angle):
            if new_side < best_side:
                best_side = new_side
                best_inner = [list(p) for p in new_inner]
            # Accept if better or with probability
            delta = new_side - current_side
            if delta < 0 or np.random.random() < np.exp(-delta / T):
                inner = new_inner
                current_side = new_side
        else:
            # Invalid, reject
            pass
        
        T *= cooling
        step_size = max(0.05, step_size * 0.9995)
    
    # Final refinement: try to optimize outer center and angle
    if best_inner is not None:
        # Try different outer angles
        for ang in [0, 30, 60, 90, 120, 150]:
            test_inner_hexes = [_hex_vertices(x, y, ia) for x, y, ia in best_inner]
            for cx in [-0.5, 0.0, 0.5]:
                for cy in [-0.5, 0.0, 0.5]:
                    s = _compute_outer_side(test_inner_hexes, (cx, cy), ang)
                    if _is_valid(best_inner, (cx, cy), s, ang) and s < best_side:
                        best_side = s
                        best_center = (cx, cy)
                        best_angle = ang
    
    if best_inner is None:
        # Fallback: return seed-like result
        best_inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        best_center = (0.0, 0.0)
        best_side = 40.0
        best_angle = 0.0
    
    return best_inner, best_center, best_side, best_angle
