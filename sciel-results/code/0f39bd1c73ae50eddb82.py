"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 11

def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon with center (cx,cy), side length, rotation."""
    theta = np.radians(angle_deg)
    verts = []
    for i in range(6):
        a = theta + i * np.pi / 3
        verts.append((cx + side * np.cos(a), cy + side * np.sin(a)))
    return np.array(verts)

def _point_in_hex(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside hexagon with center (cx,cy), side, rotation."""
    dx = px - cx
    dy = py - cy
    theta = np.radians(angle_deg)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    lx = dx * cos_t + dy * sin_t
    ly = -dx * sin_t + dy * cos_t
    s = side
    sqrt3 = np.sqrt(3)
    if abs(lx) > s + 1e-9:
        return False
    if abs(ly) > s * sqrt3 / 2 + 1e-9:
        return False
    if abs(lx) + abs(ly) / sqrt3 > s + 1e-9:
        return False
    return True

def _segments_intersect(p1, p2, p3, p4):
    """Check if segment p1-p2 intersects segment p3-p4."""
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    d1 = cross(p3, p4, p1)
    d2 = cross(p3, p4, p2)
    d3 = cross(p1, p2, p3)
    d4 = cross(p1, p2, p4)
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    return False

def _hexagons_overlap(h1, h2):
    """Check if two hexagons overlap. h1, h2 are (cx,cy,angle_deg)."""
    v1 = _hex_vertices(h1[0], h1[1], 1.0, h1[2])
    v2 = _hex_vertices(h2[0], h2[1], 1.0, h2[2])
    for v in v1:
        if _point_in_hex(v[0], v[1], h2[0], h2[1], 1.0, h2[2]):
            return True
    for v in v2:
        if _point_in_hex(v[0], v[1], h1[0], h1[1], 1.0, h1[2]):
            return True
    for i in range(6):
        for j in range(6):
            if _segments_intersect(v1[i], v1[(i+1)%6], v2[j], v2[(j+1)%6]):
                return True
    return False

def _all_inside_outer(inner_list, outer_center, outer_side, outer_angle):
    """Check all inner hexagons are inside the outer hexagon."""
    ocx, ocy = outer_center
    for (cx, cy, angle) in inner_list:
        v = _hex_vertices(cx, cy, 1.0, angle)
        for (px, py) in v:
            if not _point_in_hex(px, py, ocx, ocy, outer_side, outer_angle):
                return False
    return True

def _min_outer_side(inner_list, outer_center, outer_angle):
    """Find minimum outer side length such that all inner hexagons fit inside."""
    lo = 0.0
    hi = 50.0
    for _ in range(50):
        mid = (lo + hi) / 2
        if _all_inside_outer(inner_list, outer_center, mid, outer_angle):
            hi = mid
        else:
            lo = mid
    return hi

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    sqrt3 = np.sqrt(3)
    eps = 0.02  # small gap to avoid touching

    # 4-3-4 honeycomb arrangement with small gaps
    # Row 0 (4 hexagons) at y = -sqrt3 - eps
    row0 = [(-3.0, -sqrt3 - eps), (-1.0, -sqrt3 - eps), (1.0, -sqrt3 - eps), (3.0, -sqrt3 - eps)]
    # Row 1 (3 hexagons) at y = 0
    row1 = [(-2.0, 0.0), (0.0, 0.0), (2.0, 0.0)]
    # Row 2 (4 hexagons) at y = sqrt3 + eps
    row2 = [(-3.0, sqrt3 + eps), (-1.0, sqrt3 + eps), (1.0, sqrt3 + eps), (3.0, sqrt3 + eps)]

    all_positions = row0 + row1 + row2

    # Center the arrangement
    xs = [p[0] for p in all_positions]
    ys = [p[1] for p in all_positions]
    cx_arr = (min(xs) + max(xs)) / 2
    cy_arr = (min(ys) + max(ys)) / 2

    # Inner hexagons, all aligned (0 degrees)
    inner = [[p[0] - cx_arr, p[1] - cy_arr, 0.0] for p in all_positions]

    # Try different outer hexagon orientations and find the best
    best_side = 100.0
    best_outer_angle = 0.0
    best_outer_center = (0.0, 0.0)

    # Coarse search
    for angle_deg in np.arange(0, 180, 5):
        if time.time() > deadline:
            break
        side = _min_outer_side(inner, (0.0, 0.0), angle_deg)
        if side < best_side:
            best_side = side
            best_outer_angle = angle_deg

    # Refine around best angle
    if time.time() < deadline:
        for angle_deg in np.arange(best_outer_angle - 5, best_outer_angle + 5, 0.5):
            if time.time() > deadline:
                break
            side = _min_outer_side(inner, (0.0, 0.0), angle_deg)
            if side < best_side:
                best_side = side
                best_outer_angle = angle_deg

    # Further refinement
    if time.time() < deadline:
        for angle_deg in np.arange(best_outer_angle - 0.5, best_outer_angle + 0.5, 0.05):
            if time.time() > deadline:
                break
            side = _min_outer_side(inner, (0.0, 0.0), angle_deg)
            if side < best_side:
                best_side = side
                best_outer_angle = angle_deg

    # Final high-precision polish
    if time.time() < deadline:
        for angle_deg in np.arange(best_outer_angle - 0.05, best_outer_angle + 0.05, 0.01):
            if time.time() > deadline:
                break
            side = _min_outer_side(inner, (0.0, 0.0), angle_deg)
            if side < best_side:
                best_side = side
                best_outer_angle = angle_deg

    return inner, best_outer_center, best_side, best_outer_angle
# EVOLVE-BLOCK-END
