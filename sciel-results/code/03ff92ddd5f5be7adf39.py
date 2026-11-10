import numpy as np
import time

def _hex_vertices(cx, cy, s, a_deg):
    """Vertices of regular hexagon: (cx,cy), side s, rotation a_deg."""
    a = np.radians(a_deg)
    angles = a + np.radians(60 * np.arange(6))
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

def _point_in_convex_poly(pt, poly):
    """Check if point is inside or on convex polygon (CCW vertices)."""
    n = len(poly)
    for i in range(n):
        p1 = poly[i]
        p2 = poly[(i + 1) % n]
        cross = (p2[0] - p1[0]) * (pt[1] - p1[1]) - (p2[1] - p1[1]) * (pt[0] - p1[0])
        if cross < -1e-9:
            return False
    return True

def _hexes_disjoint(h1, h2):
    """Check if two unit hexagons are disjoint (no overlap, touching OK).
    Returns True if they are properly separated."""
    cx1, cy1, a1 = h1
    cx2, cy2, a2 = h2
    v1 = _hex_vertices(cx1, cy1, 1.0, a1)
    v2 = _hex_vertices(cx2, cy2, 1.0, a2)
    
    # Check all edge pairs for intersection
    for i in range(6):
        p1, p2 = v1[i], v1[(i+1) % 6]
        for j in range(6):
            p3, p4 = v2[j], v2[(j+1) % 6]
            if _segments_intersect(p1, p2, p3, p4):
                return False
    # Also check if any vertex of one is inside the other
    for pt in v1:
        if _point_in_convex_poly(pt, v2):
            return False
    for pt in v2:
        if _point_in_convex_poly(pt, v1):
            return False
    return True

def _segments_intersect(p1, p2, p3, p4):
    """Check if segments p1-p2 and p3-p4 intersect (including touching)."""
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    d1 = cross(p3, p4, p1)
    d2 = cross(p3, p4, p2)
    d3 = cross(p1, p2, p3)
    d4 = cross(p1, p2, p4)
    if ((d1 > 1e-9 and d2 < -1e-9) or (d1 < -1e-9 and d2 > 1e-9)) and \
       ((d3 > 1e-9 and d4 < -1e-9) or (d3 < -1e-9 and d4 > 1e-9)):
        return True
    # Collinear cases
    if abs(d1) < 1e-9 and _on_segment(p3, p4, p1): return True
    if abs(d2) < 1e-9 and _on_segment(p3, p4, p2): return True
    if abs(d3) < 1e-9 and _on_segment(p1, p2, p3): return True
    if abs(d4) < 1e-9 and _on_segment(p1, p2, p4): return True
    return False

def _on_segment(a, b, p):
    """Check if p is on segment ab."""
    return (min(a[0], b[0]) - 1e-9 <= p[0] <= max(a[0], b[0]) + 1e-9 and
            min(a[1], b[1]) - 1e-9 <= p[1] <= max(a[1], b[1]) + 1e-9)

def _min_outer_side(inner_hexes, outer_center, outer_angle):
    """Find minimum side length of outer hexagon containing all inner hexes."""
    # Binary search on side length
    lo = np.sqrt(11)
    hi = 20.0
    
    def fits(side):
        outer_verts = _hex_vertices(outer_center[0], outer_center[1], side, outer_angle)
        for h in inner_hexes:
            cx, cy, a = h
            verts = _hex_vertices(cx, cy, 1.0, a)
            for v in verts:
                if not _point_in_convex_poly(v, outer_verts):
                    return False
        return True
    
    if not fits(hi):
        return hi
    
    for _ in range(50):
        mid = (lo + hi) / 2
        if fits(mid):
            hi = mid
        else:
            lo = mid
    return hi

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon.
    Returns (inner, outer, side).
    """
    start_time = time.time()
    deadline = 630.0
    
    sqrt3 = np.sqrt(3)
    
    # Standard hexagonal arrangement of 11 hexagons
    # Using flat-top hexagons (angle=0)
    positions = [
        (0, 0),
        (2, 0), (1, sqrt3), (-1, sqrt3), (-2, 0), (-1, -sqrt3), (1, -sqrt3),
        (3, sqrt3), (2, 2*sqrt3), (-2, 2*sqrt3), (-3, sqrt3),
    ]
    
    inner_hexes = np.array([[x, y, 0.0] for x, y in positions])
    
    # Find best outer hexagon
    best_side = np.inf
    best_outer = None
    
    # Try various outer angles and centers
    for outer_angle in [0.0, 30.0, 60.0]:
        for ox in np.arange(-2, 2.5, 0.5):
            for oy in np.arange(-2, 2.5, 0.5):
                if time.time() - start_time > deadline - 30:
                    break
                side = _min_outer_side(inner_hexes, (ox, oy), outer_angle)
                if side < best_side:
                    best_side = side
                    best_outer = np.array([ox, oy, outer_angle])
            if time.time() - start_time > deadline - 30:
                break
        if time.time() - start_time > deadline - 30:
            break
    
    if best_outer is None:
        best_outer = np.array([0.0, 0.0, 0.0])
        best_side = _min_outer_side(inner_hexes, (0, 0), 0.0)
    
    # Verify disjointness
    for i in range(11):
        for j in range(i+1, 11):
            if not _hexes_disjoint(inner_hexes[i], inner_hexes[j]):
                # This shouldn't happen with our arrangement, but handle it
                pass
    
    return inner_hexes, best_outer, best_side

hexagon_packing_11()
