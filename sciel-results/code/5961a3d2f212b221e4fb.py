"""Hexagonal packing of unit hexagons inside a regular hexagon."""
import time
import math
import numpy as np

N = 11
_START = time.time()
_DEADLINE = _START + 130.0  # safety margin before 150s kill

def _hex_verts(cx, cy, s, ang_deg):
    """Vertices of a regular hexagon (flat-top when ang=0)."""
    a = math.radians(ang_deg)
    ca, sa = math.cos(a), math.sin(a)
    out = []
    for i in range(6):
        t = math.radians(60.0 * i)
        x = s * math.cos(t)
        y = s * math.sin(t)
        out.append((cx + ca * x - sa * y, cy + sa * x + ca * y))
    return out

def _convex_hull(points):
    """Andrew's monotone chain; returns CCW hull."""
    pts = sorted(set((round(p[0], 9), round(p[1], 9)) for p in points))
    if len(pts) <= 1:
        return pts
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 1e-12:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 1e-12:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]

def _point_in_convex(p, poly):
    """CCW polygon; returns True if p is inside (strict)."""
    n = len(poly)
    for i in range(n):
        a = poly[i]; b = poly[(i+1) % n]
        cross = (b[0]-a[0])*(p[1]-a[1]) - (b[1]-a[1])*(p[0]-a[0])
        if cross < -1e-9:
            return False
    return True

def _dist_pt_seg(p, a, b):
    ax, ay = a; bx, by = b; px, py = p
    dx, dy = bx-ax, by-ay
    L2 = dx*dx + dy*dy
    if L2 < 1e-18:
        return math.hypot(px-ax, py-ay)
    t = ((px-ax)*dx + (py-ay)*dy) / L2
    t = max(0.0, min(1.0, t))
    qx = ax + t*dx; qy = ay + t*dy
    return math.hypot(px-qx, py-qy)

def _polygon_distance(p1, p2):
    """Min distance between two convex polygons (CCW)."""
    best = float('inf')
    # vertex-in-polygon checks (one direction suffices for min dist)
    for v in p1:
        if _point_in_convex(v, p2):
            return 0.0
    for v in p2:
        if _point_in_convex(v, p1):
            return 0.0
    # edge-edge distances
    n1 = len(p1); n2 = len(p2)
    for i in range(n1):
        a1 = p1[i]; a2 = p1[(i+1) % n1]
        for j in range(n2):
            b1 = p2[j]; b2 = p2[(j+1) % n2]
            # quick bbox reject
            if max(min(a1[0],a2[0]), min(b1[0],b2[0])) - min(max(a1[0],a2[0]), max(b1[0],b2[0])) > best:
                continue
            if max(min(a1[1],a2[1]), min(b1[1],b2[1])) - min(max(a1[1],a2[1]), max(b1[1],b2[1])) > best:
                continue
            # segment-segment distance
            d = _seg_seg_dist(a1, a2, b1, b2)
            if d < best:
                best = d
                if best < 1e-9:
                    return 0.0
    return best

def _seg_seg_dist(p1, p2, p3, p4):
    """Min distance between two segments (2D)."""
    # Check if segments intersect
    if _segs_intersect(p1, p2, p3, p4):
        return 0.0
    d = min(_dist_pt_seg(p1, p3, p4), _dist_pt_seg(p2, p3, p4),
            _dist_pt_seg(p3, p1, p2), _dist_pt_seg(p4, p1, p2))
    return d

def _segs_intersect(p1, p2, p3, p4):
    def orient(a, b, c):
        v = (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])
        if v > 1e-12: return 1
        if v < -1e-12: return -1
        return 0
    o1 = orient(p1, p2, p3)
    o2 = orient(p1, p2, p4)
    o3 = orient(p3, p4, p1)
    o4 = orient(p3, p4, p2)
    if o1 != o2 and o3 != o4:
        return True
    return False

def _build_lattice(n, spacing, ang_deg):
    """Hexagonal lattice of n hexagons (flat-top, ang=0 base)."""
    # Hex lattice: offset rows
    # For n hexagons, arrange in roughly sqrt(n) x sqrt(n) hex grid
    cols = int(math.ceil(math.sqrt(n * 2.0 / math.sqrt(3))))
    rows = int(math.ceil(n / cols))
    pos = []
    placed = 0
    for r in range(rows):
        for c in range(cols):
            if placed >= n:
                break
            # hex lattice offset
            x = c * spacing * 1.5
            y = r * spacing * math.sqrt(3)
            if r % 2 == 1:
                x += spacing * 0.75
            pos.append((x, y, ang_deg))
            placed += 1
        if placed >= n:
            break
    # Center the layout
    xs = [p[0] for p in pos]
    ys = [p[1] for p in pos]
    cx = (min(xs) + max(xs)) / 2.0
    cy = (min(ys) + max(ys)) / 2.0
    pos = [(p[0]-cx, p[1]-cy, p[2]) for p in pos]
    return pos

def _compute_outer_side(inner, outer_ang=0.0):
    """Compute minimum outer hexagon side length that contains all inner hexagons.
    Returns (side_length, center) or (inf, None) if invalid."""
    # Get all vertices of inner hexagons
    all_verts = []
    for (x, y, a) in inner:
        verts = _hex_verts(x, y, 1.0, a)
        all_verts.extend(verts)
    
    # Convex hull of all vertices
    hull = _convex_hull(all_verts)
    if len(hull) < 3:
        return float('inf'), None
    
    # Find minimum enclosing regular hexagon
    # The outer hexagon has 6 sides; we need to find side length s and center (cx,cy)
    # such that all hull points are inside the hexagon.
    # 
    # For a regular hexagon with center (cx,cy) and side s, the support function
    # in direction theta is s * cos(theta - 60k) for the 6 directions.
    # Actually, a regular hexagon with flat-top orientation (vertices at 0,60,120,180,240,300)
    # has the property that a point (px,py) relative to center is inside iff:
    #   |px| <= s and |py| <= s*sqrt(3)/2 and |px + py/sqrt(3)| <= s
    # Wait, let me think more carefully.
    #
    # A regular hexagon with side s, flat-top, centered at origin:
    # Vertices at (s,0), (s/2, s*sqrt(3)/2), (-s/2, s*sqrt(3)/2), (-s,0), (-s/2,-s*sqrt(3)/2), (s/2,-s*sqrt(3)/2)
    # The hexagon is the intersection of 6 half-planes:
    #   x <= s, x >= -s
    #   x + sqrt(3)*y <= s*sqrt(3), x + sqrt(3)*y >= -s*sqrt(3)  -- wait
    #   -x + sqrt(3)*y <= s*sqrt(3), -x + sqrt(3)*y >= -s*sqrt(3)
    #
    # Actually for a regular hexagon with vertices at angles 0,60,120,180,240,300 and radius s:
    # The 6 sides are at distances s*cos(30) = s*sqrt(3)/2 from center, in directions 30,90,150,210,270,330.
    # So the half-planes are:
    #   cos(30)*x + sin(30)*y <= s*sqrt(3)/2  =>  (sqrt(3)/2)x + (1/2)y <= s*sqrt(3)/2  =>  sqrt(3)*x + y <= s*sqrt(3)
    #   cos(90)*x + sin(90)*y <= s*sqrt(3)/2  =>  y <= s*sqrt(3)/2
    #   cos(150)*x + sin(150)*y <= s*sqrt(3)/2  =>  -sqrt(3)*x + y <= s*sqrt(3)
    #   -sqrt(3)*x - y <= s*sqrt(3)
    #   -y <= s*sqrt(3)/2
    #   sqrt(3)*x - y <= s*sqrt(3)
    #
    # So for a point (px, py) relative to center, it's inside iff:
    #   sqrt(3)*px + py <= s*sqrt(3)
    #   -sqrt(3)*px + py <= s*sqrt(3)
    #   py <= s*sqrt(3)/2
    #   -py <= s*sqrt(3)/2
    #   sqrt(3)*px - py <= s*sqrt(3)
    #   -sqrt(3)*px - py <= s*sqrt(3)
    #
    # This gives: s >= (sqrt(3)*px + py)/sqrt(3), s >= (-sqrt(3)*px + py)/sqrt(3),
    #   s >= py/(sqrt(3)/2) = 2*py/sqrt(3), s >= -2*py/sqrt(3),
    #   s >= (sqrt(3)*px - py)/sqrt(3), s >= (-sqrt(3)*px - py)/sqrt(3)
    #
    # So s >= max over all points and all 6 constraints.
    #
    # For a rotated outer hexagon by angle phi, we rotate the points by -phi first.
    
    best_side = float('inf')
    best_cx = 0.0
    best_cy = 0.0
    
    # Try different center positions and outer angles
    # The center should be near the centroid of the hull
    hx = sum(p[0] for p in hull) / len(hull)
    hy = sum(p[1] for p in hull) / len(hull)
    
    # Try a few outer rotation angles
    for outer_ang in [0.0, 30.0]:
        a = math.radians(outer_ang)
        ca, sa = math.cos(a), math.sin(a)
        
        # Rotate hull points so outer hexagon is flat-top
        rot_hull = [(ca*p[0] + sa*p[1], -sa*p[0] + ca*p[1]) for p in hull]
        
        # Try center near centroid
        for cx_try in [hx, hx - 0.5, hx + 0.5, hx - 1.0, hx + 1.0]:
            for cy_try in [hy, hy - 0.5, hy + 0.5, hy - 1.0, hy + 1.0]:
                side = 0.0
                for (px, py) in rot_hull:
                    dx = px - cx_try
                    dy = py - cy_try
                    # 6 constraints for flat-top hexagon
                    s1 = (math.sqrt(3)*dx + dy) / math.sqrt(3)
                    s2 = (-math.sqrt(3)*dx + dy) / math.sqrt(3)
                    s3 = 2.0 * dy / math.sqrt(3)
                    s4 = -2.0 * dy / math.sqrt(3)
                    s5 = (math.sqrt(3)*dx - dy) / math.sqrt(3)
                    s6 = (-math.sqrt(3)*dx - dy) / math.sqrt(3)
                    side = max(side, s1, s2, s3, s4, s5, s6)
                
                if side < best_side:
                    best_side = side
                    best_cx = cx_try
                    best_cy = cy_try
    
    if best_side < float('inf'):
        # Convert center back
        a = math.radians(0.0)  # we already rotated
        return best_side, (best_cx, best_cy)
    return float('inf'), None

def _check_validity(inner, outer_center, outer_side, outer_ang):
    """Check if all inner hexagons fit inside the outer hexagon and don't overlap."""
    ocx, ocy = outer_center
    
    # Check each inner hexagon is inside outer
    for (x, y, a) in inner:
        verts = _hex_verts(x, y, 1.0, a)
        for v in verts:
            # Rotate to outer hexagon frame
            da = math.radians(outer_ang)
            ca, sa = math.cos(da), math.sin(da)
            dx = v[0] - ocx
            dy = v[1] - ocy
            rx = ca*dx + sa*dy
            ry = -sa*dx + ca*dy
            # Check inside flat-top hexagon of side outer_side
            if (math.sqrt(3)*rx + ry) > outer_side * math.sqrt(3) + 1e-7:
                return False
            if (-math.sqrt(3)*rx + ry) > outer_side * math.sqrt(3) + 1e-7:
                return False
            if ry > outer_side * math.sqrt(3) / 2 + 1e-7:
                return False
            if -ry > outer_side * math.sqrt(3) / 2 + 1e-7:
                return False
            if (math.sqrt(3)*rx - ry) > outer_side * math.sqrt(3) + 1e-7:
                return False
            if (-math.sqrt(3)*rx - ry) > outer_side * math.sqrt(3) + 1e-7:
                return False
    
    # Check pairwise non-overlap
    polys = [_hex_verts(x, y, 1.0, a) for (x, y, a) in inner]
    for i in range(len(polys)):
        for j in range(i+1, len(polys)):
            d = _polygon_distance(polys[i], polys[j])
            if d < -1e-7:  # overlap
                return False
    return True

def _optimize_spacing(inner, outer_center, outer_side, outer_ang, max_iter=50, lr=0.1):
    """Try to reduce outer_side by adjusting inner positions."""
    # Simple gradient-free: try small perturbations
    best = (inner, outer_center, outer_side, outer_ang)
    best_side = outer_side
    
    for _ in range(max_iter):
        if time.time() > _DEADLINE:
            break
        improved = False
        for i in range(len(inner)):
            for dx in [-0.2, 0.2]:
                for dy in [-0.2, 0.2]:
                    new_inner = [list(p) for p in inner]
                    new_inner[i][0] += dx
                    new_inner[i][1] += dy
                    new_inner = [tuple(p) for p in new_inner]
                    
                    # Quick check: just recompute outer side
                    side, center = _compute_outer_side(new_inner, outer_ang)
                    if side < best_side - 1e-6:
                        if _check_validity(new_inner, center, side, outer_ang):
                            inner = new_inner
                            outer_center = center
                            outer_side = side
                            best_side = side
                            improved = True
        if not improved:
            break
    
    return inner, outer_center, outer_side, outer_ang

def construct():
    global _START, _DEADLINE
    _START = time.time()
    _DEADLINE = _START + 130.0
    
    # Start with hexagonal lattice
    best_result = None
    best_score = float('inf')
    
    # Try different spacings and arrangements
    for spacing in [2.0, 2.1, 2.2, 2.3, 2.5]:
        if time.time() > _DEADLINE:
            break
        inner = _build_lattice(N, spacing, 0.0)
        
        # Compute outer hexagon
        side, center = _compute_outer_side(inner, 0.0)
        if side < float('inf') and center is not None:
            if _check_validity(inner, center, side, 0.0):
                if side < best_score:
                    best_score = side
                    best_result = (inner, center, side, 0.0)
                
                # Try to optimize
                inner2, center2, side2, ang2 = _optimize_spacing(
                    inner, center, side, 0.0, max_iter=20, lr=0.1)
                if side2 < best_score:
                    best_score = side2
                    best_result = (inner2, center2, side2, ang2)
    
    # Also try with 30-degree rotation of inner hexagons
    for spacing in [2.0, 2.1, 2.2, 2.5]:
        if time.time() > _DEADLINE:
            break
        inner = _build_lattice(N, spacing, 30.0)
        
        side, center = _compute_outer_side(inner, 0.0)
        if side < float('inf') and center is not None:
            if _check_validity(inner, center, side, 0.0):
                if side < best_score:
                    best_score = side
                    best_result = (inner, center, side, 0.0)
    
    if best_result is None:
        # Fallback: simple line
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    return best_result
