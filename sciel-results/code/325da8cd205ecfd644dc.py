"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.spatial import ConvexHull
import time

N = 12

def hexagon_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon centered at (cx, cy) with given side length and rotation."""
    angle_rad = np.radians(angle_deg)
    angles = angle_rad + np.arange(6) * np.pi / 3
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def check_validity(inner, outer_center, outer_side, outer_angle):
    """Check if all inner hexagons are disjoint and inside the outer hexagon."""
    outer_verts = hexagon_vertices(outer_center[0], outer_center[1], outer_side, outer_angle)
    
    # Check each inner hexagon is inside outer hexagon
    for x, y, ang in inner:
        v = hexagon_vertices(x, y, 1.0, ang)
        # Point-in-polygon test using winding number or ray casting
        for px, py in v:
            # Check if point is inside outer hexagon
            inside = False
            n_outer = len(outer_verts)
            j = n_outer - 1
            for i in range(n_outer):
                xi, yi = outer_verts[i]
                xj, yj = outer_verts[j]
                if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
                    inside = not inside
                j = i
            if not inside:
                return False
    
    # Check pairwise disjointness of inner hexagons
    for i in range(len(inner)):
        vi = hexagon_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2])
        for j in range(i + 1, len(inner)):
            vj = hexagon_vertices(inner[j][0], inner[j][1], 1.0, inner[j][2])
            # Check if any edge of vi intersects any edge of vj
            if _polygons_intersect(vi, vj):
                return False
    return True

def _polygons_intersect(p1, p2):
    """Check if two convex polygons intersect."""
    # Check if any vertex of p1 is inside p2 or vice versa
    if _point_in_polygon(p1[0], p2):
        return True
    if _point_in_polygon(p2[0], p1):
        return True
    # Check edge intersections
    for i in range(len(p1)):
        for j in range(len(p2)):
            if _segments_intersect(p1[i], p1[(i+1)%len(p1)], p2[j], p2[(j+1)%len(p2)]):
                return True
    return False

def _point_in_polygon(point, polygon):
    """Check if point is inside a convex polygon."""
    px, py = point
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i+1) % n]
        cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
        if cross < -1e-9:
            return False
    return True

def _segments_intersect(p1, p2, p3, p4):
    """Check if line segments p1-p2 and p3-p4 intersect."""
    def orientation(a, b, c):
        val = (b[1]-a[1])*(c[0]-b[0]) - (b[0]-a[0])*(c[1]-b[1])
        if val > 1e-9: return 1
        if val < -1e-9: return -1
        return 0
    
    def on_segment(a, b, c):
        if a[0] <= max(b[0], c[0]) and a[0] >= min(b[0], c[0]) and \
           a[1] <= max(b[1], c[1]) and a[1] >= min(b[1], c[1]):
            return True
        return False
    
    o1 = orientation(p1, p2, p3)
    o2 = orientation(p1, p2, p4)
    o3 = orientation(p3, p4, p1)
    o4 = orientation(p3, p4, p2)
    
    if o1 != o2 and o3 != o4:
        return True
    
    if o1 == 0 and on_segment(p1, p3, p2): return True
    if o2 == 0 and on_segment(p1, p4, p2): return True
    if o3 == 0 and on_segment(p3, p1, p4): return True
    if o4 == 0 and on_segment(p3, p2, p4): return True
    
    return False

def compute_min_side(inner, outer_center, outer_angle):
    """Compute minimum side length needed for given inner configuration."""
    # For a fixed outer hexagon orientation and center, we need to find the minimum
    # side length such that all inner hexagons are contained.
    
    # The outer hexagon is regular with side length s. Its inradius is s*sqrt(3)/2.
    # But containment of rotated hexagons is complex. Let's use a simpler approach:
    # For each inner hexagon, compute the maximum distance from outer_center to any vertex,
    # then find the minimum outer hexagon side that contains all points.
    
    all_points = []
    for x, y, ang in inner:
        v = hexagon_vertices(x, y, 1.0, ang)
        all_points.append(v)
    
    all_points = np.vstack(all_points)
    
    # For a regular hexagon centered at outer_center with rotation outer_angle,
    # a point (px, py) is inside if for all 6 sides, the point is on the correct side.
    # The support function approach: for direction theta, the distance from center to the
    # side perpendicular to theta must be >= max over all points of (point - center) dot theta.
    
    # For a regular hexagon with side s and rotation alpha, the support in direction theta is:
    # h(theta) = s * max over 6 vertices of (vertex - center) dot theta / s = s * max(cos(theta - alpha - k*pi/3))
    
    # Actually, for a regular hexagon with inradius r = s*sqrt(3)/2, the support in direction theta
    # is r * max over the 6 normal directions of cos(theta - normal_angle).
    
    # Let's binary search on side length s.
    def is_contained(s):
        outer_verts = hexagon_vertices(outer_center[0], outer_center[1], s, outer_angle)
        for px, py in all_points:
            if not _point_in_polygon((px, py), outer_verts):
                return False
        return True
    
    lo, hi = 0.0, 50.0
    for _ in range(50):
        mid = (lo + hi) / 2
        if is_contained(mid):
            hi = mid
        else:
            lo = mid
    return hi + 0.05

def construct():
    start_time = time.time()
    deadline = start_time + 120.0
    
    best_score = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_side = 40.0
    best_angle = 0.0
    
    # Try a hexagonal packing arrangement
    # Unit hexagons in a hexagonal lattice: neighbors are at distance 2*sqrt(3) apart
    # for flat-top hexagons with side 1.
    
    # Let's try several configurations and pick the best one.
    
    # Configuration 1: Linear arrangement
    inner1 = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
    center1 = (0.0, 0.0)
    angle1 = 0.0
    side1 = compute_min_side(inner1, center1, angle1)
    if side1 < best_score:
        best_score = side1
        best_inner = inner1
        best_center = center1
        best_side = side1
        best_angle = angle1
    
    if time.time() > deadline:
        return best_inner, best_center, best_side, best_angle
    
    # Configuration 2: Hexagonal cluster (7 in center + 5 more)
    # Hexagonal lattice positions for flat-top hexagons with side 1:
    # Horizontal spacing: 2*sqrt(3) ≈ 3.464
    # Vertical spacing: 3
    # Offset for odd rows: sqrt(3) ≈ 1.732
    
    # Let's place them in a compact hexagonal shape.
    # For 12 hexagons, a good arrangement might be a hexagon with radius 2 (which has 1+6+12=19, too many)
    # or a rectangle-like shape.
    
    # Try a 3x4 grid in hexagonal coordinates
    sqrt3 = np.sqrt(3)
    inner2 = []
    positions = []
    for i in range(3):
        for j in range(4):
            x = j * 2 * sqrt3 + (i % 2) * sqrt3
            y = i * 3
            positions.append((x, y))
    
    # Center them
    cx = np.mean([p[0] for p in positions])
    cy = np.mean([p[1] for p in positions])
    inner2 = [[p[0] - cx, p[1] - cy, 0.0] for p in positions]
    center2 = (0.0, 0.0)
    angle2 = 0.0
    side2 = compute_min_side(inner2, center2, angle2)
    if side2 < best_score:
        best_score = side2
        best_inner = inner2
        best_center = center2
        best_side = side2
        best_angle = angle2
    
    if time.time() > deadline:
        return best_inner, best_center, best_side, best_angle
    
    # Configuration 3: Try different outer hexagon rotations
    for outer_ang in [0, 30, 60, 90, 120, 150]:
        if time.time() > deadline:
            break
        for inner_cfg, center_cfg in [(inner1, (0.0, 0.0)), (inner2, (0.0, 0.0))]:
            if time.time() > deadline:
                break
            side = compute_min_side(inner_cfg, center_cfg, outer_ang)
            if side < best_score:
                best_score = side
                best_inner = inner_cfg
                best_center = center_cfg
                best_side = side
                best_angle = outer_ang
    
    return best_inner, best_center, best_side, best_angle
# EVOLVE-BLOCK-END
