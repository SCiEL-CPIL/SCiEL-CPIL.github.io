"""Packing 11 unit hexagons inside a regular hexagon via combinatorial search."""
import numpy as np
from itertools import product
import math

N = 11

def hex_to_point(cx, cy, angle_deg, s=1.0):
    """Get the 6 vertices of a regular hexagon with center (cx,cy), side s, rotation angle_deg."""
    a = math.radians(angle_deg)
    pts = []
    for k in range(6):
        theta = a + k * math.pi / 3
        pts.append((cx + s * math.cos(theta), cy + s * math.sin(theta)))
    return pts

def point_in_hex(px, py, hx, hy, s, angle_deg):
    """Check if point (px,py) is inside regular hexagon centered at (hx,hy) with side s and rotation angle_deg."""
    # Transform to hexagon's local frame
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    dx = px - hx
    dy = py - hy
    lx = ca * dx + sa * dy
    ly = -sa * dx + ca * dy
    
    # Regular hexagon with flat-top orientation (vertices at 0,60,120,180,240,300)
    # The hexagon can be described as intersection of 6 half-planes.
    # For a regular hexagon with side s and "pointy-top" vs "flat-top":
    # Let's use the standard: vertices at angles 0, 60, ..., 300 from center.
    # The half-plane constraints: for each edge, the point must be on the interior side.
    
    # Actually, let's use a simpler approach: check distance to each edge.
    # A point is inside if it's on the correct side of all 6 edges.
    
    vertices = []
    for k in range(6):
        theta = k * math.pi / 3
        vertices.append((s * math.cos(theta), s * math.sin(theta)))
    
    for i in range(6):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % 6]
        # Edge from (x1,y1) to (x2,y2)
        # Normal pointing inward
        ex = x2 - x1
        ey = y2 - y1
        # Left normal (for CCW polygon, interior is to the left)
        nx = -ey
        ny = ex
        # Check if point is on the interior side
        # The center (0,0) should be on the interior side
        # Vector from vertex to center: (-x1, -y1)
        # Dot with normal should be >= 0 for interior
        # Vector from vertex to point: (lx-x1, ly-y1)
        dot = nx * (lx - x1) + ny * (ly - y1)
        if dot < -1e-9:
            return False
    return True

def hexes_disjoint(h1, h2, eps=1e-9):
    """Check if two unit hexagons are disjoint (no overlap in interior)."""
    # h1, h2 are (cx, cy, angle_deg)
    # We need to check if they overlap.
    # Simple approach: check if any vertex of one is inside the other,
    # and if any edge of one intersects any edge of other.
    
    cx1, cy1, a1 = h1
    cx2, cy2, a2 = h2
    
    # Quick bounding box check
    dx = abs(cx1 - cx2)
    dy = abs(cy1 - cy2)
    if dx > 2 * (1 + eps) or dy > 2 * (1 + eps):
        return True  # Disjoint
    
    # Get vertices
    v1 = hex_to_point(cx1, cy1, a1)
    v2 = hex_to_point(cx2, cy2, a2)
    
    # Check if any vertex of h1 is inside h2
    for px, py in v1:
        if point_in_hex(px, py, cx2, cy2, 1.0, a2):
            return False
    
    # Check if any vertex of h2 is inside h1
    for px, py in v2:
        if point_in_hex(px, py, cx1, cy1, 1.0, a1):
            return False
    
    # Check edge intersections
    def seg_intersect(p1, p2, p3, p4):
        """Check if segment p1-p2 intersects segment p3-p4."""
        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
        
        d1 = cross(p3, p4, p1)
        d2 = cross(p3, p4, p2)
        d3 = cross(p1, p2, p3)
        d4 = cross(p1, p2, p4)
        
        if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
           ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
            return True
        
        # Collinear cases
        if d1 == 0 and point_on_seg(p3, p4, p1):
            return True
        if d2 == 0 and point_on_seg(p3, p4, p2):
            return True
        if d3 == 0 and point_on_seg(p1, p2, p3):
            return True
        if d4 == 0 and point_on_seg(p1, p2, p4):
            return True
        
        return False
    
    def point_on_seg(a, b, p):
        """Check if point p is on segment a-b."""
        if cross(a, b, p) != 0:
            return False
        if min(a[0], b[0]) - eps > p[0] or max(a[0], b[0]) + eps < p[0]:
            return False
        if min(a[1], b[1]) - eps > p[1] or max(a[1], b[1]) + eps < p[1]:
            return False
        return True
    
    for i in range(6):
        for j in range(6):
            if seg_intersect(v1[i], v1[(i+1)%6], v2[j], v2[(j+1)%6]):
                return False
    
    return True

def min_enclosing_hex_side(hexagons):
    """Find the minimum side length of a regular hexagon that can enclose all given hexagons.
    Returns (min_side, center, angle) or just min_side for simplicity."""
    # This is complex. Let's use a simpler lower bound: 
    # The minimum enclosing hexagon side is at least half the maximum distance
    # between any two points in the union, projected appropriately.
    
    # Get all vertices
    all_pts = []
    for cx, cy, a in hexagons:
        for px, py in hex_to_point(cx, cy, a):
            all_pts.append((px, py))
    
    if not all_pts:
        return 0.0
    
    # Compute diameter
    max_d = 0
    for i in range(len(all_pts)):
        for j in range(i+1, len(all_pts)):
            dx = all_pts[i][0] - all_pts[j][0]
            dy = all_pts[i][1] - all_pts[j][1]
            d = math.sqrt(dx*dx + dy*dy)
            if d > max_d:
                max_d = d
    
    # For a regular hexagon with side s, the maximum distance between any two points
    # is 2s (opposite vertices). So s >= max_d / 2.
    # But this is a lower bound, not tight.
    
    # Better: the width of a hexagon in any direction is 2s*cos(angle_diff) for some angle.
    # The minimum enclosing hexagon problem is non-trivial.
    
    # For now, use the lower bound: s >= max_d / 2
    return max_d / 2

def construct():
    """Construct a packing of 11 unit hexagons inside a minimal outer hexagon."""
    
    # Strategy: Try to pack 11 unit hexagons in a honeycomb-like arrangement.
    # A hexagonal cluster of 19 hexagons (3-4-5-4-3) has outer hexagon side 3.
    # For 11 hexagons, we can try a smaller cluster.
    
    # Let's try to find a good configuration using a simple heuristic:
    # Place hexagons on a triangular lattice with spacing sqrt(3) (center-to-center for touching hexagons).
    
    # For unit hexagons, the center-to-center distance when touching is 2 * (sqrt(3)/2) = sqrt(3) ≈ 1.732.
    # Wait, for regular hexagons of side 1, the distance between centers of adjacent touching hexagons is 2 * apothem = 2 * (sqrt(3)/2) = sqrt(3).
    
    # Let's define the triangular lattice positions.
    # In a hexagonal packing, the centers form a triangular lattice.
    # The six neighbors of a center at origin are at:
    # (sqrt(3), 0), (sqrt(3)/2, 3/2), (-sqrt(3)/2, 3/2), (-sqrt(3), 0), (-sqrt(3)/2, -3/2), (sqrt(3)/2, -3/2)
    
    # Let's try a compact arrangement: a hexagonal number pattern.
    # 1 + 6 + 12 + ... For 11 hexagons: 1 + 6 = 7, then add 4 more.
    
    # Actually, let's just try a few known good configurations.
    
    # Configuration 1: A 3x3 grid of hexagons in hexagonal arrangement, minus corners.
    # Centers in axial coordinates (q, r) where actual position is:
    # x = sqrt(3) * (q + r/2)
    # y = 1.5 * r
    
    # Let's enumerate small hexagonal clusters and compute the enclosing hexagon.
    
    sqrt3 = math.sqrt(3)
    
    def axial_to_xy(q, r):
        x = sqrt3 * (q + r / 2)
        y = 1.5 * r
        return x, y
    
    # Try various clusters of 11 cells
    # We'll search over small bounding boxes in axial coordinates
    
    best_side = float('inf')
    best_config = None
    best_center = (0, 0)
    best_angle = 0
    
    # Generate candidate clusters: all sets of 11 cells within a small region
    # This is expensive, so let's try specific patterns
    
    # Pattern: centered hexagonal cluster
    # Radius 2 hexagonal cluster has 1 + 6 + 12 = 19 cells.
    # Let's take a subset of 11.
    
    # For simplicity, let's try a rectangular-ish block in hexagonal coordinates.
    
    # Let's try: rows of 3, 4, 3, 1 = 11? No.
    # Rows: 2, 3, 3, 2, 1 = 11
    
    # Let's just try a few hardcoded configurations and evaluate them.
    
    # Configuration A: A 4-3-4 arrangement (rows in hexagonal lattice)
    # Row r=-1: q in {-1, 0, 1, 2} -> 4 cells
    # Row r=0: q in {-1, 0, 1} -> 3 cells  
    # Row r=1: q in {-1, 0, 1, 2} -> 4 cells
    # Total: 11
    
    configs = []
    
    # Config 1: 4-3-4
    c1 = []
    for r in [-1, 0, 1]:
        if r == 0:
            qs = [-1, 0, 1]
        else:
            qs = [-1, 0, 1, 2]
        for q in qs:
            x, y = axial_to_xy(q, r)
            c1.append((x, y, 0.0))
    configs.append(c1)
    
    # Config 2: 3-5-3
    c2 = []
    for r in [-1, 0, 1]:
        if r == 0:
            qs = [-2, -1, 0, 1, 2]
        else:
            qs = [-1, 0, 1]
        for q in qs:
            x, y = axial_to_xy(q, r)
            c2.append((x, y, 0.0))
    configs.append(c2)
    
    # Config 3: 2-3-4-2
    c3 = []
    for r, n in [(-2, 2), (-1, 3), (0, 4), (1, 2)]:
        start = -n // 2
        for i in range(n):
            q = start + i
            x, y = axial_to_xy(q, r)
            c3.append((x, y, 0.0))
    configs.append(c3)
    
    # Config 4: 1-3-5-2
    c4 = []
    for r, n in [(-2, 1), (-1, 3), (0, 5), (1, 2)]:
        start = -n // 2
        for i in range(n):
            q = start + i
            x, y = axial_to_xy(q, r)
            c4.append((x, y, 0.0))
    configs.append(c4)
    
    # For each config, compute the minimum enclosing hexagon side
    # and verify no overlaps (they shouldn't overlap on the lattice)
    
    for config in configs:
        if len(config) != 11:
            continue
        
        # Check for overlaps (should be fine on lattice)
        valid = True
        for i in range(len(config)):
            for j in range(i+1, len(config)):
                if not hexes_disjoint(config[i], config[j]):
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            continue
        
        # Compute bounding box to estimate center
        xs = [p[0] for p in config]
        ys = [p[1] for p in config]
        cx = (min(xs) + max(xs)) / 2
        cy = (min(ys) + max(ys)) / 2
        
        # For a hexagonal cluster on the lattice, the enclosing hexagon
        # can be estimated. The outer hexagon should be concentric with the cluster.
        
        # Compute the minimum enclosing hexagon more carefully.
        # For a hexagonal cluster, the enclosing hexagon is also hexagonal.
        
        # Let's compute the maximum extent in the 6 hexagonal directions.
        # The 6 directions are at angles 0, 60, 120, 180, 240, 300 degrees.
        
        # Actually, for a regular hexagon with side s centered at (cx, cy) with angle a,
        # a point (px, py) is inside if and only if for all 6 directions,
        # the projection is within s * cos(30°) = s * sqrt(3)/2 from the center.
        
        # Let's use the lower bound from diameter.
        side_lb = min_enclosing_hex_side(config)
        
        # For a hexagonal cluster, the enclosing hexagon side is approximately
        # the maximum distance from center to any vertex, divided by cos(30°) * 2... 
        # This is getting complicated. Let's just use the lower bound.
        
        if side_lb < best_side:
            best_side = side_lb
            best_config = config
            best_center = (cx, cy)
            best_angle = 0.0
    
    if best_config is None:
        # Fallback: linear arrangement
        inner = [[sqrt3 * k, 0.0, 0.0] for k in range(-(N//2), N - N//2)]
        return inner, (0.0, 0.0), 20.0, 0.0
    
    # Use the best config found
    # The outer hexagon side should be at least best_side, but we need a valid enclosing hexagon.
    # Let's use a slightly larger side to be safe.
    
    outer_side = best_side * 1.01  # Small margin
    
    return best_config, best_center, outer_side, best_angle

# EVOLVE-BLOCK-START
def construct():
    return _construct_impl()

def _construct_impl():
    """Actual implementation."""
    N = 11
    sqrt3 = math.sqrt(3)
    
    def axial_to_xy(q, r):
        x = sqrt3 * (q + r / 2)
        y = 1.5 * r
        return x, y
    
    # Generate several candidate configurations
    configs = []
    
    # Config 1: 4-3-4
    c1 = []
    for r in [-1, 0, 1]:
        if r == 0:
            qs = [-1, 0, 1]
        else:
            qs = [-1, 0, 1, 2]
        for q in qs:
            x, y = axial_to_xy(q, r)
            c1.append((x, y, 0.0))
    configs.append(c1)
    
    # Config 2: 3-5-3
    c2 = []
    for r in [-1, 0, 1]:
        if r == 0:
            qs = [-2, -1, 0, 1, 2]
        else:
            qs = [-1, 0, 1]
        for q in qs:
            x, y = axial_to_xy(q, r)
            c2.append((x, y, 0.0))
    configs.append(c2)
    
    # Config 3: 2-3-4-2
    c3 = []
    for r, n in [(-2, 2), (-1, 3), (0, 4), (1, 2)]:
        start = -n // 2
        for i in range(n):
            q = start + i
            x, y = axial_to_xy(q, r)
            c3.append((x, y, 0.0))
    configs.append(c3)
    
    # Config 4: 1-4-5-1
    c4 = []
    for r, n in [(-2, 1), (-1, 4), (0, 5), (1, 1)]:
        start = -n // 2
        for i in range(n):
            q = start + i
            x, y = axial_to_xy(q, r)
            c4.append((x, y, 0.0))
    configs.append(c4)
    
    # Config 5: 2-4-4-1
    c5 = []
    for r, n in [(-2, 2), (-1, 4), (0, 4), (1, 1)]:
        start = -n // 2
        for i in range(n):
            q = start + i
            x, y = axial_to_xy(q, r)
            c5.append((x, y, 0.0))
    configs.append(c5)
    
    def compute_enclosing_hex_side(hexagons, center):
        """Compute the minimum side length of a regular hexagon centered at 'center' 
        with 0 rotation that contains all hexagons."""
        max_proj = 0
        for cx, cy, a in hexagons:
            for px, py in hex_to_point(cx, cy, a):
                # Project onto 6 hexagonal axes
                for k in range(3):
                    angle = k * math.pi / 3
                    ca, sa = math.cos(angle), math.sin(angle)
                    # Distance from center in this direction
                    dx = px - center[0]
                    dy = py - center[1]
                    proj = abs(dx * ca + dy * sa)
                    if proj > max_proj:
                        max_proj = proj
        # For a regular hexagon with side s, the apothem is s * sqrt(3)/2
        # So s = max_proj / (sqrt(3)/2) = 2 * max_proj / sqrt(3)
        return 2 * max_proj / sqrt3
    
    best_side = float('inf')
    best_config = None
    best_center = (0, 0)
    
    for config in configs:
        if len(config) != 11:
            continue
        
        xs = [p[0] for p in config]
        ys = [p[1] for p in config]
        cx = (min(xs) + max(xs)) / 2
        cy = (min(ys) + max(ys)) / 2
        
        side = compute_enclosing_hex_side(config, (cx, cy))
        
        if side < best_side:
            best_side = side
            best_config = config
            best_center = (cx, cy)
    
    if best_config is None:
        inner = [[sqrt3 * k, 0.0, 0.0] for k in range(-(N//2), N - N//2)]
        return inner, (0.0, 0.0), 20.0, 0.0
    
    return best_config, best_center, best_side, 0.0
# EVOLVE-BLOCK-END
