"""Improved program: pack unit hexagons in a regular hexagon, minimize outer side."""
import time
import math
import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull

N = 11
START_TIME = time.time()
TIME_LIMIT = 130.0  # safety margin before 150s kill

def time_left():
    return TIME_LIMIT - (time.time() - START_TIME)

def hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon centered at (cx, cy) with side length 'side' rotated by angle_deg."""
    angle = math.radians(angle_deg)
    vertices = []
    for i in range(6):
        theta = angle + i * math.pi / 3.0
        x = cx + side * math.cos(theta)
        y = cy + side * math.sin(theta)
        vertices.append((x, y))
    return np.array(vertices)

def point_in_hexagon(p, cx, cy, side, angle_deg):
    """Check if point p is inside or on the boundary of the hexagon."""
    # Transform point to hexagon's local coordinate system
    px, py = p
    dx = px - cx
    dy = py - cy
    angle = math.radians(angle_deg)
    cos_a = math.cos(-angle)
    sin_a = math.sin(-angle)
    lx = dx * cos_a - dy * sin_a
    ly = dx * sin_a + dy * cos_a
    
    # In local coords, hexagon with side s has vertices at angles 0, 60, 120, 180, 240, 300
    # The hexagon is defined by 6 half-planes. For a flat-top hexagon (vertex at angle 0),
    # the constraints are: |lx| <= s, |ly| <= s * sqrt(3)/2, and combined constraints.
    # Actually, for a regular hexagon with circumradius R = side, the 6 supporting lines are at distance R * cos(30) = R * sqrt(3)/2 from center.
    # Let me use the half-plane representation.
    # The 6 outward normals are at angles 30, 90, 150, 210, 270, 330 degrees for a pointy-top, or 0, 60, ... for flat-top.
    # Let's use the fact that a regular hexagon with circumradius R (side = R) can be described as:
    # max(|lx|, |ly|/sqrt(3), |(lx+ly)/sqrt(3)|) <= R  ... this is getting complicated.
    
    # Simpler: use the 6 half-plane inequalities.
    # For a hexagon with vertices at angles 0, 60, 120, 180, 240, 300 (flat-top in local coords),
    # the edges are between consecutive vertices. The outward normal to each edge.
    # Edge between vertex i and i+1.
    # Let's just check all 6 half-planes.
    
    # The 6 supporting lines: for k in 0..5, the line is at distance d = side * cos(pi/6) = side * sqrt(3)/2 from center,
    # with normal direction at angle k*pi/3 + pi/6.
    
    d = side * math.sqrt(3) / 2.0
    for k in range(6):
        normal_angle = k * math.pi / 3.0 + math.pi / 6.0
        nx = math.cos(normal_angle)
        ny = math.sin(normal_angle)
        # The constraint is nx * lx + ny * ly <= d
        if nx * lx + ny * ly > d + 1e-9:
            return False
    return True

def all_vertices_in_hexagon(vertices, cx, cy, side, angle_deg):
    """Check if all vertices of a hexagon are inside the outer hexagon."""
    for v in vertices:
        if not point_in_hexagon(v, cx, cy, side, angle_deg):
            return False
    return True

def check_configuration(inner, center, side, angle_deg):
    """Check if the configuration is valid: all inner hexagons are inside the outer hexagon and disjoint."""
    cx, cy = center
    outer_verts = hex_vertices(cx, cy, side, angle_deg)
    
    inner_verts_list = []
    for (ix, iy, iangle) in inner:
        iv = hex_vertices(ix, iy, 1.0, iangle)
        inner_verts_list.append(iv)
        if not all_vertices_in_hexagon(iv, cx, cy, side, angle_deg):
            return False
    
    # Check disjointness: no two inner hexagons overlap
    # For convex polygons, they are disjoint if no edge of one intersects an edge of the other
    # and neither contains a vertex of the other.
    # Simpler check: for each pair, check if any vertex of one is strictly inside the other,
    # or if any edges intersect.
    
    def point_in_polygon(p, poly):
        """Ray casting to check if point p is inside polygon poly."""
        x, y = p
        n = len(poly)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = poly[i]
            xj, yj = poly[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside
    
    def segments_intersect(p1, p2, p3, p4):
        """Check if segment p1-p2 intersects segment p3-p4."""
        def ccw(a, b, c):
            return (c[1] - a[1]) * (b[0] - a[0]) - (b[1] - a[1]) * (c[0] - a[0])
        
        d1 = ccw(p3, p4, p1)
        d2 = ccw(p3, p4, p2)
        d3 = ccw(p1, p2, p3)
        d4 = ccw(p1, p2, p4)
        
        if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
           ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
            return True
        return False
    
    for i in range(len(inner_verts_list)):
        vi = inner_verts_list[i]
        for j in range(i + 1, len(inner_verts_list)):
            vj = inner_verts_list[j]
            # Check if any vertex of vi is inside vj or vice versa
            for v in vi:
                if point_in_polygon(v, vj):
                    return False
            for v in vj:
                if point_in_polygon(v, vi):
                    return False
            # Check edge intersections
            for k in range(6):
                e1 = (vi[k], vi[(k+1)%6])
                for l in range(6):
                    e2 = (vj[l], vj[(l+1)%6])
                    if segments_intersect(e1[0], e1[1], e2[0], e2[1]):
                        return False
    return True

def generate_hexagonal_lattice_config(spacing, offset_x=0, offset_y=0, angle=0):
    """Generate a hexagonal lattice of unit hexagons."""
    inner = []
    # Hexagonal lattice: rows offset
    # For unit hexagons, the distance between centers in a hex lattice:
    # Horizontal: 3 * side = 3 (for flat-top)
    # Vertical: 3 * sqrt(3)/2 * side
    # Let's use a simple grid first, then optimize.
    
    # Actually, for non-overlapping unit hexagons, the minimum center-to-center distance
    # depends on orientation. If all are axis-aligned, the minimum distance is 3 (along flat directions)
    # or 3*sqrt(3)/2 ≈ 2.598 (along pointy directions).
    
    # Let's generate a hexagonal packing.
    # Row 0: centers at x = 0, 3, 6, ...
    # Row 1: centers at x = 1.5, 4.5, ... y = 3*sqrt(3)/2
    # etc.
    
    sqrt3 = math.sqrt(3)
    dx = 3.0  # horizontal spacing for flat-top
    dy = 3.0 * sqrt3 / 2.0  # vertical spacing
    
    n_rows = 7
    n_cols = 7
    for row in range(-n_rows//2, n_rows//2 + 1):
        for col in range(-n_cols//2, n_cols//2 + 1):
            x = col * dx + (row % 2) * dx/2 + offset_x
            y = row * dy + offset_y
            inner.append([x, y, angle])
    
    # Filter to keep exactly N hexagons closest to origin
    inner.sort(key=lambda p: p[0]**2 + p[1]**2)
    inner = inner[:N]
    return inner

def try_configuration(inner, center, side, angle_deg):
    """Try a configuration and return (valid, side) if valid."""
    if check_configuration(inner, center, side, angle_deg):
        return True, side
    return False, None

def optimize_with_lp(inner, center, angle_deg, initial_side):
    """Use LP to find minimum side length for a fixed configuration."""
    # This is complex because the containment constraints are nonlinear in side.
    # For a fixed outer hexagon center and angle, the constraint that each inner vertex
    # is inside the outer hexagon gives linear constraints on side.
    # For each inner vertex v, and for each of the 6 half-planes of the outer hexagon:
    # n_k · (v - center) <= side * sqrt(3)/2
    # So side >= 2 * n_k · (v - center) / sqrt(3)
    # The minimum side is the max over all such constraints.
    
    cx, cy = center
    sqrt3 = math.sqrt(3)
    d_factor = sqrt3 / 2.0  # distance from center to each edge
    
    max_side = 0.0
    for (ix, iy, iangle) in inner:
        verts = hex_vertices(ix, iy, 1.0, iangle)
        for v in verts:
            dx = v[0] - cx
            dy = v[1] - cy
            # Transform to outer hexagon local coords
            angle = math.radians(angle_deg)
            cos_a = math.cos(-angle)
            sin_a = math.sin(-angle)
            lx = dx * cos_a - dy * sin_a
            ly = dx * sin_a + dy * cos_a
            
            for k in range(6):
                normal_angle = k * math.pi / 3.0 + math.pi / 6.0
                nx = math.cos(normal_angle)
                ny = math.sin(normal_angle)
                val = nx * lx + ny * ly
                if val > 0:
                    required_side = val / d_factor
                    if required_side > max_side:
                        max_side = required_side
    
    # Check if this side works (disjointness is already assumed)
    if max_side < initial_side * 0.5:
        max_side = initial_side * 0.5
    
    return max_side

def main_search():
    """Main search routine."""
    best_side = float('inf')
    best_config = None
    
    # Strategy: Try hexagonal lattice packings with different orientations and offsets
    # Then try to optimize the outer hexagon.
    
    sqrt3 = math.sqrt(3)
    
    # Try different outer hexagon angles
    outer_angles = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165]
    
    for outer_angle in outer_angles:
        if time_left() < 5.0:
            break
        
        # Try hexagonal lattice with different inner hexagon angles
        inner_angles = [0, 30, 60, 90, 120, 150]
        
        for inner_angle in inner_angles:
            if time_left() < 3.0:
                break
            
            # Generate hexagonal lattice
            dx = 3.0
            dy = 3.0 * sqrt3 / 2.0
            
            # Try different offsets
            for ox in [0.0, 0.5, 1.0, 1.5, 2.0]:
                for oy in [0.0, 0.5, 1.0, 1.5, 2.0]:
                    if time_left() < 1.0:
                        break
                    
                    inner = []
                    n_rows = 5
                    n_cols = 5
                    for row in range(-n_rows//2, n_rows//2 + 1):
                        for col in range(-n_cols//2, n_cols//2 + 1):
                            x = col * dx + (row % 2) * dx/2 + ox
                            y = row * dy + oy
                            inner.append([x, y, inner_angle])
                    
                    # Keep N closest to origin
                    inner.sort(key=lambda p: p[0]**2 + p[1]**2)
                    inner = inner[:N]
                    
                    # Estimate required side
                    center = (0.0, 0.0)
                    # Compute bounding box
                    all_pts = []
                    for (ix, iy, ia) in inner:
                        verts = hex_vertices(ix, iy, 1.0, ia)
                        all_pts.extend(verts.tolist())
                    all_pts = np.array(all_pts)
                    bbox_diag = np.max(np.linalg.norm(all_pts, axis=1))
                    est_side = bbox_diag / (sqrt3/2) + 0.1
                    
                    # Use LP to get exact minimum side for this config
                    min_side = optimize_with_lp(inner, center, outer_angle, est_side)
                    
                    # Verify
                    valid, _ = try_configuration(inner, center, min_side, outer_angle)
                    if valid and min_side < best_side:
                        best_side = min_side
                        best_config = (inner, center, min_side, outer_angle)
    
    # Also try a simple linear arrangement
    for outer_angle in [0, 30, 60]:
        if time_left() < 3.0:
            break
        for inner_angle in [0, 30, 60]:
            if time_left() < 1.0:
                break
            # Linear arrangement along x-axis
            inner = [[3.0 * k, 0.0, inner_angle] for k in range(-(N//2), N - N//2)]
            center = (0.0, 0.0)
            min_side = optimize_with_lp(inner, center, outer_angle, 20.0)
            valid, _ = try_configuration(inner, center, min_side, outer_angle)
            if valid and min_side < best_side:
                best_side = min_side
                best_config = (inner, center, min_side, outer_angle)
    
    if best_config is not None:
        return best_config
    else:
        # Fallback: return a valid but large configuration
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

# EVOLVE-BLOCK-START
def construct():
    """Construct the packing configuration."""
    result = main_search()
    return result
# EVOLVE-BLOCK-END
