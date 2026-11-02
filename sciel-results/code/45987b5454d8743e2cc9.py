"""Optimized hexagon packing using gradient-based optimization."""
import numpy as np
from scipy.optimize import minimize
from itertools import product
import math

N = 11

def hex_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon centered at (cx, cy) with given side length and rotation."""
    angle = math.radians(angle_deg)
    vertices = []
    for i in range(6):
        theta = angle + i * math.pi / 3
        x = cx + side * math.cos(theta)
        y = cy + side * math.sin(theta)
        vertices.append((x, y))
    return vertices

def point_in_hexagon(px, py, cx, cy, side, angle_deg):
    """Check if point (px, py) is inside or on boundary of hexagon. Returns True if inside/on boundary."""
    # Transform point to hexagon's local coordinate system
    angle = math.radians(angle_deg)
    dx = px - cx
    dy = py - cy
    # Rotate by -angle
    cos_a = math.cos(-angle)
    sin_a = math.sin(-angle)
    lx = dx * cos_a - dy * sin_a
    ly = dx * sin_a + dy * cos_a
    
    # For a regular hexagon with side length s centered at origin with flat top:
    # The hexagon can be described by |x| <= s and |y| <= s*sqrt(3)/2 and |x| + |y|/sqrt(3) <= s
    # Actually, let's use the standard form. A regular hexagon with vertices at angles 0,60,...
    # has the property that for each of the 6 sides, the point must be on the correct side.
    
    # Simpler: compute distance from center in the hexagon's local frame
    # A regular hexagon with circumradius R = side has the equation:
    # max over i of (lx*cos(60i) + ly*sin(60i)) <= R
    # Wait, that's for the circumscribed circle. Let me think more carefully.
    
    # For a regular hexagon with side length s, the apothem is s*sqrt(3)/2.
    # In local coordinates (with one vertex pointing right), the hexagon is defined by:
    # For each edge, the dot product of (lx,ly) with the outward normal <= apothem
    # The outward normals are at angles 0, 60, 120, 180, 240, 300 degrees (for a hexagon with flat top)
    # Actually for a hexagon with a vertex at angle 0, the edges have normals at 30, 90, 150, 210, 270, 330
    
    # Let me use a different approach: check if the point satisfies all 6 half-plane constraints
    R = side
    apothem = R * math.sqrt(3) / 2
    
    # Normals to the 6 sides, pointing outward. For a hexagon with vertex at angle 0:
    # The sides are perpendicular to directions 30, 90, 150, 210, 270, 330 degrees
    for i in range(6):
        normal_angle = math.radians(30 + i * 60)
        nx = math.cos(normal_angle)
        ny = math.sin(normal_angle)
        if lx * nx + ly * ny > apothem + 1e-9:
            return False
    return True

def hex_hex_overlap(d1, d2, s1, s2, a1, a2):
    """Check if two hexagons overlap. d1,d2 are centers, s1,s2 side lengths, a1,a2 angles in degrees."""
    # For two regular hexagons, they don't overlap iff the distance between centers
    # is at least the minimum distance needed. This is complex, so let's use a simpler check:
    # Check if any vertex of one is inside the other, or vice versa.
    v1 = hex_vertices(d1[0], d1[1], s1, a1)
    v2 = hex_vertices(d2[0], d2[1], s2, a2)
    
    for vx, vy in v1:
        if point_in_hexagon(vx, vy, d2[0], d2[1], s2, a2):
            return True
    for vx, vy in v2:
        if point_in_hexagon(vx, vy, d1[0], d1[1], s1, a1):
            return True
    return False

def outer_constraint(x, y, side, angle_deg):
    """Check if point (x,y) is inside the outer hexagon. Returns True if inside/on boundary."""
    angle = math.radians(angle_deg)
    dx = x - 0.0  # outer center is at (0,0) for simplicity
    dy = y - 0.0
    cos_a = math.cos(-angle)
    sin_a = math.sin(-angle)
    lx = dx * cos_a - dy * sin_a
    ly = dx * sin_a + dy * cos_a
    
    apothem = side * math.sqrt(3) / 2
    for i in range(6):
        normal_angle = math.radians(30 + i * 60)
        nx = math.cos(normal_angle)
        ny = math.sin(normal_angle)
        if lx * nx + ly * ny > apothem + 1e-9:
            return False
    return True

def compute_score(inner, outer_side):
    """Compute a score: 0 if valid, negative if invalid. We want to maximize minimum clearance."""
    # Check all inner hexagons are inside outer
    for (cx, cy, angle) in inner:
        verts = hex_vertices(cx, cy, 1.0, angle)
        for (vx, vy) in verts:
            if not outer_constraint(vx, vy, outer_side, 0.0):
                return -1.0
    
    # Check no two inner hexagons overlap
    for i in range(len(inner)):
        for j in range(i + 1, len(inner)):
            if hex_hex_overlap((inner[i][0], inner[i][1]), (inner[j][0], inner[j][1]), 1.0, 1.0, inner[i][2], inner[j][2]):
                return -1.0
    
    return 0.0

def construct():
    """Construct the packing configuration."""
    # Start with a reasonable initial guess
    # Place hexagons in a honeycomb-like pattern
    inner = []
    for k in range(-N//2, N - N//2):
        x = 4.0 * k
        y = 0.0
        angle = 0.0
        inner.append([x, y, angle])
    
    # Try to optimize by adjusting positions
    # Use a simple local search to improve the outer side length
    
    def get_outer_side(inner_config):
        """Compute the minimum outer side length that contains all inner hexagons."""
        # The outer hexagon must contain all vertices of all inner hexagons
        # For a hexagon centered at origin with side s, a point (x,y) is inside iff
        # for all 6 directions, the projection is within the apothem
        # We need to find minimum s such that all points are inside
        
        all_points = []
        for (cx, cy, angle) in inner_config:
            verts = hex_vertices(cx, cy, 1.0, angle)
            all_points.extend(verts)
        
        if not all_points:
            return 0.0
        
        # For each point, compute the required side length for the outer hexagon
        # Point (px, py) is inside hexagon with side s centered at origin iff
        # max over 6 normals of (px*nx + py*ny) <= s*sqrt(3)/2
        # So s >= 2/sqrt(3) * max over points and normals of (px*nx + py*ny)
        
        max_proj = 0.0
        for (px, py) in all_points:
            for i in range(6):
                normal_angle = math.radians(30 + i * 60)
                nx = math.cos(normal_angle)
                ny = math.sin(normal_angle)
                proj = px * nx + py * ny
                if proj > max_proj:
                    max_proj = proj
        
        return 2.0 / math.sqrt(3) * max_proj
    
    # Try several initial configurations and pick the best
    best_inner = inner
    best_side = get_outer_side(inner)
    
    # Try some variations
    for offset_y in [0.0, 2.0, -2.0, 3.0, -3.0]:
        for offset_x in [0.0, 1.0, -1.0]:
            test_inner = [[x + offset_x, y + offset_y, 0.0] for (x, y, a) in inner]
            side = get_outer_side(test_inner)
            if side < best_side:
                best_side = side
                best_inner = test_inner
    
    # Try rotated configurations
    for rot in [0.0, 30.0, 60.0, 90.0, 120.0, 150.0]:
        test_inner = [[x, y, rot] for (x, y, a) in inner]
        side = get_outer_side(test_inner)
        if side < best_side:
            best_side = side
            best_inner = test_inner
    
    # Try honeycomb pattern
    honeycomb = []
    for row in range(-2, 3):
        for col in range(-2, 3):
            x = col * 3.0 + (0.5 if row % 2 != 0 else 0.0) * 3.0
            y = row * 2.598  # sqrt(3) * 1.5
            honeycomb.append([x, y, 0.0])
    side = get_outer_side(honeycomb)
    if side < best_side:
        best_side = side
        best_inner = honeycomb
    
    return best_inner, (0.0, 0.0), best_side, 0.0
