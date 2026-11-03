import numpy as np
import math
import time

N = 11

def hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon centered at (cx,cy) with given side and rotation."""
    angle_rad = math.radians(angle_deg)
    verts = []
    for k in range(6):
        a = angle_rad + k * math.pi / 3
        verts.append((cx + side * math.cos(a), cy + side * math.sin(a)))
    return verts

def point_in_hex(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside hexagon centered at (cx,cy)."""
    # Transform to hexagon's local frame
    angle_rad = math.radians(angle_deg)
    dx = px - cx
    dy = py - cy
    # Rotate by -angle
    ca = math.cos(-angle_rad)
    sa = math.sin(-angle_rad)
    lx = ca * dx - sa * dy
    ly = sa * dx + ca * dy
    # In local frame, hexagon with flat top has vertices at angles 0,60,...
    # Actually with our vertex generation, vertex 0 is at angle 0 (pointing right)
    # The hexagon has "pointy top" in local frame if angle=0 means vertex at 0 deg
    # For a hexagon with vertices at 0,60,120,180,240,300 degrees:
    # The width in x direction is 2*side, in y direction is sqrt(3)*side
    # Check: |lx| <= side AND |ly| <= sqrt(3)/2 * side for the "box" part
    # More precisely, use the half-plane tests
    # The 6 edges are at distances side*cos(30) = side*sqrt(3)/2 from center
    # Edge normals are at angles 30, 90, 150, 210, 270, 330
    for k in range(6):
        a = math.radians(30 + k * 60)
        nx = math.cos(a)
        ny = math.sin(a)
        d = lx * nx + ly * ny
        if d > side * math.sqrt(3) / 2 + 1e-9:
            return False
    return True

def hexagon_contains_hexagon(cx1, cy1, s1, a1, cx2, cy2, s2, a2, tol=1e-9):
    """Check if hexagon 2 is entirely inside hexagon 1."""
    # Check all vertices of hexagon 2 are inside hexagon 1
    verts2 = hex_vertices(cx2, cy2, s2, a2)
    for (vx, vy) in verts2:
        if not point_in_hex(vx, vy, cx1, cy1, s1, a1):
            return False
    return True

def min_enclosing_hexagon(points, max_angle_steps=360, max_side_steps=200):
    """Find minimum side length regular hexagon containing all points.
    Try various rotations and compute the minimum side for each."""
    best_side = float('inf')
    best_angle = 0.0
    
    # For a given rotation angle, the minimum enclosing hexagon side is determined
    # by projecting points onto the 6 normal directions
    for angle_deg in range(0, 30, 1):  # Hexagon symmetry is 60 deg, so 0-30 is enough
        angle_rad = math.radians(angle_deg)
        # The 6 edge normals are at angles 30, 90, 150, 210, 270, 330 + angle_deg
        max_proj = 0
        for k in range(6):
            a = math.radians(30 + k * 60) + angle_rad
            nx = math.cos(a)
            ny = math.sin(a)
            for (px, py) in points:
                proj = px * nx + py * ny
                max_proj = max(max_proj, proj)
        # The side length s satisfies s * sqrt(3)/2 = max_proj (distance from center to edge)
        # But we also need to center the hexagon optimally
        # For each direction, we need both +normal and -normal constraints
        # Let's do it properly: for each of 6 directions, compute min and max projection
        side_for_angle = 0
        for k in range(6):
            a = math.radians(30 + k * 60) + angle_rad
            nx = math.cos(a)
            ny = math.sin(a)
            projs = [px * nx + py * ny for (px, py) in points]
            spread = max(projs) - min(projs)
            # The hexagon must span this projection range, and the hexagon's extent
            # in direction (nx,ny) is s * sqrt(3) (from one edge to opposite edge)
            # Actually the distance between parallel edges is s * sqrt(3)
            # So we need s * sqrt(3) >= spread, i.e., s >= spread / sqrt(3)
            side_for_angle = max(side_for_angle, spread / math.sqrt(3))
        
        if side_for_angle < best_side:
            best_side = side_for_angle
            best_angle = angle_deg
    
    return best_side, best_angle

def get_all_hexagon_vertices(inner_hexes):
    """Get all vertices of all inner hexagons."""
    all_verts = []
    for (cx, cy, angle) in inner_hexes:
        verts = hex_vertices(cx, cy, 1.0, angle)
        all_verts.extend(verts)
    return all_verts

def find_best_packing(deadline):
    """Try various packings and find the one with smallest enclosing hexagon."""
    best_result = None
    best_side = float('inf')
    
    # Generate candidate packings
    candidates = []
    
    # Packing 1: Honeycomb cluster - central + 6 around + 4 more
    # Use axial coordinates (q, r) -> Cartesian: x = q + r/2, y = r * sqrt(3)/2
    # Scale by 2 for unit hexagons (center-to-center distance = 2)
    sqrt3 = math.sqrt(3)
    
    def axial_to_cartesian(q, r):
        x = 2.0 * (q + r / 2.0)
        y = 2.0 * r * sqrt3 / 2.0
        return (x, y)
    
    # Various 11-hexagon shapes in axial coordinates
    # Shape A: H2 (7) + 4 extending right
    shape_a = [(0,0), (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1),
               (2,0), (2,-1), (2,-2), (1,-2)]
    
    # Shape B: More symmetric - rows of 3,4,3,1
    shape_b = [(-1,-1),(0,-1),(1,-1),
               (-1,0),(0,0),(1,0),(2,0),
               (-1,1),(0,1),(1,1),
               (0,2)]
    
    # Shape C: 5-3-3 arrangement
    shape_c = [(-2,0),(-1,0),(0,0),(1,0),(2,0),
               (-1,1),(0,1),(1,1),
               (-1,-1),(0,-1),(1,-1)]
    
    # Shape D: Compact diamond-like
    shape_d = [(0,0),(1,0),(1,-1),(0,-1),(-1,0),(-1,1),(0,1),
               (2,0),(2,-1),(-2,0),(-2,1)]
    
    # Shape E: Row-based 4-3-4
    shape_e = [(-1,-1),(0,-1),(1,-1),(2,-1),
               (-1,0),(0,0),(1,0),
               (-1,1),(0,1),(1,1),(2,1)]
    
    # Shape F: 3-5-3
    shape_f = [(-1,-1),(0,-1),(1,-1),
               (-2,0),(-1,0),(0,0),(1,0),(2,0),
               (-1,1),(0,1),(1,1)]
    
    for shape in [shape_a, shape_b, shape_c, shape_d, shape_e, shape_f]:
        if len(shape) != N:
            continue
        if time.time() > deadline:
            break
        
        # Convert to Cartesian
        inner = []
        for (q, r) in shape:
            x, y = axial_to_cartesian(q, r)
            inner.append([x, y, 0.0])
        
        # Get all vertices
        all_verts = get_all_hexagon_vertices(inner)
        
        # Find minimum enclosing hexagon
        side, angle = min_enclosing_hexagon(all_verts)
        
        if side < best_side:
            best_side = side
            # Compute center of bounding box
            xs = [v[0] for v in all_verts]
            ys = [v[1] for v in all_verts]
            cx = (min(xs) + max(xs)) / 2.0
            cy = (min(ys) + max(ys)) / 2.0
            # Adjust inner hexes relative to this center
            inner_centered = [[x - cx, y - cy, a] for (x, y, a) in inner]
            best_result = (inner_centered, (0.0, 0.0), side, angle)
    
    return best_result, best_side

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    result, best_side = find_best_packing(deadline)
    
    if result is None:
        # Fallback: simple linear arrangement
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    inner, center, side, angle = result
    return inner, center, side, angle
# EVOLVE-BLOCK-END
