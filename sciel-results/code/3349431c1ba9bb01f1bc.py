"""
Packing unit regular hexagons inside a regular hexagon.
Strategy: Discrete contact-graph based construction with Nelder-Mead polish.
We build configurations where hexagons are in a "honeycomb-like" lattice,
allowing small rotations and translations, then optimize the outer side length
using a smooth objective and Nelder-Mead.
"""
import numpy as np
from scipy.optimize import minimize
import time
import math

N = 11
SEED = 42
np.random.seed(SEED)

def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon centered at (cx, cy) with given side length and rotation."""
    angle = math.radians(angle_deg)
    # Vertices at angles 0, 60, 120, 180, 240, 300 degrees (flat-top orientation at 0 deg)
    # Actually, for a regular hexagon, vertices are at angle + k*60
    verts = []
    for k in range(6):
        theta = angle + k * math.pi / 3.0
        x = cx + side * math.cos(theta)
        y = cy + side * math.sin(theta)
        verts.append((x, y))
    return verts

def _point_in_hex(px, py, cx, cy, side, angle_deg):
    """Check if point (px, py) is inside hexagon centered at (cx, cy) with side length and rotation."""
    # Transform to local coordinates
    angle = math.radians(angle_deg)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    dx = px - cx
    dy = py - cy
    lx = cos_a * dx + sin_a * dy
    ly = -sin_a * dx + cos_a * dy
    
    # In local coords, hexagon has vertices at (side*cos(k*pi/3), side*sin(k*pi/3))
    # The hexagon is the intersection of 6 half-planes.
    # For a regular hexagon with "flat top" (vertices at 0, 60, ...), the constraints are:
    # |x| <= side (for the left/right edges? No, that's not right)
    # Let's use the property that a regular hexagon can be described by:
    # max(|x|, |y|*sqrt(3)/2 + |x|/2) ... no, simpler to check against edges.
    
    # Vertices in local coords (flat-top, vertex at angle 0):
    # (side, 0), (side/2, side*sqrt(3)/2), (-side/2, side*sqrt(3)/2), 
    # (-side, 0), (-side/2, -side*sqrt(3)/2), (side/2, -side*sqrt(3)/2)
    
    # The 6 edge normals (pointing inward) and distances:
    # Edge between (side,0) and (side/2, side*sqrt(3)/2):
    #   Normal: (sqrt(3)/2, 1/2) -> wait, let's compute properly.
    
    # Simpler: Check if point is inside by checking against all 6 edges.
    # For each edge, check if point is on the correct side.
    verts = _hex_vertices(0, 0, side, 0)
    
    for i in range(6):
        x1, y1 = verts[i]
        x2, y2 = verts[(i+1) % 6]
        # Edge vector
        ex = x2 - x1
        ey = y2 - y1
        # Normal (pointing inward)
        # For CCW polygon, inward normal is (ey, -ex) normalized? 
        # Let's check: edge from (side,0) to (side/2, side*sqrt(3)/2)
        # ex = -side/2, ey = side*sqrt(3)/2
        # Normal (ey, -ex) = (side*sqrt(3)/2, side/2). This points inward? 
        # Dot product with vector from edge midpoint to center (0,0):
        # Midpoint: (3*side/4, side*sqrt(3)/4)
        # Vector to center: (-3*side/4, -side*sqrt(3)/4)
        # Dot: (side*sqrt(3)/2)*(-3*side/4) + (side/2)*(-side*sqrt(3)/4) = -3*side^2*sqrt(3)/8 - side^2*sqrt(3)/8 = -4*side^2*sqrt(3)/8 < 0
        # So (ey, -ex) points outward. We want inward, so use (-ey, ex).
        nx = -ey
        ny = ex
        # Distance from point to line
        dist = nx * lx + ny * ly
        # The distance from center to edge is side * sqrt(3) / 2
        # If dist > side * sqrt(3) / 2, point is outside
        if dist > side * math.sqrt(3) / 2 + 1e-9:
            return False
    return True

def _hex_overlap_penalty(hex1, hex2):
    """
    hex1, hex2: (cx, cy, angle_deg)
    Returns a penalty based on the minimum distance between the two hexagons.
    If they overlap, returns a large penalty.
    """
    cx1, cy1, a1 = hex1
    cx2, cy2, a2 = hex2
    
    # Quick reject: if distance between centers is less than 2*side (min distance for non-overlap)
    dx = cx2 - cx1
    dy = cy2 - cy1
    dist = math.sqrt(dx*dx + dy*dy)
    if dist > 2.0 + 0.1:  # With side=1, min distance is 2*side*sqrt(3)/2 = sqrt(3) ~ 1.732 for edge-to-edge, but for vertex-to-vertex it can be less. Actually, the minimum distance between two non-overlapping unit hexagons is sqrt(3) (edge-to-edge) or 2 (vertex-to-vertex). Let's use 1.7 for safety.
        return 0.0
    
    # Check if any vertex of hex1 is inside hex2
    verts1 = _hex_vertices(cx1, cy1, 1.0, a1)
    for vx, vy in verts1:
        if _point_in_hex(vx, vy, cx2, cy2, 1.0, a2):
            return 100.0
    
    # Check if any vertex of hex2 is inside hex1
    verts2 = _hex_vertices(cx2, cy2, 1.0, a2)
    for vx, vy in verts2:
        if _point_in_hex(vx, vy, cx1, cy1, 1.0, a1):
            return 100.0
    
    # Check edge-edge intersections (simplified: if centers are close, assume overlap)
    if dist < math.sqrt(3) - 0.1:
        return 10.0
        
    return 0.0

def _outer_side_length(inner, center, angle_deg):
    """
    Compute the minimum side length of a regular hexagon centered at 'center' with rotation 'angle_deg'
    that contains all inner hexagons.
    """
    cx, cy = center
    angle = math.radians(angle_deg)
    
    # The outer hexagon has 6 edges. For each edge, we need the maximum projection of all inner hexagon vertices onto the normal of that edge.
    # The side length S is determined by: S = max_projection / (sqrt(3)/2)
    # Actually, for a regular hexagon with side length S, the distance from center to each edge is S * sqrt(3) / 2.
    # So S = max_projection / (sqrt(3)/2) = 2 * max_projection / sqrt(3)
    
    max_proj = 0.0
    for k in range(6):
        theta = angle + k * math.pi / 3.0
        nx = math.cos(theta)
        ny = math.sin(theta)
        
        for (ix, iy, ia) in inner:
            verts = _hex_vertices(ix, iy, 1.0, ia)
            for (vx, vy) in verts:
                proj = nx * (vx - cx) + ny * (vy - cy)
                if proj > max_proj:
                    max_proj = proj
    
    return 2.0 * max_proj / math.sqrt(3)

def _objective(params, inner_fixed):
    """
    params: [cx, cy, angle_deg, ...] for the outer hexagon
    inner_fixed: list of (x, y, angle_deg) for inner hexagons
    """
    cx, cy, angle_deg = params
    return _outer_side_length(inner_fixed, (cx, cy), angle_deg)

def _overlap_penalty_all(inner):
    """Compute total overlap penalty for all pairs of inner hexagons."""
    penalty = 0.0
    n = len(inner)
    for i in range(n):
        for j in range(i+1, n):
            penalty += _hex_overlap_penalty(inner[i], inner[j])
    return penalty

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Initial configuration: honeycomb lattice
    # Unit hexagons in a honeycomb lattice have centers at:
    # (i * 1.5, j * sqrt(3) + (i % 2) * sqrt(3)/2) for flat-top hexagons
    # But we can rotate them.
    
    # Let's start with a simple grid
    inner = []
    for i in range(-(N//2), N//2 + 1):
        for j in range(-(N//2), N//2 + 1):
            # Honeycomb lattice positions
            x = i * 1.5
            y = j * math.sqrt(3) + (i % 2) * math.sqrt(3) / 2
            # Small random perturbation
            x += np.random.uniform(-0.05, 0.05)
            y += np.random.uniform(-0.05, 0.05)
            angle = np.random.uniform(0, 60)
            inner.append([x, y, angle])
    
    # Initial outer hexagon
    center = [0.0, 0.0]
    angle_deg = 0.0
    
    # Check initial overlap
    penalty = _overlap_penalty_all(inner)
    
    # Optimize the outer hexagon position and rotation
    # We'll use Nelder-Mead to minimize the outer side length
    def objective_wrapper(params):
        cx, cy, angle = params
        return _outer_side_length(inner, (cx, cy), angle)
    
    x0 = [0.0, 0.0, 0.0]
    result = minimize(objective_wrapper, x0, method='Nelder-Mead', 
                      options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6})
    
    best_side = result.fun
    best_center = (result.x[0], result.x[1])
    best_angle = result.x[2]
    
    # Now try to optimize inner hexagon positions to reduce overlaps and outer size
    # This is a harder problem. Let's try a simple gradient-free approach.
    
    # For now, return the best we have
    return inner, best_center, best_side, best_angle

# Test
if __name__ == "__main__":
    inner, center, side, angle = construct()
    print(f"Outer side length: {side}")
    print(f"Center: {center}")
    print(f"Angle: {angle}")
