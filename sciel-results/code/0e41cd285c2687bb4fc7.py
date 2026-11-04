"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START

def hex_vertices(cx, cy, side, angle_deg=0.0):
    """Return vertices of a regular hexagon centered at (cx,cy) with given side and rotation."""
    a = np.radians(angle_deg)
    # Flat-top hexagon: vertices at angles 0, 60, 120, 180, 240, 300
    angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180 + a
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def point_in_hexagon(px, py, cx, cy, side, angle_deg=0.0):
    """Check if point (px,py) is inside hexagon centered at (cx,cy)."""
    # Transform to hexagon's local frame
    dx = px - cx
    dy = py - cy
    a = np.radians(angle_deg)
    # Rotate back
    ca, sa = np.cos(-a), np.sin(-a)
    lx = ca * dx - sa * dy
    ly = sa * dx + ca * dy
    # For a regular hexagon with side s, the condition in local coords:
    # The hexagon has vertices at (s,0), (s/2, s*sqrt(3)/2), (-s/2, s*sqrt(3)/2), (-s,0), (-s/2,-s*sqrt(3)/2), (s/2,-s*sqrt(3)/2)
    # The support function / half-widths:
    # For angle 0 (flat top): |lx| <= s, |ly| <= s*sqrt(3)/2, and |lx|/s + |ly|/(s*sqrt(3)/2) <= 1 for the diagonal constraints
    # Actually, let's use the half-plane method
    s3 = side * np.sqrt(3)
    # The six half-planes (outward normals)
    normals = np.array([
        [1, 0],
        [0.5, np.sqrt(3)/2],
        [-0.5, np.sqrt(3)/2],
        [-1, 0],
        [-0.5, -np.sqrt(3)/2],
        [0.5, -np.sqrt(3)/2]
    ])
    # Distance from center to each side = side * sqrt(3)/2
    dist = side * np.sqrt(3) / 2
    # For each normal, dot(normal, local_point) <= dist
    vals = normals @ np.array([lx, ly])
    return np.all(vals <= dist + 1e-10)

def hexagon_contains_points(hex_verts, points):
    """Check if all points are inside the hexagon defined by hex_verts (6 vertices CCW)."""
    # Use cross product method for convex polygon
    n = len(hex_verts)
    for p in points:
        inside = True
        for i in range(n):
            v1 = hex_verts[i]
            v2 = hex_verts[(i+1) % n]
            # Cross product to check if p is to the left of edge v1->v2
            cross = (v2[0]-v1[0])*(p[1]-v1[1]) - (v2[1]-v1[1])*(p[0]-v1[0])
            if cross < -1e-10:
                inside = False
                break
        if not inside:
            return False
    return True

def get_hex_corners(cx, cy, side, angle_deg=0.0):
    """Get the 6 corners of a unit hexagon (side=1) centered at (cx,cy) with rotation."""
    return hex_vertices(cx, cy, side, angle_deg)

def objective(params, inner_positions):
    """
    params: [outer_cx, outer_cy, outer_side, outer_angle_deg]
    inner_positions: list of (cx, cy, angle_deg) for each inner hexagon
    Returns the outer side length if all inner hexagons fit, else penalty.
    """
    outer_cx, outer_cy, outer_side, outer_angle = params
    if outer_side <= 0:
        return 1e6
    
    outer_verts = hex_vertices(outer_cx, outer_cy, outer_side, outer_angle)
    
    # Collect all corners of all inner hexagons
    all_corners = []
    for (cx, cy, ang) in inner_positions:
        corners = get_hex_corners(cx, cy, 1.0, ang)
        all_corners.append(corners)
    
    all_corners = np.vstack(all_corners)
    
    # Check if all corners are inside the outer hexagon
    if not hexagon_contains_points(outer_verts, all_corners):
        # Penalty: how far outside are the points?
        # Use a soft penalty based on maximum violation
        return outer_side + 10.0
    
    return outer_side

def initial_honeycomb():
    """Create a compact honeycomb arrangement of 12 unit hexagons."""
    # Honeycomb: center at origin, 6 around it, then 5 more
    # For flat-top hexagons, neighbors are at:
    # (2, 0), (1, sqrt(3)), (-1, sqrt(3)), (-2, 0), (-1, -sqrt(3)), (1, -sqrt(3))
    s3 = np.sqrt(3)
    
    # 7-hexagon flower
    positions = [(0, 0, 0.0)]
    for k in range(6):
        angle = k * 60
        x = 2.0 * np.cos(np.radians(angle))
        y = 2.0 * np.sin(np.radians(angle))
        positions.append((x, y, 0.0))
    
    # Add 5 more hexagons attached to the outer ring
    # Attach to the "outer" sides of the 6 ring hexagons
    # For a hexagon at angle theta, its outer neighbor is at distance 2 in direction theta
    # Let's add 5 more by extending some of the ring hexagons
    # Better: add a second ring partially
    # Positions for second ring (distance 4 from center in 6 directions, but we only need 5)
    for k in range(5):
        angle = k * 60 + 30  # offset to fit in gaps or extend
        # Actually, let's extend the ring: place at distance 4 in the 6 main directions
        # But that gives 6 positions, we only need 5
        x = 4.0 * np.cos(np.radians(k * 60))
        y = 4.0 * np.sin(np.radians(k * 60))
        positions.append((x, y, 0.0))
    
    return positions[:N]

def initial_honeycomb_v2():
    """Better: 2-3-4-3-2 row arrangement or compact cluster."""
    s3 = np.sqrt(3)
    positions = []
    
    # Row-based honeycomb (offset rows)
    # Row 0: 3 hexagons
    # Row 1: 4 hexagons (offset)
    # Row 2: 3 hexagons
    # Row 3: 2 hexagons
    # Total: 12
    
    # Vertical spacing between rows: 1.5 (for pointy-top) or use flat-top
    # Let's use flat-top hexagons
    # Horizontal spacing: 2.0
    # Vertical spacing between adjacent rows: sqrt(3)
    # Offset for alternating rows: 1.0
    
    row_counts = [3, 4, 3, 2]
    y_spacing = s3  # vertical distance between row centers
    x_spacing = 2.0
    x_offset = 1.0
    
    y = 0.0
    for i, count in enumerate(row_counts):
        x_start = -(count - 1) * x_spacing / 2.0
        if i % 2 == 1:
            x_start += x_offset
        for j in range(count):
            x = x_start + j * x_spacing
            positions.append((x, y, 0.0))
        y += y_spacing
    
    return positions[:N]

def initial_compact():
    """Most compact: try to minimize bounding box."""
    s3 = np.sqrt(3)
    positions = []
    
    # Try a 4-3-2-3 arrangement or similar
    # Actually, the most compact packing of 12 hexagons is a hexagonal number pattern
    # Hex numbers: 1, 7, 19... so 12 is between 7 and 19
    # Best: 7-hex flower + 5 attached optimally
    
    # Let's use the 7-flower and attach 5 to minimize the overall size
    # Attach to 5 of the 6 outer faces
    positions = [(0, 0, 0.0)]
    for k in range(6):
        angle = k * 60
        x = 2.0 * np.cos(np.radians(angle))
        y = 2.0 * np.sin(np.radians(angle))
        positions.append((x, y, 0.0))
    
    # Add 5 more: extend 5 of the 6 ring hexagons outward
    # To keep it compact, extend in a contiguous arc
    for k in range(5):
        angle = k * 60
        x = 4.0 * np.cos(np.radians(angle))
        y = 4.0 * np.sin(np.radians(angle))
        positions.append((x, y, 0.0))
    
    return positions[:N]

def construct():
    start_time = time.time()
    deadline = 150.0 - 10.0  # 10s safety margin
    
    # Start with a good initial configuration
    inner_positions = initial_compact()
    
    # Compute initial outer hexagon that contains all
    all_corners = []
    for (cx, cy, ang) in inner_positions:
        corners = get_hex_corners(cx, cy, 1.0, ang)
        all_corners.append(corners)
    all_corners = np.vstack(all_corners)
    
    # Find center and initial side
    cx0 = np.mean(all_corners[:, 0])
    cy0 = np.mean(all_corners[:, 1])
    
    # Initial side: max distance from center to any corner
    dists = np.sqrt((all_corners[:, 0] - cx0)**2 + (all_corners[:, 1] - cy0)**2)
    side0 = np.max(dists) + 0.1
    
    # Optimize outer hexagon parameters
    x0 = [cx0, cy0, side0, 0.0]
    
    def obj(params):
        return objective(params, inner_positions)
    
    # Try multiple angle initializations
    best_result = None
    best_side = obj(x0)
    
    angles_to_try = [0, 30, 60, 15, 45]
    for ang in angles_to_try:
        if time.time() > deadline:
            break
        x0 = [cx0, cy0, side0, float(ang)]
        try:
            res = minimize(obj, x0, method='Nelder-Mead', 
                          options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6})
            if res.fun < best_side:
                best_side = res.fun
                best_result = res.x
        except Exception:
            pass
    
    if best_result is None:
        best_result = x0
    
    outer_cx, outer_cy, outer_side, outer_angle = best_result
    
    # Ensure validity: if still not fitting, increase side
    outer_verts = hex_vertices(outer_cx, outer_cy, outer_side, outer_angle)
    if not hexagon_contains_points(outer_verts, all_corners):
        # Increase side until it fits
        for _ in range(100):
            outer_side *= 1.05
            outer_verts = hex_vertices(outer_cx, outer_cy, outer_side, outer_angle)
            if hexagon_contains_points(outer_verts, all_corners):
                break
    
    # Convert to required format
    inner = [[pos[0], pos[1], pos[2]] for pos in inner_positions]
    
    return inner, (outer_cx, outer_cy), outer_side, outer_angle

# EVOLVE-BLOCK-END
