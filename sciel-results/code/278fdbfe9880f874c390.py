import numpy as np
from scipy.optimize import minimize
import time

N = 12

def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon."""
    angle = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = angle + i * np.pi / 3
        x = cx + side * np.cos(theta)
        y = cy + side * np.sin(theta)
        verts.append([x, y])
    return np.array(verts)

def _hexagon_contains_point(cx, cy, side, angle_deg, px, py):
    """Check if point (px,py) is inside hexagon centered at (cx,cy) with given side and angle."""
    # Transform to hexagon's local frame
    dx = px - cx
    dy = py - cy
    angle = np.radians(angle_deg)
    # Rotate point by -angle
    cos_a = np.cos(-angle)
    sin_a = np.sin(-angle)
    lx = dx * cos_a - dy * sin_a
    ly = dx * sin_a + dy * cos_a
    
    # In local frame, hexagon has vertices at angles 0, 60, 120, 180, 240, 300
    # The hexagon is the intersection of 6 half-planes
    # For a regular hexagon with circumradius = side, the half-planes are:
    # For each edge, the distance from center to edge (inradius) = side * sqrt(3)/2
    # Edge normals are at angles 30, 90, 150, 210, 270, 330 degrees
    inradius = side * np.sqrt(3) / 2
    for i in range(6):
        theta = np.radians(30 + i * 60)
        nx = np.cos(theta)
        ny = np.sin(theta)
        dist = lx * nx + ly * ny
        if dist > inradius + 1e-10:
            return False
    return True

def _min_enclosing_hexagon(points):
    """Find the minimum side length regular hexagon (any rotation) that contains all points.
    Returns (center_x, center_y, side_length, angle_deg)."""
    # For a given rotation angle, we can compute the minimum enclosing hexagon
    # A regular hexagon of side s, centered at (cx, cy), rotated by angle a:
    # A point (px, py) is inside if for all 6 edge normals, the projection <= s * sqrt(3)/2
    
    def hexagon_contains_all(cx, cy, s, angle_deg):
        angle = np.radians(angle_deg)
        inradius = s * np.sqrt(3) / 2
        for i in range(6):
            theta = angle + np.radians(30 + i * 60)
            nx = np.cos(theta)
            ny = np.sin(theta)
            # Project all points onto this normal
            projections = (points[:, 0] - cx) * nx + (points[:, 1] - cy) * ny
            if np.max(projections) > inradius + 1e-10:
                return False
        return True
    
    def compute_min_side_for_angle(cx, cy, angle_deg):
        """Given center and angle, find minimum side length to contain all points."""
        angle = np.radians(angle_deg)
        max_proj = 0
        for i in range(6):
            theta = angle + np.radians(30 + i * 60)
            nx = np.cos(theta)
            ny = np.sin(theta)
            projections = (points[:, 0] - cx) * nx + (points[:, 1] - cy) * ny
            max_proj = max(max_proj, np.max(projections))
        # inradius = s * sqrt(3)/2, so s = max_proj / (sqrt(3)/2) = 2 * max_proj / sqrt(3)
        return 2 * max_proj / np.sqrt(3)
    
    # Optimize over center and angle
    # Initial guess: centroid of points, angle = 0
    cx0 = np.mean(points[:, 0])
    cy0 = np.mean(points[:, 1])
    
    def objective(params):
        cx, cy, angle_deg = params
        return compute_min_side_for_angle(cx, cy, angle_deg)
    
    # Try multiple initial angles
    best_result = None
    best_obj = np.inf
    for init_angle in [0, 30, 60, 90, 120, 150]:
        x0 = [cx0, cy0, float(init_angle)]
        res = minimize(objective, x0, method='Nelder-Mead', 
                      options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8})
        if res.fun < best_obj:
            best_obj = res.fun
            best_result = res.x
    
    if best_result is None:
        return (cx0, cy0, 10.0, 0.0)
    
    cx, cy, angle_deg = best_result
    side = compute_min_side_for_angle(cx, cy, angle_deg)
    return (cx, cy, side, angle_deg)

def _all_hexagon_vertices(inner):
    """Get all vertices of all inner hexagons."""
    all_verts = []
    for x, y, angle in inner:
        verts = _hex_vertices(x, y, 1.0, angle)
        all_verts.append(verts)
    return np.vstack(all_verts)

def _check_disjoint(inner):
    """Check that all inner hexagons are pairwise disjoint (no overlap).
    Two convex polygons are disjoint if their interiors don't overlap.
    We check if any vertex of one is inside the other, or if edges intersect."""
    n = len(inner)
    hexes = [_hex_vertices(x, y, 1.0, a) for x, y, a in inner]
    
    for i in range(n):
        for j in range(i + 1, n):
            # Check if any vertex of hex i is inside hex j
            for v in hexes[i]:
                if _hexagon_contains_point(inner[j][0], inner[j][1], 1.0, inner[j][2], v[0], v[1]):
                    return False
            # Check if any vertex of hex j is inside hex i
            for v in hexes[j]:
                if _hexagon_contains_point(inner[i][0], inner[i][1], 1.0, inner[i][2], v[0], v[1]):
                    return False
            # Check edge-edge intersections
            if _edges_intersect(hexes[i], hexes[j]):
                return False
    return True

def _edges_intersect(poly1, poly2):
    """Check if any edge of poly1 intersects any edge of poly2."""
    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])
    
    def on_segment(a, b, c):
        return min(a[0], b[0]) <= c[0] <= max(a[0], b[0]) and min(a[1], b[1]) <= c[1] <= max(a[1], b[1])
    
    def segments_intersect(p1, p2, p3, p4):
        d1 = ccw(p3, p4, p1)
        d2 = ccw(p3, p4, p2)
        d3 = ccw(p1, p2, p3)
        d4 = ccw(p1, p2, p4)
        
        if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
           ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
            return True
        if d1 == 0 and on_segment(p3, p4, p1): return True
        if d2 == 0 and on_segment(p3, p4, p2): return True
        if d3 == 0 and on_segment(p1, p2, p3): return True
        if d4 == 0 and on_segment(p1, p2, p4): return True
        return False
    
    for i in range(6):
        for j in range(6):
            p1 = poly1[i]
            p2 = poly1[(i + 1) % 6]
            p3 = poly2[j]
            p4 = poly2[(j + 1) % 6]
            if segments_intersect(p1, p2, p3, p4):
                return True
    return False

def construct():
    start_time = time.time()
    deadline = start_time + 120  # Must finish within 120s
    
    # Start with a honeycomb-like arrangement
    # Place hexagons in a pattern that minimizes the bounding hexagon
    
    # Honeycomb arrangement for 12 hexagons:
    # A central hexagon, surrounded by 6, then 5 more
    # Or: two rows of 6, offset
    
    # Let's try a more systematic approach:
    # Arrange in a hexagonal grid pattern
    
    # For 12 hexagons, a good arrangement:
    # Row 1: 4 hexagons
    # Row 2: 4 hexagons (offset)
    # Row 3: 4 hexagons
    # Or: 3-4-3-2 pattern
    
    # Let me try a circular/symmetric arrangement
    # Central hexagon + ring of 6 + 5 more
    
    # Actually, let's use optimization
    # Parameters: positions (x,y) and angles for each hexagon
    
    # Initial placement: honeycomb grid
    # Hex grid spacing: 2 * sqrt(3) for horizontal, 3 for vertical (for unit hexagons)
    # Actually for regular hexagons of side 1, the center-to-center distance for adjacent is 2
    
    # Let's place them in a hexagonal pattern
    # Using axial coordinates
    
    # For 12 hexagons, let's try:
    # A 3x4 grid in hex coordinates
    
    # Hex grid: adjacent centers are distance 2 apart
    # Horizontal spacing: 2 * cos(30°) = sqrt(3) ≈ 1.732
    # Vertical spacing: 2 * sin(30°) = 1, but rows are offset
    
    # Let me just place them in a compact arrangement and optimize
    
    # Initial positions: place in a hexagonal pattern
    positions = []
    angles = []
    
    # Hex grid coordinates for 12 hexagons
    # Using a pattern: 
    #   o o o
    #  o o o o
    #   o o o
    #  o o
    # That's 3+4+3+2 = 12
    
    # Or simpler: a 4x3 grid with hex offset
    # Row 0: x = 0, 2*sqrt(3), 4*sqrt(3)  (3 hexagons)
    # Row 1: x = sqrt(3), 3*sqrt(3), 5*sqrt(3), 7*sqrt(3)  (4 hexagons) 
    # This is getting complex. Let me just use a simple grid and optimize.
    
    # Simple initial placement: 4x3 grid
    # Horizontal spacing: 2*sqrt(3) ≈ 3.464 (for non-overlapping hexagons side by side)
    # Wait, two hexagons of side 1 side by side: center distance = 2
    # But they need to be disjoint, so center distance >= 2
    
    # For a hexagon of side 1, the width (vertex to opposite vertex) = 2
    # The height (flat to flat) = sqrt(3)
    
    # So if I place them in a grid with horizontal spacing 2 and vertical spacing sqrt(3),
    # with alternating rows offset by 1, they should be disjoint.
    
    # Actually, for regular hexagons, the minimum center-to-center distance for disjointness
    # depends on orientation. For aligned hexagons, it's 2.
    
    # Let me place them in a honeycomb pattern
    # Using the fact that in a honeycomb, adjacent hexagons have center distance 2
    
    # 12 hexagons in a hexagonal arrangement:
    # Center: (0, 0)
    # Ring 1 (6 hexagons): at distance 2, angles 0, 60, 120, 180, 240, 300
    # Ring 2 partial (5 hexagons): at distance 4, some angles
    
    # This gives 1 + 6 + 5 = 12
    
    # Let me compute these positions
    pos = []
    ang = []
    
    # Central hexagon
    pos.append([0.0, 0.0])
    ang.append(0.0)
    
    # Ring 1: 6 hexagons at distance 2
    for i in range(6):
        theta = i * np.pi / 3
        x = 2.0 * np.cos(theta)
        y = 2.0 * np.sin(theta)
        pos.append([x, y])
        ang.append(0.0)
    
    # Ring 2: 5 hexagons at distance 4
    # Place them at angles that fill gaps
    for i in range(5):
        theta = i * 2 * np.pi / 5 + np.pi / 5  # offset to avoid symmetry issues
        x = 4.0 * np.cos(theta)
        y = 4.0 * np.sin(theta)
        pos.append([x, y])
        ang.append(0.0)
    
    inner = [[pos[i][0], pos[i][1], ang[i]] for i in range(12)]
    
    # Verify disjointness
    if not _check_disjoint(inner):
        # Adjust if needed
        # For now, proceed and let optimization handle it
        pass
    
    # Compute initial outer hexagon
    all_verts = _all_hexagon_vertices(inner)
    cx, cy, side, angle_deg = _min_enclosing_hexagon(all_verts)
    
    best_side = side
    best_inner = [list(x) for x in inner]
    best_center = (cx, cy)
    best_angle = angle_deg
    
    # Now optimize
    # We'll optimize the positions and angles of the inner hexagons
    # to minimize the outer hexagon side length while keeping them disjoint
    
    def flatten_params(inner_list):
        params = []
        for x, y, a in inner_list:
            params.extend([x, y, a])
        return np.array(params)
    
    def unflatten_params(params):
        inner_list = []
        for i in range(N):
            x = params[3*i]
            y = params[3*i + 1]
            a = params[3*i + 2]
            inner_list.append([x, y, a])
        return inner_list
    def objective(params):
        if time.time() > deadline - 10:
            return best_side + 1000  # Penalize if we're running out of time
        
        inner_list = unflatten_params(params)
        
        # Check disjointness
        if not _check_disjoint(inner_list):
            return 1e10  # Large penalty for overlapping
        
        # Compute outer hexagon
        all_verts = _all_hexagon_vertices(inner_list)
        _, _, s, _ = _min_enclosing_hexagon(all_verts)
        return s
    
    x0 = flatten_params(inner)
    
    # Use Nelder-Mead for optimization
    try:
        res = minimize(objective, x0, method='Nelder-Mead',
                      options={'maxiter': 5000, 'xatol': 1e-5, 'fatol': 1e-7,
                               'adaptive': True})
        
        if res.fun < best_side and res.fun < 1e9:
            best_inner = unflatten_params(res.x)
            all_verts = _all_hexagon_vertices(best_inner)
            cx, cy, best_side, best_angle = _min_enclosing_hexagon(all_verts)
            best_center = (cx, cy)
    except Exception:
        pass
    
    # Verify final result
    if _check_disjoint(best_inner):
        all_verts = _all_hexagon_vertices(best_inner)
        cx, cy, final_side, final_angle = _min_enclosing_hexagon(all_verts)
        return best_inner, (cx, cy), final_side, final_angle
    else:
        # Fallback to initial
        return inner, (0.0, 0.0), 10.0, 0.0

# EVOLVE-BLOCK-END
