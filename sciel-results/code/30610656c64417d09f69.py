import numpy as np
from scipy.optimize import minimize
import time

N = 12

# EVOLVE-BLOCK-START
def _hex_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon centered at (cx,cy) with given side and rotation."""
    a = np.radians(angle_deg)
    angles = a + np.pi/6 + np.arange(6) * np.pi/3
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def _point_in_hexagon(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside hexagon. Returns True if inside."""
    a = np.radians(angle_deg)
    dx = px - cx
    dy = py - py  # placeholder
    dx = px - cx
    dy = py - cy
    # Rotate point into hexagon's local frame
    rx = dx * np.cos(-a) - dy * np.sin(-a)
    ry = dx * np.sin(-a) + dy * np.cos(-a)
    # Hexagon with flat top: sides at angles 0, 60, 120, 180, 240, 300 from center
    # Inradius = side * sqrt(3)/2
    inradius = side * np.sqrt(3) / 2
    # For a regular hexagon, point is inside if for each of 6 sides, the projection is within inradius
    # Side normals are at angles 0, 60, 120, 180, 240, 300
    for k in range(6):
        theta = k * np.pi / 3
        nx = np.cos(theta)
        ny = np.sin(theta)
        proj = rx * nx + ry * ny
        if proj > inradius:
            return False
    return True

def _hexagon_sdf(px, py, cx, cy, side, angle_deg):
    """Signed distance from point to hexagon. Negative if inside."""
    a = np.radians(angle_deg)
    dx = px - cx
    dy = py - cy
    rx = dx * np.cos(-a) - dy * np.sin(-a)
    ry = dx * np.sin(-a) + dy * np.cos(-a)
    inradius = side * np.sqrt(3) / 2
    # Distance to hexagon boundary
    # For regular hexagon, SDF can be computed using the 6 half-planes
    # The SDF is the minimum over all 6 sides of (inradius - proj) if inside, or min distance to edges
    # Simplified: use the fact that for a regular hexagon, the distance from center in direction theta
    # to the boundary is inradius / cos(min_angle_to_side_normal)
    # This is complex; use a simpler approach for optimization
    # For optimization, we can use the maximum of the 6 linear functions as a proxy
    max_proj = 0
    for k in range(6):
        theta = k * np.pi / 3
        nx = np.cos(theta)
        ny = np.sin(theta)
        proj = rx * nx + ry * ny
        max_proj = max(max_proj, proj)
    # If max_proj < inradius, point is inside. SDF ≈ inradius - max_proj (not exact but smooth)
    # Actually this gives distance to the "hexagonal norm" boundary
    return inradius - max_proj

def _min_outer_side(inner_positions, outer_center, outer_side, outer_angle):
    """Compute minimum outer side length needed to contain all inner hexagons.
    inner_positions: list of (x, y, angle) for N unit hexagons
    Returns the required outer side length."""
    ocx, ocy = outer_center
    oa = np.radians(outer_angle)
    # For each inner hexagon, find its furthest vertex from outer center in the direction of each outer side
    # The outer hexagon contains a point if for all 6 sides, the projection is within inradius_outer
    # inradius_outer = outer_side * sqrt(3)/2
    max_required_inradius = 0
    for (ix, iy, ia) in inner_positions:
        # Get vertices of inner hexagon
        verts = _hex_vertices(ix, iy, 1.0, ia)
        for vx, vy in verts:
            # Transform to outer hexagon's local frame
            dx = vx - ocx
            dy = vy - ocy
            rx = dx * np.cos(-oa) - dy * np.sin(-oa)
            ry = dx * np.sin(-oa) + dy * np.cos(-oa)
            # Check all 6 sides of outer hexagon
            for k in range(6):
                theta = k * np.pi / 3
                nx = np.cos(theta)
                ny = np.sin(theta)
                proj = rx * nx + ry * ny
                max_required_inradius = max(max_required_inradius, proj)
    # Required inradius is max_required_inradius, side = inradius * 2 / sqrt(3)
    return max_required_inradius * 2 / np.sqrt(3)

def _check_validity(inner_positions, outer_center, outer_side, outer_angle, tol=1e-6):
    """Check if configuration is valid: no overlaps, all inside outer hexagon."""
    ocx, ocy = outer_center
    # Check containment
    for (ix, iy, ia) in inner_positions:
        verts = _hex_vertices(ix, iy, 1.0, ia)
        for vx, vy in verts:
            if not _point_in_hexagon(vx, vy, ocx, ocy, outer_side, outer_angle):
                return False
    # Check pairwise non-overlap using SAT
    for i in range(N):
        for j in range(i+1, N):
            if _polygons_overlap(inner_positions[i], inner_positions[j]):
                return False
    return True

def _polygons_overlap(hex1, hex2):
    """SAT test for two convex polygons (hexagons)."""
    v1 = _hex_vertices(hex1[0], hex1[1], 1.0, hex1[2])
    v2 = _hex_vertices(hex2[0], hex2[1], 1.0, hex2[2])
    polygons = [v1, v2]
    for poly in polygons:
        for i in range(6):
            p1 = poly[i]
            p2 = poly[(i+1) % 6]
            # Edge normal
            ex = p2[0] - p1[0]
            ey = p2[1] - p1[1]
            nx = -ey
            ny = ex
            # Project both polygons onto this normal
            proj1 = np.dot(v1, [nx, ny])
            proj2 = np.dot(v2, [nx, ny])
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            if max1 < min2 or max2 < min1:
                return False  # Found separating axis
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Initial configuration: hexagonal grid pattern
    # Place 12 hexagons in a 3x4 hexagonal lattice
    # Spacing: sqrt(3) horizontally, 1.5 vertically for offset rows
    sqrt3 = np.sqrt(3)
    
    # Try a few initial arrangements
    best_score = float('inf')
    best_result = None
    
    # Arrangement 1: 4x3 grid
    # Row 0: 4 hexagons
    # Row 1: 3 hexagons (offset)
    # Row 2: 4 hexagons
    # Row 3: 3 hexagons (offset) -- that's 14, too many
    # Let's do: 4+4+4 in 3 rows, or 3+4+3+2, etc.
    
    # Simple grid: 4 columns x 3 rows
    # Horizontal spacing: 2*sqrt(3) for same orientation? No, sqrt(3) for flat-to-flat
    # Actually for hexagons in a grid with same orientation:
    # If aligned, horizontal distance between centers for non-overlap: 2*sqrt(3) (across vertices)
    # or sqrt(3) (across flats) depending on direction
    
    # Let's use a hexagonal lattice:
    # a = sqrt(3) (distance between adjacent hexagon centers in honeycomb)
    # Actually in a honeycomb, adjacent hexagons have center distance = 2 * inradius = sqrt(3)
    
    a = sqrt3  # center-to-center distance for touching hexagons
    
    # 3 rows of 4
    positions = []
    for row in range(3):
        for col in range(4):
            x = col * a
            y = row * 1.5  # vertical spacing for hex grid
            if row % 2 == 1:
                x += a / 2
            positions.append([x, y, 0.0])
    
    # Center the configuration
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    positions = [[p[0] - cx, p[1] - cy, p[2]] for p in positions]
    
    # Estimate outer size
    max_r = max(np.sqrt(p[0]**2 + p[1]**2) + 1.0 for p in positions)
    est_side = max_r * 2 / np.sqrt(3) * 1.1  # rough estimate with margin
    
    # Optimization variables: positions (2N) + outer center (2) + outer side (1) + outer angle (1)
    # Total: 2*12 + 2 + 1 + 1 = 28 variables
    # Objective: outer_side
    # Constraints: no overlap (SAT), containment
    
    # For SLSQP, we need smooth constraints. Let's use a penalty approach with SLSQP
    # or use the SDF-based approach.
    
    # Let's try: minimize outer_side subject to all inner hexagons being inside outer
    # and no overlaps. Use SLSQP with inequality constraints.
    
    def objective(vars):
        if time.time() > deadline:
            return 1e10
        pos = []
        for i in range(N):
            x = vars[2*i]
            y = vars[2*i+1]
            ang = vars[2*N + 2*i]
            pos.append([x, y, ang])
        ocx = vars[2*N + 2*N]
        ocy = vars[2*N + 2*N + 1]
        oside = vars[2*N + 2*N + 2]
        oang = vars[2*N + 2*N + 3]
        return oside
    
    # This is getting complex. Let me use a simpler approach:
    # Fix the outer hexagon at origin with angle 0, and just optimize inner positions
    # and outer side.
    
    # Even simpler: try several analytical configurations and pick the best.
    
    configs = []
    
    # Config 1: 4x3 rectangular grid of hexagons, all same orientation
    # For hexagons with flat top, width = 2, height = sqrt(3)
    # To pack without overlap: horizontal spacing = 2, vertical spacing = sqrt(3)
    # But we can do better with offset rows
    # In a hexagonal packing: horizontal = sqrt(3), vertical = 1.5 for offset
    # But that's for the honeycomb where hexagons interlock
    
    # Actually, for non-rotated hexagons in a grid:
    # If all have same orientation, the minimum horizontal spacing is 2*sqrt(3) (vertex to vertex)
    # or sqrt(3) (flat to flat) -- it depends on the direction
    
    # Let me just use the hexagonal lattice spacing
    # In a regular hexagonal tiling, adjacent hexagon centers are 2*inradius = sqrt(3) apart
    
    # Configuration: hexagonal arrangement
    # 12 hexagons in a roughly hexagonal shape
    # Center + 6 around + 5 more? Or 3+4+3+2?
    
    # Let me try: place in a 2x6 rectangle with hexagonal offset
    for config_idx in range(5):
        if time.time() > deadline:
            break
        
        if config_idx == 0:
            # 4x3 grid, hexagonal lattice
            pos = []
            for row in range(3):
                for col in range(4):
                    x = col * sqrt3
                    y = row * 1.5
                    if row % 2 == 1:
                        x += sqrt3 / 2
                    pos.append([x, y, 0.0])
        elif config_idx == 1:
            # 6x2 grid
            pos = []
            for row in range(2):
                for col in range(6):
                    x = col * sqrt3
                    y = row * 1.5
                    if row % 2 == 1:
                        x += sqrt3 / 2
                    pos.append([x, y, 0.0])
        elif config_idx == 2:
            # 3x4 grid, rectangular
            pos = []
            for row in range(3):
                for col in range(4):
                    x = col * 2.0  # vertex-to-vertex horizontal
                    y = row * sqrt3  # flat-to-flat vertical
                    pos.append([x, y, 0.0])
        elif config_idx == 3:
            # Hexagonal cluster: center + ring of 6 + ring of 5
            pos = [[0, 0, 0.0]]
            # Ring 1: 6 hexagons at distance sqrt(3)
            for k in range(6):
                theta = k * np.pi / 3
                x = sqrt3 * np.cos(theta)
                y = sqrt3 * np.sin(theta)
                pos.append([x, y, 0.0])
            # Ring 2: add 5 more
            for k in range(5):
                theta = k * 2 * np.pi / 5 + np.pi / 5
                x = 2 * sqrt3 * np.cos(theta)
                y = 2 * sqrt3 * np.sin(theta)
                pos.append([x, y, 0.0])
        else:
            # 4x3 with slight rotation of inner hexagons
            pos = []
            for row in range(3):
                for col in range(4):
                    x = col * sqrt3
                    y = row * 1.5
                    if row % 2 == 1:
                        x += sqrt3 / 2
                    ang = 30.0 if (row + col) % 2 == 0 else 0.0
                    pos.append([x, y, ang])
        
        # Center the configuration
        xs = [p[0] for p in pos]
        ys = [p[1] for p in pos]
        cx = (min(xs) + max(xs)) / 2
        cy = (min(ys) + max(ys)) / 2
        pos = [[p[0] - cx, p[1] - cy, p[2]] for p in pos]
        
        # Find minimum enclosing hexagon
        # Try several outer angles
        best_side = float('inf')
        best_oang = 0
        for oang in range(0, 180, 5):
            side = _min_outer_side(pos, (0.0, 0.0), 100.0, float(oang))
            if side < best_side:
                best_side = side
                best_oang = oang
        
        # Verify validity
        if _check_validity(pos, (0.0, 0.0), best_side, best_oang):
            if best_side < best_score:
                best_score = best_side
                best_result = (pos, (0.0, 0.0), best_side, best_oang)
    
    # If no valid config found, use a large safe one
    if best_result is None:
        pos = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return pos, (0.0, 0.0), 40.0, 0.0
    
    return best_result
# EVOLVE-BLOCK-END
