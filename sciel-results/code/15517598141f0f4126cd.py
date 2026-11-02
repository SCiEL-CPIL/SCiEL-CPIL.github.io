"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 11

def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon centered at (cx,cy) with given side and rotation."""
    angle_rad = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = angle_rad + i * np.pi / 3
        verts.append((cx + side * np.cos(theta), cy + side * np.sin(theta)))
    return np.array(verts)

def _point_in_hex(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside hexagon centered at (cx,cy) with side and rotation."""
    # Transform to hexagon's local frame
    dx = px - cx
    dy = py - cy
    angle_rad = np.radians(angle_deg)
    # Rotate back
    cos_a = np.cos(-angle_rad)
    sin_a = np.sin(-angle_rad)
    lx = dx * cos_a - dy * sin_a
    ly = dx * sin_a + dy * cos_a
    # In local frame, hexagon has vertices at angles 0, 60, 120, 180, 240, 300 with radius=side
    # Check if point is inside: for each edge, check if point is on the correct side
    for i in range(6):
        theta1 = i * np.pi / 3
        theta2 = (i + 1) * np.pi / 3
        vx1, vy1 = side * np.cos(theta1), side * np.sin(theta1)
        vx2, vy2 = side * np.cos(theta2), side * np.sin(theta2)
        # Edge from v1 to v2
        ex, ey = vx2 - vx1, vy2 - vy1
        # Normal pointing inward
        nx, ny = -ey, ex
        # Check if point is on the inward side
        val = (lx - vx1) * nx + (ly - vy1) * ny
        if val < -1e-9:
            return False
    return True

def _hexagons_disjoint(inner, eps=1e-9):
    """Check if all inner hexagons are pairwise disjoint (no interior overlap)."""
    n = len(inner)
    for i in range(n):
        for j in range(i+1, n):
            xi, yi, ai = inner[i]
            xj, yj, aj = inner[j]
            # Check if any vertex of hex i is inside hex j, or vice versa
            vi = _hex_vertices(xi, yi, 1.0, ai)
            vj = _hex_vertices(xj, yj, 1.0, aj)
            # Check vertices of i in j
            for vx, vy in vi:
                if _point_in_hex(vx, vy, xj, yj, 1.0, aj):
                    return False
            # Check vertices of j in i
            for vx, vy in vj:
                if _point_in_hex(vx, vy, xi, yi, 1.0, ai):
                    return False
    return True

def _outer_side_length(inner, outer_center, outer_angle):
    """Compute the minimum outer hexagon side length that contains all inner hexagons.
    The outer hexagon is centered at outer_center with rotation outer_angle."""
    cx, cy = outer_center
    # For a regular hexagon with side length s, centered at (cx,cy), rotated by outer_angle,
    # we need to find the minimum s such that all vertices of all inner hexagons are inside.
    # A point (px,py) is inside a hexagon of side s centered at (cx,cy) rotated by angle a iff:
    # In the local frame, the point satisfies the hexagon inequalities.
    # The "radius" in direction theta for a hexagon is s / cos((theta - nearest_vertex_angle) mod 60 - 30)
    # Actually, the support function: for direction u, the maximum extent is s * max over vertices of (v·u)
    # For a hexagon with vertices at angles 0,60,...,300 (radius s), the support in direction phi is:
    # s * max_i cos(phi - i*pi/3) ... but that's not quite right for the hexagon shape.
    
    # Simpler: for a given direction from center, the distance to the hexagon boundary is:
    # d(phi) = s * (sqrt(3)/2) / cos(((phi - angle_offset) mod (pi/3)) - pi/6)
    # where angle_offset accounts for rotation.
    
    # Let me use a different approach: for each inner hexagon vertex, compute the required outer side length.
    # The outer hexagon with side s, center (cx,cy), rotation a contains point (px,py) iff
    # when we transform to the hexagon's local frame, the point is inside the unit hexagon scaled by s.
    
    # In local frame (rotated by -a), a point (lx, ly) is inside hexagon of side s iff:
    # |lx| <= s and |ly| <= s*sqrt(3)/2 and |lx*sqrt(3)/2 + ly/2| <= s and |lx*sqrt(3)/2 - ly/2| <= s
    # Wait, that's for a hexagon with flat top/bottom. Let me be more careful.
    
    # For a regular hexagon with vertices at angles 0, 60, 120, 180, 240, 300 degrees, radius s:
    # The edges are at distances s*sqrt(3)/2 from center.
    # The hexagon is the intersection of 6 half-planes.
    # Edge i (between vertex i and i+1) has normal direction (i+0.5)*60 degrees.
    # The half-plane is: (x,y)·n_i <= s*sqrt(3)/2 where n_i is the unit normal.
    
    angle_rad = np.radians(outer_angle)
    cos_a = np.cos(-angle_rad)
    sin_a = np.sin(-angle_rad)
    
    max_s = 0.0
    for hex_data in inner:
        hx, hy, ha = hex_data
        verts = _hex_vertices(hx, hy, 1.0, ha)
        for vx, vy in verts:
            # Transform to outer hexagon's local frame
            dx = vx - cx
            dy = vy - cy
            lx = dx * cos_a - dy * sin_a
            ly = dx * sin_a + dy * cos_a
            # In local frame, hexagon of side s contains (lx,ly) iff for all 6 edge normals:
            # (lx,ly)·n_k <= s * sqrt(3)/2
            # n_k directions: 30, 90, 150, 210, 270, 330 degrees (normals to edges)
            for k in range(6):
                theta_n = np.radians(30 + k * 60)
                nx = np.cos(theta_n)
                ny = np.sin(theta_n)
                dot = lx * nx + ly * ny
                if dot > 0:
                    required_s = dot / (np.sqrt(3) / 2)
                    if required_s > max_s:
                        max_s = required_s
    return max_s

def _objective(params, inner_fixed_positions, outer_center, outer_angle):
    """Objective: outer side length. params are the angles of inner hexagons (or positions if optimizing)."""
    # For now, just compute based on fixed positions
    return _outer_side_length(inner_fixed_positions, outer_center, outer_angle)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Strategy: Arrange hexagons in a hexagonal lattice pattern
    # For N hexagons, find the best hexagonal arrangement
    
    # Hexagonal numbers: 1, 7, 19, 37, 61, 91, 127, ...
    # For N=11, we can use a 4-4-3 arrangement or similar
    
    # Let's try a simple grid arrangement first
    # Hexagonal lattice: rows with alternating offsets
    # Row spacing: 1.5 (vertical), horizontal spacing: sqrt(3)
    # Alternating rows offset by sqrt(3)/2
    
    sqrt3 = np.sqrt(3.0)
    
    # Try to arrange in a hexagonal pattern
    # For N hexagons, find rows that sum to N
    # Row lengths: r, r+1, r+2, ... or r, r-1, r-2, ...
    
    best_result = None
    best_side = float('inf')
    
    # Try different arrangements
    # Arrangement 1: centered hexagonal pattern
    # For a hexagonal cluster with max row length m:
    # Rows: m-2, m-1, m, m+1, m+2, ... or similar
    
    # Let's try a few configurations
    configs = []
    
    # Config: rows of lengths that sum to N
    # Try rows like [3,4,4], [2,4,5], [3,3,5], [4,3,4], etc.
    # For N=11:
    row_splits = [
        [3, 4, 4],
        [4, 3, 4],
        [2, 4, 5],
        [5, 4, 2],
        [3, 3, 5],
        [5, 3, 3],
        [4, 4, 3],
        [1, 5, 5],
        [5, 5, 1],
        [2, 3, 6],
        [6, 3, 2],
        [3, 5, 3],
        [4, 2, 5],
        [5, 2, 4],
        [1, 4, 6],
        [6, 4, 1],
        [2, 5, 4],
        [4, 5, 2],
        [3, 2, 6],
        [6, 2, 3],
    ]
    
    for split in row_splits:
        if sum(split) != N:
            continue
        
        # Build positions
        positions = []
        # Row spacing: 1.5 vertically
        # Within a row: sqrt(3) horizontally
        # Alternating rows offset by sqrt(3)/2
        
        total_rows = len(split)
        for row_idx, row_len in enumerate(split):
            # Vertical position
            y = (row_idx - (total_rows - 1) / 2.0) * 1.5
            # Horizontal offset for this row
            x_offset = 0.0
            if row_idx % 2 == 1:
                x_offset = sqrt3 / 2.0
            # Positions in this row
            for col_idx in range(row_len):
                x = (col_idx - (row_len - 1) / 2.0) * sqrt3 + x_offset
                positions.append([x, y, 0.0])
        
        if len(positions) != N:
            continue
        
        # Compute outer side length
        # Try different outer center and angle
        # First, try center at origin, angle 0
        side = _outer_side_length(positions, (0.0, 0.0), 0.0)
        
        if side < best_side:
            best_side = side
            best_result = (positions, (0.0, 0.0), side, 0.0)
        
        # Check deadline
        if time.time() > deadline - 20.0:
            break
    
    if best_result is None:
        # Fallback: simple linear arrangement
        positions = [[k * sqrt3, 0.0, 0.0] for k in range(N)]
        side = _outer_side_length(positions, (0.0, 0.0), 0.0)
        return positions, (0.0, 0.0), side, 0.0
    
    inner, center, side, angle = best_result
    
    # Try to optimize outer angle
    if time.time() < deadline - 10.0:
        best_angle = angle
        best_side_opt = side
        for test_angle in [0, 15, 30, 45, 60, 90, -15, -30, -45, -60, -90]:
            s = _outer_side_length(inner, center, test_angle)
            if s < best_side_opt:
                best_side_opt = s
                best_angle = test_angle
        angle = best_angle
        side = best_side_opt
    
    return inner, center, side, angle

