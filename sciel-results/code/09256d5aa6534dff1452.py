"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START

def hex_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon centered at (cx, cy) with given side length and rotation."""
    angle_rad = np.radians(angle_deg)
    angles = angle_rad + np.pi / 6 + np.arange(6) * np.pi / 3
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def polygon_area(vertices):
    """Compute area of a polygon given its vertices using the shoelace formula."""
    x = vertices[:, 0]
    y = vertices[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

def point_in_hexagon(px, py, cx, cy, side, angle_deg):
    """Check if point (px, py) is inside a regular hexagon."""
    dx = px - cx
    dy = py - cy
    angle_rad = -np.radians(angle_deg)
    x_local = dx * np.cos(angle_rad) - dy * np.sin(angle_rad)
    y_local = dx * np.sin(angle_rad) + dy * np.cos(angle_rad)
    
    vertices = hex_vertices(0, 0, side, 0)
    for i in range(6):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % 6]
        ex = x2 - x1
        ey = y2 - y1
        nx = ey
        ny = -ex
        val = nx * x_local + ny * y_local
        if val < -1e-10:
            return False
    return True

def is_point_in_outer_hexagon(px, py, cx, cy, side, angle_deg):
    """Check if a point is inside the outer hexagon."""
    return point_in_hexagon(px, py, cx, cy, side, angle_deg)

def min_enclosing_hexagon_side(points, center=None, angle_candidates=None):
    """Find the minimum side length of a regular hexagon (with optional fixed center) that contains all points.
    
    Args:
        points: (M, 2) array of points
        center: (cx, cy) or None to optimize center
        angle_candidates: list of angles in degrees to try, or None for default
    
    Returns:
        (min_side, best_center, best_angle)
    """
    if angle_candidates is None:
        angle_candidates = np.linspace(0, 60, 13)
    
    if center is None:
        center = np.mean(points, axis=0)
    
    min_side = float('inf')
    best_center = center
    best_angle = angle_candidates[0]
    
    for angle in angle_candidates:
        # For a regular hexagon with side s, center c, angle a,
        # a point p is inside if for all 6 edges, the point is on the inside.
        # The hexagon is the intersection of 6 half-planes.
        # For each edge i, the constraint is: n_i · (p - c) <= d_i * s
        # where n_i is the inward normal and d_i is the apothem-like distance.
        
        # Actually, let's use the vertex-based check with binary search.
        # For a given angle and center, find min s such that all points are inside.
        
        lo = 0.0
        hi = 20.0
        for _ in range(40):
            mid = (lo + hi) / 2
            # Check if all points are inside hexagon with side mid, center, angle
            all_inside = True
            for p in points:
                if not is_point_in_outer_hexagon(p[0], p[1], center[0], center[1], mid, angle):
                    all_inside = False
                    break
            if all_inside:
                hi = mid
            else:
                lo = mid
        
        if hi < min_side:
            min_side = hi
            best_angle = angle
    
    return min_side, best_center, best_angle

def construct():
    """Construct 12 unit hexagons inside a minimal outer hexagon."""
    sqrt3 = np.sqrt(3)
    
    # Hexagonal lattice arrangement for 12 hexagons
    # Using axial coordinates (q, r) -> Cartesian:
    # x = 1.5 * q
    # y = sqrt(3) * (r + q/2)
    # Adjacent hexagons (side length 1) are at distance sqrt(3) in the hex grid
    
    # A compact arrangement of 12 hexagons:
    # Center + 6 in first ring + 5 in second ring (partial)
    # Or: rows of 4, 4, 4 (with offsets)
    
    # Let's use a hexagonal cluster:
    # Axial coords for a hexagonal number pattern
    # For 12 hexagons, a good shape is:
    # Row 0 (top): 2 hexagons
    # Row 1: 3 hexagons
    # Row 2: 3 hexagons
    # Row 3: 2 hexagons
    # Row 4: 2 hexagons
    # Total: 12
    
    # Actually, let's use the standard hexagonal lattice positions
    # and pick the 12 most central ones
    
    # Generate a grid of hex lattice positions
    positions_axial = []
    for q in range(-3, 4):
        for r in range(-3, 4):
            # Convert to Cartesian
            x = 1.5 * q
            y = sqrt3 * (r + q / 2.0)
            positions_axial.append((x, y, q, r))
    
    # Sort by distance from origin
    positions_axial.sort(key=lambda p: p[0]**2 + p[1]**2)
    
    # Take the 12 closest
    selected = positions_axial[:N]
    
    # Convert to (x, y, angle) with angle=0 (all aligned)
    inner_positions = [(p[0], p[1], 0.0) for p in selected]
    
    # Get all vertices of inner hexagons
    all_vertices = []
    for x, y, a in inner_positions:
        verts = hex_vertices(x, y, 1.0, a)
        all_vertices.append(verts)
    all_vertices = np.vstack(all_vertices)
    
    # Find the center of mass
    center = np.mean(all_vertices, axis=0)
    
    # Find minimum enclosing hexagon
    min_side, best_center, best_angle = min_enclosing_hexagon_side(
        all_vertices, center=center, angle_candidates=np.linspace(0, 60, 25)
    )
    
    inner = [[p[0], p[1], p[2]] for p in inner_positions]
    return inner, (best_center[0], best_center[1]), min_side, best_angle

# EVOLVE-BLOCK-END
