import numpy as np

N = 11

# EVOLVE-BLOCK-START
def _unit_hexagon_vertices():
    """Return vertices of a unit regular hexagon centered at origin, flat-top orientation."""
    # Regular hexagon with side 1, vertices at angles 0, 60, ..., 300 degrees
    angles = np.deg2rad(np.arange(0, 360, 60))
    return np.column_stack([np.cos(angles), np.sin(angles)])

def _hexagon_vertices(center, angle_deg):
    """Return vertices of unit hexagon at given center and rotation angle (degrees)."""
    verts = _unit_hexagon_vertices()
    theta = np.deg2rad(angle_deg)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    return verts @ R.T + np.array(center)

def _convex_hull(points):
    """Compute convex hull of 2D points using monotone chain. Returns list of (x,y) in CCW order."""
    pts = sorted(set([(round(p[0], 12), round(p[1], 12)) for p in points]))
    if len(pts) <= 2:
        return pts
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    return hull

def _width_in_direction(hull, angle_deg):
    """Compute width of convex hull in a given direction (angle in degrees)."""
    theta = np.deg2rad(angle_deg)
    u = np.array([np.cos(theta), np.sin(theta)])
    projections = []
    for p in hull:
        projections.append(p[0] * u[0] + p[1] * u[1])
    return max(projections) - min(projections)

def _min_enclosing_hexagon_side(hull, global_angle_deg):
    """Compute minimal enclosing regular hexagon side length for a given global rotation angle.
    
    The enclosing hexagon has side-normal directions at global_angle + k*60 for k=0..5.
    For each direction, the width of the hull must be <= L * sqrt(3).
    So L = max_k(width_k) / sqrt(3).
    """
    max_width = 0.0
    for k in range(6):
        dir_angle = global_angle_deg + k * 60.0
        w = _width_in_direction(hull, dir_angle)
        if w > max_width:
            max_width = w
    return max_width / np.sqrt(3.0)

def construct():
    # Place 11 unit hexagons in a 3x3 hexagonal grid (centered at origin)
    # Hexagonal grid: rows offset by 30 degrees, spacing = 2 (side length 1, so adjacent hexagons touch at distance 2)
    # Actually for regular hexagons of side 1, the distance between centers of adjacent hexagons in a honeycomb is 2.
    # A 3x3 hexagonal arrangement: center hexagon at (0,0), and 6 surrounding, plus 4 more? 
    # Let's use a compact cluster: 1 center + 6 surrounding + 4 more to make 11.
    # Simpler: use a 3-row hexagonal grid: row 0: 4 hexagons, row 1: 3, row 2: 4? No, that's 11.
    # Let me use: positions in a hexagonal lattice.
    # Hex lattice with spacing 2: neighbors at distance 2 in directions 0, 60, 120, 180, 240, 300.
    # Let's place them in a diamond/rhombus shape for compactness.
    
    # Use axial coordinates for hex grid. Convert to Cartesian.
    # For a hex grid with pointy-top orientation, but our hexagons are flat-top.
    # Let's just place them in a compact cluster manually.
    
    # 11 hexagons: a 3x3 square grid with the center removed? No, that's 8.
    # Let's use: a hexagonal number H(2) = 7 (center + 6) plus 4 more on the perimeter.
    # Or simpler: a 5x3 rectangle in hex grid? 
    # Let me just use a linear chain first to get a baseline, then optimize.
    # Actually, let's use a compact 3-row arrangement:
    # Row 0 (y=0): 4 hexagons at x = -3, -1, 1, 3
    # Row 1 (y=sqrt(3)): 3 hexagons at x = -2, 0, 2
    # Row 2 (y=2*sqrt(3)): 4 hexagons at x = -3, -1, 1, 3
    # Wait, that's 11 hexagons. Let me verify spacing.
    # For flat-top hexagons of side 1, the horizontal distance between adjacent hexagons in the same row is 3 (since width = 2, but they interlock).
    # Actually, for flat-top hexagons, the distance between centers of horizontally adjacent hexagons is 3*side = 3? No.
    # The width of a flat-top hexagon is 2*side = 2 (vertex to vertex horizontally).
    # The distance between centers of adjacent hexagons sharing a vertical edge is 3*side*cos(30) = 3*sqrt(3)/2 ≈ 2.598.
    # This is getting complicated. Let me just use a simple linear arrangement for now and rely on the rotation optimization.
    
    # Linear chain of 11 hexagons, all axis-aligned, spaced by 3 units (so they don't overlap).
    # Actually, for non-overlapping, center distance must be >= 2*inradius = 2*sqrt(3)/2 = sqrt(3) ≈ 1.732 for adjacent.
    # But for a chain, if they're in a line, center distance >= 2*side = 2? No, for hexagons sharing an edge, center distance = 2*inradius = sqrt(3).
    # Let me just space them by 2.0 to be safe.
    
    positions = []
    for k in range(-(N // 2), N - N // 2):
        positions.append((3.0 * k, 0.0, 0.0))
    
    inner = positions
    
    # Compute convex hull of all vertices
    all_vertices = []
    for (x, y, ang) in inner:
        verts = _hexagon_vertices((x, y), ang)
        all_vertices.extend(verts)
    
    hull = _convex_hull(all_vertices)
    
    # Minimize the enclosing hexagon side length over global rotation angle
    # Using ternary search on [0, 60) degrees
    lo, hi = 0.0, 60.0
    for _ in range(200):  # high precision ternary search
        m1 = lo + (hi - lo) / 3.0
        m2 = hi - (hi - lo) / 3.0
        f1 = _min_enclosing_hexagon_side(hull, m1)
        f2 = _min_enclosing_hexagon_side(hull, m2)
        if f1 < f2:
            hi = m2
        else:
            lo = m1
    
    best_angle = (lo + hi) / 2.0
    best_side = _min_enclosing_hexagon_side(hull, best_angle)
    
    # Center of the enclosing hexagon: centroid of the hull
    cx = sum(p[0] for p in hull) / len(hull)
    cy = sum(p[1] for p in hull) / len(hull)
    
    return inner, (cx, cy), best_side, best_angle
# EVOLVE-BLOCK-END
