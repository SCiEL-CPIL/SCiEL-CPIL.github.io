import numpy as np
import time

N = 12

def hex_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon with center (cx,cy), side length, rotation in degrees."""
    a = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * np.pi / 3
        verts.append((cx + side * np.cos(theta), cy + side * np.sin(theta)))
    return np.array(verts)

def get_all_points(inner):
    """Get all vertices of all inner hexagons as a single array."""
    pts = []
    for (x, y, ang) in inner:
        v = hex_vertices(x, y, 1.0, ang)
        pts.append(v)
    return np.vstack(pts)

def min_outer_side(points, cx, cy, angle_deg):
    """Given points, center, and outer hex rotation, find minimum side length to contain all points."""
    a = np.radians(angle_deg)
    # The 6 edge normals of a regular hexagon with rotation a
    # Edge normals are at angles a + pi/6 + i*pi/3 for i=0..5
    # For containment: dot(p - center, n_i) <= s * sqrt(3)/2 for all i
    inradius_factor = np.sqrt(3) / 2
    max_proj = 0
    for i in range(6):
        theta = a + np.pi / 6 + i * np.pi / 3
        nx = np.cos(theta)
        ny = np.sin(theta)
        projections = (points[:, 0] - cx) * nx + (points[:, 1] - cy) * ny
        m = np.max(projections)
        if m > max_proj:
            max_proj = m
    if max_proj <= 0:
        return 0.0
    return max_proj / inradius_factor

def evaluate_arrangement(inner):
    """Find minimal outer hexagon side for a given inner arrangement."""
    points = get_all_points(inner)
    # Optimize over center and rotation
    # Start with centroid and try several rotations
    cx0 = np.mean(points[:, 0])
    cy0 = np.mean(points[:, 1])
    
    best_side = float('inf')
    best_cx, best_cy, best_angle = cx0, cy0, 0.0
    
    # Try a grid of rotations
    for angle_deg in np.linspace(0, 60, 13):
        # For this rotation, optimize center
        # The optimal center for a given rotation minimizes max of projections
        # This is a linear programming problem, but we can approximate
        # Try centroid and nearby points
        for dx in np.linspace(-1, 1, 5):
            for dy in np.linspace(-1, 1, 5):
                cx = cx0 + dx
                cy = cy0 + dy
                s = min_outer_side(points, cx, cy, angle_deg)
                if s < best_side:
                    best_side = s
                    best_cx, best_cy, best_angle = cx, cy, angle_deg
    
    return best_side, (best_cx, best_cy), best_angle

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    # Generate hexagonal lattice arrangements
    # Hexagonal lattice: positions at (i + j/2, j*sqrt(3)/2) for integer i,j
    # but for hexagons, the spacing is 2 (diameter) in the appropriate directions
    
    # Actually, for unit hexagons to be disjoint, centers must be at least 2 apart
    # on a hexagonal lattice, the nearest neighbor distance is 2
    
    # Let's generate candidate sets of 12 lattice points
    # Hexagonal lattice basis: e1 = (2, 0), e2 = (1, sqrt(3))
    # Points: i*e1 + j*e2 for integers i,j
    
    sqrt3 = np.sqrt(3)
    e1 = np.array([2.0, 0.0])
    e2 = np.array([1.0, sqrt3])
    
    # Generate lattice points in a region
    lattice_points = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            p = i * e1 + j * e2
            lattice_points.append(p)
    lattice_points = np.array(lattice_points)
    
    # Find all subsets of 12 that form "nice" clusters
    # Strategy: use hexagonal number patterns
    # For 12 hexagons, try:
    # 1. A 3x4 rectangular block on hex lattice
    # 2. A hexagonal cluster
    # 3. Other compact arrangements
    
    best_result = None
    best_score = float('inf')
    
    # Define several candidate arrangements
    arrangements = []
    
    # Arrangement 1: Hexagonal cluster - center + ring of 6 + 5 more
    # Actually, let's use a systematic approach: try all connected subsets of 12
    # from the lattice that are "compact"
    
    # Simpler: try specific known-good patterns
    # Pattern A: 3 rows of 4
    pattern_a = []
    for row in range(3):
        for col in range(4):
            i = col - 1
            j = row - 1
            p = i * e1 + j * e2
            pattern_a.append((p[0], p[1], 0.0))
    arrangements.append(pattern_a)
    
    # Pattern B: Hexagonal shape - 1+3+4+3+1 = 12? No, 1+3+5+3 = 12
    # Hexagonal numbers: 1, 7, 19... 
    # For 12: try a hexagon of side 2 (7 hexagons) + 5 more
    # Or: rows of 2,3,4,3 = 12
    pattern_b = []
    rows = [2, 3, 4, 3]
    for row_idx, count in enumerate(rows):
        center_row = row_idx - 1.5  # center the rows
        for col in range(count):
            i = col - (count - 1) / 2
            j = row_idx - 1.5
            p = i * e1 + j * e2
            pattern_b.append((p[0], p[1], 0.0))
    arrangements.append(pattern_b)
    
    # Pattern C: 4 rows of 3
    pattern_c = []
    for row in range(4):
        for col in range(3):
            i = col - 1
            j = row - 1.5
            p = i * e1 + j * e2
            pattern_c.append((p[0], p[1], 0.0))
    arrangements.append(pattern_c)
    
    # Pattern D: More hexagonal - 3,4,4,1 or similar
    # Let's try: center at origin, then add neighbors
    # Hex lattice neighbors of (0,0): 6 points at distance 2
    # Then next ring: 12 points at distance 2*sqrt(3) or 4
    
    # Let me just try all subsets of size 12 from a small set of lattice points
    # that are within a reasonable bounding box
    
    # Generate more lattice points
    all_lattice = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            p = i * e1 + j * e2
            if np.abs(p[0]) <= 8 and np.abs(p[1]) <= 8:
                all_lattice.append(tuple(p))
    
    all_lattice = list(set(all_lattice))
    
    # Try to find compact arrangements by greedy or specific patterns
    # For now, evaluate the patterns we have
    
    for arr in arrangements:
        if time.time() > deadline:
            break
        side, center, angle = evaluate_arrangement(arr)
        if side < best_score:
            best_score = side
            best_result = (arr, center, side, angle)
    
    # If we have time, try more arrangements
    if time.time() < deadline - 10:
        # Try shifted and rotated versions
        for base_arr in arrangements[:]:
            for shift_x in np.linspace(-1, 1, 3):
                for shift_y in np.linspace(-1, 1, 3):
                    shifted = [[x + shift_x, y + shift_y, a] for (x, y, a) in base_arr]
                    if time.time() > deadline:
                        break
                    side, center, angle = evaluate_arrangement(shifted)
                    if side < best_score:
                        best_score = side
                        best_result = (shifted, center, side, angle)
    
    if best_result is None:
        # Fallback: just place them in a line
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    inner, center, side, angle = best_result
    return inner, center, side, angle

