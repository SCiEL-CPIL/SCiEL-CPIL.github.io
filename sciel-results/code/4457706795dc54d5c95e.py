"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import itertools

N = 11

# EVOLVE-BLOCK-START
def _hexagon_vertices(cx, cy, side, angle_rad):
    """Return vertices of a regular hexagon centered at (cx,cy) with given side and rotation."""
    angles = angle_rad + np.pi / 6 * np.arange(6)
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def _polygon_area(vertices):
    """Shoelace formula for polygon area."""
    x = vertices[:, 0]
    y = vertices[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

def _point_in_polygon(pt, vertices):
    """Ray casting point-in-polygon test."""
    x, y = pt
    n = len(vertices)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = vertices[i]
        xj, yj = vertices[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside

def _polygon_distance(p1, p2):
    """Minimum distance between two convex polygons (hexagons). Returns 0 if they overlap."""
    # Check if any vertex of p1 is inside p2 or vice versa
    for v in p1:
        if _point_in_polygon(v, p2):
            return 0.0
    for v in p2:
        if _point_in_polygon(v, p1):
            return 0.0
    
    # Check edge-edge distances
    min_dist = np.inf
    for i in range(6):
        a1 = p1[i]
        a2 = p1[(i + 1) % 6]
        for j in range(6):
            b1 = p2[j]
            b2 = p2[(j + 1) % 6]
            # Segment-segment distance
            d = _segment_distance(a1, a2, b1, b2)
            if d < min_dist:
                min_dist = d
    return min_dist

def _segment_distance(p1, p2, p3, p4):
    """Minimum distance between two line segments."""
    # Check if segments intersect
    if _segments_intersect(p1, p2, p3, p4):
        return 0.0
    # Distance from each endpoint to the other segment
    d1 = _point_segment_distance(p1, p3, p4)
    d2 = _point_segment_distance(p2, p3, p4)
    d3 = _point_segment_distance(p3, p1, p2)
    d4 = _point_segment_distance(p4, p1, p2)
    return min(d1, d2, d3, d4)

def _segments_intersect(p1, p2, p3, p4):
    """Check if two segments intersect."""
    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])
    return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)

def _point_segment_distance(p, a, b):
    """Distance from point p to segment ab."""
    ab = b - a
    ap = p - a
    ab_len_sq = np.dot(ab, ab)
    if ab_len_sq == 0:
        return np.linalg.norm(p - a)
    t = np.dot(ap, ab) / ab_len_sq
    t = max(0, min(1, t))
    proj = a + t * ab
    return np.linalg.norm(p - proj)

def _min_dist_to_boundary(inner_verts, outer_verts):
    """Minimum distance from inner polygon to outer boundary (outer is containment)."""
    min_dist = np.inf
    for v in inner_verts:
        # Distance from vertex to each edge of outer
        for i in range(6):
            a = outer_verts[i]
            b = outer_verts[(i + 1) % 6]
            d = _point_segment_distance(v, a, b)
            if d < min_dist:
                min_dist = d
    return min_dist

def _compute_min_clearance(params, n_inner):
    """
    Compute minimum clearance for a given parameter vector.
    params: [x1, y1, theta1, x2, y2, theta2, ..., xn, yn, thetan, cx, cy, side]
    Returns the minimum clearance (can be negative if overlapping).
    """
    inner_positions = []
    for i in range(n_inner):
        x = params[3 * i]
        y = params[3 * i + 1]
        theta = params[3 * i + 2]
        inner_positions.append((x, y, theta))
    
    cx = params[3 * n_inner]
    cy = params[3 * n_inner + 1]
    side = params[3 * n_inner + 2]
    
    # Compute all inner hexagon vertices
    inner_hexes = []
    for (x, y, theta) in inner_positions:
        verts = _hexagon_vertices(x, y, 1.0, theta)
        inner_hexes.append(verts)
    
    # Outer hexagon
    outer_verts = _hexagon_vertices(cx, cy, side, 0.0)
    
    min_clearance = np.inf
    
    # Check inner-inner distances
    for i in range(n_inner):
        for j in range(i + 1, n_inner):
            d = _polygon_distance(inner_hexes[i], inner_hexes[j])
            if d < min_clearance:
                min_clearance = d
    
    # Check inner-outer distances
    for i in range(n_inner):
        d = _min_dist_to_boundary(inner_hexes[i], outer_verts)
        if d < min_clearance:
            min_clearance = d
    
    return min_clearance

def _objective(params, n_inner):
    """
    Objective: minimize outer side length, penalize violations.
    We use a penalty approach: minimize side + penalty * max(0, -min_clearance)^2
    """
    min_clearance = _compute_min_clearance(params, n_inner)
    side = params[3 * n_inner + 2]
    
    # Penalty for negative clearance (overlap)
    penalty = 0.0
    if min_clearance < 0:
        penalty = 100.0 * min_clearance ** 2
    
    return side + penalty

def construct():
    """
    Construct packing of N unit hexagons inside a regular hexagon.
    Returns: inner (list of [x, y, angle_degrees]), center (x, y), side, angle
    """
    n_inner = N
    
    # Initial guess: hexagonal lattice arrangement
    # For a hexagonal packing, the optimal arrangement for n hexagons
    # in a hexagonal container has a known structure
    
    # Start with a simple grid-like arrangement
    # For N=11, use a pattern: 1 in center, 6 around, 4 more
    # Actually, let's use a honeycomb pattern
    
    # Initial positions in a hexagonal lattice
    # Hex lattice spacing: for unit hexagons, center-to-center distance is 2*sqrt(3)/2 * 2 = 2*sqrt(3) ≈ 3.464
    # Wait, for regular hexagons of side 1, the distance between centers of adjacent hexagons in a honeycomb is 2*sqrt(3)/2 * 2? 
    # Actually, the distance between centers of two adjacent unit hexagons sharing an edge is 2 * apothem = 2 * (sqrt(3)/2) = sqrt(3) ≈ 1.732
    # No wait. The apothem of a unit hexagon is sqrt(3)/2. Two hexagons sharing an edge have centers 2*apothem = sqrt(3) apart.
    
    # Let me reconsider. For regular hexagons of side length 1:
    # - Circumradius = 1
    # - Apothem = sqrt(3)/2
    # - Distance between centers of adjacent hexagons (sharing an edge) = 2 * apothem = sqrt(3)
    
    # Honeycomb lattice vectors:
    # v1 = (sqrt(3), 0)  -- actually no
    # In a honeycomb, the lattice is triangular with spacing sqrt(3)
    # v1 = (sqrt(3), 0)
    # v2 = (sqrt(3)/2, 3/2)
    
    # Let's place hexagons in a small cluster
    # For N=11, a good initial guess is a hexagonal arrangement
    # Center + 6 neighbors + 4 more
    
    sqrt3 = np.sqrt(3)
    
    # Hexagonal cluster positions
    # Ring 0: center
    # Ring 1: 6 neighbors at distance sqrt(3)
    # Ring 2: 12 neighbors at distance 2*sqrt(3), but we only need 4 more
    
    positions = []
    # Center
    positions.append((0.0, 0.0, 0.0))
    
    # Ring 1: 6 hexagons
    for k in range(6):
        angle = k * np.pi / 3
        x = sqrt3 * np.cos(angle)
        y = sqrt3 * np.sin(angle)
        positions.append((x, y, 0.0))
    
    # Ring 2: we need 4 more. Place them at alternating positions in ring 2
    # Ring 2 positions are at distance 2*sqrt(3) from center, at angles 0, 30, 60, ... degrees
    # Actually ring 2 in honeycomb has 12 positions
    for k in range(0, 12, 3):  # Pick every 3rd
        if len(positions) >= n_inner:
            break
        angle = k * np.pi / 6  # 30 degree increments
        # But ring 2 is actually at 2*sqrt(3) distance
        # Hmm, let me just use a simpler initial guess
        pass
    
    # Let me just use a simple linear arrangement as initial guess and let optimization work
    # Actually, let's use a better initial guess
    
    # For N hexagons, a hexagonal number arrangement:
    # 1, 7, 19, 37, ...
    # N=11 is between 7 and 19
    
    # Let's use: center (1), ring 1 (6), and 4 from ring 2
    # Ring 2 positions: 12 positions at distance 2*sqrt(3)
    # The angles for ring 2 are at 0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330 degrees
    # But actually in a honeycomb, ring 2 is not a simple circle
    
    # Let me just initialize with a grid and let the optimizer figure it out
    
    # Simple initial guess: place hexagons in a line
    spacing = 2.0  # Slightly more than sqrt(3) to avoid initial overlap
    initial_params = []
    for i in range(n_inner):
        x = (i - n_inner // 2) * spacing
        y = 0.0
        theta = 0.0
        initial_params.extend([x, y, theta])
    
    # Outer hexagon initial guess
    cx = 0.0
    cy = 0.0
    # Estimate side length: for N hexagons in a line of length L, outer hex needs to contain them
    # L = (n_inner-1)*spacing + 2 (for the hexagons themselves)
    # For a hexagon containing a line of length L along one direction, side ≈ L/2 + something
    side_est = (n_inner - 1) * spacing / 2 + 3.0
    initial_params.extend([cx, cy, side_est])
    
    initial_params = np.array(initial_params)
    
    # Bounds: positions can be anywhere reasonable, side must be positive
    bounds = []
    for i in range(n_inner):
        bounds.append((-20, 20))  # x
        bounds.append((-20, 20))  # y
        bounds.append((-np.pi, np.pi))  # angle
    bounds.append((-20, 20))  # cx
    bounds.append((-20, 20))  # cy
    bounds.append((1.0, 50.0))  # side
    
    # Optimize
    best_result = None
    best_side = np.inf
    
    # Try multiple starting points
    n_starts = 3
    for start in range(n_starts):
        if start == 0:
            params0 = initial_params.copy()
        elif start == 1:
            # Perturbed version
            params0 = initial_params.copy()
            np.random.seed(start)
            noise = np.random.normal(0, 0.5, size=params0.shape)
            params0 += noise
        else:
            # Different arrangement: hexagonal cluster
            params0 = np.zeros_like(initial_params)
            # Place in hexagonal pattern
            pos_list = [(0, 0, 0)]
            for k in range(6):
                angle = k * np.pi / 3
                pos_list.append((sqrt3 * np.cos(angle), sqrt3 * np.sin(angle), 0))
            # Add 4 more at ring 2
            for k in range(4):
                angle = k * np.pi / 3 + np.pi / 6
                pos_list.append((2 * sqrt3 * np.cos(angle), 2 * sqrt3 * np.sin(angle), 0))
            for i, (x, y, theta) in enumerate(pos_list[:n_inner]):
                params0[3*i] = x
                params0[3*i+1] = y
                params0[3*i+2] = theta
            params0[3*n_inner] = 0.0
            params0[3*n_inner+1] = 0.0
            params0[3*n_inner+2] = 5.0
        
        result = minimize(
            _objective,
            params0,
            args=(n_inner,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        if result.success or result.fun < best_side:
            side_val = result.x[3 * n_inner + 2]
            clearance = _compute_min_clearance(result.x, n_inner)
            if clearance >= -1e-6:  # Allow small numerical tolerance
                if side_val < best_side:
                    best_side = side_val
                    best_result = result.x
    
    if best_result is None:
        # Fallback to initial guess
        best_result = initial_params
        best_side = initial_params[3 * n_inner + 2]
    
    # Extract results
    inner = []
    for i in range(n_inner):
        x = best_result[3 * i]
        y = best_result[3 * i + 1]
        theta_rad = best_result[3 * i + 2]
        theta_deg = np.degrees(theta_rad) % 360.0
        inner.append([float(x), float(y), float(theta_deg)])
    
    cx = float(best_result[3 * n_inner])
    cy = float(best_result[3 * n_inner + 1])
    side = float(best_result[3 * n_inner + 2])
    outer_angle = 0.0
    
    return inner, (cx, cy), side, outer_angle
# EVOLVE-BLOCK-END
