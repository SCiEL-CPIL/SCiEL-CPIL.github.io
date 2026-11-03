"""Unit regular hexagons inside a regular hexagon - optimized packing."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 11

def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon with center (cx,cy), side length, rotation in degrees."""
    theta = np.radians(angle_deg)
    # Regular hexagon vertices at angles 0, 60, ..., 300 degrees
    angles = np.arange(6) * np.pi / 3.0 + theta
    xs = cx + side * np.cos(angles)
    ys = cy + side * np.sin(angles)
    return np.column_stack([xs, ys])

def _outer_hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of outer regular hexagon."""
    return _hex_vertices(cx, cy, side, angle_deg)

def _point_in_hexagon(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside regular hexagon with center (cx,cy), side, rotation."""
    # Transform to hexagon's local frame
    theta = np.radians(angle_deg)
    dx = px - cx
    dy = py - cy
    # Rotate by -theta
    cos_t = np.cos(-theta)
    sin_t = np.sin(-theta)
    lx = dx * cos_t - dy * sin_t
    ly = dx * sin_t + dy * cos_t
    # In local frame, hexagon has vertices at angles 0,60,...,300 with radius=side
    # The hexagon is defined by |lx*cos(60k) + ly*sin(60k)| <= side*cos(30deg) for k=0,1,2
    # Actually for a regular hexagon with side s, the apothem (inradius) = s * sqrt(3)/2
    # The hexagon can be described as intersection of 3 strips:
    # |x| <= s*cos(30) = s*sqrt(3)/2
    # |x/2 + y*sqrt(3)/2| <= s*sqrt(3)/2
    # |x/2 - y*sqrt(3)/2| <= s*sqrt(3)/2
    # Wait, let me reconsider. A regular hexagon with side length s has:
    # - Circumradius = s
    # - Inradius (apothem) = s * sqrt(3)/2
    # In local coordinates with vertices at 0, 60, ..., 300 degrees:
    # The hexagon is the set of points where for each of the 6 edge normals,
    # the projection is <= apothem.
    # Edge normals are at angles 30, 90, 150, 210, 270, 330 degrees
    # So constraints: |lx*cos(30+60k) + ly*sin(30+60k)| <= s*sqrt(3)/2 for k=0,1,2
    apothem = side * np.sqrt(3) / 2.0
    for k in range(3):
        ang = np.radians(30 + 60 * k)
        val = lx * np.cos(ang) + ly * np.sin(ang)
        if abs(val) > apothem + 1e-10:
            return False
    return True

def _hexagon_distance(v1, v2):
    """Compute minimum distance between two convex polygons (hexagons) given as vertex arrays.
    Returns the minimum distance between any point in poly1 and any point in poly2.
    If they overlap, returns a negative value indicating penetration depth (approximate)."""
    # For two convex polygons, min distance can be found by checking:
    # 1. If any vertex of one is inside the other -> negative (overlap)
    # 2. Min distance between all edge pairs
    # This is a simplified approach - for hexagons we can use the separating axis theorem
    # but for optimization we need a smooth-ish objective.
    
    # Simple approach: compute min distance between all vertex-edge pairs and vertex-vertex pairs
    min_d = np.inf
    
    # Check if any vertex of v1 is inside v2 or vice versa (approximate via point-in-poly)
    # For simplicity in optimization, we'll use a different approach:
    # Compute pairwise distances between all vertices, and vertex-to-edge distances
    
    for i in range(6):
        p1 = v1[i]
        # Distance from p1 to each edge of v2
        for j in range(6):
            p2a = v2[j]
            p2b = v2[(j + 1) % 6]
            # Point-to-segment distance
            d = _point_to_segment_dist(p1, p2a, p2b)
            if d < min_d:
                min_d = d
        
        p2 = v2[i]
        for j in range(6):
            p1a = v1[j]
            p1b = v1[(j + 1) % 6]
            d = _point_to_segment_dist(p2, p1a, p1b)
            if d < min_d:
                min_d = d
    
    return min_d

def _point_to_segment_dist(p, a, b):
    """Distance from point p to segment ab."""
    ab = b - a
    t = np.dot(p - a, ab) / (np.dot(ab, ab) + 1e-15)
    t = np.clip(t, 0.0, 1.0)
    proj = a + t * ab
    return np.linalg.norm(p - proj)

def _check_no_overlap(inner_params):
    """Check that no two inner hexagons overlap. inner_params: list of (x, y, angle_deg)."""
    n = len(inner_params)
    for i in range(n):
        v1 = _hex_vertices(inner_params[i][0], inner_params[i][1], 1.0, inner_params[i][2])
        for j in range(i + 1, n):
            v2 = _hex_vertices(inner_params[j][0], inner_params[j][1], 1.0, inner_params[j][2])
            # Check if any vertex of v1 is inside v2 or vice versa
            for k in range(6):
                if _point_in_hexagon(v1[k, 0], v1[k, 1], inner_params[j][0], inner_params[j][1], 1.0, inner_params[j][2]):
                    return False
                if _point_in_hexagon(v2[k, 0], v2[k, 1], inner_params[i][0], inner_params[i][1], 1.0, inner_params[i][2]):
                    return False
    return True

def _compute_min_enclosing_hex_side(inner_params):
    """Compute the minimum side length of a regular hexagon that encloses all inner hexagons.
    We optimize over the outer hexagon's center, rotation, and side length.
    Returns the minimum side length found."""
    # For each inner hexagon, compute all 6 vertices
    all_points = []
    for (x, y, ang) in inner_params:
        verts = _hex_vertices(x, y, 1.0, ang)
        all_points.append(verts)
    all_points = np.vstack(all_points)
    
    # To find minimum enclosing regular hexagon:
    # The outer hexagon has center (cx, cy), rotation theta, side s.
    # A point (px, py) is inside if for all k in 0..5:
    #   (px-cx)*cos(30+60k + theta) + (py-cy)*sin(30+60k + theta) <= s*sqrt(3)/2
    # We want to minimize s such that all points satisfy these constraints.
    # This is a linear programming problem for fixed (cx, cy, theta).
    
    # Strategy: try different rotations and centers, use LP to find min s
    # But this is expensive. Instead, use a heuristic:
    # The minimum enclosing hexagon's center is near the centroid of all points.
    # Try several rotations and compute required s.
    
    cx0 = np.mean(all_points[:, 0])
    cy0 = np.mean(all_points[:, 1])
    
    best_s = np.inf
    for theta_deg in range(0, 360, 15):
        theta = np.radians(theta_deg)
        # For each of the 6 face normals (at angles 30+60k+theta), compute max projection
        s_required = 0.0
        for k in range(3):
            ang = np.radians(30 + 60 * k) + theta
            # Project all points onto this normal
            projections = (all_points[:, 0] - cx0) * np.cos(ang) + (all_points[:, 1] - cy0) * np.sin(ang)
            # Also need the opposite direction
            max_proj = np.max(np.abs(projections))
            # s*sqrt(3)/2 >= max_proj => s >= 2*max_proj/sqrt(3)
            s_req = 2.0 * max_proj / np.sqrt(3.0)
            s_required = max(s_required, s_req)
        best_s = min(best_s, s_required)
    
    return best_s

def _objective(params):
    """Objective: minimize the outer hexagon side length, with penalty for overlaps."""
    # params: flattened array of [x0,y0,a0, x1,y1,a1, ..., x10,y10,a10]
    inner_params = [(params[i*3], params[i*3+1], params[i*3+2]) for i in range(N)]
    
    # Check overlap penalty
    overlap_penalty = 0.0
    for i in range(N):
        v1 = _hex_vertices(inner_params[i][0], inner_params[i][1], 1.0, inner_params[i][2])
        for j in range(i+1, N):
            v2 = _hex_vertices(inner_params[j][0], inner_params[j][1], 1.0, inner_params[j][2])
            # Check vertex containment
            for k in range(6):
                if _point_in_hexagon(v1[k,0], v1[k,1], inner_params[j][0], inner_params[j][1], 1.0, inner_params[j][2]):
                    overlap_penalty += 100.0
                if _point_in_hexagon(v2[k,0], v2[k,1], inner_params[i][0], inner_params[i][1], 1.0, inner_params[i][2]):
                    overlap_penalty += 100.0
            # Also check edge-edge intersection via min distance
            d = _hexagon_distance(v1, v2)
            if d < 0.01:
                overlap_penalty += (0.01 - d) * 50.0
    
    if overlap_penalty > 1e-6:
        return overlap_penalty
    
    s = _compute_min_enclosing_hex_side(inner_params)
    return s

def _make_initial_hex_lattice():
    """Create a hexagonal lattice arrangement of 11 unit hexagons.
    Central hexagon + ring of 6 + 4 more attached to alternating edges."""
    # Hexagonal lattice spacing: for unit hexagons (side=1), the distance between
    # centers of adjacent hexagons in a honeycomb is 2*apothem = sqrt(3)
    apothem = np.sqrt(3) / 2.0
    center_dist = 2 * apothem  # = sqrt(3)
    
    positions = []
    # Center
    positions.append((0.0, 0.0, 0.0))
    # Ring of 6
    for k in range(6):
        ang = np.radians(60 * k)
        x = center_dist * np.cos(ang)
        y = center_dist * np.sin(ang)
        positions.append((x, y, 0.0))
    # 4 more: attach to edges of the ring hexagons (alternating)
    # For hexagons at angles 0, 120, 240 (every other one), add one more outward
    for k in [0, 2, 4]:
        ang = np.radians(60 * k)
        x = 2 * center_dist * np.cos(ang)
        y = 2 * center_dist * np.sin(ang)
        positions.append((x, y, 0.0))
    # One more: attach to hexagon at angle 180
    positions.append((-2 * center_dist, 0.0, 0.0))
    
    return positions

def _make_initial_compact():
    """Create a more compact initial arrangement."""
    # Try a 3-4-3 or similar arrangement
    apothem = np.sqrt(3) / 2.0
    dx = 2 * apothem  # horizontal spacing
    dy = 1.5  # vertical spacing for staggered rows (hexagon height = 2*side = 2, but staggered)
    # Actually for unit hexagons with flat top:
    # Width = 2*side = 2, Height = sqrt(3)*side = sqrt(3)
    # For pointy top: Width = sqrt(3)*side, Height = 2*side
    # Let's use flat-top orientation (angle=0 means vertex at 0 degrees)
    # Hexagon with vertex at angle 0: vertices at 0, 60, 120, 180, 240, 300
    # Width (horizontal) = 2*side = 2
    # Height (vertical) = sqrt(3)*side = sqrt(3)
    
    # For flat-top hexagons in a grid:
    # Horizontal spacing = 1.5 * side = 1.5 (for staggered)
    # Vertical spacing = sqrt(3) * side = sqrt(3)
    # Odd columns shifted by sqrt(3)/2
    
    positions = []
    # Row 0: 3 hexagons
    for i in range(3):
        positions.append((i * 1.5, 0.0, 0.0))
    # Row 1: 4 hexagons, shifted
    for i in range(4):
        positions.append((i * 1.5 + 0.75, np.sqrt(3), 0.0))
    # Row 2: 3 hexagons
    for i in range(3):
        positions.append((i * 1.5, 2 * np.sqrt(3), 0.0))
    
    return positions

def _make_initial_tight_ring():
    """Tight ring configuration: center + 6 around + 4 more in gaps."""
    apothem = np.sqrt(3) / 2.0
    d = 2 * apothem  # center-to-center distance for touching hexagons = sqrt(3)
    
    positions = [(0.0, 0.0, 0.0)]
    for k in range(6):
        ang = np.radians(60 * k)
        positions.append((d * np.cos(ang), d * np.sin(ang), 0.0))
    # Add 4 more in the "gaps" between ring hexagons, slightly outward
    for k in range(6):
        ang = np.radians(60 * k + 30)  # 30, 90, 150, 210, 270, 330
        # Place at distance that touches two adjacent ring hexagons
        # The gap position is at distance d / cos(30) from center? No...
        # Two adjacent ring hexagons are at angles 60k and 60(k+1), distance d from center.
        # The point equidistant from both, at distance d from each:
        # This forms an equilateral triangle with the two ring centers.
        # The third vertex is at distance d from each, at angle 60k+30
        # Distance from origin = d * cos(30) / cos(30) ... let me compute
        # Two points at (d*cos(60k), d*sin(60k)) and (d*cos(60(k+1)), d*sin(60(k+1)))
        # Midpoint is at distance d*cos(30) from origin, at angle 60k+30
        # The third vertex of equilateral triangle is at distance d from both
        # It's at angle 60k+30, at distance d from origin? Let's verify:
        # Distance from (d, 0) to (d*cos(60), d*sin(60)) = d (equilateral)
        # So the third vertex at angle 30, distance d from origin:
        # Distance to (d,0): sqrt(d^2 + d^2 - 2d^2*cos(30)) = d*sqrt(2-2cos30) != d
        # Actually for equilateral triangle with side d, if two vertices are at distance d from origin
        # and angle 60 apart, the third vertex is at distance d from origin too.
        # Check: vertices at angles 0 and 60, both at distance d.
        # Third vertex at angle 30, distance d.
        # Dist from (d,0) to (d*cos30, d*sin30) = sqrt(d^2(1-cos30)^2 + d^2 sin^2(30))
        # = d * sqrt((1-cos30)^2 + sin^2(30)) = d*sqrt(1 - 2cos30 + cos^2(30) + sin^2(30))
        # = d*sqrt(2 - 2cos30) = d*sqrt(2-2*0.866) = d*sqrt(0.268) = 0.517d != d
        # So it's NOT at distance d. The equilateral triangle's third vertex:
        # For two points A, B at distance d from origin, angle 60 apart:
        # |AB| = d (since triangle OAB is equilateral when angle AOB=60 and OA=OB=d)
        # So AB = d. Third vertex C of equilateral triangle on AB:
        # C is at distance d from A and B.
        # The two possible positions: one closer to origin, one farther.
        # The farther one is at distance d*sqrt(3) from origin? Let me compute properly.
        # A = (d, 0), B = (d/2, d*sqrt(3)/2)
        # Midpoint M = (3d/4, d*sqrt(3)/4)
        # AB direction = (-d/2, d*sqrt(3)/2), unit = (-1/2, sqrt(3)/2)
        # Perpendicular = (sqrt(3)/2, 1/2) or (-sqrt(3)/2, -1/2)
        # Height of equilateral triangle = d*sqrt(3)/2
        # C = M + (d*sqrt(3)/2) * (sqrt(3)/2, 1/2) = (3d/4 + 3d/4, d*sqrt(3)/4 + d*sqrt(3)/4)
        #   = (3d/2, d*sqrt(3)/2)
        # Distance from origin: sqrt(9d^2/4 + 3d^2/4) = sqrt(12d^2/4) = sqrt(3)d
        # So the outer gap positions are at distance sqrt(3)*d from origin, at angles 30, 90, ...
        # But we only need 4 more, so pick 4 of the 6 gaps.
        pass
    
    # Simpler: just add 4 at the outer gap positions
    gap_dist = np.sqrt(3) * d
    for k in [0, 1, 2, 3]:
        ang = np.radians(60 * k + 30)
        positions.append((gap_dist * np.cos(ang), gap_dist * np.sin(ang), 0.0))
    
    return positions

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_result = None
    best_score = np.inf
    
    # Generate initial configurations
    inits = [
        _make_initial_hex_lattice(),
        _make_initial_compact(),
        _make_initial_tight_ring(),
    ]
    
    # Add some random perturbations
    rng = np.random.default_rng(42)
    for _ in range(5):
        base = _make_initial_compact()
        perturbed = []
        for (x, y, a) in base:
            perturbed.append((
                x + rng.normal(0, 0.3),
                y + rng.normal(0, 0.3),
                a + rng.uniform(-15, 15)
            ))
        inits.append(perturbed)
    
    for init in inits:
        if time.time() > deadline:
            break
        
        params0 = np.array([v for p in init for v in p])
        
        # Use differential evolution for global search (fast, handles non-smooth)
        bounds = [(-10, 10)] * (N * 3)
        
        try:
            result = differential_evolution(
                _objective, bounds,
                seed=42, maxiter=30, tol=1e-4,
                popsize=15, init='sobol',
                polish=False, workers=1
            )
            if time.time() > deadline:
                break
            
            score = result.fun
            if score < best_score:
                best_score = score
                best_params = result.x
                best_result = [(best_params[i*3], best_params[i*3+1], best_params[i*3+2]) for i in range(N)]
        except Exception:
            pass
    
    # Polish with L-BFGS-B if we have a good result
    if best_result is not None and time.time() < deadline - 10:
        try:
            params0 = np.array([v for p in best_result for v in p])
            result = minimize(
                _objective, params0, method='L-BFGS-B',
                options={'maxiter': 50, 'ftol': 1e-6}
            )
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x
                best_result = [(best_params[i*3], best_params[i*3+1], best_params[i*3+2]) for i in range(N)]
        except Exception:
            pass
    
    # If no valid result found, fall back to a simple valid configuration
    if best_result is None:
        # Just use the initial compact arrangement
        best_result = _make_initial_compact()
        best_score = _compute_min_enclosing_hex_side(best_result)
    
    # Compute the outer hexagon parameters
    # Find the minimum enclosing hexagon for the final arrangement
    all_points = []
    for (x, y, ang) in best_result:
        verts = _hex_vertices(x, y, 1.0, ang)
        all_points.append(verts)
    all_points = np.vstack(all_points)
    
    cx0 = np.mean(all_points[:, 0])
    cy0 = np.mean(all_points[:, 1])
    
    best_s = np.inf
    best_theta = 0.0
    for theta_deg in range(0, 360, 5):
        theta = np.radians(theta_deg)
        s_required = 0.0
        for k in range(3):
            ang = np.radians(30 + 60 * k) + theta
            projections = (all_points[:, 0] - cx0) * np.cos(ang) + (all_points[:, 1] - cy0) * np.sin(ang)
            max_proj = np.max(np.abs(projections))
            s_req = 2.0 * max_proj / np.sqrt(3.0)
            s_required = max(s_required, s_req)
        if s_required < best_s:
            best_s = s_required
            best_theta = theta_deg
    
    # Refine theta
    for theta_deg in np.linspace(best_theta - 5, best_theta + 5, 50):
        theta = np.radians(theta_deg)
        s_required = 0.0
        for k in range(3):
            ang = np.radians(30 + 60 * k) + theta
            projections = (all_points[:, 0] - cx0) * np.cos(ang) + (all_points[:, 1] - cy0) * np.sin(ang)
            max_proj = np.max(np.abs(projections))
            s_req = 2.0 * max_proj / np.sqrt(3.0)
            s_required = max(s_required, s_req)
        if s_required < best_s:
            best_s = s_required
            best_theta = theta_deg
    
    # Also try optimizing center
    for dcx in np.linspace(-0.5, 0.5, 11):
        for dcy in np.linspace(-0.5, 0.5, 11):
            cx = cx0 + dcx
            cy = cy0 + dcy
            theta = np.radians(best_theta)
            s_required = 0.0
            for k in range(3):
                ang = np.radians(30 + 60 * k) + theta
                projections = (all_points[:, 0] - cx) * np.cos(ang) + (all_points[:, 1] - cy) * np.sin(ang)
                max_proj = np.max(np.abs(projections))
                s_req = 2.0 * max_proj / np.sqrt(3.0)
                s_required = max(s_required, s_req)
            if s_required < best_s:
                best_s = s_required
                cx0 = cx
                cy0 = cy
    
    inner = [[float(x), float(y), float(a)] for (x, y, a) in best_result]
    return inner, (float(cx0), float(cy0)), float(best_s), float(best_theta)

# EVOLVE-BLOCK-START
# (construct is defined above; this block is for compatibility)
# EVOLVE-BLOCK-END
