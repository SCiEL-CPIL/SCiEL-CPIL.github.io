"""Packing 12 unit hexagons inside a regular hexagon using grid-based construction."""
import math
import numpy as np

N = 12

def _hex_verts(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon with given center, side length, and rotation."""
    angle = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = angle + i * math.pi / 3
        verts.append([cx + side * math.cos(theta), cy + side * math.sin(theta)])
    return np.array(verts)

def _point_in_hexagon(p, center, side, angle_deg):
    """Check if point p is inside a regular hexagon."""
    # Transform to hexagon's local coordinates
    dx = p[0] - center[0]
    dy = p[1] - center[1]
    angle = math.radians(-angle_deg)
    rx = dx * math.cos(angle) - dy * math.sin(angle)
    ry = dx * math.sin(angle) + dy * math.cos(angle)
    
    # For a regular hexagon with side s, the inscribed circle radius (apothem) is s*sqrt(3)/2
    # The hexagon can be described as |x| <= s*sqrt(3)/2 and |y| <= s*sqrt(3)/2 with additional constraints
    # More precisely, in the rotated frame, the hexagon vertices are at (s,0), (s/2, s*sqrt(3)/2), etc.
    # The condition for being inside: max(|x|, |x/2 + y*sqrt(3)/2|, |-x/2 + y*sqrt(3)/2|) <= s*sqrt(3)/2 ... 
    # Actually simpler: the hexagon is the intersection of 6 half-planes.
    
    apothem = side * math.sqrt(3) / 2
    # The 6 edge normals in local coords (pointing outward)
    for i in range(6):
        theta = i * math.pi / 3
        nx = math.cos(theta)
        ny = math.sin(theta)
        if rx * nx + ry * ny > apothem + 1e-10:
            return False
    return True

def _hexagons_disjoint(h1, h2, tol=1e-9):
    """Check if two hexagons are disjoint using SAT."""
    v1 = _hex_verts(h1[0], h1[1], 1.0, h1[2])
    v2 = _hex_verts(h2[0], h2[1], 1.0, h2[2])
    
    all_edges = []
    for i in range(6):
        all_edges.append(v1[i] - v1[(i+1) % 6])
        all_edges.append(v2[i] - v2[(i+1) % 6])
    
    for edge in all_edges:
        normal = np.array([-edge[1], edge[0]])
        norm = np.linalg.norm(normal)
        if norm < 1e-12:
            continue
        normal /= norm
        proj1 = v1 @ normal
        proj2 = v2 @ normal
        if proj1.max() < proj2.min() - tol or proj2.max() < proj1.min() - tol:
            return True
    return False

def _all_disjoint(hexagons):
    """Check all pairs of hexagons are disjoint."""
    for i in range(len(hexagons)):
        for j in range(i+1, len(hexagons)):
            if not _hexagons_disjoint(hexagons[i], hexagons[j]):
                return False
    return True

def _min_enclosing_hexagon(hexagons, outer_angle=0.0):
    """Find minimum side length of outer hexagon (with given angle) that contains all inner hexagons.
    
    Returns (side_length, center_x, center_y)
    """
    # For a regular hexagon with side s and center (cx, cy), the hexagon is the set of points
    # where for each of 6 edge normals n_i: (p - c) · n_i <= s * sqrt(3)/2
    
    # We need to find (cx, cy, s) minimizing s such that all vertices of all inner hexagons
    # satisfy the containment constraint.
    
    # Collect all vertices
    all_verts = []
    for h in hexagons:
        verts = _hex_verts(h[0], h[1], 1.0, h[2])
        all_verts.extend(verts.tolist())
    all_verts = np.array(all_verts)
    
    # The 6 edge normals for the outer hexagon (pointing inward, so we use negative)
    # Actually, let's define: outer hexagon centered at (cx,cy) with side s, angle a
    # Vertex i is at (cx + s*cos(a + i*pi/3), cy + s*sin(a + i*pi/3))
    # Edge from vertex i to i+1, normal pointing outward
    # The constraint is that for all inner points p: (p - c) · n_i <= s*sqrt(3)/2 for all i
    
    # This is a linear programming problem in variables (cx, cy, s)
    # Minimize s subject to:
    # For each vertex v of each inner hexagon, and for each of 6 directions i:
    #   v · n_i - c · n_i <= s * sqrt(3)/2
    #   => v · n_i - c · n_i - s * sqrt(3)/2 <= 0
    
    # Let's use scipy's linprog or solve it analytically.
    # Actually, for a given outer angle, we can solve for the optimal center and side.
    
    # The constraint for point v and direction i is:
    # (v - c) · n_i <= s * sqrt(3)/2
    # => v · n_i - c · n_i <= s * sqrt(3)/2
    # => c · n_i + s * sqrt(3)/2 >= v · n_i
    
    # We want to minimize s. Let's think of it as: for a given s, is there a center c that works?
    # The constraints on c are: c · n_i >= v · n_i - s * sqrt(3)/2 for all v, i
    
    # This is a feasibility problem. We can binary search on s.
    
    # But we can also solve it directly. The optimal s is determined by the tightest constraint.
    
    # Let me use a different approach: the minimum enclosing hexagon with fixed orientation
    # can be found by noting that the center must be in the intersection of certain half-planes
    # and the side length is determined by the maximum extent.
    
    # For each of the 6 directions, the constraint is:
    # max over all vertices v of (v · n_i) - c · n_i <= s * sqrt(3)/2
    # => c · n_i >= max_v(v · n_i) - s * sqrt(3)/2
    
    # Similarly, for the opposite direction -n_i:
    # max over all vertices v of (v · (-n_i)) - c · (-n_i) <= s * sqrt(3)/2
    # => -max_v(v · n_i) + c · n_i <= s * sqrt(3)/2  ... wait let me redo
    
    # Direction i: n_i. Constraint: (v - c) · n_i <= s*sqrt(3)/2 for all v
    # => v · n_i - c · n_i <= s*sqrt(3)/2
    # => c · n_i >= v · n_i - s*sqrt(3)/2 for all v
    # => c · n_i >= max_v(v · n_i) - s*sqrt(3)/2
    
    # Direction i+3: n_{i+3} = -n_i. Constraint: (v - c) · (-n_i) <= s*sqrt(3)/2
    # => -v · n_i + c · n_i <= s*sqrt(3)/2
    # => c · n_i <= v · n_i + s*sqrt(3)/2 for all v
    # => c · n_i <= min_v(v · n_i) + s*sqrt(3)/2
    
    # So for each direction i:
    # max_v(v · n_i) - s*sqrt(3)/2 <= c · n_i <= min_v(v · n_i) + s*sqrt(3)/2
    # => max_v(v · n_i) - min_v(v · n_i) <= 2*s*sqrt(3)/2 = s*sqrt(3)
    # => s >= (max_v(v · n_i) - min_v(v · n_i)) / sqrt(3)
    
    # So the minimum s for a given orientation is:
    # s = max over i of (max_v(v · n_i) - min_v(v · n_i)) / sqrt(3)
    
    # And the center c must satisfy:
    # c · n_i >= max_v(v · n_i) - s*sqrt(3)/2
    # c · n_i <= min_v(v · n_i) + s*sqrt(3)/2
    # for all i
    
    # The center is the midpoint of the range for each direction:
    # c · n_i = (max_v(v · n_i) + min_v(v · n_i)) / 2
    
    # This gives a linear system for c.
    
    sqrt3 = math.sqrt(3)
    best_s = float('inf')
    best_cx, best_cy = 0, 0
    
    for outer_angle_deg in [0.0]:  # We'll try multiple angles
        n = math.radians(outer_angle_deg)
        # 6 edge normals (outward)
        normals = []
        for i in range(6):
            theta = n + i * math.pi / 3 + math.pi / 6  # Edge normals are at 30° offset from vertices
            normals.append((math.cos(theta), math.sin(theta)))
        
        # For each direction, compute max and min projection
        ranges = []
        for nx, ny in normals:
            projs = all_verts[:, 0] * nx + all_verts[:, 1] * ny
            ranges.append((projs.max(), projs.min()))
        
        s_candidate = max((r[0] - r[1]) / sqrt3 for r in ranges)
        
        if s_candidate < best_s:
            best_s = s_candidate
            # Find center
            # c · n_i = (max + min) / 2 for each i
            # We have 6 equations, 2 unknowns. Use least squares.
            A = np.array(normals)  # 6x2
            b = np.array([(r[0] + r[1]) / 2 for r in ranges])  # 6
            c, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            best_cx, best_cy = c[0], c[1]
    
    return best_s, best_cx, best_cy

def _is_valid(hexagons, outer_center, outer_side, outer_angle, tol=1e-7):
    """Check if the packing is valid."""
    # Check all hexagons are inside the outer hexagon
    for h in hexagons:
        verts = _hex_verts(h[0], h[1], 1.0, h[2])
        for v in verts:
            if not _point_in_hexagon(v, outer_center, outer_side, outer_angle):
                return False
    # Check all pairs are disjoint
    if not _all_disjoint(hexagons):
        return False
    return True

def _construct_grid_packing():
    """Construct a grid-like packing of 12 hexagons."""
    # Flat-top hexagons: vertices at angles 0, 60, 120, 180, 240, 300
    # For flat-top, the horizontal spacing between centers is 3/2 * side = 1.5
    # The vertical spacing is sqrt(3) * side = sqrt(3)
    # Adjacent rows are offset by 0.75 in x
    
    # Let's try a 3x4 grid or similar arrangement
    # For 12 hexagons, a natural arrangement is 3 rows of 4, or 4 rows of 3
    
    # Flat-top hexagon: width = 2*side = 2, height = sqrt(3)*side = sqrt(3)
    # Horizontal center-to-center distance for adjacent columns: 1.5
    # Vertical center-to-center distance for adjacent rows: sqrt(3)
    # Offset for alternating rows: 0.75
    
    hexagons = []
    
    # Try 4 columns x 3 rows
    for row in range(3):
        for col in range(4):
            x = col * 1.5 + (0.75 if row % 2 == 1 else 0)
            y = row * math.sqrt(3)
            hexagons.append((x, y, 0.0))  # angle 0 = flat-top
    
    return hexagons

def _construct_hexagonal_packing():
    """Construct a hexagonal cluster packing: 1 center + 6 ring + 5 more."""
    hexagons = []
    
    # Center
    hexagons.append((0, 0, 0.0))
    
    # First ring: 6 hexagons at distance 2 (for flat-top, adjacent centers are 2 apart? No)
    # For flat-top hexagons with side 1, the distance between centers of adjacent hexagons is 2
    # Actually, for two adjacent flat-top hexagons sharing an edge, the center-to-center distance is 2*apothem = 2*sqrt(3)/2 = sqrt(3)
    # Wait, let me reconsider. Two unit hexagons sharing an edge: the distance between centers is 2 * apothem = 2 * (sqrt(3)/2) = sqrt(3)
    
    # Hmm, but they need to be disjoint, not touching. So we need distance > sqrt(3)
    
    # For a hexagonal lattice of flat-top unit hexagons:
    # The lattice vectors are (3/2, sqrt(3)/2) and (0, sqrt(3))
    # No wait. Let me think again.
    
    # Flat-top hexagon vertices: (1,0), (1/2, sqrt(3)/2), (-1/2, sqrt(3)/2), (-1,0), (-1/2, -sqrt(3)/2), (1/2, -sqrt(3)/2)
    # The rightmost point is at x=1, leftmost at x=-1. Width = 2.
    # The topmost point is at y=sqrt(3)/2, bottommost at y=-sqrt(3)/2. Height = sqrt(3).
    
    # Two adjacent flat-top hexagons sharing a vertical edge: centers are 1.5 apart horizontally?
    # No. If they share the right edge of one and left edge of the other, the edge is at x=1 for the first and x=-1 for the second.
    # So centers are at 0 and 2? No, that's not right either.
    
    # Let me just use the fact that for a hexagonal tiling, the center-to-center distance is 2*side = 2 for hexagons of side 1.
    # Actually no. For regular hexagons of side s, the distance between centers in a tiling is 2*s*sqrt(3)/2 = s*sqrt(3).
    
    # Let me verify: two hexagons sharing an edge. The edge has length s. The distance from center to edge midpoint is the apothem = s*sqrt(3)/2.
    # So center-to-center distance = 2 * s*sqrt(3)/2 = s*sqrt(3).
    
    # For s=1, center-to-center distance = sqrt(3) ≈ 1.732
    
    # But for disjointness, we need strictly greater than sqrt(3).
    
    # For a hexagonal arrangement:
    # Center at (0,0)
    # 6 neighbors at distance sqrt(3) in 6 directions (0°, 60°, 120°, 180°, 240°, 300°)
    # But these would be touching, not disjoint. So we need to add a small epsilon.
    
    # Actually, the problem says "disjoint" which typically means non-overlapping, and touching might be allowed.
    # Let me assume touching is allowed (closed sets that don't overlap in interior).
    
    d = math.sqrt(3)  # Center-to-center distance for adjacent hexagons
    
    # First ring
    for i in range(6):
        theta = i * math.pi / 3
        hexagons.append((d * math.cos(theta), d * math.sin(theta), 0.0))
    
    # Second ring: we need 5 more to make 12 total (1 + 6 + 5)
    # In a hexagonal packing, the second ring has 12 hexagons. We pick 5 of them.
    # The second ring hexagons are at distance 2*d from center, at angles 0, 30, 60, ..., 330
    # But actually, the second ring in a hexagonal lattice consists of hexagons at:
    # Distance 2*d at angles 0, 60, 120, 180, 240, 300 (6 hexagons)
    # And distance sqrt(3)*d at angles 30, 90, 150, 210, 270, 330 (6 hexagons)
    # Total 12 in second ring.
    
    # Let me just place 5 more in a reasonable pattern
    # Place them at the "corners" of the hexagonal cluster
    for i in range(5):
        theta = i * 2 * math.pi / 5 + math.pi / 10
        hexagons.append((2 * d * math.cos(theta), 2 * d * math.sin(theta), 0.0))
    
    return hexagons

def _try_packing(hexagons, outer_angle=0.0):
    """Try to fit hexagons in a minimal outer hexagon. Returns (side, center, angle) or None."""
    side, cx, cy = _min_enclosing_hexagon(hexagons, outer_angle)
    
    # Verify validity
    if _is_valid(hexagons, (cx, cy), side, outer_angle):
        return side, (cx, cy), outer_angle
    return None

def construct():
    """Main entry point: construct the packing."""
    best_result = None
    best_side = float('inf')
    
    # Try different outer hexagon angles
    for outer_angle in [0.0, 30.0]:
        # Try grid packing
        hexagons = _construct_grid_packing()
        result = _try_packing(hexagons, outer_angle)
        if result is not None:
            side, center, angle = result
            if side < best_side:
                best_side = side
                best_result = (hexagons, center, side, angle)
        
        # Try hexagonal packing
        hexagons = _construct_hexagonal_packing()
        result = _try_packing(hexagons, outer_angle)
        if result is not None:
            side, center, angle = result
            if side < best_side:
                best_side = side
                best_result = (hexagons, center, side, angle)
    
    # If no valid packing found, use a large fallback
    if best_result is None:
        # Very large container with hexagons spread out
        inner = []
        d = math.sqrt(3) + 0.01  # Slightly more than touching distance
        positions = [(0, 0)]
        for i in range(6):
            theta = i * math.pi / 3
            positions.append((d * math.cos(theta), d * math.sin(theta)))
        for i in range(5):
            theta = i * 2 * math.pi / 5 + math.pi / 10
            positions.append((2 * d * math.cos(theta), 2 * d * math.sin(theta)))
        
        for px, py in positions:
            inner.append([px, py, 0.0])
        
        center = (0.0, 0.0)
        side = 10.0
        angle = 0.0
        
        return inner, center, side, angle
    
    hexagons, center, side, angle = best_result
    inner = [[h[0], h[1], h[2]] for h in hexagons]
    
    return inner, center, side, angle

