"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize, LinearConstraint
from scipy.spatial.distance import cdist

# Helper to check if a point is inside a regular hexagon
def point_in_hexagon(p, center, side, angle_degrees):
    """
    Check if point p is inside a regular hexagon defined by center, side, angle.
    The hexagon has flat top/bottom when angle=0.
    Vertices at angles 0, 60, 120, 180, 240, 300 degrees relative to center,
    scaled by 'side' (circumradius = side for regular hexagon).
    """
    # Transform to hexagon's local frame
    dx = p[0] - center[0]
    dy = p[1] - center[1]
    
    theta = np.radians(angle_degrees)
    ct = np.cos(theta)
    st = np.sin(theta)
    
    # Rotate point by -theta
    x_local = dx * ct + dy * st
    y_local = -dx * st + dy * ct
    
    # For a regular hexagon with circumradius R (side length s = R),
    # the hexagon is defined by |x| <= R/2 * sqrt(3) ? No.
    # Let's use the standard definition:
    # Vertices at (R, 0), (R/2, R*sqrt(3)/2), (-R/2, R*sqrt(3)/2), (-R, 0), etc.
    # The hexagon is the intersection of 6 half-planes.
    # For angle=0, vertices are at 0, 60, 120, 180, 240, 300.
    # Normal vectors to the edges point inwards.
    # Edge 1: between v0(R,0) and v1(R/2, R*sqrt(3)/2).
    # The normal to this edge (pointing in) is at angle 30 degrees?
    # Let's just check distance to each edge.
    
    R = side
    # The 6 normals for a flat-top hexagon (angle=0) point at 0, 60, 120, 180, 240, 300?
    # No, the vertices are at those angles. The edge normals are at 30, 90, 150, 210, 270, 330.
    # For a point to be inside, its projection onto each normal must be <= apothem.
    # Apothem a = R * sqrt(3) / 2.
    
    a = R * np.sqrt(3) / 2.0
    # Normals angles: 30, 90, 150, 210, 270, 330 degrees.
    # Or simply: check if |x_local| <= a and |y_local| <= a and ... ?
    # Actually, for a regular hexagon aligned with axes (flat top), the conditions are:
    # |x| <= R
    # |y| <= R * sqrt(3) / 2
    # |x|/2 + |y| * sqrt(3)/2 <= R * sqrt(3) / 2 ? No.
    
    # Let's use the robust method: check against the 6 half-planes.
    # Vertices v_i = R * (cos(i*60), sin(i*60)).
    # Edge i is between v_i and v_{i+1}.
    # Inward normal n_i is perpendicular to edge, pointing to center.
    # For v0=(R,0), v1=(R/2, R*sqrt(3)/2).
    # Edge vector e = v1 - v0 = (-R/2, R*sqrt(3)/2).
    # Normal n = (R*sqrt(3)/2, R/2) ? Dot with e is -R^2/4 + 3R^2/4 = R^2/2 != 0.
    # Perpendicular to e is (R*sqrt(3)/2, R/2) or (-R*sqrt(3)/2, -R/2).
    # We want the one pointing towards origin. Origin is (0,0).
    # Midpoint m = (3R/4, R*sqrt(3)/4).
    # Vector m to origin is (-3R/4, -R*sqrt(3)/4).
    # Dot with (sqrt(3)/2, 1/2) is -3R/8 - R*sqrt(3)/8 < 0.
    # So inward normal is (-sqrt(3)/2, -1/2) ? No, that points away from first quadrant.
    # Let's just compute the distance to the line segment or use the standard inequality.
    
    # Standard inequality for regular hexagon centered at origin, circumradius R, flat-top:
    # |x| <= R
    # |y| <= R * sqrt(3) / 2
    # |x| + |y| / sqrt(3) <= R * sqrt(3) / 2 ? No.
    
    # Let's use the half-plane method properly.
    # The hexagon is the set of points p such that for all i in 0..5:
    # dot(p, n_i) <= a
    # where n_i are unit normals to the edges, and a is the apothem.
    # For flat-top hexagon (vertices at 0, 60, ...), the edge normals are at 30, 90, 150, 210, 270, 330.
    # n_0 = (cos(30), sin(30)) = (sqrt(3)/2, 1/2)
    # n_1 = (cos(90), sin(90)) = (0, 1)
    # n_2 = (cos(150), sin(150)) = (-sqrt(3)/2, 1/2)
    # n_3 = (-sqrt(3)/2, -1/2)
    # n_4 = (0, -1)
    # n_5 = (sqrt(3)/2, -1/2)
    
    # Wait, if vertices are at 0, 60..., the edge between v0(R,0) and v1(R/2, R*sqrt(3)/2)
    # has midpoint (3R/4, R*sqrt(3)/4).
    # The normal pointing outwards is (sqrt(3)/2, 1/2).
    # dot(midpoint, normal) = 3R/4 * sqrt(3)/2 + R*sqrt(3)/4 * 1/2 = 3R*sqrt(3)/8 + R*sqrt(3)/8 = R*sqrt(3)/2 = a.
    # So the condition is dot(p, n_i) <= a for all i.
    
    normals = []
    for i in range(6):
        angle_rad = np.radians(30 + i * 60)
        normals.append([np.cos(angle_rad), np.sin(angle_rad)])
    normals = np.array(normals)
    
    # Project local point onto normals
    projections = x_local * normals[:, 0] + y_local * normals[:, 1]
    
    return np.all(projections <= a + 1e-9)

def hexagon_vertices(center, side, angle_degrees):
    """Return the 6 vertices of a regular hexagon."""
    R = side
    theta = np.radians(angle_degrees)
    vertices = []
    for i in range(6):
        angle = theta + np.radians(60 * i)
        x = center[0] + R * np.cos(angle)
        y = center[1] + R * np.sin(angle)
        vertices.append([x, y])
    return np.array(vertices)

def hexagon_overlap_area_approx(hex1_center, hex1_side, hex1_angle, hex2_center, hex2_side, hex2_angle):
    """
    Approximate overlap by checking if any vertex of one is inside the other.
    This is not exact but good enough for a heuristic search.
    For a more robust check, we can use the fact that if they overlap,
    at least one vertex of one is inside the other OR they are deeply nested.
    Given unit hexagons, if they are disjoint, the distance between centers must be large enough.
    """
    # Check vertices of hex1 in hex2
    v1 = hexagon_vertices(hex1_center, hex1_side, hex1_angle)
    for v in v1:
        if point_in_hexagon(v, hex2_center, hex2_side, hex2_angle):
            return True
    
    # Check vertices of hex2 in hex1
    v2 = hexagon_vertices(hex2_center, hex2_side, hex2_angle)
    for v in v2:
        if point_in_hexagon(v, hex1_center, hex1_side, hex1_angle):
            return True
    
    return False

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    N = 11
    # Initial guess: side length of outer hexagon
    # A single unit hexagon has side 1. Circumradius 1.
    # To fit N hexagons in a line, we need roughly N * sqrt(3) ?
    # Let's start with a large side and minimize.
    
    # Parameters:
    # For each inner hexagon: x, y, angle (3 params)
    # Outer hexagon: center_x, center_y, side, angle (4 params)
    # Total: 3*N + 4
    
    # We can fix the outer center to (0,0) and angle to 0 to reduce symmetry?
    # The task allows outer to move/rotate. But fixing it to (0,0) and angle 0
    # is WLOG if we allow inner to move/rotate freely?
    # Actually, if we fix outer to (0,0) and angle 0, we lose the ability to shift the cluster.
    # But since the objective is just the side length, and the outer hexagon can be anywhere,
    # we can fix the outer center to (0,0) and angle to 0, and let the inner hexagons
    # be positioned relative to it. This reduces 4 parameters to 0 (fixed).
    # Wait, the score is the outer side length. If we fix the outer center and angle,
    # we are effectively defining the container. The inner hexagons must fit inside.
    # So we minimize the side length 'S' such that there exist positions/angles for inner hexagons.
    
    # Let's fix outer center to (0,0) and outer angle to 0.
    # Variables:
    # S (outer side length)
    # For each inner hexagon i: x_i, y_i, theta_i
    
    # We will use a heuristic placement first, then optimize S.
    
    # Heuristic: Place hexagons in a honeycomb pattern.
    # Unit hexagons in honeycomb:
    # Distance between centers of adjacent hexagons is sqrt(3) * side = sqrt(3).
    # Orientation: 0 degrees.
    
    # Let's generate a honeycomb layout for N hexagons.
    # Hexagonal numbers: 1, 7, 19, 37...
    # N=11 is not a hexagonal number.
    # We can place them in rows.
    # Row 0: 3 hexagons
    # Row 1: 4 hexagons
    # Row 2: 4 hexagons
    # Total 11.
    
    # Honeycomb coordinates:
    # Axial coordinates (q, r).
    # Distance between (q1,r1) and (q2,r2) is sqrt(3) * hex_dist(q1,r1,q2,r2).
    # hex_dist = (|q1-q2| + |r1-r2| + |q1+r1-q2-r2|) / 2
    
    # Let's pick 11 axial coordinates.
    # Center (0,0).
    # Neighbors: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1).
    # That's 7.
    # Next ring: (2,0), (1,1), (0,2), (-1,2), (-2,2), (-2,1), (-2,0), (-2,-1), (-1,-2), (0,-2), (1,-2), (2,-1).
    # We need 4 more.
    # Let's take (2,0), (1,1), (0,2), (-1,2).
    
    coords = [
        (0,0),
        (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
        (2,0), (1,1), (0,2), (-1,2)
    ]
    
    # Convert axial to Cartesian.
    # x = sqrt(3) * (q + r/2)
    # y = 3/2 * r
    # This assumes "pointy top" or "flat top"?
    # Standard axial to cartesian for pointy top:
    # x = size * sqrt(3) * (q + r/2)
    # y = size * 3/2 * r
    # For flat top, swap x and y or rotate.
    # Let's use flat top orientation (angle 0).
    # In flat top, the "up" direction is y.
    # Axial coordinates for flat top:
    # x = size * 3/2 * q
    # y = size * sqrt(3) * (r + q/2)
    # Let's verify distance.
    # (0,0) to (1,0): x=1.5, y=sqrt(3)/2. dist = sqrt(2.25 + 0.75) = sqrt(3). Correct.
    # (0,0) to (0,1): x=0, y=sqrt(3). dist = sqrt(3). Correct.
    
    positions = []
    for q, r in coords:
        x = 1.5 * q
        y = np.sqrt(3) * (r + q / 2.0)
        positions.append([x, y, 0.0]) # angle 0
    
    # Now we have a valid packing of 11 unit hexagons.
    # We need to find the minimum side length S of an outer hexagon (center 0,0, angle 0)
    # that contains all these.
    
    # The outer hexagon is defined by S.
    # We need to find the minimum S such that all inner hexagons are inside.
    
    # Function to check if a packing fits in outer hexagon of side S.
    def fits_in_outer(positions, S):
        for pos in positions:
            x, y, angle = pos
            # Check all 6 vertices of the inner hexagon
            v = hexagon_vertices([x, y], 1.0, angle)
            for vertex in v:
                if not point_in_hexagon(vertex, [0.0, 0.0], S, 0.0):
                    return False
        return True
    
    # Find minimum S using binary search or optimization.
    # Lower bound: The distance from center to the farthest vertex of any inner hexagon.
    # Upper bound: Large enough.
    
    # Calculate initial S.
    max_dist = 0
    for pos in positions:
        x, y, angle = pos
        v = hexagon_vertices([x, y], 1.0, angle)
        for vertex in v:
            d = np.linalg.norm(vertex)
            if d > max_dist:
                max_dist = d
    
    # For a hexagon of side S, the max distance from center to a vertex is S.
    # But the hexagon is not a circle.
    # The apothem is S * sqrt(3) / 2.
    # The distance to a vertex is S.
    # So S must be at least max_dist.
    # But also, the point must be inside the hexagon, which is tighter than just distance <= S.
    
    S_low = max_dist
    S_high = max_dist * 1.5 + 5.0
    
    # Binary search for S
    S_opt = S_high
    for _ in range(50):
        S_mid = (S_low + S_high) / 2.0
        if fits_in_outer(positions, S_mid):
            S_opt = S_mid
            S_high = S_mid
        else:
            S_low = S_mid
        if time.time() > deadline - 10:
            break
    
    # Now we have a feasible S_opt.
    # Can we improve by moving/rotating the inner hexagons?
    # The current packing is a honeycomb. It is likely near-optimal for N=11.
    # Let's try a simple local optimization to see if we can reduce S further.
    # This is complex. For now, return the honeycomb result.
    
    # Return format:
    # inner: list of [x, y, angle_degrees]
    # center: [x, y]
    # side: S_opt
    # angle: 0.0
    
    inner_list = [[pos[0], pos[1], pos[2]] for pos in positions]
    center = [0.0, 0.0]
    side = S_opt
    angle = 0.0
    
    return inner_list, center, side, angle

# EVOLVE-BLOCK-START
# The construct function is defined above.
# EVOLVE-BLOCK-END
