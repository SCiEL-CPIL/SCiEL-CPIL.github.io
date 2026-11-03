"""Packing 12 unit regular hexagons inside a regular hexagon, minimizing outer side length.

Approach: Use a structured grid-based packing with careful geometry.
A regular hexagon of side length S can contain a honeycomb arrangement of unit hexagons.
For n=12, we can arrange them in a compact cluster.

Key insight: The minimal container for a cluster of unit hexagons depends on the
convex hull of the cluster. For a regular hexagon container, we need to find
the minimum side length S such that the container (rotated optimally) contains
all 12 unit hexagons.

We'll use a known efficient packing: arrange 12 hexagons in a roughly circular
cluster. One good arrangement is a central hexagon surrounded by 6, then 5 more
around the outside, forming a larger hexagonal shape.

Actually, let's think about this more carefully. A regular hexagon of side n contains
3n^2 - 3n + 1 unit hexagons in a perfect honeycomb. For n=3, that's 19. For n=2, that's 7.

So 12 hexagons fit inside a side-2 hexagon? No, side-2 contains 7. Side-3 contains 19.
So the answer should be between 2 and 3.

Let me try a different approach: place 12 hexagons in a pattern that minimizes the
bounding hexagon. We can use optimization to find good positions.

Strategy: Start with a reasonable initial configuration and use scipy optimization
to minimize the outer hexagon side length needed to contain all inner hexagons.

The outer hexagon side length needed for a set of points (with their hexagon extents)
can be computed by checking all 6 directions.
"""
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 12

def hexagon_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon centered at (cx,cy) with given side and rotation."""
    angle_rad = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = angle_rad + i * np.pi / 3
        x = cx + side * np.cos(theta)
        y = cy + side * np.sin(theta)
        verts.append((x, y))
    return np.array(verts)

def point_in_hexagon(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside hexagon centered at (cx,cy)."""
    # Transform to hexagon's local frame
    dx = px - cx
    dy = py - cy
    angle_rad = np.radians(angle_deg)
    cos_a = np.cos(-angle_rad)
    sin_a = np.sin(-angle_rad)
    lx = dx * cos_a - dy * sin_a
    ly = dx * sin_a + dy * cos_a
    
    # In local frame, hexagon has vertices at angles 0, 60, 120, 180, 240, 300 degrees
    # The hexagon is defined by |lx| <= side, |ly| <= side*sqrt(3)/2, and |lx*sqrt(3)/2 + ly/2| <= side
    # Actually, for a regular hexagon with "flat top" orientation (vertex at angle 0):
    # The constraints are:
    # |lx| <= side
    # |ly| <= side * sqrt(3)/2
    # |lx * sqrt(3)/2 + ly/2| <= side  ... wait, let me reconsider.
    
    # A regular hexagon with circumradius = side (vertex-to-center distance = side)
    # In the frame where one vertex is at angle 0:
    # Vertices at (side, 0), (side/2, side*sqrt(3)/2), (-side/2, side*sqrt(3)/2), 
    #             (-side, 0), (-side/2, -side*sqrt(3)/2), (side/2, -side*sqrt(3)/2)
    # 
    # The hexagon is the intersection of 6 half-planes.
    # For a vertex at angle theta_i, the corresponding edge is perpendicular to that direction.
    # The half-plane constraint is: dot((lx,ly), (cos(theta_i), sin(theta_i))) <= side
    # where theta_i are the angles to the vertices (which are also the normals to the opposite edges).
    
    # Actually, for a regular hexagon with circumradius R=side, the apothem (inradius) is R*sqrt(3)/2.
    # The 6 edge normals are at angles 30, 90, 150, 210, 270, 330 degrees (offset by 30 from vertices).
    # Each edge constraint: dot((lx,ly), n_i) <= apothem = side * sqrt(3)/2
    
    apothem = side * np.sqrt(3) / 2
    for i in range(6):
        theta = np.radians(30 + i * 60)
        nx = np.cos(theta)
        ny = np.sin(theta)
        if lx * nx + ly * ny > apothem + 1e-9:
            return False
    return True

def hexagon_contains_hexagon(center1, angle1, side1, center2, angle2, side2):
    """Check if hexagon 2 is entirely inside hexagon 1."""
    # For this to work well, we need to check all vertices of hex2 are inside hex1
    # AND all edges of hex2 don't cross edges of hex1. For convex shapes, 
    # checking vertices of one inside the other is sufficient if we also check
    # that no edge of the inner crosses an edge of the outer.
    # Simpler: check all vertices of inner hex are inside outer hex.
    verts2 = hexagon_vertices(center2[0], center2[1], side2, angle2)
    for vx, vy in verts2:
        if not point_in_hexagon(vx, vy, center1[0], center1[1], side1, angle1):
            return False
    return True

def min_outer_side(inner_positions, inner_angles, outer_center, outer_angle):
    """Find minimum side length of outer hexagon (at given center/angle) that contains all inner hexagons."""
    # For each inner hexagon, find the maximum "extent" in each of the 6 directions
    # of the outer hexagon.
    
    outer_angle_rad = np.radians(outer_angle)
    
    # The 6 edge normals of the outer hexagon
    apothem_dirs = []
    for i in range(6):
        theta = np.radians(30 + i * 60) + outer_angle_rad
        apothem_dirs.append((np.cos(theta), np.sin(theta)))
    
    max_extents = [0.0] * 6
    
    for (cx, cy), angle_deg in zip(inner_positions, inner_angles):
        verts = hexagon_vertices(cx, cy, 1.0, angle_deg)
        for vx, vy in verts:
            for j in range(6):
                nx, ny = apothem_dirs[j]
                extent = (vx - outer_center[0]) * nx + (vy - outer_center[1]) * ny
                if extent > max_extents[j]:
                    max_extents[j] = extent
    
    # The side length must satisfy: side * sqrt(3)/2 >= max_extents[j] for all j
    # So side >= max(max_extents) * 2 / sqrt(3)
    max_extent = max(max_extents)
    if max_extent < 0:
        max_extent = 0
    return max_extent * 2 / np.sqrt(3)

def check_disjointness(inner_positions, inner_angles):
    """Check if all inner hexagons are pairwise disjoint (allowing touching)."""
    n = len(inner_positions)
    for i in range(n):
        for j in range(i + 1, n):
            if hexagons_overlap(inner_positions[i], inner_angles[i], 
                               inner_positions[j], inner_angles[j]):
                return False
    return True

def hexagons_overlap(pos1, ang1, pos2, ang2):
    """Check if two unit hexagons overlap (interior intersection)."""
    # Check if any vertex of one is strictly inside the other
    verts1 = hexagon_vertices(pos1[0], pos1[1], 1.0, ang1)
    verts2 = hexagon_vertices(pos2[0], pos2[1], 1.0, ang2)
    
    # Check vertices of hex1 inside hex2
    for vx, vy in verts1:
        if point_in_hexagon_strict(vx, vy, pos2[0], pos2[1], 1.0, ang2):
            return True
    
    # Check vertices of hex2 inside hex1
    for vx, vy in verts2:
        if point_in_hexagon_strict(vx, vy, pos1[0], pos1[1], 1.0, ang1):
            return True
    
    # Check edge intersections
    if edges_intersect(verts1, verts2):
        return True
    
    return False

def point_in_hexagon_strict(px, py, cx, cy, side, angle_deg):
    """Check if point is strictly inside hexagon (not on boundary)."""
    dx = px - cx
    dy = py - cy
    angle_rad = np.radians(angle_deg)
    cos_a = np.cos(-angle_rad)
    sin_a = np.sin(-angle_rad)
    lx = dx * cos_a - dy * sin_a
    ly = dx * sin_a + dy * cos_a
    
    apothem = side * np.sqrt(3) / 2
    for i in range(6):
        theta = np.radians(30 + i * 60)
        nx = np.cos(theta)
        ny = np.sin(theta)
        if lx * nx + ly * ny > apothem - 1e-10:
            return False
    return True

def edges_intersect(v1, v2):
    """Check if any edge of polygon v1 intersects any edge of polygon v2."""
    for i in range(6):
        p1 = v1[i]
        p2 = v1[(i + 1) % 6]
        for j in range(6):
            p3 = v2[j]
            p4 = v2[(j + 1) % 6]
            if line_segments_intersect(p1, p2, p3, p4):
                return True
    return False

def line_segments_intersect(p1, p2, p3, p4):
    """Check if line segment p1-p2 intersects line segment p3-p4."""
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    
    d1 = cross(p3, p4, p1)
    d2 = cross(p3, p4, p2)
    d3 = cross(p1, p2, p3)
    d4 = cross(p1, p2, p4)
    
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    
    # Check collinear cases
    if d1 == 0 and on_segment(p3, p1, p4):
        return True
    if d2 == 0 and on_segment(p3, p2, p4):
        return True
    if d3 == 0 and on_segment(p1, p3, p2):
        return True
    if d4 == 0 and on_segment(p1, p4, p2):
        return True
    
    return False

def on_segment(p, q, r):
    """Check if point q lies on segment pr."""
    if min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and \
       min(p[1], r[1]) <= q[1] <= max(p[1], r[1]):
        return True
    return False

def objective(params):
    """Objective function: outer hexagon side length + penalty for overlaps."""
    n = N
    # params: 3*n + 2 = 38 parameters
    # First 3n: positions and angles of inner hexagons
    # Last 2: outer center (x, y)
    
    inner_pos = []
    inner_ang = []
    for i in range(n):
        x = params[3*i]
        y = params[3*i + 1]
        ang = params[3*i + 2]
        inner_pos.append((x, y))
        inner_ang.append(ang)
    
    outer_cx = params[3*n]
    outer_cy = params[3*n + 1]
    
    # Check disjointness
    overlap_penalty = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            if hexagons_overlap(inner_pos[i], inner_ang[i], 
                               inner_pos[j], inner_ang[j]):
                # Compute overlap distance (approximate)
                dist = np.sqrt((inner_pos[i][0] - inner_pos[j][0])**2 + 
                              (inner_pos[i][1] - inner_pos[j][1])**2)
                overlap_penalty += max(0, 2.0 - dist) ** 2
    
    # Compute outer side length
    side = min_outer_side(inner_pos, inner_ang, (outer_cx, outer_cy), 0.0)
    
    return side + 100.0 * overlap_penalty

def construct():
    """Construct a packing of 12 unit hexagons inside a regular hexagon."""
    
    # Initial configuration: honeycomb-like cluster
    # Place hexagons in a compact arrangement
    # Use a hexagonal grid with spacing sqrt(3) (distance between centers of adjacent hexes)
    
    # A good starting point: arrange in a roughly hexagonal cluster
    # For 12 hexagons, try a pattern like:
    # Row 0: 3 hexagons
    # Row 1: 3 hexagons  
    # Row 2: 3 hexagons
    # Row 3: 3 hexagons
    # But offset for honeycomb packing
    
    # Let's use a more systematic approach
    # Start with a known good configuration and optimize
    
    # Initial guess: spread out in a circle
    np.random.seed(42)
    
    # Better initial guess: honeycomb arrangement
    # Centers of adjacent hexagons are at distance sqrt(3) apart
    sqrt3 = np.sqrt(3)
    
    # Create a hexagonal cluster of 12 hexagons
    # Use axial coordinates for hex grid
    # A hexagon of "radius" 2 has 1 + 6 + 12 = 19 cells, we want 12
    # Let's pick 12 cells from a larger hex grid
    
    # Axial to pixel: x = sqrt(3) * (q + r/2), y = 3/2 * r
    positions = []
    # Select 12 cells from a hex grid centered at origin
    # Use cells with q^2 + q*r + r^2 <= some threshold
    candidates = []
    for q in range(-3, 4):
        for r in range(-3, 4):
            dist_sq = q*q + q*r + r*r
            candidates.append((dist_sq, q, r))
    candidates.sort()
    
    # Take the 12 closest cells
    selected = candidates[:12]
    for _, q, r in selected:
        x = sqrt3 * (q + r / 2.0)
        y = 1.5 * r
        positions.append([x, y, 0.0])  # angle 0
    
    # Now optimize
    # Parameters: 3*12 + 2 = 38
    # Use differential evolution for robustness
    
    bounds = [(-5, 5)] * 36 + [(-5, 5), (-5, 5)]
    # Angles should be in [-180, 180]
    for i in range(12):
        bounds[3*i + 2] = (-180, 180)
    
    initial_params = []
    for p in positions:
        initial_params.extend(p)
    initial_params.extend([0.0, 0.0])
    
    # Try optimization
    result = differential_evolution(
        objective, 
        bounds, 
        seed=42,
        maxiter=200,
        tol=1e-6,
        polish=True,
        popsize=15
    )
    
    if result.success or result.fun < 10.0:
        params = result.x
        inner = []
        for i in range(N):
            x = params[3*i]
            y = params[3*i + 1]
            ang = params[3*i + 2]
            inner.append([x, y, ang])
        outer_cx = params[3*N]
        outer_cy = params[3*N + 1]
        outer_angle = 0.0
        
        # Compute final side length
        side = min_outer_side(
            [(p[0], p[1]) for p in inner],
            [p[2] for p in inner],
            (outer_cx, outer_cy),
            outer_angle
        )
        
        # Verify disjointness
        if check_disjointness([(p[0], p[1]) for p in inner], [p[2] for p in inner]):
            return inner, (outer_cx, outer_cy), side, outer_angle
    
    # Fallback: use the initial honeycomb configuration
    inner = positions
    outer_cx, outer_cy = 0.0, 0.0
    outer_angle = 0.0
    side = min_outer_side(
        [(p[0], p[1]) for p in inner],
        [p[2] for p in inner],
        (outer_cx, outer_cy),
        outer_angle
    )
    
    return inner, (outer_cx, outer_cy), side, outer_angle
