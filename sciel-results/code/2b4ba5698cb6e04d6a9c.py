"""Packing 11 unit regular hexagons in a minimal regular hexagon."""
import time
import math
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 11
DEADLINE = 115.0  # safety margin before 120s kill

def hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon with center (cx,cy), side length, rotation."""
    ang = math.radians(angle_deg)
    cos_a, sin_a = math.cos(ang), math.sin(ang)
    # Standard hexagon vertices at angles 0, 60, 120, 180, 240, 300 degrees
    pts = []
    for i in range(6):
        theta = ang + i * math.pi / 3.0
        x = cx + side * math.cos(theta)
        y = cy + side * math.sin(theta)
        pts.append((x, y))
    return np.array(pts)

def polygon_area_signed(pts):
    """Signed area of polygon."""
    n = len(pts)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += pts[i][0] * pts[j][1] - pts[j][0] * pts[i][1]
    return 0.5 * area

def point_in_polygon(pt, poly):
    """Ray casting point in polygon test."""
    x, y = pt
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside

def hexagons_disjoint(v1, v2):
    """Check if two hexagons (arrays of 6 vertices) are disjoint (no overlap).
    Returns True if disjoint, False if overlapping."""
    # Check if any vertex of v1 is inside v2 or vice versa
    for pt in v1:
        if point_in_polygon(pt, v2):
            return False
    for pt in v2:
        if point_in_polygon(pt, v1):
            return False
    # Check edge intersections
    def segments_intersect(p1, p2, p3, p4):
        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
        d1 = cross(p3, p4, p1)
        d2 = cross(p3, p4, p2)
        d3 = cross(p1, p2, p3)
        d4 = cross(p1, p2, p4)
        if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
           ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
            return True
        return False
    
    for i in range(6):
        j = (i + 1) % 6
        for k in range(6):
            l = (k + 1) % 6
            if segments_intersect(v1[i], v1[j], v2[k], v2[l]):
                return False
    return True

def inner_hexagons_in_outer(inner_verts, outer_verts):
    """Check all inner hexagons are inside the outer hexagon."""
    for iv in inner_verts:
        for pt in iv:
            if not point_in_polygon(pt, outer_verts):
                return False
    return True

def all_constraints_satisfied(inner_verts, outer_verts):
    """Check all constraints: inner hexagons inside outer, and pairwise disjoint."""
    if not inner_hexagons_in_outer(inner_verts, outer_verts):
        return False
    for i in range(len(inner_verts)):
        for j in range(i + 1, len(inner_verts)):
            if not hexagons_disjoint(inner_verts[i], inner_verts[j]):
                return False
    return True

def compute_outer_side(inner_positions, outer_center, outer_angle):
    """Given inner hexagon positions (centers and angles), find minimal outer side length
    such that all inner hexagons fit inside the outer hexagon.
    Returns the outer side length."""
    # For a regular hexagon with side s and rotation, the support function in direction theta
    # is s * max over vertices of dot(vertex_dir, theta_dir)
    # Actually, for a regular hexagon centered at origin with side s, the distance from center
    # to the edge in direction theta is:
    # h(theta) = s * cos(pi/6) * ... let me think more carefully.
    
    # A regular hexagon with side s has vertices at distance s from center.
    # The apothem (distance from center to edge) is s * sqrt(3)/2.
    # The support function h(theta) = max over edges of the distance from center to edge
    # projected onto direction theta.
    
    # For a regular hexagon with vertices at angles 0, 60, 120, 180, 240, 300 (rotated by outer_angle),
    # the edges are at angles 30, 90, 150, 210, 270, 330 (rotated by outer_angle).
    # The distance from center to each edge is the apothem = s * sqrt(3)/2.
    # The normal to edge k points at angle (30 + 60*k + outer_angle) degrees.
    
    # For an inner hexagon at position (cx, cy) with angle inner_angle, its vertices are
    # at (cx + cos(inner_angle + 60*i), cy + sin(inner_angle + 60*i)) for i=0..5.
    # The constraint is that for each inner vertex v, it must be inside the outer hexagon.
    # v is inside the outer hexagon centered at (ox, oy) with side s and rotation outer_angle
    # iff for all 6 edges, the dot product of (v - center) with the edge normal <= apothem.
    
    # So: for each inner vertex v, for each edge normal n_k:
    #   dot(v - center, n_k) <= s * sqrt(3)/2
    #   => s >= dot(v - center, n_k) / (sqrt(3)/2) for all k where dot > 0
    
    # The minimal s is the max over all inner vertices and all edge normals of:
    #   dot(v - center, n_k) / (sqrt(3)/2)
    
    outer_ang_rad = math.radians(outer_angle)
    # Edge normals of outer hexagon (pointing outward)
    # Vertices at angles: outer_angle + 0, 60, 120, 180, 240, 300
    # Edges between consecutive vertices; edge normal bisects the angle between the two vertex directions
    # Edge between vertex i and i+1 has normal at angle outer_angle + 30 + 60*i degrees
    edge_normals = []
    for k in range(6):
        theta = outer_ang_rad + math.radians(30 + 60 * k)
        edge_normals.append((math.cos(theta), math.sin(theta)))
    
    apothem_factor = math.sqrt(3) / 2.0
    max_s = 0.0
    
    for (cx, cy, inner_ang_deg) in inner_positions:
        inner_ang_rad = math.radians(inner_ang_deg)
        for i in range(6):
            theta_v = inner_ang_rad + i * math.pi / 3.0
            vx = cx + math.cos(theta_v)
            vy = cy + math.sin(theta_v)
            for (nx, ny) in edge_normals:
                d = (vx - outer_center[0]) * nx + (vy - outer_center[1]) * ny
                if d > 0:
                    s_needed = d / apothem_factor
                    if s_needed > max_s:
                        max_s = s_needed
    return max_s

def objective(params, outer_center, outer_angle):
    """Objective: compute outer side length given inner hexagon parameters.
    params: array of shape (3*N,) = [x1, y1, a1, x2, y2, a2, ..., xN, yN, aN]
    """
    inner_positions = []
    for i in range(N):
        x = params[3*i]
        y = params[3*i+1]
        a = params[3*i+2]
        inner_positions.append((x, y, a))
    return compute_outer_side(inner_positions, outer_center, outer_angle)

def constraint_check(params, outer_center, outer_angle):
    """Check if all inner hexagons are pairwise disjoint.
    Returns True if valid."""
    inner_verts = []
    for i in range(N):
        x = params[3*i]
        y = params[3*i+1]
        a = params[3*i+2]
        inner_verts.append(hex_vertices(x, y, 1.0, a))
    
    for i in range(N):
        for j in range(i+1, N):
            if not hexagons_disjoint(inner_verts[i], inner_verts[j]):
                return False
    return True

def hexagonal_lattice_seed():
    """Create a seed based on hexagonal lattice arrangement."""
    # Standard hexagonal lattice: centers at (2*sqrt(3)*i + sqrt(3)*j, 3*j) for unit hexagons
    # Actually for unit side length hexagons, the distance between adjacent centers is 2*sqrt(3)
    # Wait, for regular hexagons with side 1, the distance between centers of adjacent hexagons
    # in a honeycomb is 2*sqrt(3)/... let me recalculate.
    # A regular hexagon with side s has "width" (distance between parallel sides) = s*sqrt(3).
    # In a honeycomb packing, adjacent hexagons share edges, so center-to-center distance = s*sqrt(3).
    # Wait no. Two hexagons sharing an edge: the edge has length s. The centers are on opposite
    # sides of the edge, each at distance apothem = s*sqrt(3)/2 from the edge.
    # So center-to-center distance = 2 * apothem = s*sqrt(3).
    
    # For N=11, a common arrangement is a "flower" pattern: 1 center + 6 around + 4 more
    # Or a 3-4-4-3 arrangement in a hexagonal grid.
    
    # Let's use a hexagonal grid pattern.
    # Rows: -1, 0, 1 (3 rows) with 4, 4, 3 hexagons? That's 11.
    # Or 2-3-3-3-2? That's 13, too many.
    # 3-4-4-3 = 14, too many.
    # 2-3-3-3-2 = 13.
    # 1-3-5-3-1 = 13.
    # Let's do: row 0 has 3, row 1 has 4, row 2 has 4 -> 11? No, that's 11.
    # Actually let me just place them in a compact hexagonal arrangement.
    
    # Hexagonal lattice basis vectors for unit hexagons:
    # v1 = (sqrt(3), 0)  -- horizontal spacing
    # v2 = (sqrt(3)/2, 1.5) -- diagonal spacing
    # Wait, let me be more careful.
    
    # For a regular hexagon with side 1, vertices at (cos(60k), sin(60k)) for k=0..5.
    # The "flat-top" hexagon has vertices at angles 0, 60, 120, 180, 240, 300.
    # Width (left to right) = 2, height (top to bottom) = sqrt(3).
    # In a honeycomb: horizontal spacing = 3/2 (for pointy-top) or 2 (for flat-top)?
    
    # Let me use pointy-top hexagons. Vertices at angles 30, 90, 150, 210, 270, 330.
    # Height = 2, width = sqrt(3).
    # Honeycomb spacing: horizontal = 3/2 * width/... 
    
    # Actually, let me just use a simple approach: place centers in a hexagonal pattern
    # with appropriate spacing to avoid overlap.
    
    # For pointy-top hexagons with side 1:
    # - Height (vertex to vertex) = 2
    # - Width (flat to flat) = sqrt(3)
    # - In honeycomb, row spacing = 1.5 (vertical), column spacing = sqrt(3) (horizontal)
    # - Odd rows offset by sqrt(3)/2
    
    # Let's place 11 hexagons in a compact arrangement:
    # Row 0 (y=0): 3 hexagons at x = -sqrt(3), 0, sqrt(3)
    # Row 1 (y=1.5): 4 hexagons at x = -3*sqrt(3)/2, -sqrt(3)/2, sqrt(3)/2, 3*sqrt(3)/2
    # Row 2 (y=3): 3 hexagons at x = -sqrt(3), 0, sqrt(3)
    # Total: 3+4+3 = 10. Need one more.
    
    # Let me try:
    # Row 0 (y=0): 4 hexagons
    # Row 1 (y=1.5): 4 hexagons
    # Row 2 (y=3): 3 hexagons
    # Total: 11
    
    sqrt3 = math.sqrt(3)
    positions = []
    # Row 0: y = 0, 4 hexagons
    for i in range(4):
        positions.append((i * sqrt3 - 1.5 * sqrt3, 0.0, 90.0))  # pointy-top: angle 90
    # Row 1: y = 1.5, 4 hexagons, offset
    for i in range(4):
        positions.append((i * sqrt3 - 1.5 * sqrt3 + sqrt3/2, 1.5, 90.0))
    # Row 2: y = 3.0, 3 hexagons
    for i in range(3):
        positions.append((i * sqrt3 - sqrt3, 3.0, 90.0))
    
    # Center the arrangement
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    positions = [(x - cx, y - cy, a) for (x, y, a) in positions]
    
    return positions

def construct():
    start_time = time.time()
    
    # Start with hexagonal lattice seed
    positions = hexagonal_lattice_seed()
    
    # Flatten to parameter vector
    params = np.array([p[0] for p in positions] + [p[1] for p in positions] + [p[2] for p in positions])
    
    # Initial outer center at origin, angle 0
    outer_center = (0.0, 0.0)
    outer_angle = 0.0
    
    # Compute initial side length
    inner_positions = [(params[3*i], params[3*i+1], params[3*i+2]) for i in range(N)]
    initial_side = compute_outer_side(inner_positions, outer_center, outer_angle)
    
    best_side = initial_side
    best_params = params.copy()
    
    # Check if seed is valid
    if constraint_check(params, outer_center, outer_angle):
        best_side = initial_side
    else:
        # Seed might have overlaps; try to fix
        best_side = float('inf')
    
    # Try optimization with different strategies
    # Strategy 1: Local optimization from seed
    try:
        result = minimize(
            lambda p: objective(p, outer_center, outer_angle),
            params,
            method='L-BFGS-B',
            options={'maxiter': 2000, 'ftol': 1e-10}
        )
        elapsed = time.time() - start_time
        if elapsed < DEADLINE:
            if constraint_check(result.x, outer_center, outer_angle):
                side = objective(result.x, outer_center, outer_angle)
                if side < best_side:
                    best_side = side
                    best_params = result.x.copy()
    except Exception:
        pass
    
    # Strategy 2: Try different outer angles
    if time.time() - start_time < DEADLINE - 5:
        for oa in [0, 30, 60, 90, 150]:
            if time.time() - start_time > DEADLINE - 5:
                break
            oc = (0.0, 0.0)
            try:
                result = minimize(
                    lambda p: objective(p, oc, oa),
                    best_params,
                    method='L-BFGS-B',
                    options={'maxiter': 1000, 'ftol': 1e-10}
                )
                if constraint_check(result.x, oc, oa):
                    side = objective(result.x, oc, oa)
                    if side < best_side:
                        best_side = side
                        best_params = result.x.copy()
                        outer_angle = oa
                        outer_center = oc
            except Exception:
                pass
    
    # Strategy 3: Differential evolution for global search (if time permits)
    if time.time() - start_time < DEADLINE - 10:
        # Define bounds
        bounds = [(-5, 5)] * (2 * N) + [(-180, 180)] * N
        try:
            de_result = differential_evolution(
                lambda p: objective(p, outer_center, outer_angle),
                bounds,
                maxiter=50,
                tol=1e-6,
                seed=42,
                polish=True,
                popsize=15,
                mutation=(0.5, 1.5)
            )
            elapsed = time.time() - start_time
            if elapsed < DEADLINE:
                if constraint_check(de_result.x, outer_center, outer_angle):
                    side = objective(de_result.x, outer_center, outer_angle)
                    if side < best_side:
                        best_side = side
                        best_params = de_result.x.copy()
        except Exception:
            pass
    
    # Final refinement
    if time.time() - start_time < DEADLINE - 3 and best_side < float('inf'):
        try:
            result = minimize(
                lambda p: objective(p, outer_center, outer_angle),
                best_params,
                method='L-BFGS-B',
                options={'maxiter': 5000, 'ftol': 1e-12}
            )
            if constraint_check(result.x, outer_center, outer_angle):
                side = objective(result.x, outer_center, outer_angle)
                if side < best_side:
                    best_side = side
                    best_params = result.x.copy()
        except Exception:
            pass
    
    # Build output
    if best_side >= float('inf'):
        # Fallback: use seed even if not perfectly valid
        best_side = initial_side
        best_params = params.copy()
    
    inner = [[best_params[3*i], best_params[3*i+1], best_params[3*i+2]] for i in range(N)]
    return inner, outer_center, best_side, outer_angle

# EVOLVE-BLOCK-START
# (construct is defined above)
# EVOLVE-BLOCK-END
