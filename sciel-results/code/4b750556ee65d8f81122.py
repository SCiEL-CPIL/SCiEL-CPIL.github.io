"""Packing 12 unit hexagons in a regular hexagon via SDF-based optimization."""
import numpy as np
from scipy.optimize import minimize
import time

N = 12

# EVOLVE-BLOCK-START

def _hex_vertices(cx, cy, side, angle_deg):
    """Vertices of a regular hexagon centered at (cx,cy) with given side and rotation."""
    a = np.radians(angle_deg)
    angles = a + np.arange(6) * np.pi / 3.0
    return np.column_stack([cx + side * np.cos(angles),
                            cy + side * np.sin(angles)])

def _point_in_hex_dist(px, py, cx, cy, side, angle_deg):
    """
    Signed distance from point (px,py) to hexagon centered at (cx,cy),
    side length `side`, rotated by angle_deg.
    Positive = outside, negative = inside.
    Uses the standard hexagon SDF.
    """
    # Transform point into hexagon's local frame
    dx = px - cx
    dy = py - cy
    a = np.radians(angle_deg)
    ca, sa = np.cos(a), np.sin(a)
    lx = ca * dx + sa * dy
    ly = -sa * dx + ca * dy
    
    # Regular hexagon with side length s: apothem = s * sqrt(3)/2
    apothem = side * np.sqrt(3.0) / 2.0
    
    # For a regular hexagon, the SDF can be computed by checking against 3 pairs of parallel edges
    # The hexagon in local coords (pointing right) has vertices at:
    # (s, 0), (s/2, s*sqrt(3)/2), (-s/2, s*sqrt(3)/2), (-s, 0), (-s/2, -s*sqrt(3)/2), (s/2, -s*sqrt(3)/2)
    # Edge normals (outward):
    # Right edge: (1, 0), at x = s
    # Upper-right: (1/2, sqrt(3)/2), at distance apothem
    # Upper-left: (-1/2, sqrt(3)/2), at distance apothem
    # etc.
    
    # Use the formula: for a regular hexagon with apothem a, the SDF is:
    # min over 3 directions of (dot(p, n_i) - a) where n_i are the 3 outward normal directions
    # But we need to also check the "corner" regions.
    
    # Standard hexagon SDF:
    # q = abs(p) - [a, a/sqrt(3)] in a rotated frame... 
    # Let me use a more direct approach:
    
    # The hexagon is the intersection of 6 half-planes.
    # In local coordinates, the 6 edge constraints are:
    # |lx| <= s  (left and right edges, but actually these are at x=±s for a flat-top)
    # Wait, for a pointy-top hexagon (vertex at top), the edges are at different angles.
    
    # Let me use the standard regular hexagon SDF formula:
    # For a regular hexagon with circumradius R = side (since for regular hexagon, side = circumradius):
    # The SDF is: min of distances to each of the 6 edges, but properly clipped.
    
    # Actually, for a regular hexagon with side length s:
    # circumradius = s, inradius (apothem) = s * sqrt(3)/2
    # The 6 edge normals point at angles 0, 60, 120, 180, 240, 300 degrees
    # Each edge is at distance apothem from center along its normal.
    
    # SDF = max over the 6 half-plane distances, but with corner correction.
    # Simpler: use the formula from Inigo Quilez:
    # vec2 p = abs(local_point);
    # float d = max(p.x * cos(PI/6) + p.y * sin(PI/6), p.y) - apothem;
    # But this is for a specific orientation.
    
    # Let me just compute distance to the hexagon properly:
    # The hexagon in local coords (with vertex pointing in +x direction after our rotation):
    # Vertices: (s,0), (s/2, s√3/2), (-s/2, s√3/2), (-s,0), (-s/2,-s√3/2), (s/2,-s√3/2)
    
    # I'll use the method: project onto the 3 main axes and take the max of the 3 "slab" distances.
    # For a regular hexagon, it's the intersection of 3 strips.
    
    # The 3 strip normals (in local frame) are at angles 0°, 60°, 120° (and their opposites)
    # Strip 1: |lx| <= s  (normals at 0° and 180°)
    # Strip 2: |lx*cos(60°) + ly*sin(60°)| <= s*cos(60°) = s/2 ... no wait.
    
    # Let me think again. The hexagon with vertices at angles 0, 60, 120, 180, 240, 300 (radius s):
    # Edge between vertex 0 (s,0) and vertex 60 (s/2, s√3/2):
    #   midpoint = (3s/4, s√3/4), normal direction = (√3/2, -1/2) rotated... 
    #   Actually the outward normal to this edge points at angle 30° (bisector of 0° and 60°)
    #   Distance from center to this edge = s * cos(30°) = s * √3/2 = apothem
    #   So constraint: lx*cos(30°) + ly*sin(30°) <= apothem
    # Similarly for all 6 edges, normals at 30°, 90°, 150°, 210°, 270°, 330°
    
    # So the 6 half-plane constraints in local coords:
    # For k = 0..5: lx*cos(30°+k*60°) + ly*sin(30°+k*60°) <= apothem
    
    # The SDF (signed distance, negative inside) is:
    # d = max_k [lx*cos(θ_k) + ly*sin(θ_k) - apothem]
    # But this gives the correct SDF only for convex polygons when we also handle the "beyond vertex" cases.
    # For a regular hexagon, this max-of-linear-functions IS the exact SDF because it's a convex polygon
    # and the distance to a convex polygon from outside is indeed the max of the edge distances when
    # the projection falls on an edge. But when the closest point is a vertex, we need the vertex distance.
    
    # For simplicity and differentiability, I'll use a smooth approximation or just the max formulation
    # which is exact for the hexagon (since it's the support function dual).
    
    # Actually, for a convex polygon, the signed distance function from outside is:
    # d(p) = max_i (n_i · p - h_i) where n_i are unit outward normals and h_i are support distances
    # This is EXACT for convex polygons. Inside, it's negative and equals the min distance to boundary.
    
    theta = np.radians(30) + np.arange(6) * np.radians(60)
    nx = np.cos(theta)
    ny = np.sin(theta)
    
    dists = lx * nx + ly * ny - apothem
    d = np.max(dists)
    
    return d

def _hex_hex_overlap_distance(v1, v2, side, a1, a2):
    """
    Approximate minimum distance between two hexagons.
    Positive = separated, negative = overlapping.
    We use the fact that for two convex shapes, if they overlap, the SDF of one evaluated
    at vertices of the other will be negative.
    For a proper packing, we need all vertices of each hex to be outside the other.
    """
    verts1 = _hex_vertices(v1[0], v1[1], side, a1)
    verts2 = _hex_vertices(v2[0], v2[1], side, a2)
    
    # Check all vertices of hex1 against hex2's SDF
    d1 = np.min([_point_in_hex_dist(px, py, v2[0], v2[1], side, a2) for px, py in verts1])
    # Check all vertices of hex2 against hex1's SDF
    d2 = np.min([_point_in_hex_dist(px, py, v1[0], v1[1], side, a1) for px, py in verts2])
    
    # The minimum separation is the max of these (most positive = most separated)
    # If both are positive, hexagons are separated
    # If either is negative, they overlap
    return max(d1, d2)

def _outer_hex_constraint(px, py, ocx, ocy, side, angle_deg):
    """
    Signed distance from point to OUTER hexagon boundary.
    Negative = inside outer hex, positive = outside.
    We want all inner hex vertices to be inside, so this should be <= 0.
    """
    return -_point_in_hex_dist(px, py, ocx, ocy, side, angle_deg)

def _objective(params, outer_side, outer_angle=0.0, outer_cx=0.0, outer_cy=0.0):
    """
    Objective: minimize outer_side such that all constraints are satisfied.
    params: [x0,y0,a0, x1,y1,a1, ..., x11,y11,a11] (36 values)
    We minimize a penalty function that encourages feasibility.
    """
    n_params = len(params)
    if n_params != 3 * N:
        return 1e6
    
    xs = params[0::3]
    ys = params[1::3]
    angles = params[2::3]
    
    penalty = 0.0
    max_violation = 0.0
    
    # Check inner hexagons don't overlap each other
    for i in range(N):
        for j in range(i + 1, N):
            v1 = (xs[i], ys[i])
            v2 = (xs[j], ys[j])
            d = _hex_hex_overlap_distance(v1, v2, 1.0, angles[i], angles[j])
            if d < 0:
                # Overlapping - penalize
                violation = -d
                penalty += violation * violation * 100.0
                max_violation = max(max_violation, violation)
    
    # Check all inner hex vertices are inside outer hexagon
    for i in range(N):
        verts = _hex_vertices(xs[i], ys[i], 1.0, angles[i])
        for px, py in verts:
            # Distance from point to outer hex boundary (negative = inside)
            d_out = _point_in_hex_dist(px, py, outer_cx, outer_cy, outer_side, outer_angle)
            if d_out > 0:
                violation = d_out
                penalty += violation * violation * 100.0
                max_violation = max(max_violation, violation)
    
    return penalty + max_violation * 10.0

def _solve_with_given_outer(outer_side, outer_angle=0.0, outer_cx=0.0, outer_cy=0.0):
    """Try to find a valid packing for a given outer hexagon."""
    # Initial guess: arrange hexagons in a hexagonal lattice pattern
    # For 12 hexagons, a good initial arrangement:
    # Center + 6 around it + 5 more in a ring, or two rows, etc.
    
    # Let's try a hexagonal close-packed arrangement
    # Hexagons with side 1 have center-to-center distance sqrt(3) for adjacent in same row
    # and sqrt(3) for adjacent rows (offset by sqrt(3)/2 in y, 1.5 in x)
    
    # Actually for unit hexagons (side=1), the minimum center distance for non-overlapping
    # hexagons with same orientation is sqrt(3) ≈ 1.732 (distance between parallel edges = sqrt(3))
    
    # Let me place them in a pattern:
    # Row 0: 4 hexagons
    # Row 1: 4 hexagons (offset)
    # Row 2: 4 hexagons
    # Total: 12
    
    # For hexagons with same orientation, center spacing:
    # Horizontal: 2 * (side * cos(30°)) = 2 * sqrt(3)/2 = sqrt(3) for adjacent in row
    # Wait, for flat-top hexagons, horizontal spacing between centers of adjacent hexes = 2 * apothem = sqrt(3)
    # Vertical spacing = 1.5 * side = 1.5, with horizontal offset of apothem = sqrt(3)/2
    
    # Let me use a simpler initial guess: grid-like arrangement
    dx = np.sqrt(3) + 0.01  # slight gap
    dy = 1.5 + 0.01
    offset = np.sqrt(3) / 2
    
    init = np.zeros(3 * N)
    k = 0
    for row in range(4):
        for col in range(3):
            x = (col - 1) * dx + (row % 2) * offset
            y = (row - 1.5) * dy
            init[k] = x
            init[k+1] = y
            init[k+2] = 0.0
            k += 1
    
    # Optimize positions and angles
    best_result = None
    best_penalty = 1e10
    
    for attempt in range(3):
        if attempt == 0:
            x0 = init.copy()
        elif attempt == 1:
            x0 = init + np.random.randn(3 * N) * 0.1
        else:
            x0 = init + np.random.randn(3 * N) * 0.3
        
        res = minimize(_objective, x0, args=(outer_side, outer_angle, outer_cx, outer_cy),
                      method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-8})
        
        if res.fun < best_penalty:
            best_penalty = res.fun
            best_result = res
    
    return best_result, best_penalty

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Binary search on outer side length
    # Lower bound: even 1 hexagon needs outer side >= 1 (actually > 1 due to geometry)
    # For a single unit hexagon, minimum enclosing hexagon side = 1 (if aligned) or slightly more
    # For 12 hexagons, rough estimate: they span about 4-5 units in each direction
    # So outer side might be around 4-6
    
    # Let's first try to find a valid packing, then minimize
    
    # Start with a reasonable guess for outer side
    # A hexagonal arrangement of 12 hexagons:
    # If arranged in a 3-4-3-2 or similar pattern...
    # Let's just do a direct optimization: minimize outer_side subject to constraints
    
    # Alternative: parametrize everything and minimize outer_side
    # params: outer_cx, outer_cy, outer_side, outer_angle, x0,y0,a0, ..., x11,y11,a11
    # That's 4 + 36 = 40 parameters
    
    # Objective: outer_side + penalty for violations
    
    def _full_objective(params):
        outer_cx = params[0]
        outer_cy = params[1]
        outer_side = params[2]
        outer_angle = params[3]
        
        if outer_side < 0.5:
            return 1e6
        
        xs = params[4:34:3]
        ys = params[5:35:3]
        angles = params[6:36:3]
        
        penalty = 0.0
        max_viol = 0.0
        
        # Inner-inner non-overlap
        for i in range(N):
            for j in range(i + 1, N):
                d = _hex_hex_overlap_distance((xs[i], ys[i]), (xs[j], ys[j]), 1.0, angles[i], angles[j])
                if d < 0:
                    v = -d
                    penalty += v * v * 50.0
                    max_viol = max(max_viol, v)
        
        # Inner vertices inside outer
        for i in range(N):
            verts = _hex_vertices(xs[i], ys[i], 1.0, angles[i])
            for px, py in verts:
                d_out = _point_in_hex_dist(px, py, outer_cx, outer_cy, outer_side, outer_angle)
                if d_out > 0:
                    v = d_out
                    penalty += v * v * 50.0
                    max_viol = max(max_viol, v)
        
        return outer_side + penalty + max_viol * 5.0
    
    # Initial guess
    np.random.seed(42)
    
    # Place inner hexagons in a hexagonal pattern centered at origin
    dx = np.sqrt(3)
    dy = 1.5
    offset = np.sqrt(3) / 2
    
    x0 = np.zeros(40)
    x0[0] = 0.0  # outer_cx
    x0[1] = 0.0  # outer_cy
    x0[2] = 5.0  # outer_side initial guess
    x0[3] = 0.0  # outer_angle
    
    k = 4
    positions = []
    # 7 hexagons in a hex pattern (center + 6 around)
    positions.append((0, 0))
    for ang in range(6):
        a = np.radians(60 * ang)
        positions.append((dx * np.cos(a), dx * np.sin(a)))
    # 5 more in a second ring
    for ang in range(6):
        a = np.radians(60 * ang + 30)
        positions.append((2*dx * np.cos(a), 2*dx * np.sin(a)))
    
    for i in range(N):
        x0[k] = positions[i][0]
        x0[k+1] = positions[i][1]
        x0[k+2] = 0.0
        k += 3
    
    best_result = None
    best_val = 1e10
    
    # Multiple restarts with different initial outer sizes
    initial_sides = [5.0, 5.5, 6.0, 4.5, 6.5]
    
    for side_init in initial_sides:
        if time.time() > deadline - 30:
            break
        
        x0_trial = x0.copy()
        x0_trial[2] = side_init
        x0_trial[4:] += np.random.randn(36) * 0.05
        
        res = minimize(_full_objective, x0_trial, method='L-BFGS-B',
                      options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
        
        if res.fun < best_val:
            best_val = res.fun
            best_result = res
    
    # Polish with Nelder-Mead on the best result
    if best_result is not None:
        if time.time() < deadline - 10:
            res_nm = minimize(_full_objective, best_result.x, method='Nelder-Mead',
                            options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
            if res_nm.fun < best_val:
                best_val = res_nm.fun
                best_result = res_nm
    
    # Extract solution
    if best_result is None:
        # Fallback: return the seed solution
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    params = best_result.x
    outer_cx = params[0]
    outer_cy = params[1]
    outer_side = max(params[2], 1.0)  # ensure positive
    outer_angle = params[3]
    
    inner = []
    for i in range(N):
        x = params[4 + 3*i]
        y = params[5 + 3*i]
        a = params[6 + 3*i]
        # Normalize angle to [0, 360)
        a = a % 360.0
        inner.append([float(x), float(y), float(a)])
    
    # Verify and if invalid, try a simple feasible solution
    # Quick check: compute max violation
    max_viol = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            d = _hex_hex_overlap_distance((inner[i][0], inner[i][1]), (inner[j][0], inner[j][1]), 1.0, inner[i][2], inner[j][2])
            if d < 0:
                max_viol = max(max_viol, -d)
        verts = _hex_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2])
        for px, py in verts:
            d_out = _point_in_hex_dist(px, py, outer_cx, outer_cy, outer_side, outer_angle)
            if d_out > 0:
                max_viol = max(max_viol, d_out)
    
    if max_viol > 0.01:
        # Try to fix with a few more iterations
        if time.time() < deadline - 5:
            for _ in range(100):
                res_fix = minimize(_full_objective, params, method='L-BFGS-B',
                                  options={'maxiter': 50, 'ftol': 1e-10})
                params = res_fix.x
                outer_cx = params[0]
                outer_cy = params[1]
                outer_side = max(params[2], 1.0)
                outer_angle = params[3]
                inner = []
                for i in range(N):
                    x = params[4 + 3*i]
                    y = params[5 + 3*i]
                    a = params[6 + 3*i] % 360.0
                    inner.append([float(x), float(y), float(a)])
                
                # Recheck
                max_viol = 0.0
                for i in range(N):
                    for j in range(i + 1, N):
                        d = _hex_hex_overlap_distance((inner[i][0], inner[i][1]), (inner[j][0], inner[j][1]), 1.0, inner[i][2], inner[j][2])
                        if d < 0:
                            max_viol = max(max_viol, -d)
                    verts = _hex_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2])
                    for px, py in verts:
                        d_out = _point_in_hex_dist(px, py, outer_cx, outer_cy, outer_side, outer_angle)
                        if d_out > 0:
                            max_viol = max(max_viol, d_out)
                if max_viol < 0.01:
                    break
    
    return inner, (float(outer_cx), float(outer_cy)), float(outer_side), float(outer_angle)

# EVOLVE-BLOCK-END
