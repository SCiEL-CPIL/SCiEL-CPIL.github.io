"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

# Constants for regular hexagon geometry
# A regular hexagon with side length s has:
# - Circumradius (center to vertex) = s
# - Inradius (center to side) = s * sqrt(3)/2

def hexagon_vertices(center, side, angle_deg):
    """Return 6 vertices of a regular hexagon."""
    cx, cy = center
    angle_rad = np.deg2rad(angle_deg)
    # Vertices at angles 0, 60, 120, 180, 240, 300 degrees relative to angle_rad
    angles = angle_rad + np.arange(6) * np.pi / 3
    xs = cx + side * np.cos(angles)
    ys = cy + side * np.sin(angles)
    return np.column_stack([xs, ys])

def hexagon_edges(center, side, angle_deg):
    """Return the 6 edges as (p1, p2) pairs."""
    verts = hexagon_vertices(center, side, angle_deg)
    edges = []
    for i in range(6):
        p1 = verts[i]
        p2 = verts[(i + 1) % 6]
        edges.append((p1, p2))
    return edges

def segment_segment_distance(p1, p2, p3, p4):
    """Compute minimum distance between two line segments p1-p2 and p3-p4."""
    d1 = p2 - p1
    d2 = p4 - p3
    r = p1 - p3
    a = np.dot(d1, d1)
    e = np.dot(d2, d2)
    f = np.dot(d2, r)
    
    EPS = 1e-12
    if a <= EPS and e <= EPS:
        return np.linalg.norm(p1 - p3)
    if a <= EPS:
        s = 0.0
        t = f / e
        t = np.clip(t, 0, 1)
        return np.linalg.norm(p1 - (p3 + t * d2))
    if e <= EPS:
        t = 0.0
        s = -np.dot(d1, r) / a
        s = np.clip(s, 0, 1)
        return np.linalg.norm(p1 + s * d1 - p3)
    
    c = np.dot(d1, r)
    b = np.dot(d1, d2)
    denom = a * e - b * b
    s = 0.0
    t = 0.0
    if denom != 0:
        s = np.clip((b * f - c * e) / denom, 0, 1)
    t = (b * s + f) / e
    if t < 0:
        t = 0
        s = np.clip(-c / a, 0, 1)
    elif t > 1:
        t = 1
        s = np.clip((b - c) / a, 0, 1)
    
    p1s = p1 + s * d1
    p3t = p3 + t * d2
    return np.linalg.norm(p1s - p3t)

def min_edge_distance_hexagons(center1, side1, angle1, center2, side2, angle2):
    """
    Compute the minimum distance between edges of two regular hexagons.
    This is a smooth-ish approximation for non-overlap.
    """
    edges1 = hexagon_edges(center1, side1, angle1)
    edges2 = hexagon_edges(center2, side2, angle2)
    
    min_dist = np.inf
    for e1 in edges1:
        for e2 in edges2:
            d = segment_segment_distance(e1[0], e1[1], e2[0], e2[1])
            if d < min_dist:
                min_dist = d
    return min_dist

def point_in_hexagon(px, py, center, side, angle_deg):
    """Check if point (px, py) is inside or on the boundary of a regular hexagon."""
    cx, cy = center
    angle_rad = np.deg2rad(angle_deg)
    
    # Transform point to local coordinates of the hexagon
    dx = px - cx
    dy = py - cy
    
    # Rotate by -angle_rad to align hexagon with standard orientation
    cos_a = np.cos(-angle_rad)
    sin_a = np.sin(-angle_rad)
    lx = dx * cos_a - dy * sin_a
    ly = dx * sin_a + dy * cos_a
    
    # Standard hexagon centered at origin with side length s, flat-top orientation
    # Vertices at (s, 0), (s/2, s*sqrt(3)/2), (-s/2, s*sqrt(3)/2), (-s, 0), (-s/2, -s*sqrt(3)/2), (s/2, -s*sqrt(3)/2)
    # For a point to be inside, it must satisfy 6 half-plane inequalities
    
    sqrt3 = np.sqrt(3)
    # The 6 normal vectors and offsets for the standard flat-top hexagon
    # Edge 1: right vertical edge from (s/2, s*sqrt(3)/2) to (s/2, -s*sqrt(3)/2) -> normal (1, 0), offset s/2
    # Actually, let's use the vertex-based approach or the standard inequality form
    
    # For a regular hexagon with circumradius R = side, the inequalities are:
    # |x| <= R
    # |x*sqrt(3)/2 + y/2| <= R * sqrt(3)/2  (not quite)
    
    # Let's use the half-plane method properly.
    # Standard flat-top hexagon vertices (counterclockwise from right):
    # V0 = (R, 0)
    # V1 = (R/2, R*sqrt(3)/2)
    # V2 = (-R/2, R*sqrt(3)/2)
    # V3 = (-R, 0)
    # V4 = (-R/2, -R*sqrt(3)/2)
    # V5 = (R/2, -R*sqrt(3)/2)
    
    # The 6 edges and their outward normals:
    # Edge V0-V1: normal is perpendicular to (V1-V0), rotated outward
    # V1-V0 = (-R/2, R*sqrt(3)/2), length R
    # Normal = (R*sqrt(3)/2, R/2) normalized -> (sqrt(3)/2, 1/2)
    # Check: dot(normal, V0) = sqrt(3)/2 * R = R*sqrt(3)/2
    # So inequality: sqrt(3)/2 * x + 1/2 * y <= R*sqrt(3)/2
    
    # Edge V1-V2: normal = (0, 1), offset R*sqrt(3)/2
    # y <= R*sqrt(3)/2
    
    # Edge V2-V3: normal = (-sqrt(3)/2, 1/2), offset R*sqrt(3)/2
    # -sqrt(3)/2 * x + 1/2 * y <= R*sqrt(3)/2
    
    # Edge V3-V4: normal = (-sqrt(3)/2, -1/2), offset R*sqrt(3)/2
    # -sqrt(3)/2 * x - 1/2 * y <= R*sqrt(3)/2
    
    # Edge V4-V5: normal = (0, -1), offset R*sqrt(3)/2
    # -y <= R*sqrt(3)/2  =>  y >= -R*sqrt(3)/2
    
    # Edge V5-V0: normal = (sqrt(3)/2, -1/2), offset R*sqrt(3)/2
    # sqrt(3)/2 * x - 1/2 * y <= R*sqrt(3)/2
    
    R = side
    sqrt3 = np.sqrt(3)
    
    # Check all 6 inequalities
    if sqrt3/2 * lx + 1/2 * ly > R * sqrt3/2 + 1e-9:
        return False
    if ly > R * sqrt3/2 + 1e-9:
        return False
    if -sqrt3/2 * lx + 1/2 * ly > R * sqrt3/2 + 1e-9:
        return False
    if -sqrt3/2 * lx - 1/2 * ly > R * sqrt3/2 + 1e-9:
        return False
    if -ly > R * sqrt3/2 + 1e-9:
        return False
    if sqrt3/2 * lx - 1/2 * ly > R * sqrt3/2 + 1e-9:
        return False
    
    return True

def check_validity(inner, outer_center, outer_side, outer_angle):
    """
    Check if the packing is valid:
    1. All inner hexagons are inside the outer hexagon
    2. No two inner hexagons overlap
    """
    ocx, ocy = outer_center
    
    # Check containment: each vertex of each inner hexagon must be inside outer hexagon
    for ix, iy, iangle in inner:
        verts = hexagon_vertices((ix, iy), 1.0, iangle)
        for vx, vy in verts:
            if not point_in_hexagon(vx, vy, (ocx, ocy), outer_side, outer_angle):
                return False
    
    # Check non-overlap: minimum edge distance between any pair should be >= 0
    n = len(inner)
    for i in range(n):
        for j in range(i + 1, n):
            ix_i, iy_i, ia_i = inner[i]
            ix_j, iy_j, ia_j = inner[j]
            # If centers are too far apart, skip detailed check
            dist_centers = np.sqrt((ix_i - ix_j)**2 + (iy_i - iy_j)**2)
            if dist_centers > 2.0 * np.sqrt(3):  # Max possible distance for touching hexagons
                continue
            min_d = min_edge_distance_hexagons((ix_i, iy_i), 1.0, ia_i, (ix_j, iy_j), 1.0, ia_j)
            if min_d < -1e-6:  # Negative means overlap
                return False
    
    return True

def objective(params, N, outer_angle):
    """
    Objective function: minimize outer side length.
    params: [ox, oy, side, x1, y1, a1, x2, y2, a2, ..., xN, yN, aN]
    """
    ox, oy = params[0], params[1]
    side = params[2]
    
    if side < 1.0:
        return 1e10
    
    inner = []
    for i in range(N):
        base = 3 + i * 3
        ix, iy, ia = params[base], params[base+1], params[base+2]
        inner.append((ix, iy, ia))
    
    # Check containment
    for ix, iy, iangle in inner:
        verts = hexagon_vertices((ix, iy), 1.0, iangle)
        for vx, vy in verts:
            if not point_in_hexagon(vx, vy, (ox, oy), side, outer_angle):
                return 1e10
    
    # Check non-overlap with penalty
    penalty = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            ix_i, iy_i, ia_i = inner[i]
            ix_j, iy_j, ia_j = inner[j]
            dist_centers = np.sqrt((ix_i - ix_j)**2 + (iy_i - iy_j)**2)
            if dist_centers > 2.0 * np.sqrt(3):
                continue
            min_d = min_edge_distance_hexagons((ix_i, iy_i), 1.0, ia_i, (ix_j, iy_j), 1.0, ia_j)
            if min_d < 0:
                penalty += min_d * min_d
    
    if penalty > 1e-6:
        return 1e10 + penalty
    
    return side

def construct():
    N = 11
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Initial guess: grid arrangement
    # For N=11, a 3x4 or similar grid
    # Let's start with a hexagonal grid pattern
    init_inner = []
    # Center hexagon at (0,0)
    init_inner.append((0.0, 0.0, 0.0))
    # 6 around it
    for k in range(6):
        angle = k * 60
        x = 2.0 * np.sqrt(3) * np.cos(np.deg2rad(angle))
        y = 2.0 * np.sqrt(3) * np.sin(np.deg2rad(angle))
        init_inner.append((x, y, 0.0))
    # 4 more for total 11
    # Add at distance 4*sqrt(3)
    for k in range(4):
        angle = k * 90 + 30
        x = 4.0 * np.sqrt(3) * np.cos(np.deg2rad(angle))
        y = 4.0 * np.sqrt(3) * np.sin(np.deg2rad(angle))
        init_inner.append((x, y, 0.0))
    
    # Initial outer hexagon
    outer_side_init = 15.0
    outer_center_init = (0.0, 0.0)
    outer_angle = 0.0
    
    # Build initial parameter vector
    params_init = [outer_center_init[0], outer_center_init[1], outer_side_init]
    for ix, iy, ia in init_inner:
        params_init.extend([ix, iy, ia])
    
    params_init = np.array(params_init)
    
    best_side = outer_side_init
    best_params = params_init.copy()
    
    # Try optimization
    try:
        result = minimize(
            objective,
            params_init,
            args=(N, outer_angle),
            method='L-BFGS-B',
            options={'maxiter': 200, 'maxfun': 5000, 'ftol': 1e-6, 'gtol': 1e-4}
        )
        
        if result.success or result.fun < best_side:
            if result.fun < 1e9:  # Valid solution
                best_side = result.fun
                best_params = result.x
    
    except Exception as e:
        pass
    
    # Check if we have time for a second attempt with different initial guess
    elapsed = time.time() - start_time
    if elapsed < deadline - 60:
        # Try a different initial arrangement
        init_inner2 = []
        # Linear arrangement
        for k in range(N):
            init_inner2.append((k * 3.5, 0.0, 0.0))
        
        outer_side_init2 = 25.0
        params_init2 = [0.0, 0.0, outer_side_init2]
        for ix, iy, ia in init_inner2:
            params_init2.extend([ix, iy, ia])
        
        params_init2 = np.array(params_init2)
        
        try:
            result2 = minimize(
                objective,
                params_init2,
                args=(N, outer_angle),
                method='L-BFGS-B',
                options={'maxiter': 200, 'maxfun': 5000, 'ftol': 1e-6, 'gtol': 1e-4}
            )
            
            if (result2.success or result2.fun < best_side) and result2.fun < 1e9:
                best_side = result2.fun
                best_params = result2.x
        
        except Exception as e:
            pass
    
    # Extract best solution
    ox, oy = best_params[0], best_params[1]
    side = best_params[2]
    
    inner = []
    for i in range(N):
        base = 3 + i * 3
        ix, iy, ia = best_params[base], best_params[base+1], best_params[base+2]
        inner.append([ix, iy, ia])
    
    # Verify validity
    if not check_validity(inner, (ox, oy), side, outer_angle):
        # Fall back to a simple valid arrangement
        inner = [[k * 3.5, 0.0, 0.0] for k in range(N)]
        side = 20.0
        ox, oy = 0.0, 0.0
    
    return inner, (ox, oy), side, outer_angle
# EVOLVE-BLOCK-END
