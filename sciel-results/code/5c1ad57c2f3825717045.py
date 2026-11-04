"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 12

# EVOLVE-BLOCK-START
def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon centered at (cx,cy) with given side and rotation."""
    angle = np.radians(angle_deg)
    # Regular hexagon vertices at angles 0, 60, 120, 180, 240, 300 degrees (flat-top)
    # For a regular hexagon with side s, the distance from center to vertex is s
    # Vertices at angles: 0, 60, 120, 180, 240, 300
    base_angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180.0
    angles = base_angles + angle
    xs = cx + side * np.cos(angles)
    ys = cy + side * np.sin(angles)
    return np.column_stack([xs, ys])

def _point_in_hexagon(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside or on the boundary of a regular hexagon.
    Uses the support function approach: a point is inside if its projection onto
    each of the 3 axis directions (spaced 60 deg apart) is within the apothem."""
    angle = np.radians(angle_deg)
    dx = px - cx
    dy = py - cy
    # Rotate point into hexagon's local frame
    cos_a = np.cos(-angle)
    sin_a = np.sin(-angle)
    lx = dx * cos_a - dy * sin_a
    ly = dx * sin_a + dy * cos_a
    
    # For a regular hexagon with side s, the apothem (distance from center to edge midpoint) is s * sqrt(3)/2
    # The hexagon can be described by |projection on 3 axes| <= apothem
    # Axes are at 0, 60, 120 degrees in local frame
    apothem = side * np.sqrt(3) / 2.0
    
    # Three axes directions (unit vectors)
    # Axis 1: along x (0 deg)
    # Axis 2: 60 deg
    # Axis 3: 120 deg
    proj1 = np.abs(lx)
    proj2 = np.abs(lx * np.cos(np.pi/3) + ly * np.sin(np.pi/3))
    proj3 = np.abs(lx * np.cos(2*np.pi/3) + ly * np.sin(2*np.pi/3))
    
    return (proj1 <= apothem + 1e-12) and (proj2 <= apothem + 1e-12) and (proj3 <= apothem + 1e-12)

def _hexagon_distance_to_hexagon(v1, v2, s1, a1, v2c, v2c_y, s2, a2):
    """Compute minimum distance between two hexagons. If overlapping, return negative."""
    # This is complex; for now use a simple bounding box check + vertex-to-edge distances
    # For efficiency in the optimizer, we'll use a simplified distance metric
    
    # Actually, for the smooth surrogate, we'll use a different approach:
    # We'll compute the maximum "violation" of the containment constraint.
    
    # Let's use a simpler approach: compute distance from center of hex2 to hex1's boundary
    # and vice versa, then take max of negative of these if inside
    
    # For two hexagons, the distance is the minimum distance between any edge of one and any edge of the other
    # This is expensive. Instead, let's use a smooth approximation.
    
    # For now, return a placeholder that will be replaced with proper computation
    return 0.0

def _signed_distance_to_hexagon(px, py, cx, cy, side, angle_deg):
    """Compute signed distance from point (px,py) to the hexagon boundary.
    Negative if inside, positive if outside."""
    angle = np.radians(angle_deg)
    dx = px - cx
    dy = py - cy
    cos_a = np.cos(-angle)
    sin_a = np.sin(-angle)
    lx = dx * cos_a - dy * sin_a
    ly = dx * sin_a + dy * cos_a
    
    apothem = side * np.sqrt(3) / 2.0
    
    # The signed distance for a regular hexagon can be computed using the support function
    # d = max of (|proj_i| - apothem) over the 3 axes, but this gives distance to the "box" approximation
    # For a proper hexagon, we need to handle the 6 edges
    
    # Edges of the hexagon in local coordinates:
    # The hexagon vertices in local frame (before rotation) with side s:
    # (s, 0), (s/2, s*sqrt(3)/2), (-s/2, s*sqrt(3)/2), (-s, 0), (-s/2, -s*sqrt(3)/2), (s/2, -s*sqrt(3)/2)
    
    s = side
    verts = np.array([
        [s, 0],
        [s/2, s*np.sqrt(3)/2],
        [-s/2, s*np.sqrt(3)/2],
        [-s, 0],
        [-s/2, -s*np.sqrt(3)/2],
        [s/2, -s*np.sqrt(3)/2]
    ])
    
    min_dist = np.inf
    for i in range(6):
        p1 = verts[i]
        p2 = verts[(i+1) % 6]
        # Distance from point (lx, ly) to line segment p1-p2
        # First, check if the closest point is on the segment or at an endpoint
        d = p2 - p1
        d2 = np.dot(d, d)
        t = np.dot([lx - p1[0], ly - p1[1]], d) / d2 if d2 > 0 else 0
        t = np.clip(t, 0, 1)
        closest = p1 + t * d
        dist = np.sqrt((lx - closest[0])**2 + (ly - closest[1])**2)
        min_dist = min(min_dist, dist)
    
    # Determine if inside: check if point is on the same side as the center for all edges
    inside = True
    for i in range(6):
        p1 = verts[i]
        p2 = verts[(i+1) % 6]
        # Edge vector
        ex = p2[0] - p1[0]
        ey = p2[1] - p1[1]
        # Normal pointing outward (for CCW polygon, right normal)
        nx = ey
        ny = -ex
        # Check side
        val1 = nx * (0 - p1[0]) + ny * (0 - p1[1])  # center relative to p1
        val2 = nx * (lx - p1[0]) + ny * (ly - p1[1])  # point relative to p1
        if val1 * val2 < -1e-12:
            inside = False
            break
    
    if inside:
        return -min_dist
    else:
        return min_dist

def _hexagon_overlap_penalty(v1, v2, s1, a1, v2x, v2y, s2, a2):
    """Compute a smooth penalty for overlap between two unit hexagons.
    Uses the minimum distance between vertices/edges."""
    # Get vertices of both hexagons
    verts1 = _hex_vertices(v1[0], v1[1], s1, a1)
    verts2 = _hex_vertices(v2x, v2y, s2, a2)
    
    min_dist = np.inf
    for i in range(6):
        for j in range(6):
            d = np.sqrt((verts1[i,0]-verts2[j,0])**2 + (verts1[i,1]-verts2[j,1])**2)
            min_dist = min(min_dist, d)
    
    # Also check edge-to-edge distances (simplified: just use vertex-to-vertex for now)
    # This is a rough approximation; the true minimum distance could be edge-to-edge
    
    # Smooth penalty: if min_dist < threshold, apply quadratic penalty
    threshold = 0.0
    if min_dist < threshold + 0.1:
        return (threshold + 0.1 - min_dist) ** 2
    return 0.0

def _objective(params, outer_side, outer_cx, outer_cy, outer_angle, beta=100.0):
    """Objective function: minimize outer_side + smooth penalty for violations.
    params: [x1, y1, a1, x2, y2, a2, ..., x12, y12, a12]"""
    n = len(params) // 3
    
    # Penalty for inner hexagons going outside the outer hexagon
    penalty = 0.0
    
    for i in range(n):
        cx = params[i*3]
        cy = params[i*3+1]
        ca = params[i*3+2]
        
        # Check all 6 vertices of inner hexagon against outer hexagon
        verts = _hex_vertices(cx, cy, 1.0, ca)
        for k in range(6):
            vx, vy = verts[k]
            sd = _signed_distance_to_hexagon(vx, vy, outer_cx, outer_cy, outer_side, outer_angle)
            if sd > 0:  # Outside
                penalty += sd ** 2
    
    # Penalty for inner hexagons overlapping each other
    for i in range(n):
        for j in range(i+1, n):
            cx1 = params[i*3]
            cy1 = params[i*3+1]
            ca1 = params[i*3+2]
            cx2 = params[j*3]
            cy2 = params[j*3+1]
            ca2 = params[j*3+2]
            
            # Quick bounding box check
            if abs(cx1 - cx2) > 2.5 or abs(cy1 - cy2) > 2.5:
                continue
            
            penalty += _hexagon_overlap_penalty((cx1, cy1), 1.0, ca1, cx2, cy2, 1.0, ca2)
    
    # Soft max of violations using log-sum-exp
    # For the outer containment, we want max violation to be 0
    # Use: (1/beta) * log(sum(exp(beta * violation_i)))
    
    return outer_side + 10.0 * penalty

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Initial guess: place hexagons in a honeycomb pattern
    # For 12 hexagons, a 4x3 grid or similar
    # Let's try a compact arrangement
    
    # Honeycomb positions for 12 hexagons (side length 1)
    # In a honeycomb, adjacent hexagons are separated by 2 * apothem = sqrt(3) in one direction
    # and by 1.5 in the other... let me think more carefully.
    
    # For flat-top hexagons with side 1:
    # Horizontal spacing: 1.5 (center to center for adjacent in same row)
    # Wait, for flat-top: the width is 2*side = 2, height is sqrt(3)*side = sqrt(3)
    # Center-to-center horizontal: 1.5 * side = 1.5
    # Center-to-center vertical: sqrt(3) * side = sqrt(3)
    # For staggered rows, vertical offset is sqrt(3)/2
    
    # Let's use pointy-top for easier honeycomb:
    # Actually, let's just use a grid and let the optimizer figure it out.
    
    # Initial positions: 4 columns x 3 rows
    positions = []
    for row in range(3):
        for col in range(4):
            x = col * 1.732  # approx sqrt(3)
            y = row * 1.5
            if row % 2 == 1:
                x += 0.866  # stagger
            positions.append([x, y, 0.0])
    
    # Center the configuration
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    cx_mean = (max(xs) + min(xs)) / 2
    cy_mean = (max(ys) + min(ys)) / 2
    for p in positions:
        p[0] -= cx_mean
        p[1] -= cy_mean
    
    params0 = np.array([p[0] for p in positions] + [p[1] for p in positions] + [p[2] for p in positions])
    
    # Initial outer hexagon estimate
    # Bounding box of inner hexagons
    max_extent = max(max(abs(p[0]) + 1.0 for p in positions), max(abs(p[1]) + 1.0 for p in positions))
    outer_side0 = max_extent / (np.sqrt(3)/2)  # rough estimate
    outer_cx0 = 0.0
    outer_cy0 = 0.0
    outer_angle0 = 0.0
    
    best_result = None
    best_score = float('inf')
    
    # Try multiple random initializations
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        if restart == 0:
            params = params0.copy()
        else:
            # Add some randomness
            params = params0.copy() + np.random.normal(0, 0.5, len(params0))
            # Randomize angles
            for i in range(N):
                params[i*3+2] = np.random.uniform(-30, 30)
        
        # Optimize with L-BFGS-B
        try:
            res = minimize(
                _objective,
                params,
                args=(outer_side0, outer_cx0, outer_cy0, outer_angle0, 100.0),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10}
            )
            
            if res.success or res.fun < best_score:
                best_score = res.fun
                best_params = res.x.copy()
                
                if time.time() > deadline:
                    break
        except Exception:
            continue
    
    # Use the best params found
    if best_params is not None:
        params = best_params
    else:
        params = params0
    
    # Extract inner hexagon positions
    inner = []
    for i in range(N):
        inner.append([params[i*3], params[i*3+1], params[i*3+2]])
    
    # Compute the actual outer side length needed
    # Find the minimum outer hexagon side that contains all inner hexagons
    # This requires solving: for a given outer hexagon, check if all inner vertices are inside
    
    # For now, estimate the outer side from the bounding box
    max_x = max(max(p[0] for p in inner) + 1.0, 0)
    max_y = max(max(p[1] for p in inner) + 1.0, 0)
    min_x = min(min(p[0] for p in inner) - 1.0, 0)
    min_y = min(min(p[1] for p in inner) - 1.0, 0)
    
    # The outer hexagon needs to contain all points
    # For a regular hexagon with side s centered at origin, the maximum extent in x is s, in y is s*sqrt(3)/2
    # So we need s >= max(|x|) and s*sqrt(3)/2 >= max(|y|)
    # Thus s >= max(max_x, max_y / (sqrt(3)/2))
    
    outer_side = max(max_x, max_y / (np.sqrt(3)/2))
    
    # But this is for axis-aligned. With rotation, we can do better.
    # Let's try a few rotations
    best_outer_side = outer_side
    for angle in [0, 30, 60, 90, 120, 150]:
        # Rotate all inner points by -angle and compute required outer side
        a = np.radians(angle)
        cos_a = np.cos(-a)
        sin_a = np.sin(-a)
        
        max_x_rot = 0
        max_y_rot = 0
        for p in inner:
            # Rotate each vertex of each inner hexagon
            verts = _hex_vertices(p[0], p[1], 1.0, p[2] - angle)
            for v in verts:
                rx = v[0] * cos_a - v[1] * sin_a
                ry = v[0] * sin_a + v[1] * cos_a
                max_x_rot = max(max_x_rot, abs(rx))
                max_y_rot = max(max_y_rot, abs(ry))
        
        side_needed = max(max_x_rot, max_y_rot / (np.sqrt(3)/2))
        best_outer_side = min(best_outer_side, side_needed)
    
    return inner, (0.0, 0.0), best_outer_side, 0.0
# EVOLVE-BLOCK-END
