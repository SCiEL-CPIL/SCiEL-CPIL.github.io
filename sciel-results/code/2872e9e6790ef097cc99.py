"""Fitting 12 unit hexagons inside a regular hexagon, minimizing outer side length."""
import numpy as np
from scipy.optimize import differential_evolution, minimize
import time
import itertools

N = 12
DEADLINE = 140.0  # seconds, safety margin below 150s

def hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon centered at (cx,cy) with given side length and rotation."""
    angle_rad = np.radians(angle_deg)
    # Flat-top orientation: vertices at angles 0, 60, 120, 180, 240, 300 degrees
    angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180 + angle_rad
    verts = np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])
    return verts

def polygon_overlap(v1, v2, tol=1e-9):
    """Check if two convex polygons overlap (including touching edges count as overlap)."""
    # Use separating axis theorem for convex polygons
    for verts in (v1, v2):
        for i in range(len(verts)):
            ax = verts[i+1][1] - verts[i][1]
            ay = -(verts[i+1][0] - verts[i][0])
            # Project both polygons onto this normal
            proj1 = np.dot(v1, [ax, ay])
            proj2 = np.dot(v2, [ax, ay])
            min1, max1 = proj1.min(), proj1.max()
            min2, max2 = proj2.min(), proj2.max()
            if max1 < min2 - tol or max2 < min1 - tol:
                return False
    return True

def point_in_hexagon(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside hexagon centered at (cx,cy)."""
    # Transform to hexagon's local coordinate system
    angle_rad = np.radians(angle_deg)
    dx, dy = px - cx, py - cy
    # Rotate point by -angle to align with flat-top
    rx = dx * np.cos(-angle_rad) - dy * np.sin(-angle_rad)
    ry = dx * np.sin(-angle_rad) + dy * np.cos(-angle_rad)
    
    # For a flat-top hexagon, the boundaries are:
    # |rx| <= side (left/right flat sides)
    # |ry| <= side * sqrt(3) / 2 (top/bottom flat sides)
    # and |ry| <= (sqrt(3)/2) * (side - |rx|) for the angled sides
    # Actually, let me use the proper inequality.
    
    # A flat-top regular hexagon with side s has vertices at:
    # (s,0), (s/2, s*sqrt(3)/2), (-s/2, s*sqrt(3)/2), (-s,0), (-s/2, -s*sqrt(3)/2), (s/2, -s*sqrt(3)/2)
    # The hexagon is the intersection of 6 half-planes.
    
    s = side
    sqrt3 = np.sqrt(3)
    
    # The six supporting lines (normals pointing inward):
    # 1. Right side: x <= s
    # 2. Top-right: (x/2 + y*sqrt(3)/2) <= s
    # 3. Top-left: (-x/2 + y*sqrt(3)/2) <= s
    # 4. Left side: -x <= s  i.e., x >= -s
    # 5. Bottom-left: (-x/2 - y*sqrt(3)/2) <= s
    # 6. Bottom-right: (x/2 - y*sqrt(3)/2) <= s
    
    checks = [
        rx <= s + 1e-9,
        rx / 2 + ry * sqrt3 / 2 <= s + 1e-9,
        -rx / 2 + ry * sqrt3 / 2 <= s + 1e-9,
        -rx <= s + 1e-9,
        -rx / 2 - ry * sqrt3 / 2 <= s + 1e-9,
        rx / 2 - ry * sqrt3 / 2 <= s + 1e-9,
    ]
    return all(checks)

def is_inside_hexagon(v, outer_cx, outer_cy, outer_side, outer_angle_deg):
    """Check if all vertices of hexagon v are inside the outer hexagon."""
    for pt in v:
        if not point_in_hexagon(pt[0], pt[1], outer_cx, outer_cy, outer_side, outer_angle_deg):
            return False
    return True

def all_inside_outer(inner_verts_list, outer_cx, outer_cy, outer_side, outer_angle_deg):
    for v in inner_verts_list:
        if not is_inside_hexagon(v, outer_cx, outer_cy, outer_side, outer_angle_deg):
            return False
    return True

def check_validity(inner, outer_cx, outer_cy, outer_side, outer_angle_deg):
    """Check if configuration is valid: no overlaps, all inside outer hexagon."""
    if outer_side < 1.0:  # Can't fit even one unit hexagon
        return False
    inner_verts = [hex_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2]) for i in range(N)]
    
    # Check all inside outer
    if not all_inside_outer(inner_verts, outer_cx, outer_cy, outer_side, outer_angle_deg):
        return False
    
    # Check no overlaps
    for i in range(N):
        for j in range(i + 1, N):
            if polygon_overlap(inner_verts[i], inner_verts[j]):
                return False
    return True

def outer_side_needed(inner, outer_angle_deg=0.0):
    """Compute minimum outer side length needed to contain all inner hexagons, centered at origin.
    This is a rough estimate for initialization."""
    inner_verts = [hex_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2]) for i in range(N)]
    all_pts = np.vstack(inner_verts)
    # The outer hexagon centered at origin with flat-top orientation
    # We need to find the minimum scale s such that all points are inside
    # For each point, compute the "required side" based on the hexagon inequalities
    sqrt3 = np.sqrt(3)
    max_req = 0.0
    for pt in all_pts:
        rx, ry = pt[0], pt[1]
        # From the six inequalities, solve for s:
        reqs = [
            abs(rx),
            abs(rx / 2 + ry * sqrt3 / 2) * 2,
            abs(-rx / 2 + ry * sqrt3 / 2) * 2,
        ]
        # Actually we need: each of the 6 expressions <= s
        # rx <= s, -rx <= s => |rx| <= s
        # rx/2 + ry*sqrt3/2 <= s => (rx + ry*sqrt3)/2 <= s
        # etc.
        # So s >= max of all six:
        vals = [
            abs(rx),
            abs(rx / 2 + ry * sqrt3 / 2),
            abs(-rx / 2 + ry * sqrt3 / 2),
        ]
        max_req = max(max_req, max(vals))
    return max_req

def objective(x, start_time, deadline):
    """Objective function for optimization. Returns outer side length or a penalty."""
    if time.time() > deadline:
        return 1e10
    
    inner = []
    for i in range(N):
        px = x[3 * i]
        py = x[3 * i + 1]
        angle = x[3 * i + 2]
        inner.append([px, py, angle])
    outer_side = x[3 * N]
    outer_cx = x[3 * N + 1]
    outer_cy = x[3 * N + 2]
    
    if outer_side < 1.0:
        return 1e10
    
    # Check validity
    inner_verts = [hex_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2]) for i in range(N)]
    
    # Check all inside outer
    if not all_inside_outer(inner_verts, outer_cx, outer_cy, outer_side, 0.0):
        # Penalty for being outside
        penalty = 0.0
        for v in inner_verts:
            for pt in v:
                if not point_in_hexagon(pt[0], pt[1], outer_cx, outer_cy, outer_side, 0.0):
                    # Compute how far outside
                    dx = pt[0] - outer_cx
                    dy = pt[1] - outer_cy
                    penalty += abs(dx) + abs(dy)
        return outer_side + 10.0 * penalty + 5.0
    
    # Check overlaps
    for i in range(N):
        for j in range(i + 1, N):
            if polygon_overlap(inner_verts[i], inner_verts[j]):
                return outer_side + 5.0
    return outer_side

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    best_score = 1e10
    best_result = None
    
    # Initial guess: place hexagons in a roughly circular pattern
    # Try a few different initial configurations
    
    def make_initial_config(config_type):
        inner = []
        if config_type == 0:
            # Two rows of 6
            for k in range(6):
                inner.append([k * 2.1 - 5.25, -1.0, 0.0])
            for k in range(6):
                inner.append([k * 2.1 - 5.25, 1.0, 0.0])
        elif config_type == 1:
            # Ring of 12
            for k in range(12):
                angle = 2 * np.pi * k / 12
                r = 2.5
                inner.append([r * np.cos(angle), r * np.sin(angle), 0.0])
        elif config_type == 2:
            # 3x4 grid
            for i in range(3):
                for j in range(4):
                    inner.append([j * 2.1 - 3.15, i * 2.0 - 2.0, 0.0])
        elif config_type == 3:
            # Hexagonal packing
            # Center + ring of 6 + outer partial
            inner.append([0.0, 0.0, 0.0])
            for k in range(6):
                angle = 2 * np.pi * k / 6
                inner.append([2.0 * np.cos(angle), 2.0 * np.sin(angle), 0.0])
            for k in range(5):
                angle = 2 * np.pi * k / 5 + np.pi / 10
                inner.append([4.0 * np.cos(angle), 4.0 * np.sin(angle), 0.0])
        else:
            # Random
            np.random.seed(config_type)
            for k in range(N):
                inner.append([np.random.uniform(-5, 5), np.random.uniform(-5, 5), np.random.uniform(0, 360)])
        
        # Estimate outer side
        est_side = outer_side_needed(inner) + 0.5
        return inner, est_side

    # Try differential evolution for a few initial configurations
    for config_type in range(5):
        if time.time() > deadline - 20:
            break
        
        inner_init, est_side = make_initial_config(config_type)
        
        # Bounds: positions in [-10, 10], angles in [0, 360], outer_side in [2, 20]
        bounds = []
        for i in range(N):
            bounds.append((-10, 10))  # x
            bounds.append((-10, 10))  # y
            bounds.append((0, 360))   # angle
        bounds.append((2, 20))   # outer_side
        bounds.append((-5, 5))   # outer_cx
        bounds.append((-5, 5))   # outer_cy
        
        # Set initial values for DE
        x0 = []
        for i in range(N):
            x0.append(inner_init[i][0])
            x0.append(inner_init[i][1])
            x0.append(inner_init[i][2])
        x0.append(est_side)
        x0.append(0.0)
        x0.append(0.0)
        
        try:
            result = differential_evolution(
                objective,
                bounds=bounds,
                x0=x0,
                args=(start_time, deadline),
                maxiter=30,
                popsize=10,
                tol=1e-4,
                seed=config_type * 100 + 42,
                polish=True,
                options={'maxiter': 30}
            )
            
            if result.fun < best_score and result.fun < 1e9:
                inner = []
                for i in range(N):
                    x = result.x[3 * i]
                    y = result.x[3 * i + 1]
                    angle = result.x[3 * i + 2]
                    inner.append([x, y, np.degrees(angle) % 360])
                outer_side = result.x[3 * N]
                outer_cx = result.x[3 * N + 1]
                outer_cy = result.x[3 * N + 2]
                
                valid = check_validity(inner, outer_cx, outer_cy, outer_side, 0.0)
                if valid:
                    best_score = outer_side
                    best_result = (inner, (outer_cx, outer_cy), outer_side, 0.0)
        except Exception:
            pass
    
    # If we found a valid result, try to refine with local optimization
    if best_result is not None and time.time() < deadline - 10:
        inner, (ocx, ocy), oside, oang = best_result
        
        # Try to reduce outer_side with local optimization
        def refine_objective(x):
            if time.time() > deadline:
                return 1e10
            inner_r = []
            for i in range(N):
                px = x[3 * i]
                py = x[3 * i + 1]
                angle = x[3 * i + 2]
                inner_r.append([px, py, angle])
            rs = x[3 * N]
            rcx = x[3 * N + 1]
            rcy = x[3 * N + 2]
            
            if rs < 1.0:
                return 1e10
            
            inner_verts = [hex_vertices(inner_r[i][0], inner_r[i][1], 1.0, inner_r[i][2]) for i in range(N)]
            
            if not all_inside_outer(inner_verts, rcx, rcy, rs, 0.0):
                penalty = 0.0
                for v in inner_verts:
                    for pt in v:
                        if not point_in_hexagon(pt[0], pt[1], rcx, rcy, rs, 0.0):
                            dx = pt[0] - rcx
                            dy = pt[1] - rcy
                            penalty += abs(dx) + abs(dy)
                return rs + 10.0 * penalty + 5.0
            
            for i in range(N):
                for j in range(i + 1, N):
                    if polygon_overlap(inner_verts[i], inner_verts[j]):
                        return rs + 5.0
            return rs
        
        x0 = []
        for i in range(N):
            x0.append(inner[i][0])
            x0.append(inner[i][1])
            x0.append(inner[i][2])
        x0.append(oside)
        x0.append(ocx)
        x0.append(ocy)
        
        bounds = []
        for i in range(N):
            bounds.append((-15, 15))
            bounds.append((-15, 15))
            bounds.append((0, 360))
        bounds.append((1.5, 25))
        bounds.append((-10, 10))
        bounds.append((-10, 10))
        
        try:
            res = minimize(refine_objective, x0, method='Nelder-Mead', 
                          bounds=bounds, options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8})
            if res.fun < best_score and res.fun < 1e9:
                inner_r = []
                for i in range(N):
                    inner_r.append([res.x[3 * i], res.x[3 * i + 1], np.degrees(res.x[3 * i + 2]) % 360])
                rs = res.x[3 * N]
                rcx = res.x[3 * N + 1]
                rcy = res.x[3 * N + 2]
                
                valid = check_validity(inner_r, rcx, rcy, rs, 0.0)
                if valid:
                    best_score = rs
                    best_result = (inner_r, (rcx, rcy), rs, 0.0)
        except Exception:
            pass
    
    # If no valid result found, return a safe default
    if best_result is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    return best_result

# EVOLVE-BLOCK-START
# EVOLVE-BLOCK-END
