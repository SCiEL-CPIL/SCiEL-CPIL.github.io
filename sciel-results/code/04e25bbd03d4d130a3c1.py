"""Packing 12 unit hexagons in a regular hexagon, minimizing outer side length."""
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time
import math

N = 12
START_TIME = time.time()
TIME_LIMIT = 140.0  # safety margin before 150s kill

def hex_vertices(cx, cy, s, angle_deg):
    """Return vertices of a regular hexagon with center (cx,cy), side s, rotation angle_deg."""
    a = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * np.pi / 3
        verts.append((cx + s * np.cos(theta), cy + s * np.sin(theta)))
    return np.array(verts)

def point_in_polygon(x, y, polygon):
    """Ray casting algorithm for point in polygon."""
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside

def segments_intersect(p1, p2, p3, p4):
    """Check if segment p1-p2 intersects segment p3-p4 (proper intersection only)."""
    def ccw(a, b, c):
        return (c[1]-a[1])*(b[0]-a[0]) - (b[1]-a[1])*(c[0]-a[0])
    
    d1 = ccw(p3, p4, p1)
    d2 = ccw(p3, p4, p2)
    d3 = ccw(p1, p2, p3)
    d4 = ccw(p1, p2, p4)
    
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    return False

def polygons_disjoint(hex1_verts, hex2_verts, eps=1e-9):
    """Check if two hexagons are disjoint (no interior overlap)."""
    for v in hex1_verts:
        if point_in_polygon(v[0], v[1], hex2_verts):
            return False
    for v in hex2_verts:
        if point_in_polygon(v[0], v[1], hex1_verts):
            return False
    for i in range(6):
        p1 = hex1_verts[i]
        p2 = hex1_verts[(i+1)%6]
        for j in range(6):
            p3 = hex2_verts[j]
            p4 = hex2_verts[(j+1)%6]
            if segments_intersect(p1, p2, p3, p4):
                return False
    return True

def compute_min_outer_side(inner_positions, inner_angles, outer_center, outer_angle):
    """
    Compute the minimum outer hexagon side length such that all inner hexagons fit.
    Returns the minimum side length.
    """
    outer_cx, outer_cy = outer_center
    a_outer = np.radians(outer_angle)
    apothem_factor = np.sqrt(3) / 2
    
    max_s = 0.0
    
    for (ix, iy), ia in zip(inner_positions, inner_angles):
        inner_verts = hex_vertices(ix, iy, 1.0, ia)
        for vx, vy in inner_verts:
            dx = vx - outer_cx
            dy = vy - outer_cy
            for i in range(6):
                normal_angle = a_outer + (i + 0.5) * np.pi / 3
                nx = np.cos(normal_angle)
                ny = np.sin(normal_angle)
                dist = dx * nx + dy * ny
                if dist > 0:
                    s_needed = dist / apothem_factor
                    if s_needed > max_s:
                        max_s = s_needed
    
    return max_s

def evaluate_configuration(inner_positions, inner_angles, outer_center, outer_angle):
    """
    Evaluate a configuration. Returns (outer_side, is_valid).
    """
    n = len(inner_positions)
    hex_verts_list = []
    for i in range(n):
        hex_verts_list.append(hex_vertices(inner_positions[i][0], inner_positions[i][1], 1.0, inner_angles[i]))
    
    for i in range(n):
        for j in range(i+1, n):
            if not polygons_disjoint(hex_verts_list[i], hex_verts_list[j]):
                return (float('inf'), False)
    
    outer_side = compute_min_outer_side(inner_positions, inner_angles, outer_center, outer_angle)
    return (outer_side, True)

def make_initial_grid(rows, cols, scale=1.0):
    """Create a grid arrangement of hexagons."""
    positions = []
    angles = []
    for r in range(rows):
        for c in range(cols):
            x = (c - (cols-1)/2.0) * 2.0 * scale
            y = (r - (rows-1)/2.0) * np.sqrt(3) * scale
            positions.append((x, y))
            angles.append(0.0)
    return positions, angles

def make_hex_cluster(n_hex, scale=1.0):
    """Create a hexagonal cluster arrangement."""
    positions = [(0.0, 0.0)]
    angles = [0.0]
    
    # Ring 1: 6 hexagons at distance 2*scale
    for i in range(6):
        theta = i * np.pi / 3
        x = 2.0 * scale * np.cos(theta)
        y = 2.0 * scale * np.sin(theta)
        positions.append((x, y))
        angles.append(0.0)
    
    # Ring 2: add more as needed
    count = len(positions)
    while count < n_hex:
        for i in range(6):
            if count >= n_hex:
                break
            theta = i * np.pi / 3 + np.pi/6
            x = 4.0 * scale * np.cos(theta)
            y = 4.0 * scale * np.sin(theta)
            positions.append((x, y))
            angles.append(0.0)
            count += 1
    
    return positions[:n_hex], angles[:n_hex]

def make_random_positions(n, spread=5.0, seed=42):
    """Create random positions."""
    rng = np.random.RandomState(seed)
    positions = [(rng.uniform(-spread, spread), rng.uniform(-spread, spread)) for _ in range(n)]
    angles = [0.0 for _ in range(n)]
    return positions, angles

def optimize_with_slsqp(positions, angles, outer_center, outer_angle, max_iter=500):
    """Optimize using SLSQP with penalty for overlaps."""
    n = len(positions)
    
    x0 = []
    for pos in positions:
        x0.extend([pos[0], pos[1]])
    for ang in angles:
        x0.append(ang)
    x0.extend([outer_center[0], outer_center[1], outer_angle])
    
    def objective(params):
        if time.time() - START_TIME > TIME_LIMIT:
            return 1e10
        
        pos = [(params[2*i], params[2*i+1]) for i in range(n)]
        angs = list(params[2*n:2*n+n])
        oc = (params[2*n+n], params[2*n+n+1])
        oa = params[2*n+n+2]
        
        side, valid = evaluate_configuration(pos, angs, oc, oa)
        if not valid:
            return 1e10
        return side
    
    result = minimize(objective, x0, method='SLSQP', 
                      options={'maxiter': max_iter, 'ftol': 1e-8})
    
    if time.time() - START_TIME > TIME_LIMIT:
        return None
    
    if result.fun < 1e9:
        pos = [(result.x[2*i], result.x[2*i+1]) for i in range(n)]
        angs = list(result.x[2*n:2*n+n])
        oc = (result.x[2*n+n], result.x[2*n+n+1])
        oa = result.x[2*n+n+2]
        return (pos, angs, oc, oa, result.fun)
    
    return None

def optimize_with_nelder_mead(positions, angles, outer_center, outer_angle, max_iter=1000):
    """Optimize using Nelder-Mead."""
    n = len(positions)
    
    x0 = []
    for pos in positions:
        x0.extend([pos[0], pos[1]])
    for ang in angles:
        x0.append(ang)
    x0.extend([outer_center[0], outer_center[1], outer_angle])
    
    def objective(params):
        if time.time() - START_TIME > TIME_LIMIT:
            return 1e10
        
        pos = [(params[2*i], params[2*i+1]) for i in range(n)]
        angs = list(params[2*n:2*n+n])
        oc = (params[2*n+n], params[2*n+n+1])
        oa = params[2*n+n+2]
        
        side, valid = evaluate_configuration(pos, angs, oc, oa)
        if not valid:
            return 1e10
        return side
    
    result = minimize(objective, x0, method='Nelder-Mead', 
                      options={'maxiter': max_iter, 'xatol': 1e-7, 'fatol': 1e-7})
    
    if time.time() - START_TIME > TIME_LIMIT:
        return None
    
    if result.fun < 1e9:
        pos = [(result.x[2*i], result.x[2*i+1]) for i in range(n)]
        angs = list(result.x[2*n:2*n+n])
        oc = (result.x[2*n+n], result.x[2*n+n+1])
        oa = result.x[2*n+n+2]
        return (pos, angs, oc, oa, result.fun)
    
    return None

def construct():
    best_side = float('inf')
    best_result = None
    
    # Generate multiple initial configurations
    initial_configs = []
    
    # Grid arrangements
    for rows, cols in [(2,6), (3,4), (4,3), (6,2)]:
        if rows * cols >= N:
            pos, ang = make_initial_grid(rows, cols, scale=1.0)
            if len(pos) >= N:
                initial_configs.append((pos[:N], ang[:N], (0.0, 0.0), 0.0))
    
    # Hexagonal cluster
    pos, ang = make_hex_cluster(N, scale=1.0)
    initial_configs.append((pos, ang, (0.0, 0.0), 0.0))
    
    # Random configurations
    for seed in [1, 2, 3, 4, 5]:
        pos, ang = make_random_positions(N, spread=4.0, seed=seed)
        initial_configs.append((pos, ang, (0.0, 0.0), 0.0))
    
    # Different outer angles
    for base_pos, base_ang, base_oc, base_oa in [(initial_configs[0][0], initial_configs[0][1], 
                                                   initial_configs[0][2], initial_configs[0][3])]:
        for outer_angle in [0, 30, 15, 45]:
            initial_configs.append((base_pos, base_ang, base_oc, outer_angle))
    
    # Evaluate all initial configurations
    for pos, ang, oc, oa in initial_configs:
        if time.time() - START_TIME > TIME_LIMIT:
            break
        side, valid = evaluate_configuration(pos, ang, oc, oa)
        if valid and side < best_side:
            best_side = side
            best_result = (pos, ang, oc, oa, side)
    
    # Sort initial configs by their side length and optimize the best ones
    scored_configs = []
    for pos, ang, oc, oa in initial_configs:
        side, valid = evaluate_configuration(pos, ang, oc, oa)
        if valid:
            scored_configs.append((side, pos, ang, oc, oa))
    
    scored_configs.sort(key=lambda x: x[0])
    
    # Optimize top configurations with different methods
    for side, pos, ang, oc, oa in scored_configs[:4]:
        if time.time() - START_TIME > TIME_LIMIT:
            break
        
        # Try SLSQP
        opt_result = optimize_with_slsqp(pos, ang, oc, oa, max_iter=300)
        if opt_result is not None:
            o_pos, o_ang, o_oc, o_oa, o_side = opt_result
            if o_side < best_side:
                best_side = o_side
                best_result = (o_pos, o_ang, o_oc, o_oa, o_side)
        
        if time.time() - START_TIME > TIME_LIMIT:
            break
        
        # Try Nelder-Mead
        opt_result = optimize_with_nelder_mead(pos, ang, oc, oa, max_iter=500)
        if opt_result is not None:
            o_pos, o_ang, o_oc, o_oa, o_side = opt_result
            if o_side < best_side:
                best_side = o_side
                best_result = (o_pos, o_ang, o_oc, o_oa, o_side)
    
    # If we have time, do a differential evolution search
    if time.time() - START_TIME < TIME_LIMIT - 30:
        n_params = 2*N + N + 2 + 1  # positions + angles + outer center + outer angle
        
        def de_objective(params):
            if time.time() - START_TIME > TIME_LIMIT:
                return 1e10
            
            pos = [(params[2*i], params[2*i+1]) for i in range(N)]
            angs = list(params[2*N:2*N+N])
            oc = (params[2*N+N], params[2*N+N+1])
            oa = params[2*N+N+2]
            
            side, valid = evaluate_configuration(pos, angs, oc, oa)
            if not valid:
                return 1e10
            return side
        
        bounds = [(-6, 6)] * (2*N) + [(-180, 180)] * N + [(-2, 2), (-2, 2), (-180, 180)]
        
        try:
            de_result = differential_evolution(de_objective, bounds, maxiter=200, 
                                               popsize=10, tol=1e-6, 
                                               seed=42, polish=False)
            if de_result.fun < 1e9:
                pos = [(de_result.x[2*i], de_result.x[2*i+1]) for i in range(N)]
                angs = list(de_result.x[2*N:2*N+N])
                oc = (de_result.x[2*N+N], de_result.x[2*N+N+1])
                oa = de_result.x[2*N+N+2]
                if de_result.fun < best_side:
                    best_side = de_result.fun
                    best_result = (pos, angs, oc, oa, de_result.fun)
        except Exception:
            pass
    
    # Final polish with Nelder-Mead on the best result
    if best_result is not None and time.time() - START_TIME < TIME_LIMIT - 10:
        pos, ang, oc, oa, side = best_result
        opt_result = optimize_with_nelder_mead(pos, ang, oc, oa, max_iter=200)
        if opt_result is not None:
            o_pos, o_ang, o_oc, o_oa, o_side = opt_result
            if o_side < best_side:
                best_side = o_side
                best_result = (o_pos, o_ang, o_oc, o_oa, o_side)
    
    # Fallback
    if best_result is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    positions, angles, outer_center, outer_angle, side = best_result
    
    # Format output
    inner = [[pos[0], pos[1], ang] for pos, ang in zip(positions, angles)]
    
    return inner, outer_center, side, outer_angle
