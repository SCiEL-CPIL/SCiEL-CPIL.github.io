import numpy as np
from scipy.optimize import minimize, differential_evolution
import time
import math

N = 11
SQRT3 = math.sqrt(3.0)
APOTHEM = SQRT3 / 2.0  # apothem of unit hexagon

def _hex_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon centered at (cx,cy) with given side and rotation.
    Uses pointy-top convention: at angle 0, vertices at 30, 90, 150, 210, 270, 330 degrees."""
    a = np.radians(angle_deg)
    angles = a + np.pi/6 + np.arange(6) * np.pi/3
    return cx + side * np.cos(angles), cy + side * np.sin(angles)

def _hex_vertices_fast(cx, cy, side, angle_rad):
    """Fast vectorized version using radians."""
    angles = angle_rad + np.pi/6 + np.arange(6) * np.pi/3
    return cx + side * np.cos(angles), cy + side * np.sin(angles)

def _point_in_hex(px, py, cx, cy, side, angle_rad):
    """Check if point (px,py) is inside hexagon centered at (cx,cy) with side, angle (radians)."""
    dx = px - cx
    dy = py - cy
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    rx = dx * cos_a + dy * sin_a
    ry = -dx * sin_a + dy * cos_a
    apothem = side * SQRT3 / 2
    for k in range(6):
        theta = np.radians(30 + k * 60)
        val = rx * np.cos(theta) + ry * np.sin(theta)
        if val > apothem + 1e-9:
            return False
    return True

def _hexagons_disjoint(cx1, cy1, a1_rad, cx2, cy2, a2_rad, side=1.0):
    """Check if two unit hexagons are disjoint (non-overlapping) using separating axis theorem."""
    dx = cx1 - cx2
    dy = cy1 - cy2
    d2 = dx*dx + dy*dy
    if d2 > (2.0 * side + 1e-9)**2:
        return True
    
    # Get vertices
    vx1, vy1 = _hex_vertices_fast(cx1, cy1, side, a1_rad)
    vx2, vy2 = _hex_vertices_fast(cx2, cy2, side, a2_rad)
    
    # Check if any vertex of hex1 is inside hex2
    for i in range(6):
        if _point_in_hex(vx1[i], vy1[i], cx2, cy2, side, a2_rad):
            return False
    # Check if any vertex of hex2 is inside hex1
    for i in range(6):
        if _point_in_hex(vx2[i], vy2[i], cx1, cy1, side, a1_rad):
            return False
    
    # Check edge-edge intersections via separating axis
    # For two convex polygons, if there's a separating axis, they're disjoint
    # The candidate axes are the normals of all edges
    # For regular hexagons, the edge normals are at angles: angle_rad + k*pi/3 + pi/2 for k in 0..5
    # But we need to check all 12 axes (6 from each hexagon)
    
    axes = []
    for k in range(6):
        # Normal to edge k of hex1
        ang1 = a1_rad + k * np.pi/3 + np.pi/2
        axes.append((np.cos(ang1), np.sin(ang1)))
    for k in range(6):
        # Normal to edge k of hex2
        ang2 = a2_rad + k * np.pi/3 + np.pi/2
        axes.append((np.cos(ang2), np.sin(ang2)))
    
    for nx, ny in axes:
        # Project all vertices of both hexagons onto this axis
        proj1 = vx1 * nx + vy1 * ny
        proj2 = vx2 * nx + vy2 * ny
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        # Check for separation
        if max1 < min2 - 1e-10 or max2 < min1 - 1e-10:
            return True
    
    return False

def _compute_min_side(positions, angles_rad, center=(0.0, 0.0), outer_angle_rad=0.0):
    """Compute the minimum outer hexagon side length that contains all inner hexagons."""
    max_side = 0.0
    n = len(positions)
    for i in range(n):
        cx, cy = positions[i]
        a = angles_rad[i]
        vx, vy = _hex_vertices_fast(cx, cy, 1.0, a)
        cos_o = np.cos(outer_angle_rad)
        sin_o = np.sin(outer_angle_rad)
        for j in range(6):
            px = vx[j] - center[0]
            py = vy[j] - center[1]
            rx = px * cos_o + py * sin_o
            ry = -px * sin_o + py * cos_o
            for k in range(6):
                theta = np.radians(30 + k * 60)
                proj = rx * np.cos(theta) + ry * np.sin(theta)
                required_S = 2.0 * abs(proj) / SQRT3
                if required_S > max_side:
                    max_side = required_S
    return max_side

def _objective(params, center=(0.0, 0.0), outer_angle_rad=0.0):
    """Objective: outer hexagon side length plus penalty for overlaps."""
    n = N
    positions = []
    angles_rad = []
    for i in range(n):
        cx = params[3*i]
        cy = params[3*i+1]
        a = params[3*i+2]
        positions.append((cx, cy))
        angles_rad.append(a)
    
    penalty = 0.0
    for i in range(n):
        for j in range(i+1, n):
            if not _hexagons_disjoint(positions[i][0], positions[i][1], angles_rad[i], 
                                      positions[j][0], positions[j][1], angles_rad[j]):
                d = np.sqrt((positions[i][0]-positions[j][0])**2 + (positions[i][1]-positions[j][1])**2)
                penalty += max(0, 2.0 - d) ** 2 * 100
    
    side = _compute_min_side(positions, angles_rad, center, outer_angle_rad)
    return side + penalty

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin
    
    sqrt3 = SQRT3
    
    # Generate initial hexagonal lattice configuration
    # For flat-top unit hexagons: width = 2, height = sqrt(3)
    # Adjacent centers are at distance sqrt(3)
    # Using a grid pattern:
    # Row spacing: 1.5 (for pointy-top) or sqrt(3) for flat-top rows
    # Let's use a proper hexagonal grid.
    
    # For pointy-top hexagons (vertex at top), the natural grid is:
    # Horizontal distance between adjacent in same row: sqrt(3)
    # Vertical distance between rows: 1.5
    # Odd rows shifted by sqrt(3)/2
    
    grid_pos = []
    # 3-3-3-2 pattern
    for row in range(4):
        n_cols = 3 if row < 3 else 2
        for col in range(n_cols):
            x = (col - (n_cols-1)/2.0) * sqrt3
            if row % 2 == 1:
                x += sqrt3 / 2
            y = row * 1.5
            grid_pos.append((x, y))
    
    # Center the cluster
    cx_mean = np.mean([p[0] for p in grid_pos])
    cy_mean = np.mean([p[1] for p in grid_pos])
    centered_pos = np.array([(p[0] - cx_mean, p[1] - cy_mean) for p in grid_pos])
    
    # Verify this is a valid disjoint configuration
    # All angles are 0 (pointy-top, aligned)
    base_params = np.zeros(3 * N)
    for i in range(N):
        base_params[3*i] = centered_pos[i, 0]
        base_params[3*i+1] = centered_pos[i, 1]
        base_params[3*i+2] = 0.0
    
    # Check disjointness of base configuration
    valid = True
    for i in range(N):
        for j in range(i+1, N):
            if not _hexagons_disjoint(base_params[3*i], base_params[3*i+1], 0,
                                      base_params[3*j], base_params[3*j+1], 0):
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Try a different arrangement: pure hexagonal packing with center + ring1 + part of ring2
        grid_pos2 = []
        # Center
        grid_pos2.append((0.0, 0.0))
        # Ring 1: 6 hexagons at distance sqrt(3)
        for i in range(6):
            ang = i * np.pi / 3
            grid_pos2.append((sqrt3 * np.cos(ang), sqrt3 * np.sin(ang)))
        # Ring 2: pick 4 hexagons from the 12 available
        # Ring 2 centers are at distance 2*sqrt(3) in directions, but not all at same distance
        # For hexagonal grid, ring 2 has positions like:
        # (2*sqrt3, 0), (sqrt3, 1.5*sqrt3), (-sqrt3, 1.5*sqrt3), etc.
        # Let's use the lattice vectors: v1 = (sqrt3, 0), v2 = (sqrt3/2, 1.5)
        # Ring 2 positions (12 total): combinations of 2*v1, 2*v2, v1+v2, etc.
        ring2_candidates = []
        for a in range(-2, 3):
            for b in range(-2, 3):
                if abs(a) + abs(b) > 3:
                    continue
                x = a * sqrt3 + b * sqrt3 / 2
                y = b * 1.5
                # Exclude ring 1 and center
                r = np.sqrt(x*x + y*y)
                if r > 1e-6 and abs(r - sqrt3) > 0.1:
                    # Check it's at ring 2 distance (~2*sqrt3)
                    if abs(r - 2*sqrt3) < 0.5 or abs(r - 1.5*sqrt3) < 0.3:
                        ring2_candidates.append((x, y))
        
        # Deduplicate and pick 4 symmetric ones
        seen = set()
        unique_r2 = []
        for p in ring2_candidates:
            key = (round(p[0], 3), round(p[1], 3))
            if key not in seen:
                seen.add(key)
                unique_r2.append(p)
        
        # Pick 4 that are most symmetric - take ones closest to cardinal directions
        unique_r2.sort(key=lambda p: (p[0]**2 + p[1]**2))
        for p in unique_r2[:4]:
            grid_pos2.append(p)
        
        cx_mean2 = np.mean([p[0] for p in grid_pos2])
        cy_mean2 = np.mean([p[1] for p in grid_pos2])
        centered_pos = np.array([(p[0] - cx_mean2, p[1] - cy_mean2) for p in grid_pos2])
        
        base_params = np.zeros(3 * N)
        for i in range(N):
            base_params[3*i] = centered_pos[i, 0]
            base_params[3*i+1] = centered_pos[i, 1]
            base_params[3*i+2] = 0.0
    
    # Optimize: first try optimizing rotation of the entire cluster
    def calc_side_cluster_angle(cluster_angle_deg):
        cluster_angle_rad = np.radians(cluster_angle_deg)
        cos_a = np.cos(cluster_angle_rad)
        sin_a = np.sin(cluster_angle_rad)
        
        max_required_side = 0.0
        
        for i in range(N):
            cx, cy = centered_pos[i]
            # Rotate center
            rx_c = cx * cos_a - cy * sin_a
            ry_c = cx * sin_a + cy * cos_a
            
            # Inner hexagon rotated by cluster_angle
            vx_local, vy_local = _hex_vertices_fast(0.0, 0.0, 1.0, 0.0)
            vx_rot = vx_local * cos_a - vy_local * sin_a
            vy_rot = vx_local * sin_a + vy_local * cos_a
            vx_global = vx_rot + rx_c
            vy_global = vy_rot + ry_c
            
            for k in range(6):
                theta = np.radians(30 + k * 60)
                cos_t = np.cos(theta)
                sin_t = np.sin(theta)
                projs = vx_global * cos_t + vy_global * sin_t
                max_proj = np.max(np.abs(projs))
                required_S = 2.0 * max_proj / SQRT3
                if required_S > max_required_side:
                    max_required_side = required_S
                    
        return max_required_side
    
    # Golden Section Search for optimal cluster rotation
    phi = (np.sqrt(5) - 1) / 2
    a_g = 0.0
    b_g = 60.0
    c_g = b_g - phi * (b_g - a_g)
    d_g = a_g + phi * (b_g - a_g)
    fc = calc_side_cluster_angle(c_g)
    fd = calc_side_cluster_angle(d_g)
    
    for _ in range(50):
        if time.time() > deadline - 20.0:
            break
        if fc < fd:
            b_g = d_g
            d_g = c_g
            fd = fc
            c_g = b_g - phi * (b_g - a_g)
            fc = calc_side_cluster_angle(c_g)
        else:
            a_g = c_g
            c_g = d_g
            fc = fd
            d_g = a_g + phi * (b_g - a_g)
            fd = calc_side_cluster_angle(d_g)
    
    opt_cluster_angle = (a_g + b_g) / 2.0
    min_side_cluster = calc_side_cluster_angle(opt_cluster_angle)
    
    # Now try full Nelder-Mead optimization on all positions and angles
    # Initialize from the rotated cluster configuration
    cos_opt = np.cos(np.radians(opt_cluster_angle))
    sin_opt = np.sin(np.radians(opt_cluster_angle))
    
    init_params = np.zeros(3 * N)
    for i in range(N):
        cx, cy = centered_pos[i]
        rx_c = cx * cos_opt - cy * sin_opt
        ry_c = cx * sin_opt + cy * cos_opt
        init_params[3*i] = rx_c
        init_params[3*i+1] = ry_c
        init_params[3*i+2] = opt_cluster_angle * np.pi / 180.0
    
    # Nelder-Mead optimization
    def nm_objective(params):
        return _objective(params, (0.0, 0.0), 0.0)
    
    best_params = init_params.copy()
    best_val = nm_objective(init_params)
    
    if time.time() < deadline - 30.0:
        result = minimize(nm_objective, init_params, method='Nelder-Mead',
                         options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
        if result.fun < best_val:
            best_params = result.x
            best_val = result.fun
    
    # Try differential evolution for global search if time permits
    if time.time() < deadline - 40.0:
        bounds = [(-8, 8)] * (3*N)
        # Angle bounds
        for i in range(N):
            bounds[3*i+2] = (-np.pi, np.pi)
        
        def de_objective(params):
            return _objective(params, (0.0, 0.0), 0.0)
        
        try:
            de_result = differential_evolution(de_objective, bounds, maxiter=100, tol=1e-6,
                                               seed=42, polish=True, init='sobol')
            if de_result.fun < best_val:
                best_params = de_result.x
                best_val = de_result.fun
        except:
            pass
    
    # Final polish with tighter Nelder-Mead
    if time.time() < deadline - 10.0:
        try:
            final_result = minimize(nm_objective, best_params, method='Nelder-Mead',
                                   options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})
            if final_result.fun < best_val:
                best_params = final_result.x
                best_val = final_result.fun
        except:
            pass
    
    # Extract final configuration
    positions = []
    angles_rad = []
    for i in range(N):
        cx = best_params[3*i]
        cy = best_params[3*i+1]
        a = best_params[3*i+2]
        positions.append((cx, cy))
        angles_rad.append(a)
    
    # Check validity
    for i in range(N):
        for j in range(i+1, N):
            if not _hexagons_disjoint(positions[i][0], positions[i][1], angles_rad[i],
                                      positions[j][0], positions[j][1], angles_rad[j]):
                # If invalid, fall back to the cluster-rotated configuration
                positions = []
                angles_rad = []
                for i in range(N):
                    cx, cy = centered_pos[i]
                    rx_c = cx * cos_opt - cy * sin_opt
                    ry_c = cx * sin_opt + cy * cos_opt
                    positions.append((rx_c, ry_c))
                    angles_rad.append(np.radians(opt_cluster_angle))
                break
        else:
            continue
        break
    
    # Compute final side length
    final_side = _compute_min_side(positions, angles_rad, (0.0, 0.0), 0.0)
    
    # Add small buffer for floating point
    final_side += 1e-6
    
    inner = []
    for i in range(N):
        inner.append([float(positions[i][0]), float(positions[i][1]), float(np.degrees(angles_rad[i]))])
    
    return inner, (0.0, 0.0), float(final_side), 0.0
