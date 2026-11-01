"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time
from itertools import combinations

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin before 150s kill

    def hexagon_vertices(cx, cy, side, angle_deg):
        angle_rad = np.radians(angle_deg)
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

    def point_in_hexagon(px, py, cx, cy, side, angle_deg):
        dx = px - cx
        dy = py - cy
        angle_rad = np.radians(angle_deg)
        cos_a = np.cos(-angle_rad)
        sin_a = np.sin(-angle_rad)
        local_x = dx * cos_a - dy * sin_a
        local_y = dx * sin_a + dy * cos_a
        
        apothem = side * np.sqrt(3) / 2
        normal_angles = np.radians(np.arange(30, 360, 60))
        
        for na in normal_angles:
            nx = np.cos(na)
            ny = np.sin(na)
            dist = local_x * nx + local_y * ny
            if dist > apothem + 1e-9:
                return False
        return True

    def polygons_disjoint_sat(v1, v2):
        axes = []
        for i in range(len(v1)):
            p1 = v1[i]
            p2 = v1[(i + 1) % len(v1)]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-9:
                axes.append(normal / norm)
        
        for i in range(len(v2)):
            p1 = v2[i]
            p2 = v2[(i + 1) % len(v2)]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-9:
                axes.append(normal / norm)
        
        for axis in axes:
            proj1 = np.dot(v1, axis)
            proj2 = np.dot(v2, axis)
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            if max1 < min2 - 1e-9 or max2 < min1 - 1e-9:
                return True
        return False

    def all_inside_outer(inner_positions, outer_center, outer_side, outer_angle):
        for i in range(len(inner_positions)):
            cx, cy, angle = inner_positions[i]
            verts = hexagon_vertices(cx, cy, 1.0, angle)
            for vx, vy in verts:
                if not point_in_hexagon(vx, vy, outer_center[0], outer_center[1], outer_side, outer_angle):
                    return False
        return True

    def pairwise_disjoint(inner_positions):
        for i in range(len(inner_positions)):
            for j in range(i+1, len(inner_positions)):
                ci = (inner_positions[i][0], inner_positions[i][1])
                cj = (inner_positions[j][0], inner_positions[j][1])
                ai = inner_positions[i][2]
                aj = inner_positions[j][2]
                v1 = hexagon_vertices(ci[0], ci[1], 1.0, ai)
                v2 = hexagon_vertices(cj[0], cj[1], 1.0, aj)
                if not polygons_disjoint_sat(v1, v2):
                    return False
        return True

    def is_valid(inner, oc, os, oa):
        return all_inside_outer(inner, oc, os, oa) and pairwise_disjoint(inner)

    def compute_minimal_outer(inner_positions, outer_angle_deg):
        """Compute minimal side length of outer hexagon centered at origin to contain all inner hexagons."""
        all_verts = []
        for cx, cy, angle in inner_positions:
            verts = hexagon_vertices(cx, cy, 1.0, angle)
            all_verts.append(verts)
        all_verts = np.vstack(all_verts)
        
        angle_rad = np.radians(outer_angle_deg)
        apothem_factor = np.sqrt(3) / 2
        
        normal_angles = angle_rad + np.radians(np.arange(30, 360, 60))
        
        max_apothem_needed = 0
        for na in normal_angles:
            nx = np.cos(na)
            ny = np.sin(na)
            dists = all_verts[:, 0] * nx + all_verts[:, 1] * ny
            max_dist = np.max(dists)
            if max_dist > 0:
                needed_side = max_dist / apothem_factor
                max_apothem_needed = max(max_apothem_needed, needed_side)
        
        return max_apothem_needed

    def make_hex_cluster():
        positions = []
        positions.append((0.0, 0.0, 0.0))
        
        for k in range(6):
            angle = np.radians(60 * k)
            x = 2.0 * np.cos(angle)
            y = 2.0 * np.sin(angle)
            positions.append((x, y, 0.0))
        
        sqrt3 = np.sqrt(3)
        second_ring_candidates = [
            (4.0, 0.0),
            (3.0, sqrt3),
            (1.0, 2*sqrt3),
            (-1.0, 2*sqrt3),
            (-3.0, sqrt3),
            (-4.0, 0.0),
            (-3.0, -sqrt3),
            (-1.0, -2*sqrt3),
            (1.0, -2*sqrt3),
            (3.0, -sqrt3),
        ]
        
        best_combo = None
        best_side = float('inf')
        
        for combo in combinations(range(len(second_ring_candidates)), 5):
            test_positions = positions.copy()
            for idx in combo:
                x, y = second_ring_candidates[idx]
                test_positions.append((x, y, 0.0))
            
            side = compute_minimal_outer(test_positions, 30.0)
            if side < best_side:
                best_side = side
                best_combo = combo
        
        for idx in best_combo:
            x, y = second_ring_candidates[idx]
            positions.append((x, y, 0.0))
        
        return positions, best_side

    def make_params(inner, oc, os, oa):
        params = []
        for pos in inner:
            params.extend([pos[0], pos[1], pos[2]])
        params.extend([oc[0], oc[1], os, oa])
        return np.array(params, dtype=float)

    def objective(params):
        n_inner = N
        inner_pos = []
        for i in range(n_inner):
            x = params[3*i]
            y = params[3*i+1]
            a = params[3*i+2]
            inner_pos.append((x, y, a))
        
        ox = params[3*n_inner]
        oy = params[3*n_inner+1]
        os = max(params[3*n_inner+2], 0.1)
        oa = params[3*n_inner+3]
        
        outer_center = (ox, oy)
        outer_side = os
        outer_angle = oa
        
        inside_ok = all_inside_outer(inner_pos, outer_center, outer_side, outer_angle)
        disjoint_ok = pairwise_disjoint(inner_pos)
        
        if inside_ok and disjoint_ok:
            return outer_side
        
        penalty = 0.0
        if not inside_ok:
            for i in range(n_inner):
                cx, cy, angle = inner_pos[i]
                verts = hexagon_vertices(cx, cy, 1.0, angle)
                for vx, vy in verts:
                    if not point_in_hexagon(vx, vy, outer_center[0], outer_center[1], outer_side, outer_angle):
                        penalty += 1.0
        if not disjoint_ok:
            for i in range(n_inner):
                for j in range(i+1, n_inner):
                    ci = (inner_pos[i][0], inner_pos[i][1])
                    cj = (inner_pos[j][0], inner_pos[j][1])
                    ai = inner_pos[i][2]
                    aj = inner_pos[j][2]
                    v1 = hexagon_vertices(ci[0], ci[1], 1.0, ai)
                    v2 = hexagon_vertices(cj[0], cj[1], 1.0, aj)
                    if not polygons_disjoint_sat(v1, v2):
                        penalty += 1.0
                        
        return outer_side + penalty * 10.0

    # Generate multiple initial configurations
    initial_configs = []
    
    # Config 1: Hexagonal cluster (center + 6 ring + 5 second ring)
    hex_positions, _ = make_hex_cluster()
    initial_configs.append(hex_positions)
    
    # Config 2: 4x3 grid
    x_coords = [-3.0, -1.0, 1.0, 3.0]
    y_coords = [-1.8, 0.0, 1.8]
    config2 = []
    for y in y_coords:
        for x in x_coords:
            config2.append((x, y, 0.0))
    initial_configs.append(config2)
    
    # Config 3: 2 rows of 6, staggered
    config3 = []
    for i in range(6):
        config3.append((i * 2.0 - 5.0, -0.9, 0.0))
        config3.append((i * 2.0 - 4.0, 0.9, 0.0))
    initial_configs.append(config3)
    
    # Config 4: Linear arrangement
    config4 = []
    for i in range(12):
        config4.append((i * 2.0 - 11.0, 0.0, 0.0))
    initial_configs.append(config4)
    
    # Find best initial configuration
    best_params = None
    best_score = float('inf')
    best_inner_init = None
    best_oc_init = None
    best_os_init = None
    best_oa_init = None
    
    for inner_positions in initial_configs:
        for oa in [0.0, 15.0, 30.0, 45.0, 60.0]:
            side = compute_minimal_outer(inner_positions, oa)
            oc = (0.0, 0.0)
            
            # Ensure validity by expanding if needed
            os_try = side
            while not all_inside_outer(inner_positions, oc, os_try, oa):
                os_try += 0.05
                if os_try > 15:
                    break
            
            params = make_params(inner_positions, oc, os_try, oa)
            score = objective(params)
            
            if score < best_score:
                best_score = score
                best_params = params.copy()
                best_inner_init = inner_positions
                best_oc_init = oc
                best_os_init = os_try
                best_oa_init = oa
    
    if best_params is None:
        # Fallback to hex cluster
        inner_init, _ = make_hex_cluster()
        best_inner_init = inner_init
        best_oc_init = (0.0, 0.0)
        best_os_init = 6.0
        best_oa_init = 30.0
        best_params = make_params(best_inner_init, best_oc_init, best_os_init, best_oa_init)
        best_score = objective(best_params)
    
    # Optimization with multiple restarts from best initial config
    current_params = best_params.copy()
    current_score = best_score
    
    # First optimization pass
    if time.time() < deadline - 20:
        try:
            res = minimize(objective, current_params, method='Nelder-Mead', 
                         options={'maxiter': 300, 'xatol': 0.01, 'fatol': 0.01, 'adaptive': True})
            if res.fun < current_score:
                current_score = res.fun
                current_params = res.x.copy()
        except Exception:
            pass
    
    # Second optimization pass with tighter tolerance
    if time.time() < deadline - 10 and current_score < 5.0:
        try:
            res2 = minimize(objective, current_params, method='Nelder-Mead', 
                          options={'maxiter': 200, 'xatol': 0.005, 'fatol': 0.005, 'adaptive': True})
            if res2.fun < current_score:
                current_score = res2.fun
                current_params = res2.x.copy()
        except Exception:
            pass
    
    # Third optimization pass if time allows
    if time.time() < deadline - 5 and current_score < 4.5:
        try:
            res3 = minimize(objective, current_params, method='Nelder-Mead', 
                          options={'maxiter': 150, 'xatol': 0.002, 'fatol': 0.002, 'adaptive': True})
            if res3.fun < current_score:
                current_score = res3.fun
                current_params = res3.x.copy()
        except Exception:
            pass
    
    # Parse final parameters
    final_params = current_params
    inner = []
    for i in range(N):
        x = final_params[3*i]
        y = final_params[3*i+1]
        a = final_params[3*i+2]
        inner.append((x, y, a))
    oc = (final_params[3*N], final_params[3*N+1])
    os = final_params[3*N+2]
    oa = final_params[3*N+3]
    
    # Validate and fallback if needed
    if not is_valid(inner, oc, os, oa):
        inner = best_inner_init
        oc = best_oc_init
        os = best_os_init
        oa = best_oa_init
        while not all_inside_outer(inner, oc, os, oa):
            os += 0.05
            if os > 10:
                break
    
    return inner, oc, os, oa
# EVOLVE-BLOCK-END
