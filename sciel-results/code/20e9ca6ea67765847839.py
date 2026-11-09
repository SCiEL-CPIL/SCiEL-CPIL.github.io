import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    SQRT3 = math.sqrt(3)
    
    # Helper: Get vertices of a unit hexagon centered at (cx, cy) with rotation angle
    def get_hex_vertices(cx, cy, angle_deg):
        verts = []
        for i in range(6):
            theta = math.radians(angle_deg + i * 60)
            x = cx + math.cos(theta)
            y = cy + math.sin(theta)
            verts.append((x, y))
        return verts

    # Helper: Get axes for SAT (normals to edges)
    def get_hex_axes(angle_deg):
        axes = []
        for i in range(6):
            normal_angle = math.radians(angle_deg + i * 60 + 30)
            nx = math.cos(normal_angle)
            ny = math.sin(normal_angle)
            axes.append((nx, ny))
        return axes

    def project_poly(vertices, axis):
        nx, ny = axis
        min_val = float('inf')
        max_val = float('-inf')
        for vx, vy in vertices:
            proj = vx * nx + vy * ny
            if proj < min_val:
                min_val = proj
            if proj > max_val:
                max_val = proj
        return min_val, max_val

    def is_disjoint(hex1_verts, hex2_verts, axes1, axes2, d_sq):
        """
        Fast disjoint check.
        """
        if d_sq > 4.0:
            return True
        if d_sq < 3.0:
            return False
            
        # SAT check
        all_axes = axes1 + axes2
        for axis in all_axes:
            min1, max1 = project_poly(hex1_verts, axis)
            min2, max2 = project_poly(hex2_verts, axis)
            if max1 < min2 or max2 < min1:
                return True
        return False

    def calc_outer_side(hexagons, outer_angle_deg):
        """
        Calculate the minimum side length S of a regular hexagon centered at (0,0)
        with rotation outer_angle_deg that contains all given hexagons.
        """
        base_angles = [30, 90, 150, 210, 270, 330]
        axis_dirs = []
        for a in base_angles:
            rad = math.radians(a + outer_angle_deg)
            axis_dirs.append((math.cos(rad), math.sin(rad)))
        
        max_proj = 0.0
        for hex_verts in hexagons:
            for vx, vy in hex_verts:
                for nx, ny in axis_dirs:
                    proj = abs(vx * nx + vy * ny)
                    if proj > max_proj:
                        max_proj = proj
        
        # Apothem of outer hex is S * SQRT3 / 2
        # S >= 2 * max_proj / SQRT3
        return 2.0 * max_proj / SQRT3

    def axial_to_cartesian(q, r, spacing=1.0):
        x = SQRT3 * spacing * (q + r / 2.0)
        y = 1.5 * spacing * r
        return x, y

    # Define a compact base configuration: A 3x3 hex block minus corners? 
    # Let's use a 3-3-3-2 arrangement which is very compact.
    # Coords in axial system (q, r)
    # Row r=-1: q=-1, 0, 1
    # Row r=0: q=-1, 0, 1
    # Row r=1: q=-1, 0, 1
    # Row r=2: q=0, 1  (Wait, let's check connectivity)
    # A standard hex grid packing for 11:
    # Center (0,0)
    # Neighbors (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1) -> 7 total (Flower)
    # Add 4 more to the sides to make it compact.
    # Let's try a "diamond" or "rhombus" shape.
    
    # Configuration A: 3-3-3-2
    config_A = [
        (-1, -1), (0, -1), (1, -1),
        (-1, 0), (0, 0), (1, 0),
        (-1, 1), (0, 1), (1, 1),
        (0, 2), (1, 2)
    ]
    
    # Configuration B: 2-3-3-3
    config_B = [
        (0, -1), (1, -1),
        (-1, 0), (0, 0), (1, 0),
        (-1, 1), (0, 1), (1, 1),
        (-1, 2), (0, 2), (1, 2)
    ]
    
    # Configuration C: Linear-ish but compact (2-2-2-2-3)
    config_C = [
        (-1, -1), (0, -1),
        (-1, 0), (0, 0),
        (-1, 1), (0, 1),
        (-1, 2), (0, 2),
        (-1, 3), (0, 3), (1, 3)
    ]
    
    base_configs = [config_A, config_B, config_C]
    
    best_global_score = float('inf')
    best_global_params = None # (tx, ty, theta, scale, config_idx)

    for cfg_idx, config in enumerate(base_configs):
        if time.time() > deadline - 20.0:
            break
            
        # Generate base centers
        base_centers = []
        for q, r in config:
            x, y = axial_to_cartesian(q, r, 1.0)
            base_centers.append((x, y))
        
        # Center the base configuration at origin
        cx_avg = sum(c[0] for c in base_centers) / N
        cy_avg = sum(c[1] for c in base_centers) / N
        base_centers = [(x - cx_avg, y - cy_avg) for x, y in base_centers]
        
        # Pre-compute base angles (0.0 for all)
        base_angles = [0.0] * N
        
        def evaluate_rigid(tx, ty, theta_deg, scale):
            """
            Evaluate a rigid transformation of the base configuration.
            """
            theta_rad = math.radians(theta_deg)
            cos_t = math.cos(theta_rad)
            sin_t = math.sin(theta_rad)
            
            centers = []
            for i in range(N):
                bx, by = base_centers[i]
                # Scale
                rx = bx * scale
                ry = by * scale
                # Rotate
                x = rx * cos_t - ry * sin_t + tx
                y = rx * sin_t + ry * cos_t + ty
                centers.append((x, y))
            
            # Re-center to (0,0) for outer hex calculation?
            # The outer hex is centered at (0,0). The inner cluster can be anywhere.
            # However, for symmetry and simplicity, we can assume the centroid of the cluster
            # should be at (0,0) if the outer hex is symmetric. 
            # Actually, the outer hex constraint is symmetric around (0,0).
            # If we shift the cluster, the max projection changes.
            # But since the outer hex is centered at (0,0), and the cluster is symmetric-ish,
            # optimal tx, ty are likely near 0.
            # Let's allow tx, ty to vary but they will likely converge to 0.
            
            # Calculate vertices for all hexagons
            hexagons = []
            hex_axes = []
            for i in range(N):
                cx, cy = centers[i]
                a = base_angles[i] + theta_deg
                hex_verts = get_hex_vertices(cx, cy, a)
                hexagons.append(hex_verts)
                hex_axes.append(get_hex_axes(a))
            
            # Check overlaps
            overlap_penalty = 0.0
            for i in range(N):
                for j in range(i+1, N):
                    dx = centers[i][0] - centers[j][0]
                    dy = centers[i][1] - centers[j][1]
                    d_sq = dx*dx + dy*dy
                    if not is_disjoint(hexagons[i], hexagons[j], hex_axes[i], hex_axes[j], d_sq):
                        overlap_penalty += 1000.0
            
            if overlap_penalty > 0:
                return overlap_penalty
            
            # Calculate outer side
            # We can also optimize the outer hex angle, but let's fix it to 0 for now 
            # and rely on the cluster rotation to find the best fit.
            # Actually, the outer hex angle is a free parameter in the final return.
            # For the SA, we just need to minimize the side length for a given outer angle.
            # Since the cluster is rigid, rotating the cluster is equivalent to rotating the frame.
            # Let's just calc for outer_angle = 0.
            side = calc_outer_side(hexagons, 0.0)
            
            return side

        # Simulated Annealing on (tx, ty, theta, scale)
        # Initial guess
        current_params = [0.0, 0.0, 0.0, 1.0]
        current_score = evaluate_rigid(*current_params)
        
        local_best_params = current_params.copy()
        local_best_score = current_score
        
        T = 2.0
        T_min = 0.001
        cooling_rate = 0.995
        
        # Time budget per config
        config_deadline = time.time() + 40.0
        
        while T > T_min and time.time() < config_deadline:
            new_params = current_params.copy()
            
            # Perturb
            new_params[0] += random.gauss(0, 0.2) # tx
            new_params[1] += random.gauss(0, 0.2) # ty
            new_params[2] += random.gauss(0, 10.0) # theta
            new_params[3] += random.gauss(0, 0.02) # scale
            
            # Constraints
            new_params[3] = max(0.8, min(1.2, new_params[3]))
            
            new_score = evaluate_rigid(*new_params)
            
            delta_e = new_score - current_score
            if delta_e < 0 or random.random() < math.exp(-delta_e / T):
                current_params = new_params
                current_score = new_score
                if current_score < local_best_score:
                    local_best_score = current_score
                    local_best_params = current_params.copy()
            
            T *= cooling_rate
        
        # Polish with Nelder-Mead on the 4 parameters
        def objective_nm(p):
            return evaluate_rigid(p[0], p[1], p[2], p[3])
        
        try:
            res = minimize(objective_nm, local_best_params, method='Nelder-Mead', 
                           options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-5})
            if res.fun < local_best_score:
                local_best_score = res.fun
                local_best_params = res.x
        except Exception:
            pass
        
        if local_best_score < best_global_score:
            best_global_score = local_best_score
            best_global_params = (local_best_params[0], local_best_params[1], 
                                  local_best_params[2], local_best_params[3], cfg_idx)

    # Finalize best configuration
    if best_global_params is None:
        # Fallback
        tx, ty, theta, scale, cfg_idx = 0.0, 0.0, 0.0, 1.0, 0
    else:
        tx, ty, theta, scale, cfg_idx = best_global_params
    
    config = base_configs[cfg_idx]
    base_centers = []
    for q, r in config:
        x, y = axial_to_cartesian(q, r, 1.0)
        base_centers.append((x, y))
    cx_avg = sum(c[0] for c in base_centers) / N
    cy_avg = sum(c[1] for c in base_centers) / N
    base_centers = [(x - cx_avg, y - cy_avg) for x, y in base_centers]
    
    theta_rad = math.radians(theta)
    cos_t = math.cos(theta_rad)
    sin_t = math.sin(theta_rad)
    
    final_centers = []
    final_angles = []
    for i in range(N):
        bx, by = base_centers[i]
        rx = bx * scale
        ry = by * scale
        x = rx * cos_t - ry * sin_t + tx
        y = rx * sin_t + ry * cos_t + ty
        final_centers.append((x, y))
        final_angles.append(theta)
    
    # Re-center the final configuration to (0,0) to ensure minimal outer hex size?
    # The outer hex is centered at (0,0). If the cluster centroid is not at (0,0),
    # we can shift the whole cluster.
    # Since the outer hex is symmetric, shifting the cluster to center it at (0,0)
    # is always optimal or neutral.
    fx_avg = sum(c[0] for c in final_centers) / N
    fy_avg = sum(c[1] for c in final_centers) / N
    final_centers = [(x - fx_avg, y - fy_avg) for x, y in final_centers]
    
    # Verify validity
    hexagons = []
    hex_axes = []
    for i in range(N):
        cx, cy = final_centers[i]
        a = final_angles[i]
        hex_verts = get_hex_vertices(cx, cy, a)
        hexagons.append(hex_verts)
        hex_axes.append(get_hex_axes(a))
        
    valid = True
    for i in range(N):
        for j in range(i+1, N):
            dx = final_centers[i][0] - final_centers[j][0]
            dy = final_centers[i][1] - final_centers[j][1]
            d_sq = dx*dx + dy*dy
            if not is_disjoint(hexagons[i], hexagons[j], hex_axes[i], hex_axes[j], d_sq):
                valid = False
                break
        if not valid:
            break
            
    if not valid:
        # Fallback to a known valid config: Simple flower + 4
        # Just return a safe, known valid config
        # Flower: (0,0) + 6 neighbors
        # Add 4 more in a line
        safe_config = [
            (0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
            (2,0), (2,-1), (2,1), (3,0)
        ]
        safe_centers = []
        for q, r in safe_config:
            x, y = axial_to_cartesian(q, r, 1.0)
            safe_centers.append((x, y))
        sx_avg = sum(c[0] for c in safe_centers) / N
        sy_avg = sum(c[1] for c in safe_centers) / N
        safe_centers = [(x - sx_avg, y - sy_avg) for x, y in safe_centers]
        
        inner_list = [[x, y, 0.0] for x, y in safe_centers]
        hexagons_safe = [get_hex_vertices(p[0], p[1], p[2]) for p in inner_list]
        best_side = calc_outer_side(hexagons_safe, 0.0)
        return (inner_list, (0.0, 0.0), best_side, 0.0)

    # Optimize outer angle for the final static configuration
    best_side = float('inf')
    best_oa = 0.0
    for oa_deg in range(0, 600):
        oa = oa_deg * 0.1
        side = calc_outer_side(hexagons, oa)
        if side < best_side:
            best_side = side
            best_oa = oa
            
    inner_list = []
    for i in range(N):
        inner_list.append([final_centers[i][0], final_centers[i][1], final_angles[i]])
        
    return (inner_list, (0.0, 0.0), best_side, best_oa)

# EVOLVE-BLOCK-END
