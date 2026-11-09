import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

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
        1. If d_sq > 4.0 (d > 2.0), definitely disjoint.
        2. If d_sq < 3.0 (d < sqrt(3)), definitely overlap (min distance for disjoint is sqrt(3)).
        3. Otherwise, use SAT.
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

    # Initial configurations
    # 1. Flower + extensions
    coords1 = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (2,0), (2,-1), (2,1), (1,1)]
    # 2. Compact 3-5-3
    coords2 = [(-1,-1),(0,-1),(1,-1),(-2,0),(-1,0),(0,0),(1,0),(2,0),(-1,1),(0,1),(1,1)]
    # 3. Linear
    coords3 = [(0,0), (1,0), (2,0), (3,0), (4,0), (-1,0), (-2,0), (0,1), (1,1), (2,1), (3,1)]
    
    configs = [coords1, coords2, coords3]
    
    best_params = None
    best_score = float('inf')
    
    # New Structural Idea: Rigid Cluster SA
    # We treat the set of hexagons as a rigid body (fixed relative positions) 
    # but allow global rotation, translation, and a "breathing" scale factor.
    # This reduces the search space from 33 dims to 4 dims (x, y, theta, scale),
    # allowing for much deeper search within the time budget.
    
    def evaluate_rigid(base_centers, base_angles, tx, ty, theta_deg, scale):
        """
        Evaluate a rigid transformation of a base configuration.
        base_centers: list of (x,y)
        base_angles: list of angles
        tx, ty: translation
        theta_deg: global rotation
        scale: breathing factor (1.0 is exact fit for touching hexagons)
        """
        # Transform centers
        cos_t = math.cos(math.radians(theta_deg))
        sin_t = math.sin(math.radians(theta_deg))
        
        transformed_centers = []
        transformed_angles = []
        
        for i in range(N):
            bx, by = base_centers[i]
            # Scale relative to centroid (which is 0,0 in our base configs)
            sx = bx * scale
            sy = by * scale
            
            # Rotate
            rx = sx * cos_t - sy * sin_t
            ry = sx * sin_t + sy * cos_t
            
            # Translate
            fx = rx + tx
            fy = ry + ty
            
            transformed_centers.append((fx, fy))
            transformed_angles.append(base_angles[i] + theta_deg)
        
        # Calculate vertices and axes
        hexagons = []
        hex_axes = []
        for i in range(N):
            cx, cy = transformed_centers[i]
            a = transformed_angles[i]
            # Note: vertices are unit hexagons, but if we scale the cluster, 
            # do the hexagons themselves scale? 
            # The problem says "unit regular hexagons". So the hexagons themselves
            # do NOT scale. Only their positions relative to each other scale.
            # This is a "breathing" of the lattice, not a scaling of the objects.
            hex_verts = get_hex_vertices(cx, cy, a)
            hexagons.append(hex_verts)
            hex_axes.append(get_hex_axes(a))
        
        # Overlap Penalty using Smooth Log-Sum-Exp
        # We calculate the "depth" of overlap for each pair.
        # If disjoint, depth = 0. If overlapping, depth > 0.
        # For simplicity and speed, we use a soft penalty based on distance.
        # Min distance between disjoint unit hexagons is sqrt(3).
        # If d < sqrt(3), they overlap.
        
        # However, SAT is the correct check. Let's use a simpler metric for SA speed:
        # We approximate the Minkowski sum.
        # Actually, let's stick to the SAT-based is_disjoint for validity, 
        # but for the *penalty* in SA, we can use a smooth approximation.
        
        # Let's use a simple distance-based penalty for the SA step to be fast,
        # and verify with SAT at the end.
        # Distance between centers: d.
        # If d < 2.0 (diameter), they *might* overlap.
        # If d < sqrt(3), they *definitely* overlap (for aligned hexagons).
        # For arbitrary rotations, the minimum distance is variable.
        
        # To be safe and accurate, let's compute a "separation" metric.
        # We will use the SAT separation distance.
        
        total_penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dx = transformed_centers[i][0] - transformed_centers[j][0]
                dy = transformed_centers[i][1] - transformed_centers[j][1]
                d_sq = dx*dx + dy*dy
                
                # Quick reject
                if d_sq > 4.0:
                    continue
                
                # SAT check for separation
                # We need the minimum separation distance.
                # If they are disjoint, sep > 0. If overlapping, sep < 0.
                # We want to minimize the negative of the separation (i.e., maximize separation).
                # Let's compute the max overlap depth.
                
                axes1 = hex_axes[i]
                axes2 = hex_axes[j]
                all_axes = axes1 + axes2
                
                min_sep = float('inf')
                
                for axis in all_axes:
                    min1, max1 = project_poly(hexagons[i], axis)
                    min2, max2 = project_poly(hexagons[j], axis)
                    
                    # Separation on this axis
                    if max1 < min2:
                        sep = min2 - max1
                    elif max2 < min1:
                        sep = min1 - max2
                    else:
                        sep = 0.0 # Overlapping on this axis
                    
                    if sep < min_sep:
                        min_sep = sep
                
                # min_sep is the minimum separation distance.
                # If min_sep < 0, they are overlapping.
                # We add a penalty proportional to the overlap depth.
                # Use a smooth function to avoid kinks.
                # penalty = max(0, -min_sep)^2
                if min_sep < 0:
                    overlap_depth = -min_sep
                    total_penalty += overlap_depth * overlap_depth * 100.0
        
        # Calculate outer side
        side = calc_outer_side(hexagons, 0.0)
        
        return side + total_penalty

    for config in configs:
        if time.time() > deadline - 30.0:
            break
            
        # Prepare base config
        base_centers = []
        base_angles = []
        for q, r in config:
            x, y = axial_to_cartesian(q, r, spacing=1.0)
            base_centers.append((x, y))
            base_angles.append(0.0)
        
        # Initial params: tx, ty, theta, scale
        # Start with identity transform and scale 1.0
        params = np.array([0.0, 0.0, 0.0, 1.0])
        
        current_score = evaluate_rigid(base_centers, base_angles, *params)
        local_best_params = params.copy()
        local_best_score = current_score
        
        T = 1.0
        T_min = 0.001
        cooling_rate = 0.99
        steps_per_temp = 10
        
        step = 0
        while T > T_min and time.time() < deadline - 15.0:
            new_params = params.copy()
            
            # Random perturbations
            # tx, ty
            new_params[0] += random.gauss(0, 0.1)
            new_params[1] += random.gauss(0, 0.1)
            # theta
            new_params[2] += random.gauss(0, 5.0)
            # scale
            new_params[3] += random.gauss(0, 0.01)
            
            # Keep scale positive
            if new_params[3] < 0.5:
                new_params[3] = 0.5 + random.random() * 0.1
            if new_params[3] > 1.5:
                new_params[3] = 1.5 - random.random() * 0.1
                
            new_score = evaluate_rigid(base_centers, base_angles, *new_params)
            
            delta_e = new_score - current_score
            if delta_e < 0 or random.random() < math.exp(-delta_e / T):
                params = new_params
                current_score = new_score
                if current_score < local_best_score:
                    local_best_score = current_score
                    local_best_params = params.copy()
            
            step += 1
            if step % steps_per_temp == 0:
                T *= cooling_rate
        
        if local_best_score < best_score:
            best_score = local_best_score
            best_params = local_best_params.copy()
            # Store which config was best
            best_config = config

    if best_params is None:
        # Fallback to first config with identity
        best_params = np.array([0.0, 0.0, 0.0, 1.0])
        best_config = coords1

    # Final Polish with Nelder-Mead on the rigid parameters
    def objective_nm(p):
        return evaluate_rigid(base_centers, base_angles, *p)
        
    try:
        res = minimize(objective_nm, best_params, method='Nelder-Mead', 
                       options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-5})
        if res.fun < best_score:
            best_score = res.fun
            best_params = res.x
    except Exception:
        pass

    # Reconstruct final positions
    tx, ty, theta_deg, scale = best_params
    
    cos_t = math.cos(math.radians(theta_deg))
    sin_t = math.sin(math.radians(theta_deg))
    
    final_centers = []
    final_angles = []
    
    for i in range(N):
        bx, by = base_centers[i]
        sx = bx * scale
        sy = by * scale
        
        rx = sx * cos_t - sy * sin_t
        ry = sx * sin_t + sy * cos_t
        
        fx = rx + tx
        fy = ry + ty
        
        final_centers.append((fx, fy))
        final_angles.append(base_angles[i] + theta_deg)
        
    # Center the configuration
    cx_avg = sum(c[0] for c in final_centers) / N
    cy_avg = sum(c[1] for c in final_centers) / N
    shifted_centers = [(x - cx_avg, y - cy_avg) for x, y in final_centers]
    
    # Verify validity
    hexagons = []
    hex_axes = []
    for i in range(N):
        cx, cy = shifted_centers[i]
        a = final_angles[i]
        hex_verts = get_hex_vertices(cx, cy, a)
        hexagons.append(hex_verts)
        hex_axes.append(get_hex_axes(a))
        
    valid = True
    for i in range(N):
        for j in range(i+1, N):
            dx = shifted_centers[i][0] - shifted_centers[j][0]
            dy = shifted_centers[i][1] - shifted_centers[j][1]
            d_sq = dx*dx + dy*dy
            if not is_disjoint(hexagons[i], hexagons[j], hex_axes[i], hex_axes[j], d_sq):
                valid = False
                break
        if not valid:
            break
            
    if not valid:
        # Fallback to a known valid config (identity transform of config 1)
        inner_list = []
        for q, r in coords1:
            x, y = axial_to_cartesian(q, r, spacing=1.0)
            inner_list.append([x, y, 0.0])
        cx_avg_fb = sum(p[0] for p in inner_list) / N
        cy_avg_fb = sum(p[1] for p in inner_list) / N
        inner_list = [[x - cx_avg_fb, y - cy_avg_fb, a] for x, y, a in inner_list]
        hexagons_fb = [get_hex_vertices(p[0], p[1], p[2]) for p in inner_list]
        best_side = calc_outer_side(hexagons_fb, 0.0)
        return (inner_list, (0.0, 0.0), best_side, 0.0)

    # Optimize outer angle
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
        inner_list.append([shifted_centers[i][0], shifted_centers[i][1], final_angles[i]])
        
    return (inner_list, (0.0, 0.0), best_side, best_oa)

# EVOLVE-BLOCK-END
