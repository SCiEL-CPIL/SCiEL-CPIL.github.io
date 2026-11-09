import math
import time
import random
import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin (evaluator limit 120s)
    
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

    def evaluate(p, outer_angle_deg=0.0):
        centers = []
        angles = []
        for i in range(N):
            centers.append((p[i*3], p[i*3+1]))
            angles.append(p[i*3+2])
        
        cx_avg = sum(c[0] for c in centers) / N
        cy_avg = sum(c[1] for c in centers) / N
        shifted_centers = [(x - cx_avg, y - cy_avg) for x, y in centers]
        
        hexagons = []
        hex_axes = []
        for i in range(N):
            cx, cy = shifted_centers[i]
            a = angles[i]
            hex_verts = get_hex_vertices(cx, cy, a)
            hexagons.append(hex_verts)
            hex_axes.append(get_hex_axes(a))
        
        overlap_penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dx = shifted_centers[i][0] - shifted_centers[j][0]
                dy = shifted_centers[i][1] - shifted_centers[j][1]
                d_sq = dx*dx + dy*dy
                if not is_disjoint(hexagons[i], hexagons[j], hex_axes[i], hex_axes[j], d_sq):
                    overlap_penalty += 1000.0
        
        side = calc_outer_side(hexagons, outer_angle_deg)
        return overlap_penalty + side

    def axial_to_cartesian(q, r, spacing=1.0):
        x = SQRT3 * spacing * (q + r / 2.0)
        y = 1.5 * spacing * r
        return x, y

    # Initial configurations
    coords1 = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (2,0), (2,-1), (2,1), (1,1)]
    coords2 = [(-1,-1),(0,-1),(1,-1),(-2,0),(-1,0),(0,0),(1,0),(2,0),(-1,1),(0,1),(1,1)]
    coords3 = [(0,0), (1,0), (2,0), (3,0), (4,0), (-1,0), (-2,0), (0,1), (1,1), (2,1), (3,1)]
    coords4 = [(-1,-1),(0,-1),(1,-1),(-1,0),(0,0),(1,0),(-1,1),(0,1),(1,1),(2,0),(2,1)]
    coords5 = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (2,0), (2,1), (-2,1), (-2,0)]
    
    configs = [coords1, coords2, coords3, coords4, coords5]
    
    best_params = None
    best_score = float('inf')
    
    # --- Structural Change: Simulated Annealing with Strict Feasibility ---
    # Unlike the previous greedy local search which minimized (side + penalty),
    # this SA approach treats overlaps as hard constraints in the acceptance step 
    # (or with very high energy cost) and focuses the objective on minimizing 
    # the outer side length. It also optimizes the outer hexagon angle in each 
    # step to keep the container tight.
    
    def sa_search(params, max_iters=3000, start_temp=1.0, cool_rate=0.995):
        """
        Simulated Annealing with strict overlap rejection.
        """
        current_params = np.array(params, dtype=float)
        
        # Calculate initial state
        cx_avg = sum(current_params[i*3] for i in range(N)) / N
        cy_avg = sum(current_params[i*3+1] for i in range(N)) / N
        shifted_centers = [(current_params[i*3] - cx_avg, current_params[i*3+1] - cy_avg) for i in range(N)]
        angles = [current_params[i*3+2] for i in range(N)]
        
        hexagons = []
        hex_axes = []
        for i in range(N):
            hex_verts = get_hex_vertices(shifted_centers[i][0], shifted_centers[i][1], angles[i])
            hexagons.append(hex_verts)
            hex_axes.append(get_hex_axes(angles[i]))
            
        # Optimize outer angle for initial state
        best_oa = 0.0
        best_side = float('inf')
        for oa_deg in range(0, 60, 5):
            s = calc_outer_side(hexagons, oa_deg)
            if s < best_side:
                best_side = s
                best_oa = oa_deg
        
        current_side = best_side
        current_penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dx = shifted_centers[i][0] - shifted_centers[j][0]
                dy = shifted_centers[i][1] - shifted_centers[j][1]
                d_sq = dx*dx + dy*dy
                if not is_disjoint(hexagons[i], hexagons[j], hex_axes[i], hex_axes[j], d_sq):
                    current_penalty += 1.0
        
        current_energy = current_side + current_penalty * 100.0
        best_params_local = current_params.copy()
        best_energy_local = current_energy
        best_side_local = current_side
        
        temp = start_temp
        
        for it in range(max_iters):
            if time.time() > deadline - 5.0:
                break
                
            # Pick a random hexagon to move
            idx = random.randint(0, N-1)
            
            # Generate move
            dx = random.gauss(0, 0.3)
            dy = random.gauss(0, 0.3)
            da = random.gauss(0, 15.0)
            
            new_params = current_params.copy()
            new_params[idx*3] += dx
            new_params[idx*3+1] += dy
            new_params[idx*3+2] += da
            
            # Evaluate new state
            cx_avg_new = sum(new_params[i*3] for i in range(N)) / N
            cy_avg_new = sum(new_params[i*3+1] for i in range(N)) / N
            shifted_centers_new = [(new_params[i*3] - cx_avg_new, new_params[i*3+1] - cy_avg_new) for i in range(N)]
            angles_new = [new_params[i*3+2] for i in range(N)]
            
            hexagons_new = []
            hex_axes_new = []
            for i in range(N):
                hex_verts = get_hex_vertices(shifted_centers_new[i][0], shifted_centers_new[i][1], angles_new[i])
                hexagons_new.append(hex_verts)
                hex_axes_new.append(get_hex_axes(angles_new[i]))
                
            # Check overlaps
            new_penalty = 0.0
            for i in range(N):
                for j in range(i+1, N):
                    dx_c = shifted_centers_new[i][0] - shifted_centers_new[j][0]
                    dy_c = shifted_centers_new[i][1] - shifted_centers_new[j][1]
                    d_sq = dx_c*dx_c + dy_c*dy_c
                    if not is_disjoint(hexagons_new[i], hexagons_new[j], hex_axes_new[i], hex_axes_new[j], d_sq):
                        new_penalty += 1.0
            
            # If overlaps increased significantly, reject immediately (hard constraint)
            if new_penalty > current_penalty + 0.5:
                temp *= cool_rate
                continue
                
            # Optimize outer angle for new state
            best_oa_new = 0.0
            best_side_new = float('inf')
            for oa_deg in range(0, 60, 5):
                s = calc_outer_side(hexagons_new, oa_deg)
                if s < best_side_new:
                    best_side_new = s
                    best_oa_new = oa_deg
            
            new_energy = best_side_new + new_penalty * 100.0
            
            # Acceptance criterion
            delta_e = new_energy - current_energy
            if delta_e < 0 or random.random() < math.exp(-delta_e / temp):
                current_params = new_params
                current_side = best_side_new
                current_penalty = new_penalty
                current_energy = new_energy
                
                if current_energy < best_energy_local:
                    best_energy_local = current_energy
                    best_params_local = current_params.copy()
                    best_side_local = current_side
                    
            temp *= cool_rate
            
        return best_params_local, best_side_local

    # Run SA from each initial configuration
    for config in configs:
        if time.time() > deadline - 10.0:
            break
        
        params = []
        for q, r in config:
            x, y = axial_to_cartesian(q, r, spacing=1.0)
            params.extend([x, y, 0.0])
        
        sa_params, sa_score = sa_search(params, max_iters=2000, start_temp=1.0, cool_rate=0.99)
        if sa_score < best_score:
            best_score = sa_score
            best_params = sa_params

    # If no SA was run or failed, fallback to a simple configuration
    if best_params is None:
        config = configs[0]
        params = []
        for q, r in config:
            x, y = axial_to_cartesian(q, r, spacing=1.0)
            params.extend([x, y, 0.0])
        best_params = np.array(params)
        best_score = evaluate(best_params)

    # Final Validation and Outer Angle Optimization
    final_params = best_params.copy()
    
    centers = []
    angles = []
    for i in range(N):
        centers.append((final_params[i*3], final_params[i*3+1]))
        angles.append(final_params[i*3+2])
        
    cx_avg = sum(c[0] for c in centers) / N
    cy_avg = sum(c[1] for c in centers) / N
    shifted_centers = [(x - cx_avg, y - cy_avg) for x, y in centers]
    
    hexagons = []
    hex_axes = []
    for i in range(N):
        cx, cy = shifted_centers[i]
        a = angles[i]
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
        # Fallback
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

    # Optimize outer angle with finer resolution
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
        inner_list.append([shifted_centers[i][0], shifted_centers[i][1], angles[i]])
        
    return (inner_list, (0.0, 0.0), best_side, best_oa)

# EVOLVE-BLOCK-END
