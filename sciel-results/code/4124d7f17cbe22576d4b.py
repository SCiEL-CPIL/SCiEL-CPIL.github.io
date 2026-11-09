import math
import time
import random
import numpy as np
from scipy.optimize import minimize

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
    # 1. Flower + extensions
    coords1 = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (2,0), (2,-1), (2,1), (1,1)]
    # 2. Compact 3-5-3
    coords2 = [(-1,-1),(0,-1),(1,-1),(-2,0),(-1,0),(0,0),(1,0),(2,0),(-1,1),(0,1),(1,1)]
    # 3. Linear
    coords3 = [(0,0), (1,0), (2,0), (3,0), (4,0), (-1,0), (-2,0), (0,1), (1,1), (2,1), (3,1)]
    
    configs = [coords1, coords2, coords3]
    
    best_params = None
    best_score = float('inf')
    
    # --- New Structural Idea: Contact Graph Local Search ---
    # We define a contact graph for each configuration. 
    # The optimization tries to maintain these contacts while minimizing the bounding box.
    
    def get_contact_graph(coords):
        """
        Identify pairs of hexagons that are 'touching' in the initial lattice configuration.
        In a hexagonal lattice, neighbors are at distance sqrt(3).
        """
        centers = []
        for q, r in coords:
            x, y = axial_to_cartesian(q, r, spacing=1.0)
            centers.append((x, y))
        
        # Center the coordinates
        cx_avg = sum(c[0] for c in centers) / N
        cy_avg = sum(c[1] for c in centers) / N
        centers = [(x - cx_avg, y - cy_avg) for x, y in centers]
        
        contacts = []
        for i in range(N):
            for j in range(i+1, N):
                dx = centers[i][0] - centers[j][0]
                dy = centers[i][1] - centers[j][1]
                d_sq = dx*dx + dy*dy
                # If distance is close to sqrt(3), they are touching
                if abs(d_sq - 3.0) < 0.1:
                    contacts.append((i, j))
        return centers, contacts

    def evaluate_with_contacts(centers, angles, contacts, outer_angle_deg=0.0):
        """
        Evaluate configuration with a penalty for breaking contact constraints.
        """
        hexagons = []
        hex_axes = []
        for i in range(N):
            cx, cy = centers[i]
            a = angles[i]
            hex_verts = get_hex_vertices(cx, cy, a)
            hexagons.append(hex_verts)
            hex_axes.append(get_hex_axes(a))
        
        overlap_penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dx = centers[i][0] - centers[j][0]
                dy = centers[i][1] - centers[j][1]
                d_sq = dx*dx + dy*dy
                if not is_disjoint(hexagons[i], hexagons[j], hex_axes[i], hex_axes[j], d_sq):
                    overlap_penalty += 1000.0
        
        # Contact penalty: try to keep touching hexagons at distance sqrt(3)
        contact_penalty = 0.0
        for i, j in contacts:
            dx = centers[i][0] - centers[j][0]
            dy = centers[i][1] - centers[j][1]
            d_sq = dx*dx + dy*dy
            # Penalty for deviation from sqrt(3) (d_sq = 3)
            # Use a quadratic penalty
            contact_penalty += (d_sq - 3.0)**2 * 10.0
            
        side = calc_outer_side(hexagons, outer_angle_deg)
        return overlap_penalty + contact_penalty + side

    # Run Contact Graph Local Search for each config
    for config in configs:
        if time.time() > deadline - 20.0:
            break
            
        base_centers, contacts = get_contact_graph(config)
        
        # Initial params: centers and angles (0)
        params = []
        for i in range(N):
            params.extend([base_centers[i][0], base_centers[i][1], 0.0])
        
        current_score = evaluate_with_contacts(base_centers, [0.0]*N, contacts)
        local_best_params = params.copy()
        local_best_score = current_score
        
        # Local Search with Simulated Annealing on the contact-constrained space
        T = 2.0
        T_min = 0.01
        cooling_rate = 0.995
        step_size = 0.1
        
        while T > T_min and time.time() < deadline - 10.0:
            new_params = params.copy()
            
            # Perturb a random hexagon
            idx = random.randint(0, N-1)
            new_params[idx*3] += random.gauss(0, step_size)
            new_params[idx*3+1] += random.gauss(0, step_size)
            new_params[idx*3+2] += random.gauss(0, 10.0)
            
            # Extract centers and angles for evaluation
            new_centers = []
            new_angles = []
            for i in range(N):
                new_centers.append((new_params[i*3], new_params[i*3+1]))
                new_angles.append(new_params[i*3+2])
                
            new_score = evaluate_with_contacts(new_centers, new_angles, contacts)
            
            delta_e = new_score - current_score
            if delta_e < 0 or random.random() < math.exp(-delta_e / T):
                params = new_params
                current_score = new_score
                if current_score < local_best_score:
                    local_best_score = current_score
                    local_best_params = params.copy()
            
            T *= cooling_rate
            
            # Adaptive step size
            if random.random() < 0.1:
                step_size *= 0.9
                step_size = max(step_size, 0.01)

        # Update best
        if local_best_score < best_score:
            best_score = local_best_score
            best_params = local_best_params

    # If no contact graph SA was run or failed, fallback to standard SA
    if best_params is None:
        config = configs[0]
        params = []
        for q, r in config:
            x, y = axial_to_cartesian(q, r, spacing=1.0)
            params.extend([x, y, 0.0])
        best_params = np.array(params)
        
        current_score = evaluate(params)
        local_best_params = params.copy()
        local_best_score = current_score
        
        T = 10.0
        T_min = 0.01
        cooling_rate = 0.995
        
        while T > T_min and time.time() < deadline - 5.0:
            new_params = params.copy()
            idx = random.randint(0, N-1)
            dx = random.gauss(0, 0.3)
            dy = random.gauss(0, 0.3)
            da = random.gauss(0, 15.0)
            
            new_params[idx*3] += dx
            new_params[idx*3+1] += dy
            new_params[idx*3+2] += da
            
            new_score = evaluate(new_params)
            
            delta_e = new_score - current_score
            if delta_e < 0 or random.random() < math.exp(-delta_e / T):
                params = new_params
                current_score = new_score
                if current_score < local_best_score:
                    local_best_score = current_score
                    local_best_params = params.copy()
            
            T *= cooling_rate
        
        best_params = local_best_params
        best_score = local_best_score

    # Polish with Nelder-Mead on the full parameter space
    def objective_nm(p):
        return evaluate(p, outer_angle_deg=0.0)
        
    try:
        res = minimize(objective_nm, best_params, method='Nelder-Mead', 
                       options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-4})
        if res.fun < best_score:
            best_score = res.fun
            best_params = res.x
    except Exception:
        pass

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
        inner_list.append([shifted_centers[i][0], shifted_centers[i][1], angles[i]])
        
    return (inner_list, (0.0, 0.0), best_side, best_oa)

# EVOLVE-BLOCK-END
