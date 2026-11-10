"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin (evaluator limit 150s, keep 15s buffer)
    
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

    # Helper: Get the 6 unique normal axes for a hexagon
    def get_hex_axes(angle_deg):
        axes = []
        for i in range(6):
            normal_angle = math.radians(angle_deg + i * 60 + 30)
            nx = math.cos(normal_angle)
            ny = math.sin(normal_angle)
            axes.append((nx, ny))
        return axes

    # Fast disjoint check using SAT (for final validation only)
    def is_disjoint(hex1_verts, hex2_verts, axes1, axes2):
        all_axes = axes1 + axes2
        for axis in all_axes:
            nx, ny = axis
            min1 = min(vx * nx + vy * ny for vx, vy in hex1_verts)
            max1 = max(vx * nx + vy * ny for vx, vy in hex1_verts)
            min2 = min(vx * nx + vy * ny for vx, vy in hex2_verts)
            max2 = max(vx * nx + vy * ny for vx, vy in hex2_verts)
            if max1 < min2 or max2 < min1:
                return True
        return False

    # Calculate minimum outer hexagon side length containing all given hexagons
    def calc_outer_side(hexagons, outer_angle_deg):
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
        
        return 2.0 * max_proj / SQRT3

    # Axial to Cartesian conversion
    def axial_to_cartesian(q, r, spacing=1.0):
        x = SQRT3 * spacing * (q + r / 2.0)
        y = 1.5 * spacing * r
        return x, y

    # --- New Strategy: Rigid Cluster Optimization ---
    
    # Define a "rung": a pair of hexagons that are close together.
    # We optimize the positions of these rungs relative to each other.
    # Structure:
    # Rung 0: Hex 0, 1
    # Rung 1: Hex 2, 3
    # Rung 2: Hex 4, 5
    # Rung 3: Hex 6, 7
    # Rung 4: Hex 8, 9
    # Single: Hex 10 (center of mass or specific spot)
    
    # Initial guess for rung relative offsets (dx, dy, dangle)
    # We will optimize these offsets.
    
    def evaluate_cluster(params, num_outer_samples=12):
        """
        params: [dx0, dy0, da0, dx1, dy1, da1, ..., dx4, dy4, da4, dx10, dy10, da10]
        Total 15 params.
        """
        # Extract params
        rungs = []
        for i in range(5):
            dx = params[i*3]
            dy = params[i*3+1]
            da = params[i*3+2]
            rungs.append((dx, dy, da))
        
        single_dx = params[15]
        single_dy = params[16]
        single_da = params[17]
        
        # Base positions for rungs (fixed relative to origin, but we can shift/rotate whole thing? 
        # Actually, better to fix one rung at origin to remove global DOF, and optimize relative positions.
        # However, the outer hexagon center is also variable. 
        # Let's fix Rung 0 at (0,0,0).
        # Rung 1..4 and Single 10 are optimized relative to Rung 0.
        
        # Rung 0 fixed at 0,0,0
        rung_pos = [(0.0, 0.0, 0.0)]
        for i in range(1, 5):
            rung_pos.append(rungs[i])
        
        single_pos = (single_dx, single_dy, single_da)
        
        # Generate hexagon centers
        # Within a rung, hexagon 2i is at (0,0) relative to rung, hexagon 2i+1 is at (1.5, SQRT3/2) relative?
        # Standard hex tiling: neighbors are at distance SQRT3.
        # Let's say in a rung, hex i is at (0,0) and hex i+1 is at (SQRT3, 0) with same rotation.
        # This creates a "dumbbell" of width 2*SQRT3 approx.
        
        hex_centers = []
        hex_angles = []
        
        for i in range(5):
            rx, ry, ra = rung_pos[i]
            # Hex 2i
            hex_centers.append((rx, ry))
            hex_angles.append(ra)
            # Hex 2i+1
            # Offset within rung: (SQRT3, 0) rotated by ra
            ox = SQRT3 * math.cos(math.radians(ra))
            oy = SQRT3 * math.sin(math.radians(ra))
            hex_centers.append((rx + ox, ry + oy))
            hex_angles.append(ra)
            
        # Hex 10
        sx, sy, sa = single_pos
        hex_centers.append((sx, sy))
        hex_angles.append(sa)
        
        # Center of mass correction
        cx_avg = sum(c[0] for c in hex_centers) / N
        cy_avg = sum(c[1] for c in hex_centers) / N
        shifted_centers = [(x - cx_avg, y - cy_avg) for x, y in hex_centers]
        
        # Calculate smooth objective
        # 1. Containment (Smooth Max)
        # We want to minimize the outer side length.
        # Outer side L is determined by max projection.
        # L = 2 * max_proj / SQRT3
        # We approximate max_proj with soft-max for differentiability? 
        # Actually, Nelder-Mead doesn't need gradients, but smoothness helps.
        # Let's just use the actual max projection but sample outer angles.
        
        # To speed up, we can pre-compute vertices for each hexagon
        hex_verts_list = []
        for i in range(N):
            cx, cy = shifted_centers[i]
            a = hex_angles[i]
            hex_verts_list.append(get_hex_vertices(cx, cy, a))
        
        # Find best outer angle
        best_side = float('inf')
        for k in range(num_outer_samples):
            oa_deg = k * 60.0 / num_outer_samples
            side = calc_outer_side(hex_verts_list, oa_deg)
            if side < best_side:
                best_side = side
        
        # 2. Overlap Penalty (Smooth)
        # For each pair, calculate minimum distance between edges?
        # Approximation: If center distance < 2*SQRT3, they might overlap.
        # More accurate: Check SAT. If overlapping, penalty = penetration depth squared.
        # For speed, use center distance heuristic first, then refine if close.
        
        overlap_penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dx = shifted_centers[i][0] - shifted_centers[j][0]
                dy = shifted_centers[i][1] - shifted_centers[j][1]
                dist = math.sqrt(dx*dx + dy*dy)
                
                # Hexagons of side 1 have inradius SQRT3/2, circumradius 1.
                # Min distance between centers for no overlap (same orientation) is SQRT3.
                # If dist < SQRT3, they definitely overlap.
                # If dist > 2, they definitely don't.
                # Between SQRT3 and 2, it depends on rotation.
                
                if dist < 1.5: # Definitely overlapping region
                    # Estimate penetration
                    # Approximate as circles of radius SQRT3/2? No, that's too loose.
                    # Use SAT for this pair only if close
                    axes1 = get_hex_axes(hex_angles[i])
                    axes2 = get_hex_axes(hex_angles[j])
                    if not is_disjoint(hex_verts_list[i], hex_verts_list[j], axes1, axes2):
                        # Calculate penetration depth
                        # For simplicity, use distance to SQRT3
                        pen = SQRT3 - dist
                        if pen > 0:
                            overlap_penalty += 100.0 * pen * pen + 10.0
                elif dist < 2.0: # Potential overlap region
                    axes1 = get_hex_axes(hex_angles[i])
                    axes2 = get_hex_axes(hex_angles[j])
                    if not is_disjoint(hex_verts_list[i], hex_verts_list[j], axes1, axes2):
                        # Need better penetration estimate
                        # Approximate: The overlap is roughly proportional to how much they penetrate
                        # Let's use a fixed penalty if SAT says overlap
                        overlap_penalty += 50.0
        
        return best_side + overlap_penalty

    # Initial parameters for the cluster
    # We want to form a compact shape.
    # Rungs arranged in a triangle or hexagon?
    # Let's try a 3-3-3-1-1 arrangement? No, 5 rungs + 1 single.
    # Arrange rungs in a hexagonal lattice.
    
    # Rung 0 at (0,0)
    # Rung 1 at (SQRT3*2, 0) ?
    # Rung 2 at (SQRT3, 1.5*SQRT3) ?
    
    # Let's initialize with a known good lattice configuration and convert to cluster params.
    # Config: 3-5-3
    coords1 = [(-1,-1),(0,-1),(1,-1),(-2,0),(-1,0),(0,0),(1,0),(2,0),(-1,1),(0,1),(1,1)]
    
    # Map these to cluster structure?
    # This is hard to map exactly. Let's just use random-ish initial params around a lattice.
    
    def make_initial_params_cluster():
        params = np.zeros(18)
        # Rung 0 fixed at 0
        # Rung 1: offset (3.0, 0.0, 0.0)
        params[3] = 3.0
        params[4] = 0.0
        # Rung 2: offset (1.5, 2.6, 0.0)
        params[6] = 1.5
        params[7] = 2.6
        # Rung 3: offset (-1.5, 2.6, 0.0)
        params[9] = -1.5
        params[10] = 2.6
        # Rung 4: offset (-3.0, 0.0, 0.0)
        params[12] = -3.0
        params[13] = 0.0
        # Single 10: offset (0.0, 0.0, 0.0) - center
        params[15] = 0.0
        params[16] = 0.0
        # Add small noise
        for i in range(18):
            params[i] += random.gauss(0, 0.1)
        return params

    best_params = None
    best_score = float('inf')
    
    # Multi-start
    for start_iter in range(5):
        if time.time() > deadline - 20.0:
            break
            
        params = make_initial_params_cluster()
        
        # Simulated Annealing on cluster params
        T = 2.0
        T_min = 0.01
        cooling_rate = 0.995
        current_score = evaluate_cluster(params, num_outer_samples=6)
        local_best_params = params.copy()
        local_best_score = current_score
        
        step = 0
        while T > T_min and time.time() < deadline - 15.0:
            new_params = params.copy()
            idx = random.randint(0, 17)
            
            # Perturbation
            step_scale = 0.5 + 0.2 * (T / 2.0)
            new_params[idx] += random.gauss(0, step_scale)
            
            new_score = evaluate_cluster(new_params, num_outer_samples=6)
            
            delta_e = new_score - current_score
            if delta_e < 0 or random.random() < math.exp(-delta_e / T):
                params = new_params
                current_score = new_score
                if current_score < local_best_score:
                    local_best_score = current_score
                    local_best_params = params.copy()
            
            step += 1
            if step % 10 == 0:
                T *= cooling_rate
        
        # Nelder-Mead refinement
        if time.time() < deadline - 10.0:
            def objective_nm(p):
                return evaluate_cluster(p, num_outer_samples=12)
            
            try:
                res = minimize(objective_nm, local_best_params, method='Nelder-Mead',
                             options={'maxiter': 100, 'xatol': 1e-4, 'fatol': 1e-4})
                if res.fun < local_best_score:
                    local_best_params = res.x
                    local_best_score = res.fun
            except Exception:
                pass
        
        if local_best_score < best_score:
            best_score = local_best_score
            best_params = local_best_params.copy()
    
    # Final validation and extraction
    if best_params is None:
        # Fallback to simple lattice
        coords1 = [(-1,-1),(0,-1),(1,-1),(-2,0),(-1,0),(0,0),(1,0),(2,0),(-1,1),(0,1),(1,1)]
        inner_list = []
        for q, r in coords1:
            x, y = axial_to_cartesian(q, r, spacing=1.0)
            inner_list.append([x, y, 0.0])
        cx_avg = sum(p[0] for p in inner_list) / N
        cy_avg = sum(p[1] for p in inner_list) / N
        inner_list = [[x - cx_avg, y - cy_avg, a] for x, y, a in inner_list]
        hexagons = [get_hex_vertices(p[0], p[1], p[2]) for p in inner_list]
        best_side = calc_outer_side(hexagons, 0.0)
        return (inner_list, (0.0, 0.0), best_side, 0.0)

    # Reconstruct hexagons from best cluster params
    rungs = []
    for i in range(5):
        dx = best_params[i*3]
        dy = best_params[i*3+1]
        da = best_params[i*3+2]
        rungs.append((dx, dy, da))
    
    single_dx = best_params[15]
    single_dy = best_params[16]
    single_da = best_params[17]
    
    rung_pos = [(0.0, 0.0, 0.0)]
    for i in range(1, 5):
        rung_pos.append(rungs[i])
    
    single_pos = (single_dx, single_dy, single_da)
    
    hex_centers = []
    hex_angles = []
    
    for i in range(5):
        rx, ry, ra = rung_pos[i]
        hex_centers.append((rx, ry))
        hex_angles.append(ra)
        ox = SQRT3 * math.cos(math.radians(ra))
        oy = SQRT3 * math.sin(math.radians(ra))
        hex_centers.append((rx + ox, ry + oy))
        hex_angles.append(ra)
        
    hex_centers.append((single_dx, single_dy))
    hex_angles.append(single_da)
    
    cx_avg = sum(c[0] for c in hex_centers) / N
    cy_avg = sum(c[1] for c in hex_centers) / N
    shifted_centers = [(x - cx_avg, y - cy_avg) for x, y in hex_centers]
    
    hexagons = []
    hex_axes = []
    for i in range(N):
        cx, cy = shifted_centers[i]
        a = hex_angles[i]
        hex_verts = get_hex_vertices(cx, cy, a)
        hexagons.append(hex_verts)
        hex_axes.append(get_hex_axes(a))
        
    # Check disjointness
    valid = True
    for i in range(N):
        for j in range(i+1, N):
            if not is_disjoint(hexagons[i], hexagons[j], hex_axes[i], hex_axes[j]):
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Fallback: use initial lattice
        coords1 = [(-1,-1),(0,-1),(1,-1),(-2,0),(-1,0),(0,0),(1,0),(2,0),(-1,1),(0,1),(1,1)]
        inner_list = []
        for q, r in coords1:
            x, y = axial_to_cartesian(q, r, spacing=1.0)
            inner_list.append([x, y, 0.0])
        cx_avg = sum(p[0] for p in inner_list) / N
        cy_avg = sum(p[1] for p in inner_list) / N
        inner_list = [[x - cx_avg, y - cy_avg, a] for x, y, a in inner_list]
        hexagons = [get_hex_vertices(p[0], p[1], p[2]) for p in inner_list]
        best_side = calc_outer_side(hexagons, 0.0)
        return (inner_list, (0.0, 0.0), best_side, 0.0)

    # Find best outer angle with fine resolution
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
        inner_list.append([shifted_centers[i][0], shifted_centers[i][1], hex_angles[i]])
        
    return (inner_list, (0.0, 0.0), best_side, best_oa)

# EVOLVE-BLOCK-END
