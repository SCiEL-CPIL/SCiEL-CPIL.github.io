import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 12
DEADLINE = 145.0  # Safety margin before 150s wall-clock kill
SQRT3 = math.sqrt(3)

def construct():
    t0 = time.time()
    
    # Precompute unit hexagon vertices (angle 0)
    # Vertices at 0, 60, 120, 180, 240, 360 degrees
    angles = np.radians(np.arange(0, 360, 60))
    unit_hex_verts = np.column_stack((np.cos(angles), np.sin(angles)))
    
    # Precompute outward normals for unit hexagon (perpendicular to edges)
    # Edge i goes from vertex i to vertex (i+1)%6
    # Normal points outward
    unit_hex_normals = np.zeros((6, 2))
    for i in range(6):
        ex = unit_hex_verts[(i+1)%6, 0] - unit_hex_verts[i, 0]
        ey = unit_hex_verts[(i+1)%6, 1] - unit_hex_verts[i, 1]
        # Outward normal: rotate edge vector by -90 degrees (clockwise)
        unit_hex_normals[i, 0] = ey
        unit_hex_normals[i, 1] = -ex
        norm = math.sqrt(unit_hex_normals[i, 0]**2 + unit_hex_normals[i, 1]**2)
        if norm > 1e-10:
            unit_hex_normals[i, 0] /= norm
            unit_hex_normals[i, 1] /= norm
    
    def get_all_vertices(positions, angles_rad):
        c = np.cos(angles_rad)
        s = np.sin(angles_rad)
        ux = unit_hex_verts[:, 0]
        uy = unit_hex_verts[:, 1]
        rx = ux[None, :] * c[:, None] - uy[None, :] * s[:, None]
        ry = ux[None, :] * s[:, None] + uy[None, :] * c[:, None]
        verts_x = rx + positions[:, 0:1]
        verts_y = ry + positions[:, 1:2]
        all_verts = np.stack((verts_x, verts_y), axis=-1)
        return all_verts

    # Helper: Check if two hexagons are disjoint using SAT with precomputed normals
    def hexagons_disjoint_fast(h1, h2):
        dx = h1[0] - h2[0]
        dy = h1[1] - h2[1]
        dist_sq = dx*dx + dy*dy
        
        if dist_sq > 12.0:
            return True
            
        a1 = h1[2]
        a2 = h2[2]
        
        c1, s1 = math.cos(a1), math.sin(a1)
        c2, s2 = math.cos(a2), math.sin(a2)
        
        ux = unit_hex_verts[:, 0]
        uy = unit_hex_verts[:, 1]
        
        v1x = h1[0] + ux * c1 - uy * s1
        v1y = h1[1] + ux * s1 + uy * c1
        
        v2x = h2[0] + ux * c2 - uy * s2
        v2y = h2[1] + ux * s2 + uy * c2
        
        for i in range(6):
            nx = unit_hex_normals[i, 0] * c1 - unit_hex_normals[i, 1] * s1
            ny = unit_hex_normals[i, 0] * s1 + unit_hex_normals[i, 1] * c1
            
            proj1 = v1x * nx + v1y * ny
            proj2 = v2x * nx + v2y * ny
            
            if proj1.max() < proj2.min() or proj2.max() < proj1.min():
                return True
        
        for i in range(6):
            nx = unit_hex_normals[i, 0] * c2 - unit_hex_normals[i, 1] * s2
            ny = unit_hex_normals[i, 0] * s2 + unit_hex_normals[i, 1] * c2
            
            proj1 = v1x * nx + v1y * ny
            proj2 = v2x * nx + v2y * ny
            
            if proj1.max() < proj2.min() or proj2.max() < proj1.min():
                return True
                
        return False

    def compute_min_outer_side(positions, angles_deg):
        """
        Compute the minimum side length of a regular hexagon containing all inner hexagons.
        Returns (side_length, optimal_outer_angle_deg, center_x, center_y)
        """
        angles_rad = np.radians(angles_deg)
        all_verts = get_all_vertices(positions, angles_rad)
        
        # Flatten to (72, 2)
        verts_flat = all_verts.reshape(-1, 2)
        
        # Center
        center = np.mean(verts_flat, axis=0)
        offsets = verts_flat - center
        
        best_side = float('inf')
        best_phi = 0.0
        
        # Coarse scan
        for phi_deg in range(0, 60, 2):
            phi = np.radians(phi_deg)
            max_extent = 0.0
            for k in range(6):
                normal_angle = phi + np.pi/6 + k * np.pi/3
                nx = math.cos(normal_angle)
                ny = math.sin(normal_angle)
                
                proj = offsets[:, 0] * nx + offsets[:, 1] * ny
                max_proj = np.max(np.abs(proj))
                
                side_k = (2.0 / SQRT3) * max_proj
                if side_k > max_extent:
                    max_extent = side_k
            
            if max_extent < best_side:
                best_side = max_extent
                best_phi = phi_deg
                
        # Fine scan
        for phi_deg in np.arange(best_phi - 2, best_phi + 2, 0.1):
            if phi_deg < 0 or phi_deg >= 60: continue
            phi = np.radians(phi_deg)
            max_extent = 0.0
            for k in range(6):
                normal_angle = phi + np.pi/6 + k * np.pi/3
                nx = math.cos(normal_angle)
                ny = math.sin(normal_angle)
                
                proj = offsets[:, 0] * nx + offsets[:, 1] * ny
                max_proj = np.max(np.abs(proj))
                
                side_k = (2.0 / SQRT3) * max_proj
                if side_k > max_extent:
                    max_extent = side_k
            
            if max_extent < best_side:
                best_side = max_extent
                best_phi = phi_deg

        return best_side, best_phi, center[0], center[1]

    def is_valid_configuration(positions, angles_deg):
        """Check if all hexagons are pairwise disjoint."""
        angles_rad = np.radians(angles_deg)
        for i in range(N):
            for j in range(i+1, N):
                h1 = (positions[i][0], positions[i][1], angles_rad[i])
                h2 = (positions[j][0], positions[j][1], angles_rad[j])
                if not hexagons_disjoint_fast(h1, h2):
                    return False
        return True

    # Boundary Tracing Objective: Optimize six support plane offsets directly
    # params: [cx, cy, d0, d1, d2, d3, d4, d5, x0, y0, a0, x1, y1, a1, ...]
    # where dk is the offset for the k-th support plane (distance from center)
    # The outer hexagon side length is max(2*d_k / SQRT3) over all k
    # We minimize this while ensuring all inner hexagons are inside all six half-planes
    
    def boundary_tracing_objective(params):
        """
        Smooth objective for boundary tracing optimization.
        Returns (objective_value, is_valid)
        """
        cx, cy = params[0], params[1]
        d = params[2:8]  # Six support plane offsets
        
        # Extract inner hexagon positions and angles
        pos = np.zeros((N, 2))
        ang = np.zeros(N)
        for i in range(N):
            pos[i, 0] = params[8 + 3*i]
            pos[i, 1] = params[9 + 3*i]
            ang[i] = params[10 + 3*i]
        
        angles_rad = np.radians(ang)
        all_verts = get_all_vertices(pos, angles_rad)
        verts_flat = all_verts.reshape(-1, 2)
        
        # Check if all vertices are inside all six half-planes
        # For plane k with normal n_k and offset d_k from center:
        # (v - center) . n_k <= d_k for all vertices v
        # n_k = (cos(phi + pi/6 + k*pi/3), sin(phi + pi/6 + k*pi/3))
        # But we're optimizing d_k directly, so we need to infer phi from the geometry
        
        # Actually, let's use a different approach: optimize the six plane normals as unit vectors
        # This is more complex. Instead, let's keep the regular hexagon constraint but optimize
        # the side length directly via a smooth approximation.
        
        # Simpler approach: use log-sum-exp to create a smooth max
        # Compute the required d_k for each plane
        required_d = np.zeros(6)
        for k in range(6):
            # Normal direction for plane k (assuming optimal orientation)
            # We'll compute this from the current best estimate
            # For now, use a fixed orientation and optimize d_k
            phi = 0.0  # Will be updated
            normal_angle = phi + np.pi/6 + k * np.pi/3
            nx = math.cos(normal_angle)
            ny = math.sin(normal_angle)
            
            proj = (verts_flat[:, 0] - cx) * nx + (verts_flat[:, 1] - cy) * ny
            required_d[k] = np.max(proj)
        
        # The side length is max(2 * required_d[k] / SQRT3)
        # Use soft-max for smoothness
        beta = 50.0  # Temperature for log-sum-exp
        log_d = np.log(np.maximum(required_d, 1e-10) + 1e-10)
        log_sum = np.log(np.sum(np.exp(beta * log_d))) / beta
        smooth_side = (2.0 / SQRT3) * np.exp(log_sum)
        
        # Penalty for violations (vertices outside planes)
        penalty = 0.0
        for k in range(6):
            violation = required_d[k] - d[k]
            if violation > 0:
                penalty += violation ** 2
        
        # Also check inner hexagon disjointness
        disjoint_penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                h1 = (pos[i, 0], pos[i, 1], angles_rad[i])
                h2 = (pos[j, 0], pos[j, 1], angles_rad[j])
                if not hexagons_disjoint_fast(h1, h2):
                    disjoint_penalty += 5.0
        
        return smooth_side + penalty + disjoint_penalty

    # Initial Configuration: Honeycomb-like with proper spacing
    cols = 4
    rows = 3
    h_spacing = 3.0
    v_spacing = SQRT3
    
    initial_positions = np.zeros((N, 2))
    initial_angles = np.zeros(N)
    
    idx = 0
    for r in range(rows):
        for c in range(cols):
            offset = (h_spacing / 2.0) if (r % 2 == 1) else 0.0
            initial_positions[idx, 0] = -1.5 * h_spacing + c * h_spacing + offset
            initial_positions[idx, 1] = -(rows - 1) / 2.0 * v_spacing + r * v_spacing
            initial_angles[idx] = 0.0
            idx += 1
            
    best_side = float('inf')
    best_positions = initial_positions.copy()
    best_angles = initial_angles.copy()
    best_phi = 0.0
    best_cx = 0.0
    best_cy = 0.0
    
    current_positions = initial_positions.copy()
    current_angles = initial_angles.copy()
    current_side, current_phi, current_cx, current_cy = compute_min_outer_side(current_positions, current_angles)
    
    if current_side < best_side:
        best_side = current_side
        best_positions = current_positions.copy()
        best_angles = current_angles.copy()
        best_phi = current_phi
        best_cx = current_cx
        best_cy = current_cy

    T = 1.0
    T_min = 0.001
    cooling = 0.9995
    step_size = 0.5
    step_size_min = 0.1
    step_size_max = 1.5
    
    step_count = 0
    max_steps = 50000
    consecutive_rejects = 0
    
    # SA Loop
    while time.time() - t0 < DEADLINE * 0.75 and step_count < max_steps:
        step_count += 1
        
        idx = random.randint(0, N - 1)
        new_positions = current_positions.copy()
        new_angles = current_angles.copy()
        
        dx = random.gauss(0, step_size)
        dy = random.gauss(0, step_size)
        da = random.gauss(0, 10.0)
        
        new_positions[idx, 0] += dx
        new_positions[idx, 1] += dy
        new_angles[idx] += da
        
        angles_rad_new = np.radians(new_angles)
        valid = True
        h_moved = (new_positions[idx, 0], new_positions[idx, 1], angles_rad_new[idx])
        for j in range(N):
            if j == idx: continue
            h_other = (new_positions[j, 0], new_positions[j, 1], angles_rad_new[j])
            if not hexagons_disjoint_fast(h_moved, h_other):
                valid = False
                break
                
        if not valid:
            consecutive_rejects += 1
            if consecutive_rejects > 100:
                step_size = max(step_size_min, step_size * 0.95)
                consecutive_rejects = 0
            T *= cooling
            if T < T_min: T = T_min
            continue
            
        consecutive_rejects = 0
        new_side, new_phi, new_cx, new_cy = compute_min_outer_side(new_positions, new_angles)
        
        if new_side < current_side:
            current_positions = new_positions
            current_angles = new_angles
            current_side = new_side
            current_phi = new_phi
            current_cx = new_cx
            current_cy = new_cy
            
            if new_side < best_side:
                best_side = new_side
                best_positions = new_positions.copy()
                best_angles = new_angles.copy()
                best_phi = new_phi
                best_cx = new_cx
                best_cy = new_cy
            step_size = min(step_size_max, step_size * 1.02)
        else:
            delta = new_side - current_side
            if delta < 0: delta = 0
            if random.random() < math.exp(-delta / T):
                current_positions = new_positions
                current_angles = new_angles
                current_side = new_side
                current_phi = new_phi
                current_cx = new_cx
                current_cy = new_cy
                
                if new_side < best_side:
                    best_side = new_side
                    best_positions = new_positions.copy()
                    best_angles = new_angles.copy()
                    best_phi = new_phi
                    best_cx = new_cx
                    best_cy = new_cy
                    
        T *= cooling
        if T < T_min: T = T_min

    # High-Precision Polish using Boundary Tracing
    # Optimize: center (2), six plane offsets (6), and inner hexagons (36) = 44 parameters
    # But we need to maintain the regular hexagon constraint, so let's optimize:
    # center (2), outer angle (1), and inner hexagons (36) = 39 parameters
    # The side length is computed from the six required plane offsets
    
    def polish_objective_v2(params):
        """
        Polish objective: minimize the outer hexagon side length with smooth constraints.
        params: [cx, cy, phi_deg, x0, y0, a0, x1, y1, a1, ...]
        """
        cx, cy = params[0], params[1]
        phi_deg = params[2]
        phi = np.radians(phi_deg)
        
        pos = np.zeros((N, 2))
        ang = np.zeros(N)
        for i in range(N):
            pos[i, 0] = params[3 + 3*i]
            pos[i, 1] = params[4 + 3*i]
            ang[i] = params[5 + 3*i]
        
        angles_rad = np.radians(ang)
        all_verts = get_all_vertices(pos, angles_rad)
        verts_flat = all_verts.reshape(-1, 2)
        
        # Compute required d_k for each of the six planes
        required_d = np.zeros(6)
        for k in range(6):
            normal_angle = phi + np.pi/6 + k * np.pi/3
            nx = math.cos(normal_angle)
            ny = math.sin(normal_angle)
            
            proj = (verts_flat[:, 0] - cx) * nx + (verts_flat[:, 1] - cy) * ny
            required_d[k] = np.max(proj)
        
        # Side length = max(2 * required_d[k] / SQRT3)
        # Use smooth max via log-sum-exp
        beta = 100.0
        d_safe = np.maximum(required_d, 1e-10)
        log_d = np.log(d_safe)
        log_sum = np.log(np.sum(np.exp(beta * log_d))) / beta
        smooth_side = (2.0 / SQRT3) * np.exp(log_sum)
        
        # Penalty for inner hexagon overlaps
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                h1 = (pos[i, 0], pos[i, 1], angles_rad[i])
                h2 = (pos[j, 0], pos[j, 1], angles_rad[j])
                if not hexagons_disjoint_fast(h1, h2):
                    penalty += 10.0
        
        return smooth_side + penalty

    # Initialize polish parameters from best SA result
    init_params = np.zeros(3 + 3*N)
    init_params[0] = best_cx
    init_params[1] = best_cy
    init_params[2] = best_phi
    for i in range(N):
        init_params[3 + 3*i] = best_positions[i, 0]
        init_params[4 + 3*i] = best_positions[i, 1]
        init_params[5 + 3*i] = best_angles[i]
    
    # Run multiple polish attempts with different initial perturbations
    if time.time() - t0 < DEADLINE * 0.95:
        for attempt in range(3):
            if time.time() - t0 > DEADLINE * 0.95:
                break
                
            # Add small perturbation for diversity
            if attempt > 0:
                perturbed = init_params.copy()
                for i in range(3, 3 + 3*N):
                    perturbed[i] += random.gauss(0, 0.05)
                init_params = perturbed
            
            try:
                res = minimize(polish_objective_v2, init_params, method='Nelder-Mead', 
                               options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-10})
                
                if res.fun < best_side + 0.5:
                    new_cx, new_cy = res.x[0], res.x[1]
                    new_phi = res.x[2]
                    new_pos = np.zeros((N, 2))
                    new_ang = np.zeros(N)
                    for i in range(N):
                        new_pos[i, 0] = res.x[3 + 3*i]
                        new_pos[i, 1] = res.x[4 + 3*i]
                        new_ang[i] = res.x[5 + 3*i]
                    
                    if is_valid_configuration(new_pos, new_ang):
                        new_side, new_phi_opt, new_cx_opt, new_cy_opt = compute_min_outer_side(new_pos, new_ang)
                        if new_side < best_side:
                            best_side = new_side
                            best_positions = new_pos
                            best_angles = new_ang
                            best_phi = new_phi_opt
                            best_cx = new_cx_opt
                            best_cy = new_cy_opt
            except Exception:
                pass

    # Final Result
    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), float(best_angles[i])])
        
    center = (float(best_cx), float(best_cy))
    side = float(best_side)
    angle = float(best_phi)
    
    return inner, center, side, angle
