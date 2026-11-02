import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 12
DEADLINE = 145.0  # Safety margin before 150s wall-clock kill

def construct():
    t0 = time.time()
    
    # Precompute unit hexagon vertices (angle 0)
    angles = np.radians(np.arange(0, 360, 60))
    unit_hex_verts = np.column_stack((np.cos(angles), np.sin(angles)))
    
    # Precompute edge normals for unit hexagon (6 normals)
    edge_normals = np.zeros((6, 2))
    for i in range(6):
        ex = unit_hex_verts[(i+1)%6, 0] - unit_hex_verts[i, 0]
        ey = unit_hex_verts[(i+1)%6, 1] - unit_hex_verts[i, 1]
        norm = math.sqrt(ex*ex + ey*ey)
        edge_normals[i, 0] = -ey / norm
        edge_normals[i, 1] = ex / norm
    
    # Helper: Get vertices of all hexagons given positions and angles
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

    def hexagons_disjoint_fast(h1, h2):
        """
        Fast SAT check for two unit hexagons.
        h1, h2: (x, y, angle_rad)
        """
        dx = h1[0] - h2[0]
        dy = h1[1] - h2[1]
        dist_sq = dx*dx + dy*dy
        
        # If far apart, disjoint. Max center distance for touching hexagons is 2*sqrt(3) ≈ 3.464
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
        v1 = np.column_stack((v1x, v1y))
        
        v2x = h2[0] + ux * c2 - uy * s2
        v2y = h2[1] + ux * s2 + uy * c2
        v2 = np.column_stack((v2x, v2y))
        
        # SAT using precomputed normals rotated by each hexagon's angle
        for i in range(6):
            nx1 = edge_normals[i, 0] * c1 - edge_normals[i, 1] * s1
            ny1 = edge_normals[i, 0] * s1 + edge_normals[i, 1] * c1
            
            proj1 = v1x * nx1 + v1y * ny1
            proj2 = v2x * nx1 + v2y * ny1
            
            if np.max(proj1) < np.min(proj2) or np.max(proj2) < np.min(proj1):
                return True
                
        for i in range(6):
            nx2 = edge_normals[i, 0] * c2 - edge_normals[i, 1] * s2
            ny2 = edge_normals[i, 0] * s2 + edge_normals[i, 1] * c2
            
            proj1 = v1x * nx2 + v1y * ny2
            proj2 = v2x * nx2 + v2y * ny2
            
            if np.max(proj1) < np.min(proj2) or np.max(proj2) < np.min(proj1):
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
        
        sqrt3 = math.sqrt(3)
        best_side = float('inf')
        best_phi = 0.0
        
        # Coarse scan with vectorized evaluation
        for phi_deg in range(0, 60, 1):
            phi = np.radians(phi_deg)
            normal_angles = phi + np.pi/6 + np.arange(6) * np.pi/3
            nx = np.cos(normal_angles)
            ny = np.sin(normal_angles)
            
            proj = offsets[:, 0:1] * nx[None, :] + offsets[:, 1:2] * ny[None, :]
            max_abs_proj = np.max(np.abs(proj), axis=0)
            side_k = (2.0 / sqrt3) * max_abs_proj
            max_extent = np.max(side_k)
            
            if max_extent < best_side:
                best_side = max_extent
                best_phi = phi_deg
                
        # Fine scan
        for phi_deg in np.arange(best_phi - 1, best_phi + 1, 0.05):
            if phi_deg < 0 or phi_deg >= 60: continue
            phi = np.radians(phi_deg)
            normal_angles = phi + np.pi/6 + np.arange(6) * np.pi/3
            nx = np.cos(normal_angles)
            ny = np.sin(normal_angles)
            
            proj = offsets[:, 0:1] * nx[None, :] + offsets[:, 1:2] * ny[None, :]
            max_abs_proj = np.max(np.abs(proj), axis=0)
            side_k = (2.0 / sqrt3) * max_abs_proj
            max_extent = np.max(side_k)
            
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

    # Initial Configuration: Try multiple and pick best
    sqrt3 = math.sqrt(3)
    configs = []
    
    # Config 1: 3x4 grid
    cols, rows = 4, 3
    x0 = -(cols - 1) / 2.0 * 3.0
    y0 = -(rows - 1) / 2.0 * sqrt3
    pos1 = np.zeros((N, 2))
    ang1 = np.zeros(N)
    idx = 0
    for r in range(rows):
        for c in range(cols):
            pos1[idx, 0] = x0 + c * 3.0
            pos1[idx, 1] = y0 + r * sqrt3
            ang1[idx] = 0.0
            idx += 1
    configs.append((pos1, ang1))
    
    # Config 2: Honeycomb with offset rows
    pos2 = np.zeros((N, 2))
    ang2 = np.zeros(N)
    for c in range(4):
        pos2[c, 0] = (c - 1.5) * 3.0
        pos2[c, 1] = 0.0
    for c in range(4):
        pos2[4+c, 0] = (c - 1.5) * 3.0 + 1.5
        pos2[4+c, 1] = 1.5 * sqrt3
    for c in range(4):
        pos2[8+c, 0] = (c - 1.5) * 3.0
        pos2[8+c, 1] = 3.0 * sqrt3
    configs.append((pos2, ang2))
    
    # Config 3: Compact 2x6
    pos3 = np.zeros((N, 2))
    ang3 = np.zeros(N)
    for r in range(2):
        for c in range(6):
            pos3[r*6+c, 0] = (c - 2.5) * 3.0
            pos3[r*6+c, 1] = (r - 0.5) * sqrt3 * 1.8
    configs.append((pos3, ang3))
    
    best_init_side = float('inf')
    best_init_pos = None
    best_init_ang = None
    for pos, ang in configs:
        side, _, _, _ = compute_min_outer_side(pos, ang)
        if side < best_init_side:
            best_init_side = side
            best_init_pos = pos.copy()
            best_init_ang = ang.copy()
            
    best_side = float('inf')
    best_positions = best_init_pos.copy()
    best_angles = best_init_ang.copy()
    best_phi = 0.0
    best_cx = 0.0
    best_cy = 0.0
    
    current_positions = best_init_pos.copy()
    current_angles = best_init_ang.copy()
    current_side, current_phi, current_cx, current_cy = compute_min_outer_side(current_positions, current_angles)
    
    if current_side < best_side:
        best_side = current_side
        best_positions = current_positions.copy()
        best_angles = current_angles.copy()
        best_phi = current_phi
        best_cx = current_cx
        best_cy = current_cy

    T = 0.5
    T_min = 0.0005
    cooling = 0.9997
    step_size = 0.3
    
    step_count = 0
    max_steps = 80000
    
    # SA Loop
    while time.time() - t0 < DEADLINE * 0.75 and step_count < max_steps:
        step_count += 1
        
        idx = random.randint(0, N - 1)
        new_positions = current_positions.copy()
        new_angles = current_angles.copy()
        
        dx = random.gauss(0, step_size)
        dy = random.gauss(0, step_size)
        da = random.gauss(0, 8.0)  # degrees
        
        new_positions[idx, 0] += dx
        new_positions[idx, 1] += dy
        new_angles[idx] += da
        
        # Check validity
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
            T *= cooling
            if T < T_min: T = T_min
            continue
            
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

    # Polish Step with multiple attempts
    def polish_objective(params):
        pos = np.zeros((N, 2))
        ang = np.zeros(N)
        for i in range(N):
            pos[i, 0] = params[3*i]
            pos[i, 1] = params[3*i+1]
            ang[i] = params[3*i+2]
            
        angles_rad = np.radians(ang)
        for i in range(N):
            for j in range(i+1, N):
                h1 = (pos[i, 0], pos[i, 1], angles_rad[i])
                h2 = (pos[j, 0], pos[j, 1], angles_rad[j])
                if not hexagons_disjoint_fast(h1, h2):
                    return 1e6
        
        side, phi, cx, cy = compute_min_outer_side(pos, ang)
        return side

    init_params = np.zeros(3*N)
    for i in range(N):
        init_params[3*i] = best_positions[i, 0]
        init_params[3*i+1] = best_positions[i, 1]
        init_params[3*i+2] = best_angles[i]
        
    polish_results = []
    for attempt in range(3):
        if time.time() - t0 > DEADLINE * 0.92:
            break
        
        try:
            if attempt == 0:
                p = init_params.copy()
            else:
                p = init_params + random.gauss(0, 0.1, 3*N)
            
            res = minimize(polish_objective, p, method='Nelder-Mead', 
                           options={'maxiter': 1000, 'xatol': 1e-5, 'fatol': 1e-7})
            
            if res.fun < 1e5:
                new_pos = np.zeros((N, 2))
                new_ang = np.zeros(N)
                for i in range(N):
                    new_pos[i, 0] = res.x[3*i]
                    new_pos[i, 1] = res.x[3*i+1]
                    new_ang[i] = res.x[3*i+2]
                
                if is_valid_configuration(new_pos, new_ang):
                    side, phi, cx, cy = compute_min_outer_side(new_pos, new_ang)
                    polish_results.append((side, new_pos, new_ang, phi, cx, cy))
        except Exception:
            pass
    
    if polish_results:
        polish_results.sort(key=lambda x: x[0])
        best_side, best_positions, best_angles, best_phi, best_cx, best_cy = polish_results[0]

    # Final Result
    # Construct inner list: (x, y, angle_degrees)
    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), float(best_angles[i])])
        
    center = (float(best_cx), float(best_cy))
    side = float(best_side)
    angle = float(best_phi)
    
    return inner, center, side, angle

# EVOLVE-BLOCK-END
