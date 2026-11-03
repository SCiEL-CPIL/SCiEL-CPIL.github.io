import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 12
DEADLINE = 140.0
SQRT3 = math.sqrt(3)

def construct():
    t0 = time.time()
    random.seed(42)
    np.random.seed(42)

    # Unit hexagon vertices (flat-top orientation, angle 0)
    hex_angles = np.radians(np.arange(0, 360, 60))
    unit_verts = np.column_stack((np.cos(hex_angles), np.sin(hex_angles)))
    # Edge normals
    edge_normals = np.zeros((6, 2))
    for i in range(6):
        ex = unit_verts[(i+1)%6, 0] - unit_verts[i, 0]
        ey = unit_verts[(i+1)%6, 1] - unit_verts[i, 1]
        nx, ny = -ey, ex
        n = math.sqrt(nx*nx + ny*ny)
        edge_normals[i] = [nx/n, ny/n]

    def hex_disjoint_fast(h1, h2):
        """Check if two hexagons (x, y, angle_rad) are disjoint using SAT."""
        dx = h1[0] - h2[0]
        dy = h1[1] - h2[1]
        dist_sq = dx*dx + dy*dy
        if dist_sq > 4.5:  # max distance where they could overlap
            return True
        a1, a2 = h1[2], h2[2]
        c1, s1 = math.cos(a1), math.sin(a1)
        c2, s2 = math.cos(a2), math.sin(a2)
        v1x = h1[0] + unit_verts[:, 0]*c1 - unit_verts[:, 1]*s1
        v1y = h1[1] + unit_verts[:, 0]*s1 + unit_verts[:, 1]*c1
        v2x = h2[0] + unit_verts[:, 0]*c2 - unit_verts[:, 1]*s2
        v2y = h2[1] + unit_verts[:, 0]*s2 + unit_verts[:, 1]*c2
        en1x = edge_normals[:, 0]*c1 - edge_normals[:, 1]*s1
        en1y = edge_normals[:, 0]*s1 + edge_normals[:, 1]*c1
        en2x = edge_normals[:, 0]*c2 - edge_normals[:, 1]*s2
        en2y = edge_normals[:, 0]*s2 + edge_normals[:, 1]*c2
        for i in range(6):
            nx, ny = en1x[i], en1y[i]
            p1 = v1x*nx + v1y*ny
            p2 = v2x*nx + v2y*ny
            if p1.max() < p2.min() or p2.max() < p1.min():
                return True
        for i in range(6):
            nx, ny = en2x[i], en2y[i]
            p1 = v1x*nx + v1y*ny
            p2 = v2x*nx + v2y*ny
            if p1.max() < p2.min() or p2.max() < p1.min():
                return True
        return False

    def compute_min_outer_side(positions, angles_deg):
        """Compute minimum outer hexagon side length to contain all inner hexagons."""
        angles_rad = np.radians(angles_deg)
        all_verts = np.zeros((N*6, 2))
        for i in range(N):
            c, s = math.cos(angles_rad[i]), math.sin(angles_rad[i])
            rx = unit_verts[:, 0]*c - unit_verts[:, 1]*s
            ry = unit_verts[:, 0]*s + unit_verts[:, 1]*c
            all_verts[i*6:(i+1)*6, 0] = positions[i, 0] + rx
            all_verts[i*6:(i+1)*6, 1] = positions[i, 1] + ry
        
        center = np.mean(all_verts, axis=0)
        offsets = all_verts - center
        
        def eval_phi(phi_deg):
            phi = np.radians(phi_deg)
            normal_angles = phi + np.pi/6 + np.arange(6)*np.pi/3
            nx = np.cos(normal_angles)
            ny = np.sin(normal_angles)
            proj = offsets[:, 0:1]*nx[None, :] + offsets[:, 1:2]*ny[None, :]
            abs_proj = np.abs(proj)
            max_projs = np.max(abs_proj, axis=0)
            side_vals = (2.0/SQRT3)*max_projs
            return np.max(side_vals)
        
        best_side = float('inf')
        best_phi = 0.0
        for phi_deg in range(0, 60, 1):
            s = eval_phi(phi_deg)
            if s < best_side:
                best_side = s
                best_phi = phi_deg
        
        try:
            lo = max(0.0, best_phi - 1.0)
            hi = min(60.0, best_phi + 1.0)
            res = minimize_scalar(eval_phi, bounds=(lo, hi), method='bounded', options={'xatol': 1e-8})
            if res.success and res.fun < best_side:
                best_side = res.fun
                best_phi = res.x
        except Exception:
            pass
        
        return best_side, best_phi, center[0], center[1]

    def is_valid(positions, angles_deg):
        angles_rad = np.radians(angles_deg)
        for i in range(N):
            for j in range(i+1, N):
                if not hex_disjoint_fast((positions[i,0], positions[i,1], angles_rad[i]),
                                         (positions[j,0], positions[j,1], angles_rad[j])):
                    return False
        return True

    # Honeycomb initial configuration
    a1 = np.array([SQRT3, 0.0])
    a2 = np.array([SQRT3/2.0, 1.5])
    lattice_coords = [(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,-1),(-1,1),(2,-1),(-2,1),(1,1),(-1,-1),(2,0)]
    
    def make_honeycomb():
        pos = np.zeros((N, 2))
        ang = np.zeros(N)
        for idx, (i, j) in enumerate(lattice_coords):
            pos[idx] = i*a1 + j*a2
        return pos, ang

    best_side = float('inf')
    best_positions, best_angles = make_honeycomb()
    best_phi = 0.0
    best_cx = best_cy = 0.0

    # Initial evaluation
    s, phi, cx, cy = compute_min_outer_side(best_positions, best_angles)
    best_side, best_phi, best_cx, best_cy = s, phi, cx, cy

    # L-BFGS-B optimization from multiple random initializations
    def objective(params):
        pos = np.zeros((N, 2))
        ang = np.zeros(N)
        for i in range(N):
            pos[i, 0] = params[3*i]
            pos[i, 1] = params[3*i+1]
            ang[i] = params[3*i+2]
        
        angles_rad = np.radians(ang)
        for i in range(N):
            for j in range(i+1, N):
                if not hex_disjoint_fast((pos[i,0], pos[i,1], angles_rad[i]),
                                         (pos[j,0], pos[j,1], angles_rad[j])):
                    return 1e6
        side, _, _, _ = compute_min_outer_side(pos, ang)
        return side

    def run_lbfgs(init_pos, init_ang, max_iter=2000):
        init_params = np.zeros(3*N)
        for i in range(N):
            init_params[3*i] = init_pos[i, 0]
            init_params[3*i+1] = init_pos[i, 1]
            init_params[3*i+2] = init_ang[i]
        
        try:
            res = minimize(objective, init_params, method='L-BFGS-B',
                          options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8})
            if res.fun < 1e5:
                pos = np.zeros((N, 2))
                ang = np.zeros(N)
                for i in range(N):
                    pos[i, 0] = res.x[3*i]
                    pos[i, 1] = res.x[3*i+1]
                    ang[i] = res.x[3*i+2]
                if is_valid(pos, ang):
                    side, phi, cx, cy = compute_min_outer_side(pos, ang)
                    return side, pos, ang, phi, cx, cy
        except Exception:
            pass
        return None

    # Run L-BFGS-B from multiple initializations
    max_init_time = DEADLINE * 0.7
    num_attempts = 0
    
    while time.time() - t0 < max_init_time:
        num_attempts += 1
        if time.time() - t0 > max_init_time:
            break
        
        # Create perturbed initialization
        if num_attempts == 1:
            init_pos, init_ang = make_honeycomb()
        else:
            init_pos, init_ang = make_honeycomb()
            # Add random perturbation
            for i in range(N):
                init_pos[i, 0] += random.gauss(0, 0.5)
                init_pos[i, 1] += random.gauss(0, 0.5)
                init_ang[i] += random.gauss(0, 10.0)
        
        result = run_lbfgs(init_pos, init_ang)
        if result and result[0] < best_side:
            best_side, best_positions, best_angles, best_phi, best_cx, best_cy = result
    
    # Polish phase with coordinate descent
    remaining_time = DEADLINE - (time.time() - t0)
    if remaining_time > 3.0:
        def polish_cd(pos, ang, time_limit):
            current_side, _, _, _ = compute_min_outer_side(pos, ang)
            best_side_cd = current_side
            best_pos_cd = pos.copy()
            best_ang_cd = ang.copy()
            
            t_start = time.time()
            step = 0.05
            min_step = 0.001
            
            while time.time() - t_start < time_limit and step > min_step:
                improved = False
                for idx in range(N):
                    for dim in range(3):
                        for direction in [1, -1]:
                            new_pos = best_pos_cd.copy()
                            new_ang = best_ang_cd.copy()
                            
                            if dim == 0:
                                new_pos[idx, 0] += direction * step
                            elif dim == 1:
                                new_pos[idx, 1] += direction * step
                            else:
                                new_ang[idx] += direction * step
                            
                            angles_rad_new = np.radians(new_ang)
                            valid = True
                            h_moved = (new_pos[idx, 0], new_pos[idx, 1], angles_rad_new[idx])
                            for j in range(N):
                                if j == idx:
                                    continue
                                h_other = (new_pos[j, 0], new_pos[j, 1], angles_rad_new[j])
                                if not hex_disjoint_fast(h_moved, h_other):
                                    valid = False
                                    break
                            
                            if not valid:
                                continue
                            
                            new_side, _, _, _ = compute_min_outer_side(new_pos, new_ang)
                            
                            if new_side < best_side_cd - 1e-10:
                                best_pos_cd = new_pos
                                best_ang_cd = new_ang
                                best_side_cd = new_side
                                improved = True
                                break
                        
                        if time.time() - t_start > time_limit:
                            return best_side_cd, best_pos_cd, best_ang_cd
                    else:
                        continue
                    break
                else:
                    step *= 0.8
                    continue
                break
            
            return best_side_cd, best_pos_cd, best_ang_cd

        cd_time = min(remaining_time * 0.5, 20.0)
        cd_side, cd_pos, cd_ang = polish_cd(best_positions, best_angles, cd_time)
        
        if cd_side < best_side:
            best_side = cd_side
            best_positions = cd_pos
            best_angles = cd_ang
            best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)

    # Final verification
    if not is_valid(best_positions, best_angles):
        best_positions, best_angles = make_honeycomb()
        best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)

    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), float(best_angles[i])])

    center = (float(best_cx), float(best_cy))
    side = float(best_side)
    angle = float(best_phi)

    return inner, center, side, angle
