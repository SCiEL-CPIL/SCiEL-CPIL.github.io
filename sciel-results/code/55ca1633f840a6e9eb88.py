import math
import time
import random
import numpy as np
from scipy.optimize import minimize, minimize_scalar

N = 12
DEADLINE = 145.0  # Safety margin before 150s wall-clock kill

# Precompute unit hexagon vertices and edge normals (angle 0)
_HEX_ANGLES = np.radians(np.arange(0, 360, 60))
_UNIT_HEX_VERTS = np.column_stack((np.cos(_HEX_ANGLES), np.sin(_HEX_ANGLES)))
_EDGE_NORMALS = np.zeros((6, 2))
for _i in range(6):
    _ex = _UNIT_HEX_VERTS[(_i+1)%6, 0] - _UNIT_HEX_VERTS[_i, 0]
    _ey = _UNIT_HEX_VERTS[(_i+1)%6, 1] - _UNIT_HEX_VERTS[_i, 1]
    _nx, _ny = -_ey, _ex
    _norm = math.sqrt(_nx*_nx + _ny*_ny)
    _EDGE_NORMALS[_i] = [_nx / _norm, _ny / _norm]

def construct():
    t0 = time.time()
    
    unit_hex_verts = _UNIT_HEX_VERTS
    edge_normals = _EDGE_NORMALS
    sqrt3 = math.sqrt(3)
    
    _ux = unit_hex_verts[:, 0]
    _uy = unit_hex_verts[:, 1]
    _enx = edge_normals[:, 0]
    _eny = edge_normals[:, 1]
    
    def get_all_vertices(positions, angles_rad):
        c = np.cos(angles_rad)
        s = np.sin(angles_rad)
        rx = _ux[None, :] * c[:, None] - _uy[None, :] * s[:, None]
        ry = _ux[None, :] * s[:, None] + _uy[None, :] * c[:, None]
        verts_x = rx + positions[:, 0:1]
        verts_y = ry + positions[:, 1:2]
        return np.stack((verts_x, verts_y), axis=-1)

    def hexagons_disjoint_fast(h1, h2):
        dx = h1[0] - h2[0]
        dy = h1[1] - h2[1]
        dist_sq = dx*dx + dy*dy
        if dist_sq > 4.5:
            return True
        a1 = h1[2]
        a2 = h2[2]
        c1, s1 = math.cos(a1), math.sin(a1)
        c2, s2 = math.cos(a2), math.sin(a2)
        v1x = h1[0] + _ux * c1 - _uy * s1
        v1y = h1[1] + _ux * s1 + _uy * c1
        v2x = h2[0] + _ux * c2 - _uy * s2
        v2y = h2[1] + _ux * s2 + _uy * c2
        en1x = _enx * c1 - _eny * s1
        en1y = _enx * s1 + _eny * c1
        en2x = _enx * c2 - _eny * s2
        en2y = _enx * s2 + _eny * c2
        for i in range(6):
            nx, ny = en1x[i], en1y[i]
            p1 = v1x * nx + v1y * ny
            p2 = v2x * nx + v2y * ny
            if p1.max() < p2.min() or p2.max() < p1.min():
                return True
        for i in range(6):
            nx, ny = en2x[i], en2y[i]
            p1 = v1x * nx + v1y * ny
            p2 = v2x * nx + v2y * ny
            if p1.max() < p2.min() or p2.max() < p1.min():
                return True
        return False

    def compute_min_outer_side(positions, angles_deg, fine_tune=True):
        angles_rad = np.radians(angles_deg)
        all_verts = get_all_vertices(positions, angles_rad)
        verts_flat = all_verts.reshape(-1, 2)
        center = np.mean(verts_flat, axis=0)
        offsets = verts_flat - center
        
        # Vectorized coarse scan
        phi_vals = np.radians(np.arange(0, 60, 1))
        normal_angles = phi_vals[:, None] + np.pi/6 + np.arange(6) * np.pi/3
        nx = np.cos(normal_angles)
        ny = np.sin(normal_angles)
        proj = offsets[:, 0:1, None] * nx[None, :, :] + offsets[:, 1:2, None] * ny[None, :, :]
        abs_proj = np.abs(proj)
        max_projs = np.max(abs_proj, axis=0)
        side_vals = (2.0 / sqrt3) * max_projs
        max_sides = np.max(side_vals, axis=1)
        
        best_idx = np.argmin(max_sides)
        best_side = float(max_sides[best_idx])
        best_phi = float(best_idx)
        
        if fine_tune:
            def eval_phi(phi_deg):
                phi = np.radians(phi_deg)
                na = phi + np.pi/6 + np.arange(6) * np.pi/3
                nxx = np.cos(na)
                nyy = np.sin(na)
                pr = offsets[:, 0:1] * nxx[None, :] + offsets[:, 1:2] * nyy[None, :]
                return (2.0 / sqrt3) * np.max(np.abs(pr), axis=0).max()
            
            try:
                lo = max(0.0, best_phi - 1.0)
                hi = min(60.0, best_phi + 1.0)
                res = minimize_scalar(eval_phi, bounds=(lo, hi), method='bounded',
                                      options={'xatol': 1e-10})
                if res.success and res.fun < best_side:
                    best_side = res.fun
                    best_phi = res.x
            except Exception:
                pass
        
        return best_side, best_phi, center[0], center[1]

    def smooth_outer_side_surrogate(positions, angles_deg, beta=80.0):
        """Smooth log-sum-exp surrogate for the outer hexagon side length."""
        angles_rad = np.radians(angles_deg)
        all_verts = get_all_vertices(positions, angles_rad)
        verts_flat = all_verts.reshape(-1, 2)
        center = np.mean(verts_flat, axis=0)
        offsets = verts_flat - center
        
        # Sample phi values
        phi_vals = np.radians(np.arange(0, 60, 0.5))
        normal_angles = phi_vals[:, None] + np.pi/6 + np.arange(6) * np.pi/3
        nx = np.cos(normal_angles)
        ny = np.sin(normal_angles)
        proj = offsets[:, None, None, 0] * nx[None, :, :, None] + offsets[:, None, None, 1] * ny[None, :, :, None]
        abs_proj = np.abs(proj)
        
        # Soft-max over vertices
        max_v = abs_proj.max(axis=0, keepdims=True)
        logsumexp_v = max_v.squeeze(0) + np.log(np.sum(np.exp(beta * (abs_proj - max_v)), axis=0)) / beta
        
        # Soft-max over edge directions
        max_e = logsumexp_v.max(axis=1, keepdims=True)
        logsumexp_e = max_e.squeeze(1) + np.log(np.sum(np.exp(beta * (logsumexp_v - max_e)), axis=1)) / beta
        
        # Soft-max over phi
        max_p = logsumexp_e.max(axis=0, keepdims=True)
        logsumexp_phi = max_p.squeeze(0) + np.log(np.sum(np.exp(beta * (logsumexp_e - max_p)), axis=0)) / beta
        
        return (2.0 / sqrt3) * logsumexp_phi

    def is_valid_configuration(positions, angles_deg):
        angles_rad = np.radians(angles_deg)
        for i in range(N):
            for j in range(i+1, N):
                h1 = (positions[i][0], positions[i][1], angles_rad[i])
                h2 = (positions[j][0], positions[j][1], angles_rad[j])
                if not hexagons_disjoint_fast(h1, h2):
                    return False
        return True

    # Initial configurations
    a1 = np.array([sqrt3, 0.0])
    a2 = np.array([sqrt3/2.0, 1.5])
    
    def make_honeycomb(scale=1.0, offset_angle=0.0):
        a1s = a1 * scale
        a2s = a2 * scale
        lattice_coords = [
            (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, -1), (-1, 1), (2, -1), (-2, 1),
            (1, 1), (-1, -1), (2, 0),
        ]
        positions = np.zeros((N, 2))
        angles = np.zeros(N) + offset_angle
        for idx, (i, j) in enumerate(lattice_coords):
            positions[idx] = i * a1s + j * a2s
        return positions, angles
    
    def make_ring_config():
        positions = np.zeros((N, 2))
        angles = np.zeros(N)
        for i in range(N):
            theta = 2 * np.pi * i / N
            r = 2.5
            positions[i, 0] = r * np.cos(theta)
            positions[i, 1] = r * np.sin(theta)
            angles[i] = theta * 180 / np.pi
        return positions, angles
    
    def make_two_rows():
        positions = np.zeros((N, 2))
        angles = np.zeros(N)
        for i in range(6):
            positions[i, 0] = i * sqrt3
            positions[i, 1] = 0
            positions[i+6, 0] = (i + 0.5) * sqrt3
            positions[i+6, 1] = 1.5
        return positions, angles
    
    def make_compact_hex():
        positions = np.zeros((N, 2))
        angles = np.zeros(N)
        coords = [(0,0), (1,0), (-1,0), (0,1), (0,-1),
                  (1,-1), (-1,1), (1,1), (-1,-1), (2,0), (-2,0), (0,2)]
        for idx, (i, j) in enumerate(coords):
            positions[idx] = i * a1 + j * a2
        return positions, angles
    
    initial_configs = [
        make_honeycomb(1.0, 0.0),
        make_honeycomb(1.0, 15.0),
        make_honeycomb(0.95, 0.0),
        make_ring_config(),
        make_two_rows(),
        make_compact_hex(),
    ]
    
    best_side = float('inf')
    best_positions = None
    best_angles = None
    best_phi = 0.0
    best_cx = 0.0
    best_cy = 0.0
    
    # SA phase
    sa_time_budget = DEADLINE * 0.55
    num_restarts = 4
    restart_time = sa_time_budget / num_restarts
    
    for restart in range(num_restarts):
        if time.time() - t0 >= sa_time_budget:
            break
        
        init_pos, init_ang = initial_configs[restart % len(initial_configs)]
        
        current_positions = init_pos.copy()
        current_angles = init_ang.copy()
        current_side, current_phi, current_cx, current_cy = compute_min_outer_side(current_positions, current_angles, fine_tune=False)
        
        if current_side < best_side:
            best_side = current_side
            best_positions = current_positions.copy()
            best_angles = current_angles.copy()
            best_phi = current_phi
            best_cx = current_cx
            best_cy = current_cy

        T = 0.3
        T_min = 0.00005
        cooling = 0.99996
        step_size = 0.2
        step_decay = 0.99999
        
        step_count = 0
        accept_count = 0
        total_count = 0
        
        restart_deadline = t0 + (restart + 1) * restart_time
        
        while time.time() < restart_deadline and time.time() - t0 < sa_time_budget:
            step_count += 1
            total_count += 1
            
            idx = random.randint(0, N - 1)
            new_positions = current_positions.copy()
            new_angles = current_angles.copy()
            
            dx = random.gauss(0, step_size)
            dy = random.gauss(0, step_size)
            da = random.gauss(0, 4.0)
            
            new_positions[idx, 0] += dx
            new_positions[idx, 1] += dy
            new_angles[idx] += da
            
            angles_rad_new = np.radians(new_angles)
            valid = True
            h_moved = (new_positions[idx, 0], new_positions[idx, 1], angles_rad_new[idx])
            for j in range(N):
                if j == idx:
                    continue
                h_other = (new_positions[j, 0], new_positions[j, 1], angles_rad_new[j])
                if not hexagons_disjoint_fast(h_moved, h_other):
                    valid = False
                    break
            
            if not valid:
                T *= cooling
                if T < T_min: T = T_min
                step_size *= step_decay
                if step_size < 0.003: step_size = 0.003
                continue
                
            new_side, new_phi, new_cx, new_cy = compute_min_outer_side(new_positions, new_angles, fine_tune=False)
            
            if new_side < current_side:
                current_positions = new_positions
                current_angles = new_angles
                current_side = new_side
                current_phi = new_phi
                current_cx = new_cx
                current_cy = new_cy
                accept_count += 1
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
                    accept_count += 1
                    if new_side < best_side:
                        best_side = new_side
                        best_positions = new_positions.copy()
                        best_angles = new_angles.copy()
                        best_phi = new_phi
                        best_cx = new_cx
                        best_cy = new_cy
            
            if total_count % 400 == 0:
                acc_rate = accept_count / max(total_count, 1)
                if acc_rate < 0.08:
                    step_size *= 0.85
                    if step_size < 0.003: step_size = 0.003
                elif acc_rate > 0.4:
                    step_size *= 1.15
                    if step_size > 0.8: step_size = 0.8
                accept_count = 0
                total_count = 0
                
            T *= cooling
            if T < T_min: T = T_min
            step_size *= step_decay
            if step_size < 0.003: step_size = 0.003

    # Polish phase with smooth surrogate and L-BFGS-B
    remaining_time = DEADLINE - (time.time() - t0)
    
    if remaining_time > 5.0 and best_positions is not None:
        def polish_with_surrogate(pos, ang, time_limit):
            t_start = time.time()
            
            params = np.zeros(3 * N)
            for i in range(N):
                params[3*i] = pos[i, 0]
                params[3*i+1] = pos[i, 1]
                params[3*i+2] = ang[i]
            
            def objective(p):
                if time.time() - t_start > time_limit:
                    return 1e6
                positions = np.zeros((N, 2))
                angles = np.zeros(N)
                for i in range(N):
                    positions[i, 0] = p[3*i]
                    positions[i, 1] = p[3*i+1]
                    angles[i] = p[3*i+2]
                
                angles_rad = np.radians(angles)
                
                # Check validity
                for i in range(N):
                    for j in range(i+1, N):
                        h1 = (positions[i, 0], positions[i, 1], angles_rad[i])
                        h2 = (positions[j, 0], positions[j, 1], angles_rad[j])
                        if not hexagons_disjoint_fast(h1, h2):
                            return 1e6
                
                return smooth_outer_side_surrogate(positions, angles, beta=100.0)
            
            try:
                res = minimize(objective, params, method='L-BFGS-B',
                              options={'maxiter': 80, 'maxfun': 800, 'gtol': 1e-7,
                                      'maxls': 40})
                if res.fun < best_side - 1e-6:
                    new_params = res.x
                    new_pos = np.zeros((N, 2))
                    new_ang = np.zeros(N)
                    for i in range(N):
                        new_pos[i, 0] = new_params[3*i]
                        new_pos[i, 1] = new_params[3*i+1]
                        new_ang[i] = new_params[3*i+2]
                    return new_pos, new_ang
            except Exception:
                pass
            
            return pos, ang
        
        cd_time = min(remaining_time * 0.7, 20.0)
        new_pos, new_ang = polish_with_surrogate(best_positions, best_angles, cd_time)
        
        # Verify
        angles_rad = np.radians(new_ang)
        valid = True
        for i in range(N):
            for j in range(i+1, N):
                h1 = (new_pos[i, 0], new_pos[i, 1], angles_rad[i])
                h2 = (new_pos[j, 0], new_pos[j, 1], angles_rad[j])
                if not hexagons_disjoint_fast(h1, h2):
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            final_side, final_phi, final_cx, final_cy = compute_min_outer_side(new_pos, new_ang, fine_tune=True)
            if final_side < best_side:
                best_side = final_side
                best_positions = new_pos
                best_angles = new_ang
                best_phi = final_phi
                best_cx = final_cx
                best_cy = final_cy
        else:
            # Fallback: coordinate descent
            def polish_coordinate_descent(pos, ang, time_limit):
                current_side, _, _, _ = compute_min_outer_side(pos, ang, fine_tune=False)
                best_side_cd = current_side
                best_pos_cd = pos.copy()
                best_ang_cd = ang.copy()
                
                t_start = time.time()
                step = 0.02
                min_step = 0.0005
                
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
                                valid2 = True
                                h_moved = (new_pos[idx, 0], new_pos[idx, 1], angles_rad_new[idx])
                                for j in range(N):
                                    if j == idx: continue
                                    h_other = (new_pos[j, 0], new_pos[j, 1], angles_rad_new[j])
                                    if not hexagons_disjoint_fast(h_moved, h_other):
                                        valid2 = False
                                        break
                                
                                if not valid2:
                                    continue
                                
                                new_side, _, _, _ = compute_min_outer_side(new_pos, new_ang, fine_tune=False)
                                
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
                        step *= 0.85
                        continue
                    break
                
                return best_side_cd, best_pos_cd, best_ang_cd
            
            cd_time2 = min(remaining_time * 0.5, 10.0)
            cd_side, cd_pos, cd_ang = polish_coordinate_descent(best_positions, best_angles, cd_time2)
            if cd_side < best_side:
                best_side = cd_side
                best_positions = cd_pos
                best_angles = cd_ang
        
        # Final fine-tune
        best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles, fine_tune=True)

    # Final verification
    if best_positions is None:
        best_positions, best_angles = initial_configs[0]
        best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)
    
    if not is_valid_configuration(best_positions, best_angles):
        best_positions, best_angles = initial_configs[0]
        best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)
    
    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), float(best_angles[i])])
        
    center = (float(best_cx), float(best_cy))
    side = float(best_side)
    angle = float(best_phi)
    
    return inner, center, side, angle
