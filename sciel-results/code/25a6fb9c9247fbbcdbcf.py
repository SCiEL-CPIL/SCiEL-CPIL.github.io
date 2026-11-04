import math
import time
import random
import numpy as np
from scipy.optimize import minimize, minimize_scalar

N = 12
DEADLINE = 142.0  # Safety margin before 150s wall-clock kill

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
    
    # Precompute flat arrays for speed
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
        # Quick bounding box check
        if abs(dx) > 2.0 or abs(dy) > 2.0:
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
        
        def eval_phi(phi_deg):
            phi = np.radians(phi_deg)
            normal_angles = phi + np.pi/6 + np.arange(6) * np.pi/3
            nx = np.cos(normal_angles)
            ny = np.sin(normal_angles)
            proj = offsets[:, 0:1] * nx[None, :] + offsets[:, 1:2] * ny[None, :]
            abs_proj = np.abs(proj)
            max_projs = np.max(abs_proj, axis=0)
            side_vals = (2.0 / sqrt3) * max_projs
            return np.max(side_vals)
        
        # Vectorized coarse scan: evaluate all 60 angles at once
        phi_vals = np.radians(np.arange(0, 60, 1))
        normal_angles = phi_vals[:, None] + np.pi/6 + np.arange(6) * np.pi/3
        nx = np.cos(normal_angles)
        ny = np.sin(normal_angles)
        proj = offsets[:, None, 0:1] * nx[None, :, :] + offsets[:, None, 1:2] * ny[None, :, :]
        abs_proj = np.abs(proj)
        max_projs = np.max(abs_proj, axis=0)
        side_vals = (2.0 / sqrt3) * max_projs
        max_sides = np.max(side_vals, axis=1)
        
        best_idx = np.argmin(max_sides)
        best_side = float(max_sides[best_idx])
        best_phi = float(best_idx)
        
        if fine_tune:
            # Fine tune with minimize_scalar
            try:
                lo = max(0.0, best_phi - 1.0)
                hi = min(60.0, best_phi + 1.0)
                res = minimize_scalar(eval_phi, bounds=(lo, hi), method='bounded',
                                      options={'xatol': 1e-8})
                if res.success and res.fun < best_side:
                    best_side = res.fun
                    best_phi = res.x
            except Exception:
                pass
            
        return best_side, best_phi, center[0], center[1]

    def smooth_outer_side_surrogate(positions, angles_deg, beta=50.0):
        """Smooth log-sum-exp surrogate for the outer hexagon side length.
        Uses soft-max over vertices and soft-max over the 6 edge directions.
        Higher beta = closer to exact min-max."""
        angles_rad = np.radians(angles_deg)
        all_verts = get_all_vertices(positions, angles_rad)
        verts_flat = all_verts.reshape(-1, 2)
        center = np.mean(verts_flat, axis=0)
        offsets = verts_flat - center  # (72, 2)

        # Sample phi values densely
        phi_vals = np.radians(np.arange(0, 60, 0.5))  # 120 values
        normal_angles = phi_vals[:, None] + np.pi/6 + np.arange(6) * np.pi/3  # (120, 6)
        nx = np.cos(normal_angles)
        ny = np.sin(normal_angles)
        # proj[v, p, e] = offset_v dot normal_{p,e}
        proj = offsets[:, None, None, 0] * nx[None, :, :, None] + offsets[:, None, None, 1] * ny[None, :, :, None]
        abs_proj = np.abs(proj)  # (72, 120, 6)

        # Soft-max over vertices (axis 0)
        beta_v = beta
        max_v = abs_proj.max(axis=0, keepdims=True)
        logsumexp_v = max_v.squeeze(0) + np.log(np.sum(np.exp(beta_v * (abs_proj - max_v)), axis=0)) / beta_v
        # Soft-max over edge directions (axis 1)
        max_e = logsumexp_v.max(axis=1, keepdims=True)
        logsumexp_e = max_e.squeeze(1) + np.log(np.sum(np.exp(beta_v * (logsumexp_v - max_e)), axis=1)) / beta_v
        # Soft-max over phi (axis 0)
        max_p = logsumexp_e.max(axis=0, keepdims=True)
        logsumexp_phi = max_p.squeeze(0) + np.log(np.sum(np.exp(beta_v * (logsumexp_e - max_p)), axis=0)) / beta_v

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

    # Honeycomb initial configuration
    a1 = np.array([sqrt3, 0.0])
    a2 = np.array([sqrt3/2.0, 1.5])
    lattice_coords = [
        (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
        (1, -1), (-1, 1), (2, -1), (-2, 1),
        (1, 1), (-1, -1), (2, 0),
    ]
    base_positions = np.zeros((N, 2))
    base_angles = np.zeros(N)
    for idx, (i, j) in enumerate(lattice_coords):
        base_positions[idx] = i * a1 + j * a2
        base_angles[idx] = 0.0

    best_side = float('inf')
    best_positions = base_positions.copy()
    best_angles = base_angles.copy()
    best_phi = 0.0
    best_cx = 0.0
    best_cy = 0.0

    # Strategy: Run multiple SA restarts with different initial configurations
    sa_time_budget = DEADLINE * 0.65
    
    # Multiple initial configurations for restarts
    def make_honeycomb(scale=1.0, offset_angle=0.0):
        a1 = np.array([sqrt3, 0.0]) * scale
        a2 = np.array([sqrt3/2.0, 1.5]) * scale
        lattice_coords = [
            (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, -1), (-1, 1), (2, -1), (-2, 1),
            (1, 1), (-1, -1), (2, 0),
        ]
        positions = np.zeros((N, 2))
        angles = np.zeros(N) + offset_angle
        for idx, (i, j) in enumerate(lattice_coords):
            positions[idx] = i * a1 + j * a2
        return positions, angles
    
    def make_ring_config():
        # 12 hexagons in a ring pattern
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
        # Two rows of 6
        positions = np.zeros((N, 2))
        angles = np.zeros(N)
        for i in range(6):
            positions[i, 0] = i * sqrt3
            positions[i, 1] = 0
            positions[i+6, 0] = (i + 0.5) * sqrt3
            positions[i+6, 1] = 1.5
        return positions, angles
    
    initial_configs = [
        make_honeycomb(1.0, 0.0),
        make_honeycomb(1.0, 15.0),
        make_honeycomb(0.9, 0.0),
        make_ring_config(),
        make_two_rows(),
    ]
    
    # Run SA with multiple restarts
    num_restarts = 2
    restart_time = sa_time_budget / num_restarts

    for restart in range(num_restarts):
        if time.time() - t0 >= sa_time_budget:
            break
        
        init_pos, init_ang = initial_configs[restart % len(initial_configs)]
        
        current_positions = init_pos.copy()
        current_angles = init_ang.copy()
        current_side, current_phi, current_cx, current_cy = compute_min_outer_side(current_positions, current_angles)
        
        if current_side < best_side:
            best_side = current_side
            best_positions = current_positions.copy()
            best_angles = current_angles.copy()
            best_phi = current_phi
            best_cx = current_cx
            best_cy = current_cy

        T = 0.5
        T_min = 0.0001
        cooling = 0.9999
        step_size = 0.3
        step_decay = 0.99998
        
        step_count = 0
        accept_count = 0
        total_count = 0
        best_in_window = current_side
        
        restart_deadline = t0 + (restart + 1) * restart_time
        
        while time.time() < restart_deadline and time.time() - t0 < sa_time_budget:
            step_count += 1
            total_count += 1
            
            idx = random.randint(0, N - 1)
            new_positions = current_positions.copy()
            new_angles = current_angles.copy()
            
            dx = random.gauss(0, step_size)
            dy = random.gauss(0, step_size)
            da = random.gauss(0, 5.0)
            
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
                if step_size < 0.005: step_size = 0.005
                continue
                
            # Use coarse scan for speed in SA loop
            new_side, new_phi, new_cx, new_cy = compute_min_outer_side(new_positions, new_angles, fine_tune=True)
            
            if new_side < current_side:
                current_positions = new_positions
                current_angles = new_angles
                current_side = new_side
                current_phi = new_phi
                current_cx = new_cx
                current_cy = new_cy
                accept_count += 1
                if new_side < best_in_window:
                    best_in_window = new_side
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
                    if new_side < best_in_window:
                        best_in_window = new_side
                    if new_side < best_side:
                        best_side = new_side
                        best_positions = new_positions.copy()
                        best_angles = new_angles.copy()
                        best_phi = new_phi
                        best_cx = new_cx
                        best_cy = new_cy
            
            # Adaptive step size with longer window
            if total_count % 1000 == 0:
                acc_rate = accept_count / total_count
                if acc_rate < 0.05:
                    step_size *= 0.7
                    if step_size < 0.003: step_size = 0.003
                elif acc_rate > 0.4:
                    step_size *= 1.3
                    if step_size > 1.5: step_size = 1.5
                accept_count = 0
                total_count = 0
                # Reset window best
                best_in_window = current_side
                
            T *= cooling
            if T < T_min: T = T_min
            step_size *= step_decay
            if step_size < 0.003: step_size = 0.003

    # Polish Phase
    remaining_time = DEADLINE - (time.time() - t0)
    
    if remaining_time > 2.0:
        # 1. Coordinate Descent Polish (Fast version)
        def polish_coordinate_descent(pos, ang, time_limit):
            current_side, _, _, _ = compute_min_outer_side(pos, ang, fine_tune=True)
            best_side_cd = current_side
            best_pos_cd = pos.copy()
            best_ang_cd = ang.copy()
            
            t_start = time.time()
            step = 0.1
            min_step = 0.0005
            
            while time.time() - t_start < time_limit and step > min_step:
                improved = False
                for idx in range(N):
                    for dim in range(3):
                        found_improvement = False
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
                                if j == idx: continue
                                h_other = (new_pos[j, 0], new_pos[j, 1], angles_rad_new[j])
                                if not hexagons_disjoint_fast(h_moved, h_other):
                                    valid = False
                                    break
                            
                            if not valid:
                                continue
                            
                            new_side, _, _, _ = compute_min_outer_side(new_pos, new_ang, fine_tune=True)
                            
                            if new_side < best_side_cd - 1e-12:
                                best_pos_cd = new_pos
                                best_ang_cd = new_ang
                                best_side_cd = new_side
                                improved = True
                                found_improvement = True
                                break
                        
                        if found_improvement:
                            break
                        if time.time() - t_start > time_limit:
                            return best_side_cd, best_pos_cd, best_ang_cd
                    else:
                        continue
                    break
                else:
                    step *= 0.75
                    continue
                break
            
            return best_side_cd, best_pos_cd, best_ang_cd

        cd_time = min(remaining_time * 0.4, 15.0)
        cd_side, cd_pos, cd_ang = polish_coordinate_descent(best_positions, best_angles, cd_time)
        
        if cd_side < best_side:
            best_side = cd_side
            best_positions = cd_pos
            best_angles = cd_ang

        # 2. L-BFGS-B Polish with smooth surrogate
        if time.time() - t0 < DEADLINE - 5.0:
            def polish_lbfgs(pos, ang, time_limit):
                # Flatten variables: x, y, angle for each hex, then outer angle
                # Outer angle is handled by compute_min_outer_side, so we just optimize inner
                
                def objective_flat(v):
                    p = v[:2*N].reshape(N, 2)
                    a = v[2*N:3*N]
                    # Check validity
                    angles_rad = np.radians(a)
                    for i in range(N):
                        for j in range(i+1, N):
                            h1 = (p[i][0], p[i][1], angles_rad[i])
                            h2 = (p[j][0], p[j][1], angles_rad[j])
                            if not hexagons_disjoint_fast(h1, h2):
                                return 1e6 # Penalty
                    side, _, _, _ = compute_min_outer_side(p, a, fine_tune=True)
                    return side
                
                def objective_smooth(v):
                    p = v[:2*N].reshape(N, 2)
                    a = v[2*N:3*N]
                    # Check validity
                    angles_rad = np.radians(a)
                    for i in range(N):
                        for j in range(i+1, N):
                            h1 = (p[i][0], p[i][1], angles_rad[i])
                            h2 = (p[j][0], p[j][1], angles_rad[j])
                            if not hexagons_disjoint_fast(h1, h2):
                                return 1e6 # Penalty
                    return smooth_outer_side_surrogate(p, a, beta=80.0)
                
                v0 = np.concatenate([pos.flatten(), ang])
                
                # Bounds: Positions in [-10, 10], Angles in [0, 360]
                bounds = [(-10, 10)] * (2*N) + [(0, 360)] * N
                
                best_result = None
                best_obj = float('inf')
                
                # Try smooth surrogate first with higher beta
                try:
                    res = minimize(objective_smooth, v0, method='L-BFGS-B', bounds=bounds,
                                   options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-10})
                    if res.success and res.fun < 1e5:
                        p_new = res.x[:2*N].reshape(N, 2)
                        a_new = res.x[2*N:3*N]
                        exact_side, _, _, _ = compute_min_outer_side(p_new, a_new, fine_tune=True)
                        if exact_side < best_obj:
                            best_obj = exact_side
                            best_result = (exact_side, p_new, a_new)
                except Exception:
                    pass
                
                # Try exact objective
                try:
                    res = minimize(objective_flat, v0, method='L-BFGS-B', bounds=bounds,
                                   options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-10})
                    if res.success and res.fun < 1e5:
                        p_new = res.x[:2*N].reshape(N, 2)
                        a_new = res.x[2*N:3*N]
                        if res.fun < best_obj:
                            best_obj = res.fun
                            best_result = (res.fun, p_new, a_new)
                except Exception:
                    pass
                
                if best_result is not None:
                    return best_result
                return None, pos, ang

            lbfgs_time = min(remaining_time * 0.35, 20.0)
            lbfgs_side, lbfgs_pos, lbfgs_ang = polish_lbfgs(best_positions, best_angles, lbfgs_time)
            if lbfgs_side is not None and lbfgs_side < best_side:
                best_side = lbfgs_side
                best_positions = lbfgs_pos
                best_angles = lbfgs_ang

        # Final fine-tune of the outer hexagon orientation
        best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles, fine_tune=True)
        
        # Additional polish: try small perturbations of each hexagon
        if time.time() - t0 < DEADLINE - 3.0:
            for idx in range(N):
                for dim in range(3):
                    for delta in [0.001, -0.001, 0.005, -0.005]:
                        new_pos = best_positions.copy()
                        new_ang = best_angles.copy()
                        if dim == 0:
                            new_pos[idx, 0] += delta
                        elif dim == 1:
                            new_pos[idx, 1] += delta
                        else:
                            new_ang[idx] += delta
                        
                        angles_rad_new = np.radians(new_ang)
                        valid = True
                        h_moved = (new_pos[idx, 0], new_pos[idx, 1], angles_rad_new[idx])
                        for j in range(N):
                            if j == idx: continue
                            h_other = (new_pos[j, 0], new_pos[j, 1], angles_rad_new[j])
                            if not hexagons_disjoint_fast(h_moved, h_other):
                                valid = False
                                break
                        
                        if not valid:
                            continue
                        
                        new_side, new_phi, new_cx, new_cy = compute_min_outer_side(new_pos, new_ang, fine_tune=True)
                        if new_side < best_side - 1e-12:
                            best_positions = new_pos
                            best_angles = new_ang
                            best_side = new_side
                            best_phi = new_phi
                            best_cx = new_cx
                            best_cy = new_cy

    # Final verification and safety check
    elapsed = time.time() - t0
    if elapsed > DEADLINE - 2.0:
        # Emergency: return best found so far without expensive verification
        pass
    else:
        if not is_valid_configuration(best_positions, best_angles):
            best_positions = base_positions.copy()
            best_angles = base_angles.copy()
            best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)
    
    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), float(best_angles[i])])
        
    center = (float(best_cx), float(best_cy))
    side = float(best_side)
    angle = float(best_phi)
    
    return inner, center, side, angle
