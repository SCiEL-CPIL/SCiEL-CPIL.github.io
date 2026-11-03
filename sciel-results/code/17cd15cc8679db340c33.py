import math
import time
import random
import numpy as np
from scipy.optimize import minimize, minimize_scalar

N = 12
DEADLINE = 115.0  # Safety margin before 120s wall-clock kill

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

    def compute_min_outer_side_fast(positions, angles_deg):
        """Fast computation of minimum outer hexagon side length."""
        angles_rad = np.radians(angles_deg)
        all_verts = get_all_vertices(positions, angles_rad)
        verts_flat = all_verts.reshape(-1, 2)
        center = np.mean(verts_flat, axis=0)
        offsets = verts_flat - center
        
        # For a regular hexagon with side s, the apothem (center to edge midpoint) is s*sqrt(3)/2
        # The distance from center to each vertex is s
        # If we project all points onto 6 directions (edge normals of the outer hexagon),
        # the max absolute projection in direction i gives the apothem in that direction.
        # For a regular hexagon, all 6 apothems are equal = s*sqrt(3)/2
        # So s = 2 * max_abs_proj / sqrt(3)
        
        # We need to find the rotation phi that minimizes the maximum of the 6 apothems
        # The outer hexagon has edge normals at angles: phi + pi/6 + k*pi/3 for k=0..5
        
        def eval_phi(phi_rad):
            normal_angles = phi_rad + np.pi/6 + np.arange(6) * np.pi/3
            nx = np.cos(normal_angles)
            ny = np.sin(normal_angles)
            proj = offsets[:, 0:1] * nx[None, :] + offsets[:, 1:2] * ny[None, :]
            abs_proj = np.abs(proj)
            max_projs = np.max(abs_proj, axis=0)
            side_vals = (2.0 / sqrt3) * max_projs
            return np.max(side_vals)
        
        # Coarse scan
        best_side = float('inf')
        best_phi = 0.0
        for phi_deg in range(0, 60, 2):
            s = eval_phi(np.radians(phi_deg))
            if s < best_side:
                best_side = s
                best_phi = phi_deg
        
        # Fine tune
        lo = max(0.0, best_phi - 2.0)
        hi = min(60.0, best_phi + 2.0)
        for _ in range(3):
            mid = (lo + hi) / 2
            s_lo = eval_phi(np.radians(lo))
            s_mid = eval_phi(np.radians(mid))
            s_hi = eval_phi(np.radians(hi))
            if s_mid <= s_lo and s_mid <= s_hi:
                lo, hi = lo, mid
            elif s_lo <= s_mid and s_lo <= s_hi:
                hi = mid
            else:
                lo = mid
        
        best_phi = (lo + hi) / 2
        best_side = eval_phi(np.radians(best_phi))
        
        return best_side, best_phi, center[0], center[1]

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

    # Compute initial side
    init_side, init_phi, init_cx, init_cy = compute_min_outer_side_fast(base_positions, base_angles)
    best_side = init_side
    best_phi = init_phi
    best_cx = init_cx
    best_cy = init_cy

    # Multi-start SA with different initial configurations
    def run_sa(initial_positions, initial_angles, time_budget, seed=0):
        """Run simulated annealing from a given initial configuration."""
        rng = random.Random(seed)
        
        current_positions = initial_positions.copy()
        current_angles = initial_angles.copy()
        current_side, current_phi, current_cx, current_cy = compute_min_outer_side_fast(current_positions, current_angles)
        
        local_best_side = current_side
        local_best_positions = current_positions.copy()
        local_best_angles = current_angles.copy()
        local_best_phi = current_phi
        local_best_cx = current_cx
        local_best_cy = current_cy
        
        T = 0.2
        T_min = 0.0001
        cooling = 0.99997
        step_size = 0.15
        step_decay = 0.99999
        
        step_count = 0
        max_steps = 200000
        accept_count = 0
        total_count = 0
        
        t_start = time.time()
        
        while time.time() - t_start < time_budget and step_count < max_steps:
            step_count += 1
            total_count += 1
            
            idx = rng.randint(0, N - 1)
            new_positions = current_positions.copy()
            new_angles = current_angles.copy()
            
            # Sometimes move two hexagons together for better exploration
            if rng.random() < 0.3 and N > 1:
                idx2 = rng.randint(0, N - 1)
                if idx2 == idx:
                    idx2 = (idx2 + 1) % N
                dx2 = rng.gauss(0, step_size)
                dy2 = rng.gauss(0, step_size)
                da2 = rng.gauss(0, 2.0)
                new_positions[idx2, 0] += dx2
                new_positions[idx2, 1] += dy2
                new_angles[idx2] += da2
            
            dx = rng.gauss(0, step_size)
            dy = rng.gauss(0, step_size)
            da = rng.gauss(0, 2.5)
            
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
                if step_size < 0.002: step_size = 0.002
                continue
                
            new_side, new_phi, new_cx, new_cy = compute_min_outer_side_fast(new_positions, new_angles)
            
            if new_side < current_side:
                current_positions = new_positions
                current_angles = new_angles
                current_side = new_side
                current_phi = new_phi
                current_cx = new_cx
                current_cy = new_cy
                accept_count += 1
                if new_side < local_best_side:
                    local_best_side = new_side
                    local_best_positions = new_positions.copy()
                    local_best_angles = new_angles.copy()
                    local_best_phi = new_phi
                    local_best_cx = new_cx
                    local_best_cy = new_cy
            else:
                delta = new_side - current_side
                if delta < 0: delta = 0
                if rng.random() < math.exp(-delta / T):
                    current_positions = new_positions
                    current_angles = new_angles
                    current_side = new_side
                    current_phi = new_phi
                    current_cx = new_cx
                    current_cy = new_cy
                    accept_count += 1
                    if new_side < local_best_side:
                        local_best_side = new_side
                        local_best_positions = new_positions.copy()
                        local_best_angles = new_angles.copy()
                        local_best_phi = new_phi
                        local_best_cx = new_cx
                        local_best_cy = new_cy
            
            # Adaptive step size
            if total_count % 200 == 0:
                acc_rate = accept_count / max(total_count, 1)
                if acc_rate < 0.03:
                    step_size *= 0.7
                    if step_size < 0.002: step_size = 0.002
                elif acc_rate > 0.35:
                    step_size *= 1.2
                    if step_size > 0.6: step_size = 0.6
                accept_count = 0
                total_count = 0
                
            T *= cooling
            if T < T_min: T = T_min
            step_size *= step_decay
            if step_size < 0.002: step_size = 0.002

        return local_best_side, local_best_positions, local_best_angles, local_best_phi, local_best_cx, local_best_cy

    # Try multiple SA runs with different seeds and initial perturbations
    sa_time_per_run = 25.0
    num_runs = 4
    
    for run_idx in range(num_runs):
        if time.time() - t0 > DEADLINE - 10.0:
            break
        
        # Perturb the initial configuration for diversity
        if run_idx == 0:
            init_pos = best_positions.copy()
            init_ang = best_angles.copy()
        else:
            init_pos = base_positions.copy()
            init_ang = base_angles.copy()
            # Add random perturbation
            for i in range(N):
                init_pos[i, 0] += random.gauss(0, 0.5)
                init_pos[i, 1] += random.gauss(0, 0.5)
                init_ang[i] += random.gauss(0, 10.0)
            # Check validity, if invalid, fall back to base
            if not is_valid_configuration(init_pos, init_ang):
                init_pos = base_positions.copy()
                init_ang = base_angles.copy()
        
        sa_side, sa_pos, sa_ang, sa_phi, sa_cx, sa_cy = run_sa(init_pos, init_ang, sa_time_per_run, seed=run_idx*100)
        
        if sa_side < best_side:
            best_side = sa_side
            best_positions = sa_pos
            best_angles = sa_ang
            best_phi = sa_phi
            best_cx = sa_cx
            best_cy = sa_cy

    # Polish Phase
    remaining_time = DEADLINE - (time.time() - t0)
    
    if remaining_time > 3.0:
        # Coordinate Descent Polish
        def polish_coordinate_descent(pos, ang, time_limit):
            current_side_cd, _, _, _ = compute_min_outer_side_fast(pos, ang)
            best_side_cd = current_side_cd
            best_pos_cd = pos.copy()
            best_ang_cd = ang.copy()
            
            t_start = time.time()
            step = 0.02
            min_step = 0.0003
            
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
                                if j == idx: continue
                                h_other = (new_pos[j, 0], new_pos[j, 1], angles_rad_new[j])
                                if not hexagons_disjoint_fast(h_moved, h_other):
                                    valid = False
                                    break
                            
                            if not valid:
                                continue
                            
                            new_side, _, _, _ = compute_min_outer_side_fast(new_pos, new_ang)
                            
                            if new_side < best_side_cd - 1e-12:
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
                    step *= 0.7
                    continue
                break
            
            return best_side_cd, best_pos_cd, best_ang_cd

        cd_time = min(remaining_time * 0.6, 15.0)
        cd_side, cd_pos, cd_ang = polish_coordinate_descent(best_positions, best_angles, cd_time)
        
        if cd_side < best_side:
            best_side = cd_side
            best_positions = cd_pos
            best_angles = cd_ang
            best_side, best_phi, best_cx, best_cy = compute_min_outer_side_fast(best_positions, best_angles)
        
        # Nelder-Mead Polish
        remaining_time = DEADLINE - (time.time() - t0)
        if remaining_time > 5.0:
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
                
                side, phi, cx, cy = compute_min_outer_side_fast(pos, ang)
                return side

            init_params = np.zeros(3*N)
            for i in range(N):
                init_params[3*i] = best_positions[i, 0]
                init_params[3*i+1] = best_positions[i, 1]
                init_params[3*i+2] = best_angles[i]
                
            try:
                res = minimize(polish_objective, init_params, method='Nelder-Mead', 
                               options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10})
                
                if res.fun < best_side:
                    new_pos = np.zeros((N, 2))
                    new_ang = np.zeros(N)
                    for i in range(N):
                        new_pos[i, 0] = res.x[3*i]
                        new_pos[i, 1] = res.x[3*i+1]
                        new_pos[i, 2] = res.x[3*i+2]
                    
                    if is_valid_configuration(new_pos, new_ang):
                        new_side, new_phi, new_cx, new_cy = compute_min_outer_side_fast(new_pos, new_ang)
                        if new_side < best_side:
                            best_side = new_side
                            best_positions = new_pos
                            best_angles = new_ang
                            best_phi = new_phi
                            best_cx = new_cx
                            best_cy = new_cy
            except Exception:
                pass

    # Final verification
    if not is_valid_configuration(best_positions, best_angles):
        best_positions = base_positions.copy()
        best_angles = base_angles.copy()
        best_side, best_phi, best_cx, best_cy = compute_min_outer_side_fast(base_positions, base_angles)
    
    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), float(best_angles[i])])
        
    center = (float(best_cx), float(best_cy))
    side = float(best_side)
    angle = float(best_phi)
    
    return inner, center, side, angle
