import math
import time
import random
import numpy as np
from scipy.optimize import minimize, minimize_scalar

N = 12
DEADLINE = 140.0  # Safety margin before 150s wall-clock kill

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

    def fast_side_length(positions, angles_deg):
        angles_rad = np.radians(angles_deg)
        all_verts = get_all_vertices(positions, angles_rad)
        verts_flat = all_verts.reshape(-1, 2)
        center = np.mean(verts_flat, axis=0)
        offsets = verts_flat - center

        phi_vals = np.radians(np.arange(0, 60, 1))
        normal_angles = phi_vals[:, None] + np.pi/6 + np.arange(6) * np.pi/3
        nx = np.cos(normal_angles)
        ny = np.sin(normal_angles)
        proj = offsets[:, None, 0:1] * nx[None, :, :] + offsets[:, None, 1:2] * ny[None, :, :]
        abs_proj = np.abs(proj)
        max_projs = np.max(abs_proj, axis=0)
        side_vals = (2.0 / sqrt3) * max_projs
        side_per_phi = np.max(side_vals, axis=1)

        best_idx = np.argmin(side_per_phi)
        best_side = side_per_phi[best_idx]
        best_phi = float(best_idx)

        return best_side, best_phi, center[0], center[1]

    def compute_min_outer_side(positions, angles_deg):
        best_side, best_phi, cx, cy = fast_side_length(positions, angles_deg)
        
        angles_rad = np.radians(angles_deg)
        all_verts = get_all_vertices(positions, angles_rad)
        verts_flat = all_verts.reshape(-1, 2)
        center = np.mean(verts_flat, axis=0)
        offsets = verts_flat - center

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
                                  options={'xatol': 1e-9})
            if res.success and res.fun < best_side:
                best_side = res.fun
                best_phi = res.x
        except Exception:
            pass

        return best_side, best_phi, cx, cy

    def is_valid_configuration(positions, angles_deg):
        angles_rad = np.radians(angles_deg)
        for i in range(N):
            for j in range(i+1, N):
                h1 = (positions[i][0], positions[i][1], angles_rad[i])
                h2 = (positions[j][0], positions[j][1], angles_rad[j])
                if not hexagons_disjoint_fast(h1, h2):
                    return False
        return True

    a1 = np.array([sqrt3, 0.0])
    a2 = np.array([sqrt3/2.0, 1.5])

    def make_honeycomb():
        lattice_coords = [
            (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, -1), (-1, 1), (2, -1), (-2, 1),
            (1, 1), (-1, -1), (2, 0),
        ]
        pos = np.zeros((N, 2))
        ang = np.zeros(N)
        for idx, (i, j) in enumerate(lattice_coords):
            pos[idx] = i * a1 + j * a2
        return pos, ang

    def make_honeycomb_rotated(seed_rot):
        lattice_coords = [
            (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, -1), (-1, 1), (2, -1), (-2, 1),
            (1, 1), (-1, -1), (2, 0),
        ]
        pos = np.zeros((N, 2))
        ang = np.zeros(N)
        for idx, (i, j) in enumerate(lattice_coords):
            pos[idx] = i * a1 + j * a2
            ang[idx] = seed_rot
        return pos, ang

    def make_ring_config():
        pos = np.zeros((N, 2))
        ang = np.zeros(N)
        for i in range(6):
            theta = i * np.pi / 3
            pos[i] = [2 * sqrt3 * math.cos(theta), 2 * sqrt3 * math.sin(theta)]
        for i in range(6):
            theta = (i + 0.5) * np.pi / 3
            pos[i + 6] = [4 * sqrt3 * math.cos(theta), 4 * sqrt3 * math.sin(theta)]
        return pos, ang

    def make_compact_hex():
        pos = np.zeros((N, 2))
        ang = np.zeros(N)
        pos[0] = [0, 0]
        for i in range(6):
            theta = i * np.pi / 3
            pos[i + 1] = [2 * sqrt3 * math.cos(theta), 2 * sqrt3 * math.sin(theta)]
        for i in range(5):
            theta = (i + 0.5) * np.pi / 3
            pos[i + 7] = [4 * sqrt3 * math.cos(theta), 4 * sqrt3 * math.sin(theta)]
        return pos, ang

    def make_tight_cluster():
        pos = np.zeros((N, 2))
        ang = np.zeros(N)
        pos[0] = [0, 0]
        d1 = 2 * sqrt3
        for i in range(6):
            theta = i * np.pi / 3
            pos[i + 1] = [d1 * math.cos(theta), d1 * math.sin(theta)]
        d2 = 3.5 * sqrt3
        for i in range(5):
            theta = (i + 0.5) * np.pi / 3
            pos[i + 7] = [d2 * math.cos(theta), d2 * math.sin(theta)]
        return pos, ang

    def make_random_cluster():
        pos = np.zeros((N, 2))
        ang = np.zeros(N)
        for i in range(N):
            pos[i] = [random.gauss(0, 2.0), random.gauss(0, 2.0)]
            ang[i] = random.uniform(-30, 30)
        return pos, ang

    init_configs = [make_honeycomb(), make_honeycomb_rotated(5.0), make_honeycomb_rotated(-5.0),
                    make_ring_config(), make_compact_hex(), make_tight_cluster(),
                    make_random_cluster(), make_random_cluster()]

    best_side = float('inf')
    best_positions = None
    best_angles = None
    best_phi = 0.0
    best_cx = 0.0
    best_cy = 0.0

    for pos0, ang0 in init_configs:
        if is_valid_configuration(pos0, ang0):
            s, phi, cx, cy = compute_min_outer_side(pos0, ang0)
            if s < best_side:
                best_side = s
                best_positions = pos0.copy()
                best_angles = ang0.copy()
                best_phi = phi
                best_cx = cx
                best_cy = cy

    if best_positions is None:
        best_positions, best_angles = make_honeycomb()
        best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)

    # SA with multi-restart
    sa_time_budget = DEADLINE * 0.45

    def run_sa(pos_init, ang_init, time_limit, t_start, best_ref):
        current_positions = pos_init.copy()
        current_angles = ang_init.copy()
        current_side, _, _, _ = compute_min_outer_side(current_positions, current_angles)
        local_best_side = current_side
        local_best_pos = current_positions.copy()
        local_best_ang = current_angles.copy()

        T = 0.25
        T_min = 0.0001
        cooling = 0.99995
        step_size = 0.3
        step_min = 0.003
        step_max = 0.8

        step_count = 0
        max_steps = 500000
        accept_count = 0
        total_count = 0
        stall_count = 0
        restart_count = 0

        while time.time() - t_start < time_limit and step_count < max_steps:
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
                step_size *= 0.99999
                if step_size < step_min: step_size = step_min
                continue

            new_side, _, _, _ = fast_side_length(new_positions, new_angles)

            if new_side < current_side:
                current_positions = new_positions
                current_angles = new_angles
                current_side = new_side
                accept_count += 1
                stall_count = 0
                if new_side < local_best_side:
                    local_best_side = new_side
                    local_best_pos = new_positions.copy()
                    local_best_ang = new_angles.copy()
            else:
                delta = new_side - current_side
                if delta < 0: delta = 0
                if random.random() < math.exp(-delta / T):
                    current_positions = new_positions
                    current_angles = new_angles
                    current_side = new_side
                    accept_count += 1
                    stall_count = 0
                    if new_side < local_best_side:
                        local_best_side = new_side
                        local_best_pos = new_positions.copy()
                        local_best_ang = new_angles.copy()
                else:
                    stall_count += 1

            # Adaptive step size
            if total_count % 200 == 0:
                acc_rate = accept_count / max(total_count, 1)
                if acc_rate < 0.05:
                    step_size *= 0.7
                    if step_size < step_min: step_size = step_min
                elif acc_rate > 0.4:
                    step_size *= 1.2
                    if step_size > step_max: step_size = step_max
                accept_count = 0
                total_count = 0

            # Restart if stuck too long
            if stall_count > 2000 and restart_count < 5:
                restart_count += 1
                T = 0.2
                step_size = 0.3
                # Perturb from local best
                current_positions = local_best_pos.copy()
                current_angles = local_best_ang.copy()
                current_side, _, _, _ = compute_min_outer_side(current_positions, current_angles)
                for k in range(N):
                    current_positions[k, 0] += random.gauss(0, 0.5)
                    current_positions[k, 1] += random.gauss(0, 0.5)
                    current_angles[k] += random.gauss(0, 10.0)
                stall_count = 0

            T *= cooling
            if T < T_min: T = T_min
            step_size *= 0.99999
            if step_size < step_min: step_size = step_min

        return local_best_side, local_best_pos, local_best_ang

    # Run SA from the best initial config
    sa_pos, sa_ang = run_sa(best_positions, best_angles, sa_time_budget, t0, best_side)
    if sa_pos is not None:
        sa_side, sa_phi, sa_cx, sa_cy = compute_min_outer_side(sa_pos, sa_ang)
        if sa_side < best_side:
            best_side = sa_side
            best_positions = sa_pos
            best_angles = sa_ang
            best_phi = sa_phi
            best_cx = sa_cx
            best_cy = sa_cy

    # Polish Phase
    remaining_time = DEADLINE - (time.time() - t0)
    
    def polish_coordinate_descent(pos, ang, time_limit, start_step=0.03, min_step=0.0003):
        current_side, _, _, _ = compute_min_outer_side(pos, ang)
        best_side_cd = current_side
        best_pos_cd = pos.copy()
        best_ang_cd = ang.copy()
        
        t_start = time.time()
        step = start_step
        
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
                        
                        new_side, _, _, _ = compute_min_outer_side(new_pos, new_ang)
                        
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
                    step *= 0.65
                    continue
                break
            
            if not improved:
                step *= 0.65
            
        return best_side_cd, best_pos_cd, best_ang_cd

    if remaining_time > 3.0:
        cd_time = min(remaining_time * 0.20, 10.0)
        cd_side, cd_pos, cd_ang = polish_coordinate_descent(best_positions, best_angles, cd_time)
        
        if cd_side < best_side:
            best_side = cd_side
            best_positions = cd_pos
            best_angles = cd_ang
            best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)

    # L-BFGS with analytic gradient
    remaining_time = DEADLINE - (time.time() - t0)
    if remaining_time > 10.0:
        _pair_indices = [(i, j) for i in range(N) for j in range(i+1, N)]
        _n_pairs = len(_pair_indices)
        _pair_i = np.array([p[0] for p in _pair_indices])
        _pair_j = np.array([p[1] for p in _pair_indices])

        # Smooth surrogate with analytic gradient
        # Objective: outer_side_smooth + penalty * soft_collision
        # outer_side_smooth = min over phi of max over (vertex, normal) of |proj|
        # We use a fixed set of phi values and approximate min with soft-min
        
        def smooth_objective_and_grad(params):
            pos = np.zeros((N, 2))
            ang = np.zeros(N)
            for i in range(N):
                pos[i, 0] = params[3*i]
                pos[i, 1] = params[3*i+1]
                ang[i] = params[3*i+2]
            
            angles_rad = np.radians(ang)
            
            # Compute all vertices
            c = np.cos(angles_rad)
            s = np.sin(angles_rad)
            rx = _ux[None, :] * c[:, None] - _uy[None, :] * s[:, None]
            ry = _ux[None, :] * s[:, None] + _uy[None, :] * c[:, None]
            verts_x = rx + pos[:, 0:1]  # (N, 6)
            verts_y = ry + pos[:, 1:2]  # (N, 6)
            
            verts_flat = np.stack((verts_x.reshape(-1), verts_y.reshape(-1)), axis=-1)  # (72, 2)
            center = np.mean(verts_flat, axis=0)
            offsets = verts_flat - center  # (72, 2)
            
            # Use 12 phi values for smooth approximation
            phi_vals = np.radians(np.arange(0, 60, 5))  # 12 values
            normal_angles = phi_vals[:, None] + np.pi/6 + np.arange(6) * np.pi/3  # (12, 6)
            nx = np.cos(normal_angles)  # (12, 6)
            ny = np.sin(normal_angles)  # (12, 6)
            
            # proj[v, phi, face] = offset_v . normal_{phi,face}
            proj = offsets[:, None, 0:1] * nx[None, :, :] + offsets[:, None, 1:2] * ny[None, :, :]  # (72, 12, 6)
            abs_proj = np.abs(proj)  # (72, 12, 6)
            
            # max over vertices for each (phi, face)
            max_projs = np.max(abs_proj, axis=0)  # (12, 6)
            side_vals = (2.0 / sqrt3) * max_projs  # (12, 6)
            
            # soft-min over phi of max over face
            # For each phi: side_phi = max over 6 faces
            side_per_phi = np.max(side_vals, axis=1)  # (12,)
            
            # soft-min over phi
            beta_sm = 50.0
            max_sp = np.max(side_per_phi)
            exp_sp = np.exp(-beta_sm * (side_per_phi - max_sp))
            soft_min = max_sp + np.log(np.sum(exp_sp)) / beta_sm
            
            # Gradient of soft-min w.r.t. side_per_phi
            # d(soft_min)/d(side_per_phi_k) = -exp_sp_k / sum(exp_sp) * beta_sm / beta_sm = -exp_sp_k/sum(exp_sp)
            # Actually: soft_min = max_sp + log(sum(exp(-beta*(sp - max_sp))))/beta
            # d/d(sp_k) = -beta * exp(-beta*(sp_k - max_sp)) / sum(exp(-beta*(sp - max_sp))) * (1/beta) * ... 
            # Let's use: w_k = exp(-beta*(sp_k - max_sp)) / sum_j exp(-beta*(sp_j - max_sp))
            # d(soft_min)/d(sp_k) = -w_k
            w = exp_sp / np.sum(exp_sp)  # (12,)
            
            # Gradient of side_per_phi_k w.r.t. params
            # side_per_phi_k = max over face f of (2/sqrt3) * max over vertex v of |proj[v,k,f]|
            # This is a max of absolute values, so gradient is at the argmax vertex and face
            # For each phi_k, find the face f* and vertex v* that achieves the max
            # d(side_per_phi_k)/d(params) = (2/sqrt3) * sign(proj[v*,k,f*]) * d(proj[v*,k,f*])/d(params)
            
            grad_params = np.zeros_like(params)
            
            for k in range(len(phi_vals)):
                # Find the face and vertex that achieve max for this phi
                face_max = np.argmax(side_vals[k, :])  # face index
                # Find vertex that achieves max abs_proj for this (k, face_max)
                vert_max = np.argmax(abs_proj[:, k, face_max])  # vertex index
                # vertex index: vert_max = hex_idx * 6 + vert_idx
                hex_idx = vert_max // 6
                vert_idx = vert_max % 6
                
                # proj[v, k, f] = offset_v . normal_{k,f}
                # offset_v = vertex_v - center
                # vertex_v = pos[hex_idx] + R(ang[hex_idx]) * unit_vert[vert_idx]
                # center = mean of all vertices
                # d(proj)/d(pos[hex_idx, d]) = normal_{k,f}[d] - (1/72) * sum over all vertices of d(vertex)/d(pos[hex_idx,d])
                # d(vertex_v)/d(pos[hex_idx, d]) = normal_{k,f}[d] if v == hex_idx*6+vert_idx, else 0
                # But center depends on all positions: d(center)/d(pos[i,d]) = (1/72) * sum_v d(vertex_v)/d(pos[i,d])
                # = (1/72) * 6 * 1 = 1/12 for each coordinate (since 6 vertices per hex, 72 total)
                # Actually: d(center_x)/d(pos[i,0]) = (1/72) * sum over 6 vertices of hex i of 1 = 6/72 = 1/12
                # d(offset_v)/d(pos[i,0]) = d(vertex_v)/d(pos[i,0]) - d(center_x)/d(pos[i,0])
                # For v in hex i: d(vertex_v)/d(pos[i,0]) = 1, so d(offset_v)/d(pos[i,0]) = 1 - 1/12 = 11/12
                # For v not in hex i: d(vertex_v)/d(pos[i,0]) = 0, so d(offset_v)/d(pos[i,0]) = -1/12
                
                normal_kf = np.array([nx[k, face_max], ny[k, face_max]])  # (2,)
                
                # d(proj[v*,k,f*])/d(pos[hex_idx, 0]) = normal_kf[0] * d(offset_v*_[0])/d(pos[hex_idx,0])
                # = normal_kf[0] * 11/12
                # d(proj[v*,k,f*])/d(pos[hex_idx, 1]) = normal_kf[1] * 11/12
                # d(proj[v*,k,f*])/d(pos[i, 0]) for i != hex_idx = normal_kf[0] * (-1/12)
                # d(proj[v*,k,f*])/d(pos[i, 1]) for i != hex_idx = normal_kf[1] * (-1/12)
                
                # d(proj)/d(ang[hex_idx]) = normal_kf . d(offset_v*)/d(ang)
                # offset_v* = pos[hex_idx] + R(ang)*unit_vert[vert_idx] - center
                # d(R(ang)*unit_vert)/d(ang) = dR/dang * unit_vert
                # dR/dang = [[-sin, -cos], [cos, -sin]]
                # unit_vert = (cos(vert_angle), sin(vert_angle)) where vert_angle = vert_idx * 60 deg
                # dR/dang * unit_vert = (-sin(ang)*cos(va) - cos(ang)*sin(va), cos(ang)*cos(va) - sin(ang)*sin(va))
                # = (-sin(ang + va), cos(ang + va))  ... wait let me redo
                # R(ang) = [[cos, -sin], [sin, cos]]
                # dR/dang = [[-sin, -cos], [cos, -sin]]
                # dR/dang * [cos(va), sin(va)] = [-sin*cos(va) - cos*sin(va), cos*cos(va) - sin*sin(va)]
                # = [-sin(ang+va), cos(ang+va)]
                va = vert_idx * np.pi / 3  # vertex angle
                dvert_dang = np.array([-math.sin(angles_rad[hex_idx] + va), math.cos(angles_rad[hex_idx] + va)])
                # d(offset_v*)/d(ang) = dvert_dang - d(center)/d(ang)
                # d(center)/d(ang) = (1/72) * sum over all vertices of d(vertex)/d(ang)
                # Only vertices of hex hex_idx depend on ang[hex_idx]
                # sum over 6 vertices of hex hex_idx of dR/dang * unit_vert[v]
                # = sum_{v=0}^{5} [-sin(ang + v*pi/3), cos(ang + v*pi/3)]
                # = [0, 0] by symmetry! So d(center)/d(ang) = 0
                # Thus d(offset_v*)/d(ang) = dvert_dang
                
                dproj_dang = float(np.dot(normal_kf, dvert_dang))
                
                # Now accumulate gradient
                sign_proj = 1.0 if proj[vert_max, k, face_max] >= 0 else -1.0
                coeff = (2.0 / sqrt3) * sign_proj * w[k]
                
                # d(soft_min)/d(params) = sum_k w_k * d(side_per_phi_k)/d(params)
                # d(side_per_phi_k)/d(params) = (2/sqrt3) * sign * dproj/dparams
                # So d(soft_min)/d(params) = sum_k (2/sqrt3) * sign_k * w_k * dproj_k/dparams
                # = sum_k coeff_k * dproj_k/dparams where coeff_k = (2/sqrt3)*sign_k*w_k
                
                # dproj_k/d(pos[hex_idx, 0]) = normal_kf[0] * 11/12
                # dproj_k/d(pos[hex_idx, 1]) = normal_kf[1] * 11/12
                # dproj_k/d(pos[i, 0]) for i != hex_idx = normal_kf[0] * (-1/12)
                # dproj_k/d(pos[i, 1]) for i != hex_idx = normal_kf[1] * (-1/12)
                # dproj_k/d(ang[hex_idx]) = dproj_dang
                
                grad_params[3*hex_idx] += coeff * normal_kf[0] * 11.0/12.0
                grad_params[3*hex_idx+1] += coeff * normal_kf[1] * 11.0/12.0
                grad_params[3*hex_idx+2] += coeff * dproj_dang
                
                for i in range(N):
                    if i == hex_idx:
                        continue
                    grad_params[3*i] += coeff * normal_kf[0] * (-1.0/12.0)
                    grad_params[3*i+1] += coeff * normal_kf[1] * (-1.0/12.0)
            
            # Collision penalty: soft collision using support function
            # For each pair (i, j), compute the minimum separation
            # We use a simple approximation: distance between centers minus 2*sqrt3
            # But this is not accurate for rotated hexagons. Use a softer approximation.
            # For now, use center distance with a soft penalty
            min_dist = 2.0 * sqrt3
            dx = pos[_pair_i, 0] - pos[_pair_j, 0]
            dy = pos[_pair_i, 1] - pos[_pair_j, 1]
            dists = np.sqrt(dx*dx + dy*dy + 1e-12)
            
            overlaps = np.maximum(0.0, min_dist - dists)
            beta_col = 30.0
            scaled = beta_col * overlaps
            max_s = np.max(scaled)
            if max_s > 400:
                lse = max_s / beta_col + np.log(np.sum(np.exp(scaled - max_s))) / beta_col
            else:
                lse = np.log(np.sum(np.exp(scaled))) / beta_col
            penalty = 5.0 * lse
            
            # Gradient of penalty
            grad_penalty = np.zeros_like(params)
            for idx_pair in range(_n_pairs):
                if overlaps[idx_pair] <= 0:
                    continue
                i_idx = _pair_i[idx_pair]
                j_idx = _pair_j[idx_pair]
                # d(overlap)/d(pos[i,0]) = -dx/dists (for the min_dist - dist term)
                # d(lse)/d(overlap_k) = exp(scaled_k)/sum(exp(scaled)) * beta_col
                # d(penalty)/d(pos[i,0]) = 5 * beta_col * exp(scaled_k)/sum(exp(scaled)) * d(overlap_k)/d(pos[i,0])
                # = 5 * beta_col * exp(scaled_k)/sum(exp(scaled)) * (-dx_k/dists_k)
                
                exp_s_k = np.exp(scaled[idx_pair] - max_s) if max_s > 400 else np.exp(scaled[idx_pair])
                sum_exp = np.sum(np.exp(scaled - max_s)) if max_s > 400 else np.sum(np.exp(scaled))
                w_col = 5.0 * beta_col * exp_s_k / sum_exp
                
                grad_penalty[3*i_idx] += w_col * (-dx[idx_pair] / dists[idx_pair])
                grad_penalty[3*i_idx+1] += w_col * (-dy[idx_pair] / dists[idx_pair])
                grad_penalty[3*j_idx] += w_col * (dx[idx_pair] / dists[idx_pair])
                grad_penalty[3*j_idx+1] += w_col * (dy[idx_pair] / dists[idx_pair])
            
            total_obj = soft_min + penalty
            total_grad = grad_params + grad_penalty
            
            return total_obj, total_grad

        def smooth_obj_only(params):
            return smooth_objective_and_grad(params)[0]

        def smooth_grad_only(params):
            return smooth_objective_and_grad(params)[1]

        init_params = np.zeros(3*N)
        for i in range(N):
            init_params[3*i] = best_positions[i, 0]
            init_params[3*i+1] = best_positions[i, 1]
            init_params[3*i+2] = best_angles[i]
            
        try:
            res = minimize(smooth_obj_only, init_params, method='L-BFGS-B',
                           jac=smooth_grad_only,
                           options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 15000})
            
            if res.fun < best_side + 2.0 and res.fun < 100:
                new_pos = np.zeros((N, 2))
                new_ang = np.zeros(N)
                for i in range(N):
                    new_pos[i, 0] = res.x[3*i]
                    new_pos[i, 1] = res.x[3*i+1]
                    new_ang[i] = res.x[3*i+2]
                
                if is_valid_configuration(new_pos, new_ang):
                    new_side, new_phi, new_cx, new_cy = compute_min_outer_side(new_pos, new_ang)
                    if new_side < best_side:
                        best_side = new_side
                        best_positions = new_pos
                        best_angles = new_ang
                        best_phi = new_phi
                        best_cx = new_cx
                        best_cy = new_cy
        except Exception:
            pass

    # Final tight CD polish
    remaining_time = DEADLINE - (time.time() - t0)
    if remaining_time > 2.0:
        cd_time2 = min(remaining_time * 0.8, 12.0)
        cd_side2, cd_pos2, cd_ang2 = polish_coordinate_descent(best_positions, best_angles, cd_time2, start_step=0.01, min_step=0.0001)
        if cd_side2 < best_side:
            best_side = cd_side2
            best_positions = cd_pos2
            best_angles = cd_ang2
            best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)

    # Final verification
    if not is_valid_configuration(best_positions, best_angles):
        for pos0, ang0 in init_configs:
            if is_valid_configuration(pos0, ang0):
                best_positions = pos0.copy()
                best_angles = ang0.copy()
                best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)
                break
        else:
            best_positions, best_angles = make_honeycomb()
            best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)

    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), float(best_angles[i])])
        
    center = (float(best_cx), float(best_cy))
    side = float(best_side)
    angle = float(best_phi)
    
    return inner, center, side, angle
