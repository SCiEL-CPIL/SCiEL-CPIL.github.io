import math
import time
import random
import numpy as np
from scipy.optimize import minimize, minimize_scalar

N = 12
DEADLINE = 143.0  # Safety margin before 150s wall-clock kill

# Unit hexagon (side 1) vertices at angle 0 (flat-top: vertices at 0,60,120,...)
_HEX_ANGLES = np.radians(np.arange(0, 360, 60))
_UNIT_HEX_VERTS = np.column_stack((np.cos(_HEX_ANGLES), np.sin(_HEX_ANGLES)))

# Edge normals for unit hexagon (outward)
_EDGE_NORMALS = np.zeros((6, 2))
for _i in range(6):
    _ex = _UNIT_HEX_VERTS[(_i+1)%6, 0] - _UNIT_HEX_VERTS[_i, 0]
    _ey = _UNIT_HEX_VERTS[(_i+1)%6, 1] - _UNIT_HEX_VERTS[_i, 1]
    _nx, _ny = -_ey, _ex
    _norm = math.sqrt(_nx*_nx + _ny*_ny)
    _EDGE_NORMALS[_i] = [_nx / _norm, _ny / _norm]

def construct():
    t0 = time.time()
    sqrt3 = math.sqrt(3)
    
    _ux = _UNIT_HEX_VERTS[:, 0]
    _uy = _UNIT_HEX_VERTS[:, 1]
    _enx = _EDGE_NORMALS[:, 0]
    _eny = _EDGE_NORMALS[:, 1]
    
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
    
    def _convex_hull_2d(points):
        """Compute convex hull of 2D points using monotone chain. Returns hull vertices."""
        n_pts = len(points)
        if n_pts <= 3:
            return points
        pts = sorted(points, key=lambda p: (p[0], p[1]))
        # Build lower hull
        lower = []
        for p in pts:
            while len(lower) >= 2:
                x1, y1 = lower[-1]
                x0, y0 = lower[-2]
                cross = (p[0]-x0)*(y1-y0) - (p[1]-y0)*(x1-x0)
                if cross <= 1e-12:
                    lower.pop()
                else:
                    break
            lower.append(p)
        # Build upper hull
        upper = []
        for p in reversed(pts):
            while len(upper) >= 2:
                x1, y1 = upper[-1]
                x0, y0 = upper[-2]
                cross = (p[0]-x0)*(y1-y0) - (p[1]-y0)*(x1-x0)
                if cross <= 1e-12:
                    upper.pop()
                else:
                    break
            upper.append(p)
        hull = lower[:-1] + upper[:-1]
        return hull
    
    def compute_min_outer_side(positions, angles_deg):
        angles_rad = np.radians(angles_deg)
        all_verts = get_all_vertices(positions, angles_rad)
        verts_flat = all_verts.reshape(-1, 2)
        center = np.mean(verts_flat, axis=0)
        offsets = verts_flat - center
        
        # Use convex hull to reduce vertices for projection
        hull_pts = _convex_hull_2d(list(verts_flat))
        if len(hull_pts) < len(verts_flat):
            hull_arr = np.array(hull_pts)
            hull_offsets = hull_arr - center
        else:
            hull_offsets = offsets
        
        def eval_phi(phi_deg):
            phi = np.radians(phi_deg)
            normal_angles = phi + np.pi/6 + np.arange(6) * np.pi/3
            nx = np.cos(normal_angles)
            ny = np.sin(normal_angles)
            proj = hull_offsets[:, 0:1] * nx[None, :] + hull_offsets[:, 1:2] * ny[None, :]
            abs_proj = np.abs(proj)
            max_projs = np.max(abs_proj, axis=0)
            side_vals = (2.0 / sqrt3) * max_projs
            return np.max(side_vals)
        
        # Coarse scan with finer steps
        best_side = float('inf')
        best_phi = 0.0
        for phi_deg in range(0, 60, 1):
            s = eval_phi(phi_deg)
            if s < best_side:
                best_side = s
                best_phi = phi_deg
        
        # Fine tune
        try:
            lo = max(0.0, best_phi - 1.5)
            hi = min(60.0, best_phi + 1.5)
            res = minimize_scalar(eval_phi, bounds=(lo, hi), method='bounded',
                                  options={'xatol': 1e-12})
            if res.success and res.fun < best_side:
                best_side = res.fun
                best_phi = res.x
        except Exception:
            pass
            
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
    
    # Initial configurations
    def make_honeycomb():
        a1 = np.array([sqrt3, 0.0])
        a2 = np.array([sqrt3/2.0, 1.5])
        lattice_coords = [
            (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, -1), (-1, 1), (2, -1), (-2, 1),
            (1, 1), (-1, -1), (2, 0),
        ]
        positions = np.zeros((N, 2))
        angles = np.zeros(N)
        for idx, (i, j) in enumerate(lattice_coords):
            positions[idx] = i * a1 + j * a2
        return positions, angles
    
    def make_compact_hex():
        positions = np.zeros((N, 2))
        angles = np.zeros(N)
        positions[0] = [0, 0]
        for k in range(6):
            a = k * np.pi / 3
            positions[1+k, 0] = 2*sqrt3 * math.cos(a)
            positions[1+k, 1] = 2*sqrt3 * math.sin(a)
        for k in range(5):
            a = k * 2*np.pi/5 + np.pi/5
            positions[7+k, 0] = 3*sqrt3 * math.cos(a)
            positions[7+k, 1] = 3*sqrt3 * math.sin(a)
        return positions, angles
    
    def make_two_rows():
        positions = np.zeros((N, 2))
        angles = np.zeros(N)
        for k in range(6):
            positions[k, 0] = k * 2*sqrt3
            positions[k, 1] = 0
            positions[k+6, 0] = k * 2*sqrt3 + sqrt3/2
            positions[k+6, 1] = 3.0
        return positions, angles
    
    def make_triangular():
        positions = np.zeros((N, 2))
        angles = np.zeros(N)
        for k in range(3):
            positions[k, 0] = k * 2*sqrt3
            positions[k, 1] = 0
        for k in range(4):
            positions[3+k, 0] = k * 2*sqrt3 + sqrt3/2
            positions[3+k, 1] = 3.0
        for k in range(3):
            positions[7+k, 0] = k * 2*sqrt3
            positions[7+k, 1] = 6.0
        return positions, angles
    
    inits = [make_honeycomb(), make_compact_hex(), make_two_rows(), make_triangular()]
    
    best_side = float('inf')
    best_positions = None
    best_angles = None
    best_phi = 0.0
    best_cx = 0.0
    best_cy = 0.0
    
    for pos_init, ang_init in inits:
        s, phi, cx, cy = compute_min_outer_side(pos_init, ang_init)
        if s < best_side:
            best_side = s
            best_positions = pos_init.copy()
            best_angles = ang_init.copy()
            best_phi = phi
            best_cx = cx
            best_cy = cy
    
    # Phase 1: Smooth nonlinear optimization (L-BFGS-B) with log-sum-exp surrogate
    # This is the structural core: use a smooth approximation of the min-max enclosing
    # hexagon objective so gradient-based methods can navigate the landscape.
    time_for_smooth = DEADLINE * 0.35
    t_phase1 = time.time()
    
    def smooth_objective(params, beta=80.0):
        pos = params[:2*N].reshape(N, 2)
        ang = params[2*N:3*N]
        angles_rad = np.radians(ang)
        
        # Hard validity check via penalty
        for i in range(N):
            for j in range(i+1, N):
                h1 = (pos[i, 0], pos[i, 1], angles_rad[i])
                h2 = (pos[j, 0], pos[j, 1], angles_rad[j])
                if not hexagons_disjoint_fast(h1, h2):
                    return 1e6
        
        all_verts = get_all_vertices(pos, angles_rad)
        verts_flat = all_verts.reshape(-1, 2)
        center = np.mean(verts_flat, axis=0)
        offsets = verts_flat - center
        
        # Use convex hull for efficiency
        hull_pts = _convex_hull_2d(list(verts_flat))
        if len(hull_pts) < len(verts_flat):
            hull_arr = np.array(hull_pts)
            hull_offsets = hull_arr - center
        else:
            hull_offsets = offsets
        
        # Evaluate at multiple phi angles; use log-sum-exp to smooth min over phi
        phi_vals = np.radians(np.arange(0, 60, 1.5))
        normal_angles = phi_vals[:, None] + np.pi/6 + np.arange(6) * np.pi/3
        nx = np.cos(normal_angles)
        ny = np.sin(normal_angles)
        proj = hull_offsets[None, :, 0:1] * nx[:, None, :] + hull_offsets[None, :, 1:2] * ny[:, None, :]
        abs_proj = np.abs(proj)
        max_projs = np.max(abs_proj, axis=1)  # (40, 6)
        
        # Smooth max over 6 edges per phi
        beta_max = 80.0
        # Numerically stable log-sum-exp
        max_m = np.max(max_projs, axis=1, keepdims=True)
        smooth_max = max_m.squeeze() + (1.0/beta_max) * np.log(np.sum(np.exp(beta_max * (max_projs - max_m)), axis=1))
        smooth_side_per_phi = (2.0/sqrt3) * smooth_max
        
        # Smooth min over phi
        beta_min = 80.0
        min_m = np.min(smooth_side_per_phi, keepdims=True)
        smooth_obj = min_m.squeeze() - (1.0/beta_min) * np.log(np.sum(np.exp(-beta_min * (smooth_side_per_phi - min_m)), axis=0))
        
        return smooth_obj
    
    for restart in range(8):
        if time.time() - t0 > t_phase1:
            break
        
        if restart == 0:
            start_pos = best_positions.copy()
            start_ang = best_angles.copy()
        elif restart < 4:
            start_pos = best_positions + np.random.normal(0, 0.4, (N, 2))
            start_ang = best_angles + np.random.normal(0, 4.0, N)
        else:
            # Try from a different initial structure
            pi = (restart - 4) % len(inits)
            start_pos = inits[pi][0].copy()
            start_ang = inits[pi][1].copy()
        
        init_params = np.zeros(3*N)
        for i in range(N):
            init_params[3*i] = start_pos[i, 0]
            init_params[3*i+1] = start_pos[i, 1]
            init_params[3*i+2] = start_ang[i]
        
        try:
            res = minimize(smooth_objective, init_params, method='L-BFGS-B',
                          options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50})
            
            if res.fun < 100:
                new_pos = res.x[:2*N].reshape(N, 2)
                new_ang = res.x[2*N:3*N]
                
                if is_valid_configuration(new_pos, new_ang):
                    new_side, new_phi, new_cx, new_cy = compute_min_outer_side(new_pos, new_ang)
                    if new_side < best_side:
                        best_side = new_side
                        best_positions = new_pos.copy()
                        best_angles = new_ang.copy()
                        best_phi = new_phi
                        best_cx = new_cx
                        best_cy = new_cy
        except Exception:
            pass
    
    # Phase 2: Simulated Annealing (optimized)
    t_phase2 = time.time()
    sa_time_budget = min(DEADLINE * 0.42, DEADLINE - (time.time() - t0) - 5.0)
    
    current_positions = best_positions.copy()
    current_angles = best_angles.copy()
    current_side = best_side
    current_phi = best_phi
    current_cx = best_cx
    current_cy = best_cy
    
    T = 0.25
    T_min = 0.00003
    cooling = 0.99994
    step_size = 0.2
    step_decay = 0.99997
    
    step_count = 0
    max_steps = 300000
    accept_count = 0
    total_count = 0
    no_improve_count = 0
    last_best_side = best_side
    
    while time.time() - t0 < sa_time_budget and step_count < max_steps:
        step_count += 1
        total_count += 1
        
        idx = random.randint(0, N - 1)
        new_positions = current_positions.copy()
        new_angles = current_angles.copy()
        
        dx = random.gauss(0, step_size)
        dy = random.gauss(0, step_size)
        da = random.gauss(0, 3.5)
        
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
            
        new_side, new_phi, new_cx, new_cy = compute_min_outer_side(new_positions, new_angles)
        
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
                no_improve_count = 0
            else:
                no_improve_count += 1
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
                    no_improve_count = 0
                else:
                    no_improve_count += 1
        
        if total_count % 200 == 0:
            acc_rate = accept_count / total_count
            if acc_rate < 0.12:
                step_size *= 0.8
                if step_size < 0.002: step_size = 0.002
            elif acc_rate > 0.50:
                step_size *= 1.2
                if step_size > 0.6: step_size = 0.6
            # If stuck for a while, try a bigger kick
            if no_improve_count > 5000:
                kick_idx = random.randint(0, N - 1)
                current_positions[kick_idx, 0] += random.gauss(0, 0.5)
                current_positions[kick_idx, 1] += random.gauss(0, 0.5)
                current_angles[kick_idx] += random.gauss(0, 8.0)
                no_improve_count = 0
            accept_count = 0
            total_count = 0
            
        T *= cooling
        if T < T_min: T = T_min
        step_size *= step_decay
        if step_size < 0.002: step_size = 0.002
    
    # Phase 3: Coordinate descent polish
    remaining_time = DEADLINE - (time.time() - t0)
    
    if remaining_time > 3.0:
        def polish_coordinate_descent(pos, ang, time_limit):
            current_side_cd, _, _, _ = compute_min_outer_side(pos, ang)
            best_side_cd = current_side_cd
            best_pos_cd = pos.copy()
            best_ang_cd = ang.copy()
            
            t_start = time.time()
            step = 0.015
            min_step = 0.0001
            
            while time.time() - t_start < time_limit and step > min_step:
                improved = False
                for idx in range(N):
                    if time.time() - t_start > time_limit:
                        break
                    for dim in range(3):
                        if time.time() - t_start > time_limit:
                            break
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
                            
                            if new_side < best_side_cd - 1e-11:
                                best_pos_cd = new_pos
                                best_ang_cd = new_ang
                                best_side_cd = new_side
                                improved = True
                                break
                if not improved:
                    step *= 0.65
                else:
                    step *= 0.92
            
            return best_side_cd, best_pos_cd, best_ang_cd
        
        cd_time = min(remaining_time * 0.4, 15.0)
        cd_side, cd_pos, cd_ang = polish_coordinate_descent(best_positions, best_angles, cd_time)
        
        if cd_side < best_side:
            best_side = cd_side
            best_positions = cd_pos.copy()
            best_angles = cd_ang.copy()
            best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)
        
        # Final Nelder-Mead with tighter tolerances
        remaining_time = DEADLINE - (time.time() - t0)
        if remaining_time > 4.0:
            def nm_objective(params):
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
                
            try:
                res = minimize(nm_objective, init_params, method='Nelder-Mead', 
                               options={'maxiter': 8000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
                
                if res.fun < best_side:
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
    
    # Final verification
    if not is_valid_configuration(best_positions, best_angles):
        for pos_f, ang_f in inits:
            if is_valid_configuration(pos_f, ang_f):
                best_positions = pos_f.copy()
                best_angles = ang_f.copy()
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
