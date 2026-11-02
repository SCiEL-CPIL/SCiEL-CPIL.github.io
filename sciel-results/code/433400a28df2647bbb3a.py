import math
import time
import random
import numpy as np
from scipy.optimize import minimize, minimize_scalar, differential_evolution

N = 12
DEADLINE = 135.0  # Safety margin before 150s wall-clock kill

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
    
    def get_all_vertices(positions, angles_rad):
        c = np.cos(angles_rad)
        s = np.sin(angles_rad)
        ux = unit_hex_verts[:, 0]
        uy = unit_hex_verts[:, 1]
        rx = ux[None, :] * c[:, None] - uy[None, :] * s[:, None]
        ry = ux[None, :] * s[:, None] + uy[None, :] * c[:, None]
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
        ux = unit_hex_verts[:, 0]
        uy = unit_hex_verts[:, 1]
        v1x = h1[0] + ux * c1 - uy * s1
        v1y = h1[1] + ux * s1 + uy * c1
        v2x = h2[0] + ux * c2 - uy * s2
        v2y = h2[1] + ux * s2 + uy * c2
        en = edge_normals
        en1x = en[:, 0] * c1 - en[:, 1] * s1
        en1y = en[:, 0] * s1 + en[:, 1] * c1
        en2x = en[:, 0] * c2 - en[:, 1] * s2
        en2y = en[:, 0] * s2 + en[:, 1] * c2
        
        eps = 1e-9
        for i in range(6):
            nx, ny = en1x[i], en1y[i]
            p1 = v1x * nx + v1y * ny
            p2 = v2x * nx + v2y * ny
            if p1.max() - eps < p2.min() or p2.max() - eps < p1.min():
                return True
        for i in range(6):
            nx, ny = en2x[i], en2y[i]
            p1 = v1x * nx + v1y * ny
            p2 = v2x * nx + v2y * ny
            if p1.max() - eps < p2.min() or p2.max() - eps < p1.min():
                return True
        return False

    def compute_min_outer_side(positions, angles_deg):
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
            res = minimize_scalar(eval_phi, bounds=(lo, hi), method='bounded',
                                  options={'xatol': 1e-10})
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

    def compute_continuous_overlap_penalty(pos, angles_rad, sharpness=100.0):
        """Compute a smooth overlap penalty using exponential distance-based penalty."""
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                h1 = (pos[i, 0], pos[i, 1], angles_rad[i])
                h2 = (pos[j, 0], pos[j, 1], angles_rad[j])
                dx = h1[0] - h2[0]
                dy = h1[1] - h2[1]
                dist = math.sqrt(dx*dx + dy*dy)
                if dist < 3.0:
                    overlap = max(0.0, 2.0 - dist)
                    penalty += math.exp(sharpness * overlap) / sharpness
        return penalty

    def fitness_function(params, outer_phi_deg=0.0):
        pos = np.zeros((N, 2))
        ang = np.zeros(N)
        for i in range(N):
            pos[i, 0] = params[3*i]
            pos[i, 1] = params[3*i+1]
            ang[i] = params[3*i+2]
        
        angles_rad = np.radians(ang)
        overlap_pen = compute_continuous_overlap_penalty(pos, angles_rad, sharpness=80.0)
        
        if overlap_pen > 1.0:
            return 1e6 + overlap_pen
        
        side, phi, cx, cy = compute_min_outer_side(pos, ang)
        total = side + 100.0 * overlap_pen
        return total

    def generate_initial_configs(num_configs=5):
        configs = []
        
        a1 = np.array([1.5, sqrt3/2.0])
        a2 = np.array([0.0, sqrt3])
        lattice_coords = [
            (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, -1), (-1, 1), (2, -1), (-2, 1),
            (1, 1), (-1, -1), (2, 0),
        ]
        pos = np.zeros((N, 2))
        ang = np.zeros(N)
        for idx, (i, j) in enumerate(lattice_coords):
            pos[idx] = i * a1 + j * a2
            ang[idx] = 0.0
        configs.append((pos, ang))
        
        for trial in range(2):
            pos = np.zeros((N, 2))
            ang = np.zeros(N)
            radius = 2.5 + 0.5 * trial
            for i in range(N):
                theta = 2 * math.pi * i / N
                pos[i, 0] = radius * math.cos(theta)
                pos[i, 1] = radius * math.sin(theta)
                ang[i] = theta * 180.0 / math.pi
            configs.append((pos, ang))
        
        pos = np.zeros((N, 2))
        ang = np.zeros(N)
        inner_count = 6
        outer_count = 6
        for i in range(inner_count):
            theta = 2 * math.pi * i / inner_count
            pos[i, 0] = 1.5 * math.cos(theta)
            pos[i, 1] = 1.5 * math.sin(theta)
            ang[i] = 0.0
        for i in range(outer_count):
            theta = 2 * math.pi * i / outer_count + math.pi / outer_count
            pos[inner_count + i, 0] = 4.0 * math.cos(theta)
            pos[inner_count + i, 1] = 4.0 * math.sin(theta)
            ang[inner_count + i] = 0.0
        configs.append((pos, ang))
        
        pos = np.zeros((N, 2))
        ang = np.zeros(N)
        for i in range(N):
            pos[i, 0] = random.uniform(-4, 4)
            pos[i, 1] = random.uniform(-4, 4)
            ang[i] = random.uniform(0, 60)
        configs.append((pos, ang))
        
        pos = np.zeros((N, 2))
        ang = np.zeros(N)
        row_y = [0, 1.5, 3.0, 4.5]
        row_counts = [4, 4, 3, 1]
        idx = 0
        for r, y in enumerate(row_y):
            count = row_counts[r]
            for c in range(count):
                x = (c - (count-1)/2.0) * 1.8
                pos[idx, 0] = x
                pos[idx, 1] = y
                ang[idx] = 0.0
                idx += 1
        configs.append((pos, ang))
        
        return configs[:num_configs]

    # Dual-Phase Rigid-Body Relaxation Strategy
    best_side = float('inf')
    best_positions = None
    best_angles = None
    best_phi = 0.0
    best_cx = 0.0
    best_cy = 0.0
    
    initial_configs = generate_initial_configs(5)
    
    for pos, ang in initial_configs:
        side, phi, cx, cy = compute_min_outer_side(pos, ang)
        if is_valid_configuration(pos, ang) and side < best_side:
            best_side = side
            best_positions = pos.copy()
            best_angles = ang.copy()
            best_phi = phi
            best_cx = cx
            best_cy = cy
    
    if best_positions is None:
        best_positions = initial_configs[0][0].copy()
        best_angles = initial_configs[0][1].copy()
        best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)
    
    # Phase 1: Differential Evolution on full parameter space
    bounds = [(-5, 5), (-5, 5), (0, 60)] * N
    de_time_budget = DEADLINE * 0.4
    de_start = time.time()
    
    try:
        num_de_restarts = 2
        for restart in range(num_de_restarts):
            if time.time() - t0 > DEADLINE - 30:
                break
            
            init_pop = []
            for pop_i in range(20):
                if pop_i == 0:
                    params = np.zeros(3*N)
                    for i in range(N):
                        params[3*i] = best_positions[i, 0]
                        params[3*i+1] = best_positions[i, 1]
                        params[3*i+2] = best_angles[i]
                elif pop_i < 5:
                    pos, ang = initial_configs[pop_i % len(initial_configs)]
                    params = np.zeros(3*N)
                    for i in range(N):
                        params[3*i] = pos[i, 0]
                        params[3*i+1] = pos[i, 1]
                        params[3*i+2] = ang[i]
                else:
                    params = np.random.uniform(-4, 4, size=3*N)
                    params[2::3] = np.random.uniform(0, 60, size=N)
                init_pop.append(params)
            
            def de_objective(params):
                return fitness_function(params)
            
            try:
                res = differential_evolution(
                    de_objective, 
                    bounds,
                    x0=init_pop,
                    maxiter=300,
                    popsize=15,
                    tol=1e-8,
                    mutation=(0.3, 1.0),
                    recombination=0.8,
                    seed=42 + restart,
                    polish=True,
                    init='latinhypercube' if pop_i != 0 else None
                )
                
                if res.fun < 1e5:
                    new_pos = np.zeros((N, 2))
                    new_ang = np.zeros(N)
                    for i in range(N):
                        new_pos[i, 0] = res.x[3*i]
                        new_pos[i, 1] = res.x[3*i+1]
                        new_ang[i] = res.x[3*i+2]
                    
                    if is_valid_configuration(new_pos, new_ang):
                        cand_side, cand_phi, cand_cx, cand_cy = compute_min_outer_side(new_pos, new_ang)
                        if cand_side < best_side:
                            best_side = cand_side
                            best_positions = new_pos
                            best_angles = new_ang
                            best_phi = cand_phi
                            best_cx = cand_cx
                            best_cy = cand_cy
            except Exception as e:
                pass
            
            if time.time() - t0 > DEADLINE - 35:
                break
        
    except Exception:
        pass
    
    # Phase 2: Rigid-Body Relaxation - optimize global translation/rotation + local adjustments
    polish_time_budget = DEADLINE - (time.time() - t0)
    if polish_time_budget > 10.0 and best_positions is not None:
        try:
            def rigid_body_objective(params):
                # params: [dx, dy, dtheta_deg, ...local adjustments (3*N params)]
                dx = params[0]
                dy = params[1]
                dtheta = np.radians(params[2])
                
                pos = np.zeros((N, 2))
                ang = np.zeros(N)
                for i in range(N):
                    x0 = best_positions[i, 0]
                    y0 = best_positions[i, 1]
                    a0 = np.radians(best_angles[i])
                    
                    # Apply local adjustments
                    lx = x0 + params[3 + 3*i]
                    ly = y0 + params[4 + 3*i]
                    la = a0 + np.radians(params[5 + 3*i])
                    
                    # Apply rigid body transform
                    c, s = math.cos(dtheta), math.sin(dtheta)
                    pos[i, 0] = lx * c - ly * s + dx
                    pos[i, 1] = lx * s + ly * c + dy
                    ang[i] = np.degrees(la + dtheta)
                
                angles_rad = np.radians(ang)
                
                # Check validity
                for i in range(N):
                    for j in range(i+1, N):
                        h1 = (pos[i, 0], pos[i, 1], angles_rad[i])
                        h2 = (pos[j, 0], pos[j, 1], angles_rad[j])
                        if not hexagons_disjoint_fast(h1, h2):
                            overlap = compute_continuous_overlap_penalty(pos, angles_rad, sharpness=100.0)
                            return 1e4 + overlap
                
                side, phi, cx, cy = compute_min_outer_side(pos, ang)
                return side
            
            init_params = np.zeros(3 + 3*N)
            init_params[0] = 0.0
            init_params[1] = 0.0
            init_params[2] = 0.0
            for i in range(N):
                init_params[3 + 3*i] = 0.0
                init_params[4 + 3*i] = 0.0
                init_params[5 + 3*i] = 0.0
            
            res = minimize(rigid_body_objective, init_params, method='Nelder-Mead',
                           options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-10})
            
            if res.fun < 1e3:
                dx = res.x[0]
                dy = res.x[1]
                dtheta = np.radians(res.x[2])
                
                new_pos = np.zeros((N, 2))
                new_ang = np.zeros(N)
                for i in range(N):
                    x0 = best_positions[i, 0]
                    y0 = best_positions[i, 1]
                    a0 = np.radians(best_angles[i])
                    
                    lx = x0 + res.x[3 + 3*i]
                    ly = y0 + res.x[4 + 3*i]
                    la = a0 + np.radians(res.x[5 + 3*i])
                    
                    c, s = math.cos(dtheta), math.sin(dtheta)
                    new_pos[i, 0] = lx * c - ly * s + dx
                    new_pos[i, 1] = lx * s + ly * c + dy
                    new_ang[i] = np.degrees(la + dtheta)
                
                if is_valid_configuration(new_pos, new_ang):
                    cand_side, cand_phi, cand_cx, cand_cy = compute_min_outer_side(new_pos, new_ang)
                    if cand_side < best_side:
                        best_side = cand_side
                        best_positions = new_pos
                        best_angles = new_ang
                        best_phi = cand_phi
                        best_cx = cand_cx
                        best_cy = cand_cy
        except Exception:
            pass
    
    # Final polish with Nelder-Mead on the best configuration
    polish_time_budget = DEADLINE - (time.time() - t0)
    if polish_time_budget > 5.0 and best_positions is not None:
        try:
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
                            overlap = compute_continuous_overlap_penalty(pos, angles_rad, sharpness=100.0)
                            return 1e4 + overlap
                
                side, phi, cx, cy = compute_min_outer_side(pos, ang)
                return side
            
            init_params = np.zeros(3*N)
            for i in range(N):
                init_params[3*i] = best_positions[i, 0]
                init_params[3*i+1] = best_positions[i, 1]
                init_params[3*i+2] = best_angles[i]
            
            res = minimize(polish_objective, init_params, method='Nelder-Mead',
                           options={'maxiter': 3000, 'xatol': 1e-9, 'fatol': 1e-11})
            
            if res.fun < 1e3:
                new_pos = np.zeros((N, 2))
                new_ang = np.zeros(N)
                for i in range(N):
                    new_pos[i, 0] = res.x[3*i]
                    new_pos[i, 1] = res.x[3*i+1]
                    new_ang[i] = res.x[3*i+2]
                
                if is_valid_configuration(new_pos, new_ang):
                    cand_side, cand_phi, cand_cx, cand_cy = compute_min_outer_side(new_pos, new_ang)
                    if cand_side < best_side:
                        best_side = cand_side
                        best_positions = new_pos
                        best_angles = new_ang
                        best_phi = cand_phi
                        best_cx = cand_cx
                        best_cy = cand_cy
        except Exception:
            pass
    
    # Final validation and computation
    if best_positions is None:
        a1 = np.array([1.5 * 1.001, (sqrt3/2.0) * 1.001])
        a2 = np.array([0.0 * 1.001, sqrt3 * 1.001])
        lattice_coords = [
            (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, -1), (-1, 1), (2, -1), (-2, 1),
            (1, 1), (-1, -1), (2, 0),
        ]
        best_positions = np.zeros((N, 2))
        best_angles = np.zeros(N)
        for idx, (i, j) in enumerate(lattice_coords):
            best_positions[idx] = i * a1 + j * a2
    
    final_valid = is_valid_configuration(best_positions, best_angles)
    if not final_valid:
        a1 = np.array([1.5 * 1.001, (sqrt3/2.0) * 1.001])
        a2 = np.array([0.0 * 1.001, sqrt3 * 1.001])
        lattice_coords = [
            (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, -1), (-1, 1), (2, -1), (-2, 1),
            (1, 1), (-1, -1), (2, 0),
        ]
        best_positions = np.zeros((N, 2))
        best_angles = np.zeros(N)
        for idx, (i, j) in enumerate(lattice_coords):
            best_positions[idx] = i * a1 + j * a2
    
    best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)
    
    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), float(best_angles[i])])
    
    center = (float(best_cx), float(best_cy))
    side = float(best_side)
    angle = float(best_phi)
    
    return inner, center, side, angle
