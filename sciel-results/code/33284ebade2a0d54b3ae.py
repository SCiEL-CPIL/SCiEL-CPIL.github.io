import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 12
DEADLINE = 145.0

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
                                  options={'xatol': 1e-8})
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

    def generate_honeycomb(seed_offset=0):
        a1 = np.array([sqrt3, 0.0])
        a2 = np.array([sqrt3/2.0, 1.5])
        patterns = [
            [(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,-1),(-1,1),(2,-1),(-2,1),(1,1),(-1,-1),(2,0)],
            [(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,-1),(2,0),(-2,0),(1,-1),(-1,1),(0,2)],
            [(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(2,-1),(-2,-1),(1,-1),(-1,-1),(0,2)],
        ]
        idx = (seed_offset) % len(patterns)
        lattice_coords = patterns[idx]
        positions = np.zeros((N, 2))
        angles = np.zeros(N)
        for i, (li, lj) in enumerate(lattice_coords):
            positions[i] = li * a1 + lj * a2
            angles[i] = 0.0
        return positions, angles

    def generate_random_cluster():
        positions = np.zeros((N, 2))
        angles = np.zeros(N)
        positions[0] = [0, 0]
        angles[0] = random.uniform(0, 60)
        
        for i in range(1, N):
            for attempt in range(50):
                ref_idx = random.randint(0, i-1)
                dx = random.uniform(-3.5, 3.5)
                dy = random.uniform(-3.5, 3.5)
                positions[i] = positions[ref_idx] + [dx, dy]
                angles[i] = random.uniform(0, 60)
                
                valid = True
                angles_rad = np.radians(angles[:i+1])
                h_new = (positions[i, 0], positions[i, 1], angles_rad[i])
                for j in range(i):
                    h_other = (positions[j, 0], positions[j, 1], angles_rad[j])
                    if not hexagons_disjoint_fast(h_new, h_other):
                        valid = False
                        break
                if valid:
                    break
            
            if not valid:
                positions[i] = [random.uniform(-5, 5), random.uniform(-5, 5)]
                angles[i] = random.uniform(0, 60)
        
        return positions, angles

    def generate_ring_pattern():
        positions = np.zeros((N, 2))
        angles = np.zeros(N)
        positions[0] = [0, 0]
        angles[0] = 0
        for i in range(1, 7):
            ang = i * 60 * np.pi / 180
            dist = 3.0
            positions[i] = [dist * np.cos(ang), dist * np.sin(ang)]
            angles[i] = 0
        for i in range(7, 12):
            ang = (i - 7) * 72 * np.pi / 180 + 30 * np.pi / 180
            dist = 5.5
            positions[i] = [dist * np.cos(ang), dist * np.sin(ang)]
            angles[i] = 0
        return positions, angles

    def smooth_objective(params, beta_side=50.0, beta_overlap=100.0):
        positions = np.zeros((N, 2))
        angles_deg = np.zeros(N)
        for i in range(N):
            positions[i, 0] = params[3*i]
            positions[i, 1] = params[3*i+1]
            angles_deg[i] = params[3*i+2]
        
        angles_rad = np.radians(angles_deg)
        all_verts = get_all_vertices(positions, angles_rad)
        verts_flat = all_verts.reshape(-1, 2)
        center = np.mean(verts_flat, axis=0)
        offsets = verts_flat - center
        
        # Compute smooth enclosing hexagon side using log-sum-exp
        best_side_smooth = 0.0
        for phi_deg in range(0, 60, 3):  # Coarser scan for speed
            phi = np.radians(phi_deg)
            normal_angles = phi + np.pi/6 + np.arange(6) * np.pi/3
            nx = np.cos(normal_angles)
            ny = np.sin(normal_angles)
            proj = offsets[:, 0:1] * nx[None, :] + offsets[:, 1:2] * ny[None, :]
            abs_proj = np.abs(proj)
            # Log-sum-exp for smooth max
            max_log = np.max(abs_proj, axis=0)
            lse = max_log + np.log(np.sum(np.exp(abs_proj - max_log[:, None]), axis=0) / len(abs_proj))
            side_vals = (2.0 / sqrt3) * lse
            if np.max(side_vals) > best_side_smooth:
                best_side_smooth = np.max(side_vals)
        
        # Compute overlap penalty
        overlap_penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                h1 = (positions[i][0], positions[i][1], angles_rad[i])
                h2 = (positions[j][0], positions[j][1], angles_rad[j])
                if not hexagons_disjoint_fast(h1, h2):
                    overlap_penalty += 1.0
        
        return best_side_smooth + beta_overlap * overlap_penalty

    def run_l_bfgs(positions, angles, time_budget, beta_side=50.0, beta_overlap=100.0):
        t_start = time.time()
        init_params = np.zeros(3*N)
        for i in range(N):
            init_params[3*i] = positions[i, 0]
            init_params[3*i+1] = positions[i, 1]
            init_params[3*i+2] = angles[i]
        
        def objective(params):
            if time.time() - t0 > DEADLINE - 5:
                return 1e6
            return smooth_objective(params, beta_side, beta_overlap)
        
        try:
            res = minimize(objective, init_params, method='L-BFGS-B',
                           options={'maxiter': 2000, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 5000})
            
            if res.fun < 1e5:
                new_pos = np.zeros((N, 2))
                new_ang = np.zeros(N)
                for i in range(N):
                    new_pos[i, 0] = res.x[3*i]
                    new_pos[i, 1] = res.x[3*i+1]
                    new_ang[i] = res.x[3*i+2]
                
                if is_valid_configuration(new_pos, new_ang):
                    side, phi, cx, cy = compute_min_outer_side(new_pos, new_ang)
                    return side, new_pos, new_ang, phi, cx, cy
        except Exception:
            pass
        
        return None

    # Phase 1: Generate diverse initial candidates
    candidates = []
    for offset in range(3):
        pos, ang = generate_honeycomb(offset)
        candidates.append((pos, ang, "honeycomb"))
    
    pos, ang = generate_ring_pattern()
    candidates.append((pos, ang, "ring"))
    
    for seed in range(5):
        random.seed(seed * 12345)
        pos, ang = generate_random_cluster()
        candidates.append((pos, ang, f"random_{seed}"))
    
    # Phase 2: Run L-BFGS on each candidate
    best_side = float('inf')
    best_positions = None
    best_angles = None
    best_phi = 0.0
    best_cx = 0.0
    best_cy = 0.0
    
    remaining_time = DEADLINE - (time.time() - t0) - 15.0
    if remaining_time < 5:
        remaining_time = 5
    
    time_per_candidate = remaining_time / max(1, len(candidates))
    
    for i, (pos, ang, name) in enumerate(candidates):
        if time.time() - t0 > DEADLINE - 15:
            break
        
        sa_time = min(time_per_candidate, 15.0)
        result = run_l_bfgs(pos, ang, sa_time)
        
        if result is not None:
            side, new_pos, new_ang, phi, cx, cy = result
            if side < best_side:
                best_side = side
                best_positions = new_pos
                best_angles = new_ang
                best_phi = phi
                best_cx = cx
                best_cy = cy
    
    # Phase 3: Final polishing with Nelder-Mead
    remaining_time = DEADLINE - (time.time() - t0)
    if remaining_time > 8.0 and best_positions is not None:
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
            
        try:
            res = minimize(polish_objective, init_params, method='Nelder-Mead', 
                           options={'maxiter': 3000, 'xatol': 1e-7, 'fatol': 1e-9})
            
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
    
    # Fallback
    if best_positions is None or not is_valid_configuration(best_positions, best_angles):
        best_positions, best_angles = generate_honeycomb(0)
        best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)
    
    if not is_valid_configuration(best_positions, best_angles):
        best_positions, best_angles = generate_honeycomb(0)
        best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)
    
    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), float(best_angles[i])])
        
    center = (float(best_cx), float(best_cy))
    side = float(best_side)
    angle = float(best_phi)
    
    return inner, center, side, angle
