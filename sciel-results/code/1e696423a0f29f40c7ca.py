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
        
        # Vectorized coarse scan: evaluate all 60 angles at once
        phi_vals = np.radians(np.arange(0, 60, 1))
        normal_angles = phi_vals[:, None] + np.pi/6 + np.arange(6) * np.pi/3
        nx = np.cos(normal_angles)  # (60, 6)
        ny = np.sin(normal_angles)  # (60, 6)
        proj = offsets[:, 0:1, None] * nx[None, :, :] + offsets[:, 1:2, None] * ny[None, :, :]
        abs_proj = np.abs(proj)
        max_projs = np.max(abs_proj, axis=0)  # (60, 6)
        side_vals = (2.0 / sqrt3) * max_projs  # (60, 6)
        best_per_angle = np.max(side_vals, axis=1)  # (60,)
        best_idx = np.argmin(best_per_angle)
        best_side = best_per_angle[best_idx]
        best_phi = float(best_idx)
        
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

    def is_valid_configuration(positions, angles_deg):
        angles_rad = np.radians(angles_deg)
        for i in range(N):
            for j in range(i+1, N):
                h1 = (positions[i][0], positions[i][1], angles_rad[i])
                h2 = (positions[j][0], positions[j][1], angles_rad[j])
                if not hexagons_disjoint_fast(h1, h2):
                    return False
        return True

    # Smooth Minkowski sum collision model
    # For two hexagons with relative angle delta, the Minkowski sum is a 12-gon
    # We compute the minimum distance for separation using support functions
    
    def smooth_collision_penalty(params, beta=50.0):
        """
        params: (3*N,) array with [x1, y1, a1, x2, y2, a2, ...]
        Returns smooth penalty for collisions using log-sum-exp approximation
        """
        pos = params.reshape(N, 3)
        penalty = 0.0
        
        for i in range(N):
            for j in range(i+1, N):
                xi, yi, ai = pos[i]
                xj, yj, aj = pos[j]
                
                dx = xi - xj
                dy = yi - yj
                dist = math.sqrt(dx*dx + dy*dy)
                
                # If far apart, no collision
                if dist > 2.5:
                    continue
                
                # Compute relative angle
                delta_a = ai - aj
                
                # For hexagons, the critical distances occur at specific angles
                # Use a smooth approximation of the collision constraint
                # The minimum distance for non-overlap depends on relative orientation
                
                # Support function approach: check all 12 edge normals
                # (6 from hex i, 6 from hex j rotated by delta_a)
                c_ai, s_ai = math.cos(ai), math.sin(ai)
                c_aj, s_aj = math.cos(aj), math.sin(aj)
                
                # Edge normals for hex i
                en1x = _enx * c_ai - _eny * s_ai
                en1y = _enx * s_ai + _eny * c_ai
                
                # Edge normals for hex j
                en2x = _enx * c_aj - _eny * s_aj
                en2y = _enx * s_aj + _eny * c_aj
                
                # Vertices
                v1x = xi + _ux * c_ai - _uy * s_ai
                v1y = yi + _ux * s_ai + _uy * c_ai
                v2x = xj + _ux * c_aj - _uy * s_aj
                v2y = yj + _ux * s_aj + _uy * c_aj
                
                # Check all 12 separating axes
                violations = []
                for k in range(6):
                    nx, ny = en1x[k], en1y[k]
                    p1 = v1x * nx + v1y * ny
                    p2 = v2x * nx + v2y * ny
                    # Violation if projections overlap
                    overlap = min(p1.max(), p2.max()) - max(p1.min(), p2.min())
                    if overlap > 0:
                        violations.append(overlap)
                
                for k in range(6):
                    nx, ny = en2x[k], en2y[k]
                    p1 = v1x * nx + v1y * ny
                    p2 = v2x * nx + v2y * ny
                    overlap = min(p1.max(), p2.max()) - max(p1.min(), p2.min())
                    if overlap > 0:
                        violations.append(overlap)
                
                if violations:
                    # Soft max using log-sum-exp for smoothness
                    max_v = max(violations)
                    if max_v > 1e-10:
                        # Smooth approximation
                        log_sum = math.log(sum(math.exp((v - max_v) * beta) for v in violations)) / beta
                        penalty += max_v + log_sum
        
        return penalty

    def smooth_objective(params, beta=50.0):
        """
        Full smooth objective: outer side length + collision penalty
        """
        pos = params.reshape(N, 3)
        positions = pos[:, :2]
        angles_deg = pos[:, 2] * 180.0 / math.pi
        
        # Compute outer side length
        side, _, _, _ = compute_min_outer_side(positions, angles_deg)
        
        # Add collision penalty
        penalty = smooth_collision_penalty(params, beta)
        
        return side + penalty * 10.0

    def smooth_gradient(params, beta=50.0):
        """
        Numerical gradient of smooth objective
        """
        eps = 1e-6
        grad = np.zeros_like(params)
        f0 = smooth_objective(params, beta)
        
        for i in range(len(params)):
            params[i] += eps
            f1 = smooth_objective(params, beta)
            params[i] -= eps
            grad[i] = (f1 - f0) / (2 * eps)
        
        return grad

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

    # Phase 1: L-BFGS-B optimization with smooth objective from multiple starts
    lbfgs_time_budget = DEADLINE * 0.5
    
    # Try multiple random initializations
    num_starts = 8
    start_time = time.time()
    time_per_start = lbfgs_time_budget / num_starts
    
    for start_idx in range(num_starts):
        if time.time() - t0 > lbfgs_time_budget:
            break
        
        # Create initial parameters
        if start_idx == 0:
            # Use honeycomb as first start
            init_params = np.zeros(3*N)
            for i in range(N):
                init_params[3*i] = base_positions[i, 0]
                init_params[3*i+1] = base_positions[i, 1]
                init_params[3*i+2] = 0.0  # radians
        else:
            # Perturbed honeycomb
            init_params = np.zeros(3*N)
            for i in range(N):
                init_params[3*i] = base_positions[i, 0] + random.gauss(0, 0.5)
                init_params[3*i+1] = base_positions[i, 1] + random.gauss(0, 0.5)
                init_params[3*i+2] = random.gauss(0, 0.1)  # radians
        
        # Run L-BFGS-B
        try:
            res = minimize(
                smooth_objective,
                init_params,
                method='L-BFGS-B',
                jac=smooth_gradient,
                options={'maxiter': 2000, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 50000}
            )
            
            if res.fun < 1e5:  # Valid result
                new_pos = res.x.reshape(N, 3)
                new_positions = new_pos[:, :2]
                new_angles_deg = new_pos[:, 2] * 180.0 / math.pi
                
                # Check validity
                if is_valid_configuration(new_positions, new_angles_deg):
                    new_side, new_phi, new_cx, new_cy = compute_min_outer_side(new_positions, new_angles_deg)
                    if new_side < best_side:
                        best_side = new_side
                        best_positions = new_positions.copy()
                        best_angles = new_angles_deg.copy()
                        best_phi = new_phi
                        best_cx = new_cx
                        best_cy = new_cy
        except Exception:
            pass
    
    # Phase 2: SA refinement on the best configuration found
    sa_time_budget = DEADLINE * 0.35
    
    current_positions = best_positions.copy()
    current_angles = best_angles.copy()
    current_side, current_phi, current_cx, current_cy = compute_min_outer_side(current_positions, current_angles)
    
    T = 0.3
    T_min = 0.0001
    cooling = 0.99995
    step_size = 0.2
    step_decay = 0.99999
    
    step_count = 0
    max_steps = 150000
    accept_count = 0
    total_count = 0
    
    while time.time() - t0 < sa_time_budget and step_count < max_steps:
        step_count += 1
        total_count += 1
        
        idx = random.randint(0, N - 1)
        new_positions = current_positions.copy()
        new_angles = current_angles.copy()
        
        dx = random.gauss(0, step_size)
        dy = random.gauss(0, step_size)
        da = random.gauss(0, 3.0)
        
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
        
        # Adaptive step size
        if total_count % 500 == 0:
            acc_rate = accept_count / total_count
            if acc_rate < 0.1:
                step_size *= 0.8
                if step_size < 0.005: step_size = 0.005
            elif acc_rate > 0.5:
                step_size *= 1.2
                if step_size > 0.8: step_size = 0.8
            accept_count = 0
            total_count = 0
            
        T *= cooling
        if T < T_min: T = T_min
        step_size *= step_decay
        if step_size < 0.005: step_size = 0.005

    # Phase 3: Polish
    remaining_time = DEADLINE - (time.time() - t0)
    
    if remaining_time > 2.0:
        # Coordinate Descent Polish
        def polish_coordinate_descent(pos, ang, time_limit):
            current_side, _, _, _ = compute_min_outer_side(pos, ang)
            best_side_cd = current_side
            best_pos_cd = pos.copy()
            best_ang_cd = ang.copy()
            
            t_start = time.time()
            step = 0.03
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
                    step *= 0.7
                    continue
                break
            
            return best_side_cd, best_pos_cd, best_ang_cd

        cd_time = min(remaining_time * 0.5, 15.0)
        cd_side, cd_pos, cd_ang = polish_coordinate_descent(best_positions, best_angles, cd_time)
        
        if cd_side < best_side:
            best_side = cd_side
            best_positions = cd_pos
            best_angles = cd_ang
            best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)
        
        # Final L-BFGS-B polish with tighter tolerances
        remaining_time = DEADLINE - (time.time() - t0)
        if remaining_time > 5.0:
            def polish_objective(params, beta=100.0):
                pos = params.reshape(N, 3)
                positions = pos[:, :2]
                angles_deg = pos[:, 2] * 180.0 / math.pi
                
                side, _, _, _ = compute_min_outer_side(positions, angles_deg)
                penalty = smooth_collision_penalty(params, beta)
                return side + penalty * 20.0
            
            init_params = np.zeros(3*N)
            for i in range(N):
                init_params[3*i] = best_positions[i, 0]
                init_params[3*i+1] = best_positions[i, 1]
                init_params[3*i+2] = best_angles[i] * math.pi / 180.0
                
            try:
                res = minimize(
                    polish_objective,
                    init_params,
                    method='L-BFGS-B',
                    jac=lambda p: smooth_gradient(p, beta=100.0),
                    options={'maxiter': 3000, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 80000}
                )
                
                if res.fun < 1e5:
                    new_pos = res.x.reshape(N, 3)
                    new_positions = new_pos[:, :2]
                    new_angles_deg = new_pos[:, 2] * 180.0 / math.pi
                    
                    if is_valid_configuration(new_positions, new_angles_deg):
                        new_side, new_phi, new_cx, new_cy = compute_min_outer_side(new_positions, new_angles_deg)
                        if new_side < best_side:
                            best_side = new_side
                            best_positions = new_positions
                            best_angles = new_angles_deg
                            best_phi = new_phi
                            best_cx = new_cx
                            best_cy = new_cy
            except Exception:
                pass

    # Final verification
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
