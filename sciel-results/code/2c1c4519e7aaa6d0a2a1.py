import math
import time
import random
import numpy as np
from scipy.optimize import minimize, minimize_scalar

N = 12
DEADLINE = 145.0 # Safety margin

# Precompute unit hexagon vertices and edge normals
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
    
    def check_overlap_vectorized(moved_idx, positions, angles_rad):
        """
        Optimized overlap check for one hexagon against all others.
        Returns True if NO overlap (valid), False if overlap (invalid).
        """
        h1x = positions[moved_idx, 0]
        h1y = positions[moved_idx, 1]
        a1 = angles_rad[moved_idx]
        
        # Pre-filter: if distance > 2.0 (sum of circumradii), no overlap possible
        # Squared distance > 4.0
        dx = h1x - positions[:, 0]
        dy = h1y - positions[:, 1]
        dist_sq = dx*dx + dy*dy
        # Mask out self and far neighbors
        mask = (dist_sq <= 4.0) & (np.arange(N) != moved_idx)
        
        if not np.any(mask):
            return True
            
        # Get indices of potential collisions
        colliding_indices = np.where(mask)[0]
        
        # Rotate edge normals of hexagon 1
        c1, s1 = math.cos(a1), math.sin(a1)
        en1x = _enx * c1 - _eny * s1
        en1y = _enx * s1 + _eny * c1
        
        # Vertices of hexagon 1
        v1x = h1x + _ux * c1 - _uy * s1
        v1y = h1y + _ux * s1 + _uy * c1
        
        # Project hexagon 1 onto its 6 edge normals
        # p1 shape: (6,)
        p1_min = np.empty(6)
        p1_max = np.empty(6)
        for k in range(6):
            proj = v1x * en1x[k] + v1y * en1y[k]
            p1_min[k] = np.min(proj)
            p1_max[k] = np.max(proj)
            
        # For each potential colliding hexagon, check separation
        for j in colliding_indices:
            a2 = angles_rad[j]
            c2, s2 = math.cos(a2), math.sin(a2)
            
            # Vertices of hexagon 2
            v2x = positions[j, 0] + _ux * c2 - _uy * s2
            v2y = positions[j, 1] + _ux * s2 + _uy * c2
            
            # Check separation using hexagon 1's normals
            # Project hexagon 2 onto hexagon 1's normals
            # p2_min/max shape: (6,)
            p2_min = np.empty(6)
            p2_max = np.empty(6)
            for k in range(6):
                proj = v2x * en1x[k] + v2y * en1y[k]
                p2_min[k] = np.min(proj)
                p2_max[k] = np.max(proj)
                
            # If any axis separates them, they are disjoint
            if np.any(p1_max < p2_min) or np.any(p2_max < p1_min):
                continue
                
            # If not separated by hex 1's normals, check hex 2's normals
            en2x = _enx * c2 - _eny * s2
            en2y = _enx * s2 + _eny * c2
            
            separated = False
            for k in range(6):
                proj1 = v1x * en2x[k] + v1y * en2y[k]
                proj2 = v2x * en2x[k] + v2y * en2y[k]
                if np.max(proj1) < np.min(proj2) or np.max(proj2) < np.min(proj1):
                    separated = True
                    break
            
            if not separated:
                return False # Overlap found
                
        return True # No overlap

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
        
        # Coarse scan
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
            if not check_overlap_vectorized(i, positions, angles_rad):
                return False
        return True

    def make_grid_2x3(scale=1.0, rotation_deg=0.0):
        a1 = np.array([sqrt3, 0.0]) * scale
        a2 = np.array([sqrt3/2.0, 1.5]) * scale
        positions = np.zeros((N, 2))
        angles = np.zeros(N)
        idx = 0
        for j in range(2):
            for i in range(3):
                pos = i * a1 + j * a2
                positions[idx] = pos
                angles[idx] = rotation_deg
                idx += 1
        return positions, angles
        
    def make_grid_3x2(scale=1.0, rotation_deg=0.0):
        a1 = np.array([sqrt3, 0.0]) * scale
        a2 = np.array([sqrt3/2.0, 1.5]) * scale
        positions = np.zeros((N, 2))
        angles = np.zeros(N)
        idx = 0
        for j in range(3):
            for i in range(2):
                pos = i * a1 + j * a2
                positions[idx] = pos
                angles[idx] = rotation_deg
                idx += 1
        return positions, angles

    # Generate initial configurations
    initial_configs = []
    
    for s in [0.98, 1.0, 1.02]:
        initial_configs.append(make_grid_2x3(scale=s))
        initial_configs.append(make_grid_3x2(scale=s))
        
    for rot in [10, 20, 30]:
        initial_configs.append(make_grid_2x3(scale=1.0, rotation_deg=rot))
        initial_configs.append(make_grid_3x2(scale=1.0, rotation_deg=rot))
        
    # Lattice based
    a1 = np.array([sqrt3, 0.0])
    a2 = np.array([sqrt3/2.0, 1.5])
    lattice_coords = [
        (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
        (1, -1), (-1, 1), (2, -1), (-2, 1),
        (1, 1), (-1, -1), (2, 0),
    ]
    for scale in [0.95, 1.0]:
        positions = np.zeros((N, 2))
        angles = np.zeros(N)
        for idx, (i, j) in enumerate(lattice_coords):
            positions[idx] = (i * a1 + j * a2) * scale
            angles[idx] = 0.0
        initial_configs.append((positions, angles))
        
    best_side = float('inf')
    best_positions = None
    best_angles = None
    best_phi = 0.0
    best_cx = 0.0
    best_cy = 0.0
    
    # Evaluate initial configs
    for pos, ang in initial_configs:
        if time.time() - t0 > DEADLINE * 0.1:
            break
        try:
            side, phi, cx, cy = compute_min_outer_side(pos, ang, fine_tune=False)
            if is_valid_configuration(pos, ang) and side < best_side:
                best_side = side
                best_positions = pos.copy()
                best_angles = ang.copy()
                best_phi = phi
                best_cx = cx
                best_cy = cy
        except Exception:
            continue
            
    if best_positions is None:
        best_positions, best_angles = make_grid_2x3(1.0)
        best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)
        
    # SA Phase
    sa_time_budget = DEADLINE * 0.5
    sa_deadline = t0 + sa_time_budget
    
    current_positions = best_positions.copy()
    current_angles = best_angles.copy()
    current_side, current_phi, current_cx, current_cy = compute_min_outer_side(current_positions, current_angles)
    
    T = 0.3
    T_min = 0.0001
    cooling = 0.99995
    step_size = 0.15
    step_decay = 0.99999
    
    while time.time() < sa_deadline:
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
        
        # Fast validity check
        if not check_overlap_vectorized(idx, new_positions, angles_rad_new):
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
        step_size *= step_decay
        if step_size < 0.003: step_size = 0.003
        
    # Gradient-based optimization
    remaining = DEADLINE - (time.time() - t0)
    if remaining > 15.0:
        grad_deadline = t0 + remaining * 0.5
        
        def smooth_outer_side_surrogate(positions, angles_deg, beta=50.0):
            angles_rad = np.radians(angles_deg)
            all_verts = get_all_vertices(positions, angles_rad)
            verts_flat = all_verts.reshape(-1, 2)
            center = np.mean(verts_flat, axis=0)
            offsets = verts_flat - center
            
            phi_vals = np.radians(np.arange(0, 60, 2))
            normal_angles = phi_vals[:, None] + np.pi/6 + np.arange(6) * np.pi/3
            nx = np.cos(normal_angles)
            ny = np.sin(normal_angles)
            proj = offsets[:, 0:1, None] * nx[None, :, :] + offsets[:, 1:2, None] * ny[None, :, :]
            abs_proj = np.abs(proj)
            max_projs = np.max(abs_proj, axis=0)
            side_vals = (2.0 / sqrt3) * max_projs
            max_sides = np.max(side_vals, axis=1)
            
            # Soft min
            m = np.max(-max_sides)
            return -m - np.log(np.sum(np.exp(beta * (-max_sides - m)))) / beta

        x0 = np.concatenate([best_positions.flatten(), best_angles])
        
        try:
            res = minimize(
                lambda x: smooth_outer_side_surrogate(x[:2*N].reshape(N, 2), x[2*N:]),
                x0,
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50}
            )
            if res.success:
                new_pos = res.x[:2*N].reshape(N, 2)
                new_ang = res.x[2*N:]
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
            
    # Coordinate Descent Polish
    remaining = DEADLINE - (time.time() - t0)
    if remaining > 5.0:
        cd_deadline = t0 + remaining * 0.6
        step = 0.02
        min_step = 0.0001
        
        while time.time() < cd_deadline and step > min_step:
            improved = False
            for idx in range(N):
                for dim in range(3):
                    for direction in [1, -1]:
                        new_pos = best_positions.copy()
                        new_ang = best_angles.copy()
                        
                        if dim == 0: new_pos[idx, 0] += direction * step
                        elif dim == 1: new_pos[idx, 1] += direction * step
                        else: new_ang[idx] += direction * step
                        
                        angles_rad_new = np.radians(new_ang)
                        if not check_overlap_vectorized(idx, new_pos, angles_rad_new):
                            continue
                            
                        new_side, _, _, _ = compute_min_outer_side(new_pos, new_ang, fine_tune=False)
                        
                        if new_side < best_side - 1e-12:
                            best_positions = new_pos
                            best_angles = new_ang
                            best_side = new_side
                            improved = True
                            break
                    if improved: break
                if improved: break
            if not improved:
                step *= 0.7
                
        best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)
        
    # Final Fine Tune
    if time.time() < t0 + DEADLINE - 2.0:
        def eval_final_phi(phi_deg):
            phi = np.radians(phi_deg)
            angles_rad = np.radians(best_angles)
            all_verts = get_all_vertices(best_positions, angles_rad)
            verts_flat = all_verts.reshape(-1, 2)
            center = np.mean(verts_flat, axis=0)
            offsets = verts_flat - center
            normal_angles = phi + np.pi/6 + np.arange(6) * np.pi/3
            nx = np.cos(normal_angles)
            ny = np.sin(normal_angles)
            proj = offsets[:, 0:1] * nx[None, :] + offsets[:, 1:2] * ny[None, :]
            abs_proj = np.abs(proj)
            max_projs = np.max(abs_proj, axis=0)
            side_vals = (2.0 / sqrt3) * max_projs
            return np.max(side_vals)
            
        try:
            res = minimize_scalar(eval_final_phi, bounds=(0, 60), method='bounded',
                                  options={'xatol': 1e-12})
            if res.success and res.fun < best_side:
                best_side = res.fun
                best_phi = res.x
                angles_rad = np.radians(best_angles)
                all_verts = get_all_vertices(best_positions, angles_rad)
                verts_flat = all_verts.reshape(-1, 2)
                center = np.mean(verts_flat, axis=0)
                best_cx = center[0]
                best_cy = center[1]
        except Exception:
            pass

    # Validate
    if not is_valid_configuration(best_positions, best_angles):
        best_positions, best_angles = make_grid_2x3(1.0)
        best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)
        
    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), float(best_angles[i])])
        
    center = (float(best_cx), float(best_cy))
    side = float(best_side)
    angle = float(best_phi)
    
    return inner, center, side, angle
