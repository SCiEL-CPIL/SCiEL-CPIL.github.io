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
    
    sqrt3 = math.sqrt(3)
    
    # Precompute for speed
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

    def check_collision(h1, h2):
        """
        Returns True if h1 and h2 overlap.
        h1, h2 are (x, y, angle_rad)
        """
        dx = h1[0] - h2[0]
        dy = h1[1] - h2[1]
        dist_sq = dx*dx + dy*dy
        # Max distance between centers for touching hexagons is 2*sqrt(3)
        if dist_sq > 12.0: # (2*sqrt(3))^2 = 12
            return False
        
        a1 = h1[2]
        a2 = h2[2]
        c1, s1 = math.cos(a1), math.sin(a1)
        c2, s2 = math.cos(a2), math.sin(a2)
        
        # Vertices of h1
        v1x = h1[0] + _ux * c1 - _uy * s1
        v1y = h1[1] + _ux * s1 + _uy * c1
        # Vertices of h2
        v2x = h2[0] + _ux * c2 - _uy * s2
        v2y = h2[1] + _ux * s2 + _uy * c2
        
        # Normals of h1
        en1x = _enx * c1 - _eny * s1
        en1y = _enx * s1 + _eny * c1
        # Normals of h2
        en2x = _enx * c2 - _eny * s2
        en2y = _enx * s2 + _eny * c2
        
        # Separation axis theorem
        for i in range(6):
            nx, ny = en1x[i], en1y[i]
            p1 = v1x * nx + v1y * ny
            p2 = v2x * nx + v2y * ny
            if p1.max() < p2.min() or p2.max() < p1.min():
                return False
        for i in range(6):
            nx, ny = en2x[i], en2y[i]
            p1 = v1x * nx + v1y * ny
            p2 = v2x * nx + v2y * ny
            if p1.max() < p2.min() or p2.max() < p1.min():
                return False
        return True

    def _eval_phi_fast(offsets, phi_rad, sqrt3):
        """Fast side-length eval for a single phi angle (radians)."""
        na = phi_rad + np.pi/6 + np.arange(6) * np.pi/3
        nx = np.cos(na)
        ny = np.sin(na)
        proj = offsets[:, 0] * nx[None, :] + offsets[:, 1] * ny[None, :]
        max_projs = np.max(np.abs(proj), axis=0)
        return (2.0 / sqrt3) * np.max(max_projs)

    def _logsumexp(a, axis=None, b=None, keepdims=False):
        """Numerically stable log-sum-exp."""
        if b is not None:
            a = a * b
        a_max = np.max(a, axis=axis, keepdims=True)
        if np.any(a_max == -np.inf):
            a_max = np.where(a_max == -np.inf, 0, a_max)
        out = np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=keepdims))
        if not keepdims:
            out = np.squeeze(out, axis=axis)
        return out + np.squeeze(a_max, axis=axis) if not keepdims else out + a_max

    def compute_outer_side_and_center(positions, angles_deg, refine=True):
        """
        Computes the minimum side length of the outer hexagon containing all inner hexagons.
        Returns (side_length, center_x, center_y, optimal_rotation_deg)
        """
        angles_rad = np.radians(angles_deg)
        all_verts = get_all_vertices(positions, angles_rad)
        verts_flat = all_verts.reshape(-1, 2)
        
        # Compute centroid
        center = np.mean(verts_flat, axis=0)
        offsets = verts_flat - center
        
        # Fully vectorized: compute side length for finer phi angles (0.5 deg resolution)
        phi_vals = np.arange(0, 60, 0.5)
        phi_rad = np.radians(phi_vals)
        
        normal_angles = phi_rad[:, None] + np.pi/6 + (np.arange(6) * np.pi/3)[None, :]
        nx = np.cos(normal_angles)
        ny = np.sin(normal_angles)
        
        proj = offsets[:, 0:1, None] * nx[None, :, :] + offsets[:, 1:2, None] * ny[None, :, :]
        abs_proj = np.abs(proj)
        max_projs = np.max(abs_proj, axis=0)
        side_vals = (2.0 / sqrt3) * max_projs
        best_sides = np.max(side_vals, axis=1)
        
        best_idx = np.argmin(best_sides)
        best_side = best_sides[best_idx]
        best_phi = phi_vals[best_idx]
        
        if refine:
            lo = max(0.0, best_phi - 0.5)
            hi = min(60.0, best_phi + 0.5)
            
            def eval_phi(phi_deg):
                phi = np.radians(phi_deg)
                na = phi + np.pi/6 + np.arange(6) * np.pi/3
                nx_r = np.cos(na)
                ny_r = np.sin(na)
                proj_r = offsets[:, 0:1] * nx_r[None, :] + offsets[:, 1:2] * ny_r[None, :]
                abs_proj_r = np.abs(proj_r)
                max_projs_r = np.max(abs_proj_r, axis=0)
                side_vals_r = (2.0 / sqrt3) * max_projs_r
                return np.max(side_vals_r)
                
            try:
                res = minimize_scalar(eval_phi, bounds=(lo, hi), method='bounded', options={'xatol': 1e-10})
                if res.success and res.fun < best_side:
                    best_side = res.fun
                    best_phi = res.x
            except Exception:
                pass
        
        return best_side, center[0], center[1], best_phi

    def fast_side_length(positions, angles_deg):
        """Fast side-length evaluation without refine (for SA loop)."""
        angles_rad = np.radians(angles_deg)
        all_verts = get_all_vertices(positions, angles_rad)
        verts_flat = all_verts.reshape(-1, 2)
        center = np.mean(verts_flat, axis=0)
        offsets = verts_flat - center
        
        phi_vals = np.arange(0, 60, 1.0)
        phi_rad = np.radians(phi_vals)
        normal_angles = phi_rad[:, None] + np.pi/6 + (np.arange(6) * np.pi/3)[None, :]
        nx = np.cos(normal_angles)
        ny = np.sin(normal_angles)
        proj = offsets[:, 0:1, None] * nx[None, :, :] + offsets[:, 1:2, None] * ny[None, :, :]
        abs_proj = np.abs(proj)
        max_projs = np.max(abs_proj, axis=0)
        side_vals = (2.0 / sqrt3) * max_projs
        best_sides = np.max(side_vals, axis=1)
        best_idx = np.argmin(best_sides)
        return best_sides[best_idx], center[0], center[1], phi_vals[best_idx]

    def smooth_side_length(offsets, phi_rad, sqrt3, temperature=0.01):
        """
        Smooth approximation of the max projection using log-sum-exp.
        Returns a differentiable proxy for the outer hexagon side length.
        """
        na = phi_rad + np.pi/6 + np.arange(6) * np.pi/3
        nx = np.cos(na)
        ny = np.sin(na)
        proj = offsets[:, 0] * nx[None, :] + offsets[:, 1] * ny[None, :]
        abs_proj = np.abs(proj)
        
        # Use log-sum-exp for smooth max
        # For each of the 6 directions, compute soft max over all vertices
        soft_maxes = np.zeros(6)
        for d in range(6):
            vals = abs_proj[:, d]
            # Add small offset to avoid log(0)
            vals = vals + 1e-10
            m = np.max(vals)
            soft_maxes[d] = m + np.log(np.sum(np.exp(vals - m)))
        
        return (2.0 / sqrt3) * np.max(soft_maxes)

    def collision_penalty(pos, ang, beta=50.0):
        """
        Compute a smooth penalty for collisions using soft-min of distances.
        Returns 0 if no collisions, positive if there are.
        Uses log-sum-exp for a smoother approximation.
        """
        angles_rad = np.radians(ang)
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                h1 = (pos[i, 0], pos[i, 1], angles_rad[i])
                h2 = (pos[j, 0], pos[j, 1], angles_rad[j])
                # Compute minimum distance between hexagons (negative if overlapping)
                # Use SAT to find minimum penetration depth
                c1, s1 = math.cos(angles_rad[i]), math.sin(angles_rad[i])
                c2, s2 = math.cos(angles_rad[j]), math.sin(angles_rad[j])
                
                v1x = h1[0] + _ux * c1 - _uy * s1
                v1y = h1[1] + _ux * s1 + _uy * c1
                v2x = h2[0] + _ux * c2 - _uy * s2
                v2y = h2[1] + _ux * s2 + _uy * c2
                
                en1x = _enx * c1 - _eny * s1
                en1y = _enx * s1 + _eny * c1
                en2x = _enx * c2 - _eny * s2
                en2y = _enx * s2 + _eny * c2
                
                gaps = []
                for i_ax in range(6):
                    nx, ny = en1x[i_ax], en1y[i_ax]
                    p1 = v1x * nx + v1y * ny
                    p2 = v2x * nx + v2y * ny
                    gap = p2.min() - p1.max()
                    gaps.append(gap)
                for i_ax in range(6):
                    nx, ny = en2x[i_ax], en2y[i_ax]
                    p1 = v1x * nx + v1y * ny
                    p2 = v2x * nx + v2y * ny
                    gap = p2.min() - p1.max()
                    gaps.append(gap)
                
                gaps_arr = np.array(gaps)
                # Use log-sum-exp to get a smooth minimum
                # We want to penalize when the minimum gap is negative (overlap)
                # soft_min = -logsumexp(-gaps)
                neg_gaps = -gaps_arr
                m = np.max(neg_gaps)
                if m == -np.inf:
                    continue
                soft_min = -(m + np.log(np.sum(np.exp(neg_gaps - m))))
                
                if soft_min < 0:
                    # Overlapping, add smooth penalty
                    penalty += math.exp(-beta * soft_min)
        
        return penalty

    def is_valid(positions, angles_deg):
        angles_rad = np.radians(angles_deg)
        for i in range(N):
            for j in range(i+1, N):
                h1 = (positions[i][0], positions[i][1], angles_rad[i])
                h2 = (positions[j][0], positions[j][1], angles_rad[j])
                if check_collision(h1, h2):
                    return False
        return True

    # --- Initialization: Honeycomb ---
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
    best_cx = 0.0
    best_cy = 0.0
    best_phi = 0.0

    # --- Phase 1: Simulated Annealing ---
    current_positions = base_positions.copy()
    current_angles = base_angles.copy()
    current_side, current_cx, current_cy, current_phi = fast_side_length(current_positions, current_angles)
    
    if current_side < best_side:
        best_side = current_side
        best_positions = current_positions.copy()
        best_angles = current_angles.copy()
        best_cx = current_cx
        best_cy = current_cy
        best_phi = current_phi

    T = 0.5
    T_min = 0.0001
    cooling = 0.99995
    step_size = 0.3
    step_decay = 0.99998
    
    sa_time_budget = DEADLINE * 0.55
    max_steps = 200000
    step_count = 0
    
    while time.time() - t0 < sa_time_budget and step_count < max_steps:
        step_count += 1
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
            if check_collision(h_moved, h_other):
                valid = False
                break
        
        if not valid:
            T *= cooling
            if T < T_min: T = T_min
            step_size *= step_decay
            if step_size < 0.005: step_size = 0.005
            continue
            
        new_side, new_cx, new_cy, new_phi = fast_side_length(new_positions, new_angles)
        
        if new_side < current_side:
            current_positions = new_positions
            current_angles = new_angles
            current_side = new_side
            current_cx = new_cx
            current_cy = new_cy
            current_phi = new_phi
            if new_side < best_side:
                best_side = new_side
                best_positions = new_positions.copy()
                best_angles = new_angles.copy()
                best_cx = new_cx
                best_cy = new_cy
                best_phi = new_phi
        else:
            delta = new_side - current_side
            if random.random() < math.exp(-delta / T):
                current_positions = new_positions
                current_angles = new_angles
                current_side = new_side
                current_cx = new_cx
                current_cy = new_cy
                current_phi = new_phi
                if new_side < best_side:
                    best_side = new_side
                    best_positions = new_positions.copy()
                    best_angles = new_angles.copy()
                    best_cx = new_cx
                    best_cy = new_cy
                    best_phi = new_phi
        
        T *= cooling
        if T < T_min: T = T_min
        step_size *= step_decay
        if step_size < 0.005: step_size = 0.005

    # --- Phase 2: Smooth Surrogate Optimization ---
    # Use L-BFGS-B on a smooth objective that combines side length and collision penalty
    
    def smooth_objective(x, beta=50.0, temperature=0.01):
        """
        Smooth objective: side_length + collision_penalty
        Uses log-sum-exp for smooth max in side length computation.
        """
        pos = np.zeros((N, 2))
        ang = np.zeros(N)
        for i in range(N):
            pos[i, 0] = x[i*3]
            pos[i, 1] = x[i*3+1]
            ang[i] = x[i*3+2]
        
        # Compute vertices
        angles_rad = np.radians(ang)
        all_verts = get_all_vertices(pos, angles_rad)
        verts_flat = all_verts.reshape(-1, 2)
        center = np.mean(verts_flat, axis=0)
        offsets = verts_flat - center
        
        # Use a fixed phi for the smooth objective (or optimize it separately)
        # For now, use the current best phi
        phi_rad = np.radians(best_phi)
        side_smooth = smooth_side_length(offsets, phi_rad, sqrt3, temperature)
        
        # Compute collision penalty
        penalty = collision_penalty(pos, ang, beta)
        
        return side_smooth + penalty
    
    def polish_smooth_lbfgs(pos, ang, time_limit):
        """
        Polish using L-BFGS-B on smooth surrogate
        """
        n_params = N * 3
        x0 = np.zeros(n_params)
        for i in range(N):
            x0[i*3] = pos[i, 0]
            x0[i*3+1] = pos[i, 1]
            x0[i*3+2] = ang[i]
        
        t_start = time.time()
        
        # Try different beta values for collision penalty
        best_side = float('inf')
        best_pos = pos.copy()
        best_ang = ang.copy()
        
        for beta in [20.0, 50.0, 100.0, 200.0]:
            if time.time() - t_start > time_limit * 0.7:
                break
            
            def obj(x):
                return smooth_objective(x, beta)
            
            try:
                res = minimize(
                    obj, x0, method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 100}
                )
                
                new_pos = np.zeros((N, 2))
                new_ang = np.zeros(N)
                for i in range(N):
                    new_pos[i, 0] = res.x[i*3]
                    new_pos[i, 1] = res.x[i*3+1]
                    new_ang[i] = res.x[i*3+2]
                
                # Verify validity
                angles_rad = np.radians(new_ang)
                valid = True
                for i in range(N):
                    for j in range(i+1, N):
                        h1 = (new_pos[i, 0], new_pos[i, 1], angles_rad[i])
                        h2 = (new_pos[j, 0], new_pos[j, 1], angles_rad[j])
                        if check_collision(h1, h2):
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    side, _, _, _ = compute_outer_side_and_center(new_pos, new_ang)
                    if side < best_side:
                        best_side = side
                        best_pos = new_pos
                        best_ang = new_ang
            except Exception:
                pass
        
        return best_side, best_pos, best_ang

    remaining_time = DEADLINE - (time.time() - t0)
    if remaining_time > 5.0:
        # Phase 2: Smooth L-BFGS-B polish
        lbfgs_side, lbfgs_pos, lbfgs_ang = polish_smooth_lbfgs(best_positions, best_angles, remaining_time * 0.4)
        if lbfgs_side < best_side:
            best_side = lbfgs_side
            best_positions = lbfgs_pos
            best_angles = lbfgs_ang
            best_side, best_cx, best_cy, best_phi = compute_outer_side_and_center(best_positions, best_angles)
        
        # Phase 3: Coordinate Descent final polish
        remaining_time = DEADLINE - (time.time() - t0)
        if remaining_time > 5.0:
            def polish_coordinate_descent(pos, ang, time_limit):
                current_side, _, _, _ = compute_outer_side_and_center(pos, ang)
                best_side_cd = current_side
                best_pos_cd = pos.copy()
                best_ang_cd = ang.copy()
                
                t_start = time.time()
                step = 0.05
                min_step = 0.0001
                
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
                                    if check_collision(h_moved, h_other):
                                        valid = False
                                        break
                                
                                if not valid:
                                    continue
                                
                                new_side, _, _, _ = fast_side_length(new_pos, new_ang)
                                
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
                            continue
                        break
                    
                    if not improved:
                        step *= 0.7
                    else:
                        step *= 0.95
                        if step < min_step:
                            step = min_step
                
                return best_side_cd, best_pos_cd, best_ang_cd
            
            cd_side, cd_pos, cd_ang = polish_coordinate_descent(best_positions, best_angles, remaining_time * 0.9)
            if cd_side < best_side:
                best_side = cd_side
                best_positions = cd_pos
                best_angles = cd_ang
                best_side, best_cx, best_cy, best_phi = compute_outer_side_and_center(best_positions, best_angles)

    # Final verification
    if not is_valid(best_positions, best_angles):
        # Fallback to base if something went wrong
        best_positions = base_positions.copy()
        best_angles = base_angles.copy()
        best_side, best_cx, best_cy, best_phi = compute_outer_side_and_center(best_positions, best_angles)
    
    # Final high-precision polish with tighter tolerances
    remaining = DEADLINE - (time.time() - t0)
    if remaining > 3.0:
        # One more round of coordinate descent with very small steps
        step = 0.001
        for _ in range(50):
            if time.time() - t0 > DEADLINE - 2.0:
                break
            improved = False
            for idx in range(N):
                for dim in range(3):
                    for direction in [1, -1]:
                        new_pos = best_positions.copy()
                        new_ang = best_angles.copy()
                        
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
                            if check_collision(h_moved, h_other):
                                valid = False
                                break
                        
                        if not valid:
                            continue
                        
                        new_side, _, _, _ = compute_outer_side_and_center(new_pos, new_ang)
                        
                        if new_side < best_side - 1e-12:
                            best_positions = new_pos
                            best_angles = new_ang
                            best_side = new_side
                            improved = True
                            break
                        
                    if improved:
                        break
                if improved:
                    break
            
            if not improved:
                step *= 0.5
                if step < 1e-6:
                    break
    
    # Recompute final center and phi
    best_side, best_cx, best_cy, best_phi = compute_outer_side_and_center(best_positions, best_angles)
    
    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), float(best_angles[i])])
        
    center = (float(best_cx), float(best_cy))
    side = float(best_side)
    angle = float(best_phi)
    
    return inner, center, side, angle


