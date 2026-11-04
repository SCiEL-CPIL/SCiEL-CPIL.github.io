import math
import time
import random
import numpy as np
from scipy.optimize import minimize_scalar

N = 12
DEADLINE = 145.0  # Safety margin before 150s kill

# Precompute unit hexagon vertices (flat-top orientation, angle 0)
# Vertices at angles 0, 60, 120, 180, 240, 300
_HEX_ANGLES = np.radians(np.arange(0, 360, 60))
_UNIT_HEX_VERTS = np.column_stack((np.cos(_HEX_ANGLES), np.sin(_HEX_ANGLES)))

# Edge normals for SAT
# Edge i is between vertex i and vertex (i+1)%6
_EDGE_NORMALS = np.zeros((6, 2))
for _i in range(6):
    _v1 = _UNIT_HEX_VERTS[_i]
    _v2 = _UNIT_HEX_VERTS[(_i + 1) % 6]
    _ex = _v2[0] - _v1[0]
    _ey = _v2[1] - _v1[1]
    # Normal is perpendicular to edge, pointing outwards
    # For convex polygon vertices in CCW order, (dy, -dx) or (-dy, dx)?
    # Let's check: (1,0) to (0.5, sqrt(3)/2). Edge vector (-0.5, sqrt(3)/2).
    # Outward normal should have positive x component roughly.
    # Normal (-dy, dx) = (-sqrt(3)/2, -0.5). This points inwards (towards origin).
    # Normal (dy, -dx) = (sqrt(3)/2, 0.5). This points outwards.
    _nx, _ny = _ey, -_ex
    _norm = math.sqrt(_nx * _nx + _ny * _ny)
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
        """
        Optimized disjointness check.
        h1, h2: tuples of (x, y, angle_rad)
        """
        dx = h1[0] - h2[0]
        dy = h1[1] - h2[1]
        dist_sq = dx * dx + dy * dy
        
        # Fast rejection: if distance > 2 (circumradius 1 + 1), they are disjoint
        if dist_sq > 4.0:
            return True
            
        # Fast acceptance of overlap: if distance < sqrt(3) (inradius sum), they MUST overlap.
        # Inradius of unit hex is sqrt(3)/2. Sum is sqrt(3) ~ 1.732.
        # If centers are closer than sum of inradii, they overlap.
        if dist_sq < 3.0:
            return False

        a1 = h1[2]
        a2 = h2[2]
        c1, s1 = math.cos(a1), math.sin(a1)
        c2, s2 = math.cos(a2), math.sin(a2)
        
        # Rotate vertices
        v1x = h1[0] + _ux * c1 - _uy * s1
        v1y = h1[1] + _ux * s1 + _uy * c1
        v2x = h2[0] + _ux * c2 - _uy * s2
        v2y = h2[1] + _ux * s2 + _uy * c2
        
        # Rotate normals
        en1x = _enx * c1 - _eny * s1
        en1y = _enx * s1 + _eny * c1
        en2x = _enx * c2 - _eny * s2
        en2y = _enx * s2 + _eny * c2
        
        # SAT Check
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
        """
        Fast evaluation for SA: coarse scan of outer rotation phi.
        Returns (side_length, best_phi_deg, center_x, center_y)
        """
        angles_rad = np.radians(angles_deg)
        all_verts = get_all_vertices(positions, angles_rad)
        verts_flat = all_verts.reshape(-1, 2)
        center = np.mean(verts_flat, axis=0)
        offsets = verts_flat - center

        # Coarse scan: 0 to 60 deg with step 1 deg
        phi_vals = np.radians(np.arange(0, 60, 1))
        # Outer hexagon normals. If outer hex is rotated by phi, its normals are at phi + 0, 60, ...
        # Wait, the outer hexagon is regular. Its side length is determined by the projection of the inner points onto the normals of the outer hexagon.
        # Let outer hexagon have rotation phi. The normals are at angles phi, phi+60, ..., phi+300.
        # The width in direction n is max(v.n) - min(v.n).
        # For a centered configuration, width is 2 * max(|v.n|).
        # Side length S relates to width W by W = S * sqrt(3). So S = W / sqrt(3).
        
        normal_angles = phi_vals[:, None] + np.arange(6) * np.pi / 3
        nx = np.cos(normal_angles)
        ny = np.sin(normal_angles)
        
        # proj[i, k, j] is projection of vertex i onto normal k for phi j
        proj = offsets[:, None, 0:1] * nx[None, :, :] + offsets[:, None, 1:2] * ny[None, :, :]
        abs_proj = np.abs(proj)
        max_projs = np.max(abs_proj, axis=0) # [6, 60]
        side_per_phi = (2.0 / sqrt3) * np.max(max_projs, axis=0) # [60]

        best_idx = np.argmin(side_per_phi)
        best_side = side_per_phi[best_idx]
        best_phi = float(best_idx)
        return best_side, best_phi, center[0], center[1]

    def compute_min_outer_side_precise(positions, angles_deg, phi_init=0.0):
        """
        High precision evaluation: local minimize_scalar around best phi.
        """
        angles_rad = np.radians(angles_deg)
        all_verts = get_all_vertices(positions, angles_rad)
        verts_flat = all_verts.reshape(-1, 2)
        center = np.mean(verts_flat, axis=0)
        offsets = verts_flat - center

        # Coarse scan first to find good start
        phi_vals = np.radians(np.arange(0, 60, 1))
        normal_angles = phi_vals[:, None] + np.arange(6) * np.pi / 3
        nx = np.cos(normal_angles)
        ny = np.sin(normal_angles)
        proj = offsets[:, None, 0:1] * nx[None, :, :] + offsets[:, None, 1:2] * ny[None, :, :]
        abs_proj = np.abs(proj)
        max_projs = np.max(abs_proj, axis=0)
        side_per_phi = (2.0 / sqrt3) * np.max(max_projs, axis=0)

        best_idx = np.argmin(side_per_phi)
        best_side = side_per_phi[best_idx]
        best_phi = float(best_idx)

        def eval_phi(phi_deg):
            phi = np.radians(phi_deg)
            na = phi + np.arange(6) * np.pi / 3
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

        return best_side, best_phi, center[0], center[1]

    def is_valid_configuration(positions, angles_deg):
        angles_rad = np.radians(angles_deg)
        for i in range(N):
            for j in range(i + 1, N):
                h1 = (positions[i][0], positions[i][1], angles_rad[i])
                h2 = (positions[j][0], positions[j][1], angles_rad[j])
                if not hexagons_disjoint_fast(h1, h2):
                    return False
        return True

    # --- Initial Configurations ---
    a1 = np.array([sqrt3, 0.0])
    a2 = np.array([sqrt3 / 2.0, 1.5])

    def make_honeycomb():
        # 12 hexagons in a hexagonal arrangement.
        # Center (0,0), 6 neighbors, 5 more? 
        # Standard honeycomb for 12:
        # Row 1: 3
        # Row 2: 4
        # Row 3: 3
        # Row 4: 2
        # Total 12.
        # Or:
        # Center + 6 ring + 5 outer?
        # Let's use the lattice coords from the seed, which seemed valid.
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

    def make_random_honeycomb(seed):
        rng = np.random.RandomState(seed)
        pos = np.zeros((N, 2))
        ang = np.zeros(N)
        lattice_coords = [
            (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, -1), (-1, 1), (2, -1), (-2, 1),
            (1, 1), (-1, -1), (2, 0),
        ]
        global_rot = rng.uniform(0, 2 * np.pi)
        c_g, s_g = math.cos(global_rot), math.sin(global_rot)
        
        for idx, (i, j) in enumerate(lattice_coords):
            x, y = i * a1[0] + j * a2[0], i * a1[1] + j * a2[1]
            rx = x * c_g - y * s_g
            ry = x * s_g + y * c_g
            # Small perturbation
            rx += rng.uniform(-0.1, 0.1)
            ry += rng.uniform(-0.1, 0.1)
            pos[idx] = [rx, ry]
            ang[idx] = rng.uniform(0, 60)
        return pos, ang

    init_configs = [make_honeycomb()]
    for s in range(10):
        init_configs.append(make_random_honeycomb(s))

    best_side = float('inf')
    best_positions = None
    best_angles = None
    best_phi = 0.0
    best_cx = 0.0
    best_cy = 0.0

    for pos0, ang0 in init_configs:
        if is_valid_configuration(pos0, ang0):
            s, phi, cx, cy = compute_min_outer_side_precise(pos0, ang0)
            if s < best_side:
                best_side = s
                best_positions = pos0.copy()
                best_angles = ang0.copy()
                best_phi = phi
                best_cx = cx
                best_cy = cy

    if best_positions is None:
        best_positions, best_angles = make_honeycomb()
        best_side, best_phi, best_cx, best_cy = compute_min_outer_side_precise(best_positions, best_angles)

    # --- Simulated Annealing ---
    sa_time_budget = DEADLINE * 0.60
    start_time = time.time()
    
    current_positions = best_positions.copy()
    current_angles = best_angles.copy()
    current_side, current_phi, current_cx, current_cy = compute_min_outer_side_fast(current_positions, current_angles)

    T = 0.3
    T_min = 0.001
    cooling = 0.9999
    step_size = 0.4
    step_decay = 0.99995

    step_count = 0
    max_steps = 1000000
    accept_count = 0
    total_count = 0

    while time.time() - start_time < sa_time_budget and step_count < max_steps:
        step_count += 1
        total_count += 1

        new_positions = current_positions.copy()
        new_angles = current_angles.copy()

        # Move 1 or 2 hexagons
        if random.random() < 0.3:
            idx1, idx2 = random.sample(range(N), 2)
            new_positions[idx1, 0] += random.gauss(0, step_size)
            new_positions[idx1, 1] += random.gauss(0, step_size)
            new_angles[idx1] += random.gauss(0, 5.0)
            
            new_positions[idx2, 0] += random.gauss(0, step_size)
            new_positions[idx2, 1] += random.gauss(0, step_size)
            new_angles[idx2] += random.gauss(0, 5.0)

            angles_rad_new = np.radians(new_angles)
            valid = True
            h1 = (new_positions[idx1, 0], new_positions[idx1, 1], angles_rad_new[idx1])
            for j in range(N):
                if j == idx1: continue
                h_j = (new_positions[j, 0], new_positions[j, 1], angles_rad_new[j])
                if not hexagons_disjoint_fast(h1, h_j):
                    valid = False
                    break
            
            if valid:
                h2 = (new_positions[idx2, 0], new_positions[idx2, 1], angles_rad_new[idx2])
                for j in range(N):
                    if j == idx2 or j == idx1: continue
                    h_j = (new_positions[j, 0], new_positions[j, 1], angles_rad_new[j])
                    if not hexagons_disjoint_fast(h2, h_j):
                        valid = False
                        break
        else:
            idx = random.randint(0, N - 1)
            new_positions[idx, 0] += random.gauss(0, step_size)
            new_positions[idx, 1] += random.gauss(0, step_size)
            new_angles[idx] += random.gauss(0, 5.0)

            angles_rad_new = np.radians(new_angles)
            valid = True
            h_moved = (new_positions[idx, 0], new_positions[idx, 1], angles_rad_new[idx])
            for j in range(N):
                if j == idx: continue
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

        new_side, new_phi, new_cx, new_cy = compute_min_outer_side_fast(new_positions, new_angles)

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

        if total_count % 200 == 0:
            acc_rate = accept_count / total_count
            if acc_rate < 0.05:
                step_size *= 0.8
                if step_size < 0.005: step_size = 0.005
            elif acc_rate > 0.4:
                step_size *= 1.1
                if step_size > 1.0: step_size = 1.0
            accept_count = 0
            total_count = 0

        T *= cooling
        if T < T_min: T = T_min
        step_size *= step_decay
        if step_size < 0.005: step_size = 0.005

    # --- Polishing Phase ---
    remaining_time = DEADLINE - (time.time() - t0)
    
    if remaining_time > 5.0:
        # 1. Coordinate Descent Polish
        def polish_coordinate_descent(pos, ang, time_limit):
            current_side, _, _, _ = compute_min_outer_side_precise(pos, ang)
            best_side_cd = current_side
            best_pos_cd = pos.copy()
            best_ang_cd = ang.copy()
            
            t_start = time.time()
            step = 0.05
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
                            
                            new_side, _, _, _ = compute_min_outer_side_precise(new_pos, new_ang)
                            
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

        cd_time = min(remaining_time * 0.4, 15.0)
        cd_side, cd_pos, cd_ang = polish_coordinate_descent(best_positions, best_angles, cd_time)
        
        if cd_side < best_side:
            best_side = cd_side
            best_positions = cd_pos
            best_angles = cd_ang
            best_side, best_phi, best_cx, best_cy = compute_min_outer_side_precise(best_positions, best_angles)
        
        # 2. Final High-Precision Phi Search
        remaining_time = DEADLINE - (time.time() - t0)
        if remaining_time > 2.0:
            def eval_phi_fine(phi_deg):
                phi = np.radians(phi_deg)
                angles_rad = np.radians(best_angles)
                all_verts = get_all_vertices(best_positions, angles_rad)
                verts_flat = all_verts.reshape(-1, 2)
                center = np.mean(verts_flat, axis=0)
                offsets = verts_flat - center
                na = phi + np.arange(6) * np.pi / 3
                nxx = np.cos(na)
                nyy = np.sin(na)
                pr = offsets[:, 0:1] * nxx[None, :] + offsets[:, 1:2] * nyy[None, :]
                return (2.0 / sqrt3) * np.max(np.abs(pr), axis=0).max()
            
            # Scan around current best phi
            best_phi_fine = best_phi
            best_side_fine = eval_phi_fine(best_phi)
            
            # Try a few local minima
            for offset in [-1.0, -0.5, 0.0, 0.5, 1.0]:
                start_phi = best_phi + offset
                lo = max(0.0, start_phi - 0.5)
                hi = min(60.0, start_phi + 0.5)
                try:
                    res = minimize_scalar(eval_phi_fine, bounds=(lo, hi), method='bounded',
                                          options={'xatol': 1e-12, 'maxiter': 200})
                    if res.success and res.fun < best_side_fine:
                        best_side_fine = res.fun
                        best_phi_fine = res.x
                except Exception:
                    pass
            
            if best_side_fine < best_side - 1e-12:
                best_side = best_side_fine
                best_phi = best_phi_fine
                _, _, best_cx, best_cy = compute_min_outer_side_precise(best_positions, best_angles)

    # Final verification
    if not is_valid_configuration(best_positions, best_angles):
        # Fallback to honeycomb if something went wrong
        best_positions, best_angles = make_honeycomb()
        best_side, best_phi, best_cx, best_cy = compute_min_outer_side_precise(best_positions, best_angles)
    
    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), float(best_angles[i])])
        
    center = (float(best_cx), float(best_cy))
    side = float(best_side)
    angle = float(best_phi)
    
    return inner, center, side, angle
