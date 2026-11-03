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
_edge_normals_raw = np.zeros((6, 2))
for _i in range(6):
    _ex = _UNIT_HEX_VERTS[(_i+1)%6, 0] - _UNIT_HEX_VERTS[_i, 0]
    _ey = _UNIT_HEX_VERTS[(_i+1)%6, 1] - _UNIT_HEX_VERTS[_i, 1]
    _nx, _ny = -_ey, _ex
    _norm = math.sqrt(_nx*_nx + _ny*_ny)
    _edge_normals_raw[_i] = [_nx / _norm, _ny / _norm]
_EDGE_NORMALS = _edge_normals_raw

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

        phi_grid = np.arange(0, 60, 0.5)
        best_side = float('inf')
        best_phi = 0.0
        for phi_deg in phi_grid:
            s = eval_phi(phi_deg)
            if s < best_side:
                best_side = s
                best_phi = phi_deg
        res = minimize_scalar(eval_phi, bounds=(best_phi - 1.0, best_phi + 1.0), method='bounded', options={'xatol': 1e-5})
        if res.success and res.fun < best_side:
            best_side = res.fun
            best_phi = res.x
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

    # Soft-max surrogate for smooth optimization
    def soft_max(x, beta):
        m = np.max(x)
        return m + np.log(np.sum(np.exp(beta * (x - m)))) / beta

    def smooth_outer_side(params, beta=20.0):
        """
        params: [x0,y0,a0, x1,y1,a1, ..., x11,y11,a11, phi]
        Returns smooth approximation of outer side length + penalty for overlaps.
        """
        n = len(params)
        pos = np.array([[params[i*3], params[i*3+1]] for i in range(N)])
        angles_deg = np.array([params[i*3+2] for i in range(N)])
        phi = params[-1]

        angles_rad = np.radians(angles_deg)
        all_verts = get_all_vertices(pos, angles_rad)
        verts_flat = all_verts.reshape(-1, 2)
        center = np.mean(verts_flat, axis=0)
        offsets = verts_flat - center

        phi_rad = np.radians(phi)
        normal_angles = phi_rad + np.pi/6 + np.arange(6) * np.pi/3
        nx = np.cos(normal_angles)
        ny = np.sin(normal_angles)
        proj = offsets[:, 0:1] * nx[None, :] + offsets[:, 1:2] * ny[None, :]
        abs_proj = np.abs(proj)
        max_projs = np.max(abs_proj, axis=0)
        side_vals = (2.0 / sqrt3) * max_projs
        smooth_side = soft_max(side_vals, beta)

        # Overlap penalty
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                h1 = (pos[i, 0], pos[i, 1], angles_rad[i])
                h2 = (pos[j, 0], pos[j, 1], angles_rad[j])
                if not hexagons_disjoint_fast(h1, h2):
                    penalty += 1.0
        return smooth_side + 10.0 * penalty

    # Initial Configuration: Honeycomb packing
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

    # === PHASE 1: Rigid-Body Relaxation ===
    def rigid_body_objective(params):
        tx, ty, theta_deg = params
        theta = np.radians(theta_deg)
        ct, st = math.cos(theta), math.sin(theta)
        new_pos = np.zeros((N, 2))
        for i in range(N):
            px, py = base_positions[i]
            new_pos[i, 0] = px * ct - py * st + tx
            new_pos[i, 1] = px * st + py * ct + ty
        new_angles = base_angles + theta_deg
        if not is_valid_configuration(new_pos, new_angles):
            return 1e6
        side, phi, cx, cy = compute_min_outer_side(new_pos, new_angles)
        return side

    rigid_candidates = [
        [0, 0, 0], [0, 0, 5], [0, 0, 10], [0, 0, 15],
        [0, 0, 30], [0, 0, 45], [0.1, 0, 0], [-0.1, 0, 0],
        [0, 0.1, 0], [0, -0.1, 0],
    ]
    for cand in rigid_candidates:
        if time.time() - t0 > DEADLINE * 0.25:
            break
        try:
            res = minimize(rigid_body_objective, cand, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-6})
            if res.fun < best_side and res.fun < 1e5:
                tx, ty, theta_deg = res.x
                theta = np.radians(theta_deg)
                ct, st = math.cos(theta), math.sin(theta)
                new_pos = np.zeros((N, 2))
                new_ang = base_angles + theta_deg
                for i in range(N):
                    px, py = base_positions[i]
                    new_pos[i, 0] = px * ct - py * st + tx
                    new_pos[i, 1] = px * st + py * ct + ty
                if is_valid_configuration(new_pos, new_ang):
                    best_side = res.fun
                    best_positions = new_pos
                    best_angles = new_ang
                    _, best_phi, best_cx, best_cy = compute_min_outer_side(new_pos, new_ang)
        except Exception:
            pass

    # === PHASE 2: SA with Individual Hexagon Moves ===
    num_restarts = 4
    time_per_restart = (DEADLINE * 0.65) / num_restarts

    for r in range(num_restarts):
        if time.time() - t0 > DEADLINE * 0.65:
            break

        current_positions = best_positions.copy()
        current_angles = best_angles.copy()
        if r > 0:
            current_positions += np.random.normal(0, 0.15, size=(N, 2))
            current_angles += np.random.normal(0, 5.0, size=N)

        current_side, current_phi, current_cx, current_cy = compute_min_outer_side(current_positions, current_angles)
        if current_side < best_side:
            best_side = current_side
            best_positions = current_positions.copy()
            best_angles = current_angles.copy()
            best_phi = current_phi
            best_cx = current_cx
            best_cy = current_cy

        T = 0.3
        T_min = 0.0005
        cooling = 0.9998
        step_size = 0.25
        step_decay = 0.99995
        step_count = 0
        max_steps = 30000

        while time.time() - t0 < DEADLINE * 0.65 and step_count < max_steps:
            step_count += 1
            idx = random.randint(0, N - 1)
            new_positions = current_positions.copy()
            new_angles = current_angles.copy()
            dx = random.gauss(0, step_size)
            dy = random.gauss(0, step_size)
            da = random.gauss(0, 6.0)
            new_positions[idx, 0] += dx
            new_positions[idx, 1] += dy
            new_angles[idx] += da

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
                if step_size < 0.01: step_size = 0.01
                continue

            new_side, new_phi, new_cx, new_cy = compute_min_outer_side(new_positions, new_angles)
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
            if step_size < 0.01: step_size = 0.01

    # === PHASE 3: Smooth gradient-based optimization ===
    if time.time() - t0 < DEADLINE - 30.0:
        # Build initial param vector
        init_params = []
        for i in range(N):
            init_params.extend([best_positions[i, 0], best_positions[i, 1], best_angles[i]])
        init_params.append(best_phi)

        # Try smooth optimization from a few starting points
        for attempt in range(3):
            if time.time() - t0 > DEADLINE - 15.0:
                break
            p0 = init_params[:]
            if attempt > 0:
                for k in range(len(p0)):
                    p0[k] += random.gauss(0, 0.05)

            try:
                res = minimize(smooth_outer_side, p0, method='L-BFGS-B',
                              options={'maxiter': 300, 'ftol': 1e-8})
                if res.fun < best_side + 0.1:  # Must be close to valid
                    # Extract and validate
                    new_pos = np.array([[res.x[i*3], res.x[i*3+1]] for i in range(N)])
                    new_ang = np.array([res.x[i*3+2] for i in range(N)])
                    new_phi = res.x[-1]
                    if is_valid_configuration(new_pos, new_ang):
                        side, phi, cx, cy = compute_min_outer_side(new_pos, new_ang)
                        if side < best_side:
                            best_side = side
                            best_positions = new_pos
                            best_angles = new_ang
                            best_phi = phi
                            best_cx = cx
                            best_cy = cy
            except Exception:
                pass

    # === PHASE 4: Coordinate Descent Polish (high precision) ===
    def polish_coordinate_descent():
        pos = best_positions.copy()
        ang = best_angles.copy()
        side, phi, cx, cy = compute_min_outer_side(pos, ang)

        step_sizes = [0.05, 0.03, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002]

        for ss in step_sizes:
            improved = True
            while improved and time.time() - t0 < DEADLINE - 3.0:
                improved = False
                for i in range(N):
                    for dx in [-ss, 0, ss]:
                        for dy in [-ss, 0, ss]:
                            for da in [-ss*20, 0, ss*20]:
                                if dx == 0 and dy == 0 and da == 0:
                                    continue
                                new_pos = pos.copy()
                                new_ang = ang.copy()
                                new_pos[i, 0] += dx
                                new_pos[i, 1] += dy
                                new_ang[i] += da

                                angles_rad = np.radians(new_ang)
                                valid = True
                                h_moved = (new_pos[i, 0], new_pos[i, 1], angles_rad[i])
                                for j in range(N):
                                    if j == i: continue
                                    h_other = (new_pos[j, 0], new_pos[j, 1], angles_rad[j])
                                    if not hexagons_disjoint_fast(h_moved, h_other):
                                        valid = False
                                        break
                                if not valid:
                                    continue
                                new_side, new_phi, new_cx, new_cy = compute_min_outer_side(new_pos, new_ang)
                                if new_side < side:
                                    pos = new_pos
                                    ang = new_ang
                                    side = new_side
                                    phi = new_phi
                                    cx = new_cx
                                    cy = new_cy
                                    improved = True
                                    if side < best_side:
                                        best_side = side
                                        best_positions = pos.copy()
                                        best_angles = ang.copy()
                                        best_phi = phi
                                        best_cx = cx
                                        best_cy = cy
                    if time.time() - t0 > DEADLINE - 3.0:
                        break
                if time.time() - t0 > DEADLINE - 3.0:
                    break

        return best_side, best_positions, best_angles, best_phi, best_cx, best_cy

    polish_time_budget = DEADLINE - (time.time() - t0)
    if polish_time_budget > 3.0:
        try:
            best_side, best_positions, best_angles, best_phi, best_cx, best_cy = polish_coordinate_descent()
        except Exception:
            pass

    # Final validation and computation
    if not is_valid_configuration(best_positions, best_angles):
        # Fallback to honeycomb
        best_positions = base_positions.copy()
        best_angles = base_angles.copy()
        best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)
    else:
        best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)

    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), float(best_angles[i])])

    center = (float(best_cx), float(best_cy))
    side = float(best_side)
    angle = float(best_phi)

    return inner, center, side, angle
