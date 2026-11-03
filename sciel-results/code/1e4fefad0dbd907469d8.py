import math
import time
import random
import numpy as np

N = 12
DEADLINE = 145.0  # Safety margin before 150s wall-clock kill

# Precompute unit hexagon vertices and edge normals (angle 0)
_HEX_ANGLES = np.radians(np.arange(0, 360, 60))
_UNIT_HEX_VERTS = np.column_stack((np.cos(_HEX_ANGLES), np.sin(_HEX_ANGLES)))
_EDGE_NORMALS = np.zeros((6, 2))
for _i in range(6):
    _ex = _UNIT_HEX_VERTS[(_i + 1) % 6, 0] - _UNIT_HEX_VERTS[_i, 0]
    _ey = _UNIT_HEX_VERTS[(_i + 1) % 6, 1] - _UNIT_HEX_VERTS[_i, 1]
    _nx, _ny = -_ey, _ex
    _norm = math.sqrt(_nx * _nx + _ny * _ny)
    _EDGE_NORMALS[_i] = [_nx / _norm, _ny / _norm]

_UX = _UNIT_HEX_VERTS[:, 0]
_UY = _UNIT_HEX_VERTS[:, 1]
_ENX = _EDGE_NORMALS[:, 0]
_ENY = _EDGE_NORMALS[:, 1]
_SQRT3 = math.sqrt(3)


def _hexagons_disjoint_fast(h1, h2):
    dx = h1[0] - h2[0]
    dy = h1[1] - h2[1]
    dist_sq = dx * dx + dy * dy
    if dist_sq > 4.5:
        return True
    a1 = h1[2]
    a2 = h2[2]
    c1, s1 = math.cos(a1), math.sin(a1)
    c2, s2 = math.cos(a2), math.sin(a2)
    v1x = h1[0] + _UX * c1 - _UY * s1
    v1y = h1[1] + _UX * s1 + _UY * c1
    v2x = h2[0] + _UX * c2 - _UY * s2
    v2y = h2[1] + _UX * s2 + _UY * c2
    en1x = _ENX * c1 - _ENY * s1
    en1y = _ENX * s1 + _ENY * c1
    en2x = _ENX * c2 - _ENY * s2
    en2y = _ENX * s2 + _ENY * c2
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


def _is_valid(positions, angles_deg):
    angles_rad = np.radians(angles_deg)
    for i in range(N):
        for j in range(i + 1, N):
            h1 = (positions[i, 0], positions[i, 1], angles_rad[i])
            h2 = (positions[j, 0], positions[j, 1], angles_rad[j])
            if not _hexagons_disjoint_fast(h1, h2):
                return False
    return True


def _compute_min_outer_side(positions, angles_deg):
    angles_rad = np.radians(angles_deg)
    c = np.cos(angles_rad)
    s = np.sin(angles_rad)
    rx = _UX[None, :] * c[:, None] - _UY[None, :] * s[:, None]
    ry = _UX[None, :] * s[:, None] + _UY[None, :] * c[:, None]
    verts_x = rx + positions[:, 0:1]
    verts_y = ry + positions[:, 1:2]
    verts_flat = np.stack((verts_x, verts_y), axis=-1).reshape(-1, 2)
    center = np.mean(verts_flat, axis=0)
    offsets = verts_flat - center

    def eval_phi(phi_deg):
        phi = np.radians(phi_deg)
        normal_angles = phi + np.pi / 6 + np.arange(6) * np.pi / 3
        nx = np.cos(normal_angles)
        ny = np.sin(normal_angles)
        proj = offsets[:, 0:1] * nx[None, :] + offsets[:, 1:2] * ny[None, :]
        abs_proj = np.abs(proj)
        max_projs = np.max(abs_proj, axis=0)
        side_vals = (2.0 / _SQRT3) * max_projs
        return np.max(side_vals)

    # Vectorized coarse scan
    phi_vals = np.radians(np.arange(0, 60, 1))
    normal_angles = phi_vals[:, None] + np.pi / 6 + np.arange(6) * np.pi / 3
    nx = np.cos(normal_angles)
    ny = np.sin(normal_angles)
    proj = offsets[:, 0:1, None] * nx[None, :, :] + offsets[:, 1:2, None] * ny[None, :, :]
    abs_proj = np.abs(proj)
    max_projs = np.max(abs_proj, axis=0)  # (60, 6)
    side_vals = (2.0 / _SQRT3) * max_projs
    best_side = np.max(side_vals).min()
    best_idx = np.argmin(np.max(side_vals, axis=1))
    best_phi = float(best_idx)

    # Fine tune around best
    try:
        lo = max(0.0, best_phi - 1.0)
        hi = min(60.0, best_phi + 1.0)
        from scipy.optimize import minimize_scalar
        res = minimize_scalar(eval_phi, bounds=(lo, hi), method='bounded',
                              options={'xatol': 1e-10, 'maxiter': 100})
        if res.success and res.fun < best_side:
            best_side = float(res.fun)
            best_phi = float(res.x)
    except Exception:
        pass

    return float(best_side), float(best_phi), float(center[0]), float(center[1])


def _push_apart(pos, ang, max_iter=20):
    """
    Iteratively resolve overlaps by pushing hexagons apart along the axis of minimum penetration.
    Returns (new_pos, new_ang, success)
    """
    new_pos = pos.copy()
    new_ang = ang.copy()
    
    for _ in range(max_iter):
        angles_rad = np.radians(new_ang)
        moved = False
        for i in range(N):
            for j in range(i + 1, N):
                h1 = (new_pos[i, 0], new_pos[i, 1], angles_rad[i])
                h2 = (new_pos[j, 0], new_pos[j, 1], angles_rad[j])
                
                # Quick check
                if _hexagons_disjoint_fast(h1, h2):
                    continue
                
                # They overlap. Find the axis of minimum separation (max penetration)
                c1, s1 = math.cos(angles_rad[i]), math.sin(angles_rad[i])
                c2, s2 = math.cos(angles_rad[j]), math.sin(angles_rad[j])
                
                v1x = new_pos[i, 0] + _UX * c1 - _UY * s1
                v1y = new_pos[i, 1] + _UX * s1 + _UY * c1
                v2x = new_pos[j, 0] + _UX * c2 - _UY * s2
                v2y = new_pos[j, 1] + _UX * s2 + _UY * c2
                
                en1x = _ENX * c1 - _ENY * s1
                en1y = _ENX * s1 + _ENY * c1
                en2x = _ENX * c2 - _ENY * s2
                en2y = _ENX * s2 + _ENY * c2
                
                min_gap = float('inf')
                best_axis = None
                best_dir = 0 # 1 means push i away from j, -1 means push j away from i? 
                             # Actually we push both apart. Let's define axis normal pointing from j to i.
                             
                # Check normals of hex 1
                for k in range(6):
                    nx, ny = en1x[k], en1y[k]
                    p1 = v1x * nx + v1y * ny
                    p2 = v2x * nx + v2y * ny
                    # If p1 is "above" p2 (p1.min > p2.max), they are separated.
                    # We want to find where they overlap the most.
                    # Overlap on this axis exists if p1.min < p2.max and p2.min < p1.max.
                    # The "gap" is min(p2.min - p1.max, p1.min - p2.max).
                    # If gap < 0, they overlap. We want to maximize the gap (make it least negative).
                    gap1 = p2.min() - p1.max()
                    gap2 = p1.min() - p2.max()
                    gap = min(gap1, gap2)
                    if gap < min_gap:
                        min_gap = gap
                        # Axis is (nx, ny). 
                        # If gap1 was the min, it means p2.min < p1.max. We should push 1 in +n direction and 2 in -n direction.
                        # If gap2 was the min, it means p1.min < p2.max. We should push 1 in -n direction and 2 in +n direction.
                        if gap1 <= gap2:
                            best_axis = (nx, ny)
                            best_dir = 1
                        else:
                            best_axis = (nx, ny)
                            best_dir = -1

                # Check normals of hex 2
                for k in range(6):
                    nx, ny = en2x[k], en2y[k]
                    p1 = v1x * nx + v1y * ny
                    p2 = v2x * nx + v2y * ny
                    gap1 = p2.min() - p1.max()
                    gap2 = p1.min() - p2.max()
                    gap = min(gap1, gap2)
                    if gap < min_gap:
                        min_gap = gap
                        if gap1 <= gap2:
                            best_axis = (nx, ny)
                            best_dir = 1
                        else:
                            best_axis = (nx, ny)
                            best_dir = -1

                if best_axis is not None:
                    # Push apart by a small amount (1.5 * penetration)
                    push = 1.5 * (-min_gap)
                    if push < 0.01:
                        push = 0.01
                    nx, ny = best_axis
                    if best_dir == 1:
                        new_pos[i, 0] += nx * push
                        new_pos[i, 1] += ny * push
                        new_pos[j, 0] -= nx * push
                        new_pos[j, 1] -= ny * push
                    else:
                        new_pos[i, 0] -= nx * push
                        new_pos[i, 1] -= ny * push
                        new_pos[j, 0] += nx * push
                        new_pos[j, 1] += ny * push
                    moved = True

        if not moved:
            return new_pos, new_ang, True
            
    return new_pos, new_ang, False


def construct():
    t0 = time.time()

    # Generate diverse initial configurations
    a1 = np.array([_SQRT3, 0.0])
    a2 = np.array([_SQRT3 / 2.0, 1.5])

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
            ang[idx] = 0.0
        return pos, ang

    def make_ring():
        pos = np.zeros((N, 2))
        ang = np.zeros(N)
        for i in range(6):
            theta = i * np.pi / 3
            pos[i, 0] = 2.5 * _SQRT3 * np.cos(theta)
            pos[i, 1] = 2.5 * _SQRT3 * np.sin(theta)
        for i in range(6):
            theta = (i + 0.5) * np.pi / 3
            pos[6 + i, 0] = 1.2 * _SQRT3 * np.cos(theta)
            pos[6 + i, 1] = 1.2 * _SQRT3 * np.sin(theta)
        return pos, ang

    def make_two_rows():
        pos = np.zeros((N, 2))
        ang = np.zeros(N)
        for i in range(6):
            pos[i, 0] = i * _SQRT3
            pos[i, 1] = 0.0
        for i in range(6):
            pos[6 + i, 0] = i * _SQRT3 + _SQRT3 / 2
            pos[6 + i, 1] = 1.5
        return pos, ang

    def make_random():
        pos = np.random.uniform(-3, 3, (N, 2))
        ang = np.random.uniform(-30, 30, N)
        return pos, ang

    # Try multiple initializations
    inits = [
        make_honeycomb(),
        make_ring(),
        make_two_rows(),
        make_random(),
        make_random(),
    ]

    best_side = float('inf')
    best_positions = None
    best_angles = None
    best_phi = 0.0
    best_cx = 0.0
    best_cy = 0.0

    for init_idx, (pos0, ang0) in enumerate(inits):
        if time.time() - t0 > DEADLINE - 10.0:
            break

        # Check initial validity
        if not _is_valid(pos0, ang0):
            # Perturb to make valid
            for _ in range(100):
                if time.time() - t0 > DEADLINE - 10.0:
                    break
                pos0 = pos0 + np.random.uniform(-0.1, 0.1, (N, 2))
                if _is_valid(pos0, ang0):
                    break

        if not _is_valid(pos0, ang0):
            continue

        # Evaluate initial
        s0, phi0, cx0, cy0 = _compute_min_outer_side(pos0, ang0)
        if s0 < best_side:
            best_side = s0
            best_positions = pos0.copy()
            best_angles = ang0.copy()
            best_phi = phi0
            best_cx = cx0
            best_cy = cy0

        # Simulated Annealing
        pos = pos0.copy()
        ang = ang0.copy()
        cur_side, _, _, _ = _compute_min_outer_side(pos, ang)
        
        T = 0.5
        T_min = 0.001
        alpha = 0.99
        steps = 0
        
        # Time limit for this SA run: 20 seconds per init roughly
        time_per_init = (DEADLINE - 15.0) / max(1, len(inits) - init_idx)
        t_init_start = time.time()

        while time.time() - t_init_start < time_per_init:
            steps += 1
            # Pick a random hexagon and move it
            idx = random.randint(0, N - 1)
            dx = random.uniform(-0.5, 0.5)
            dy = random.uniform(-0.5, 0.5)
            da = random.uniform(-10, 10)
            
            new_pos = pos.copy()
            new_ang = ang.copy()
            new_pos[idx, 0] += dx
            new_pos[idx, 1] += dy
            new_ang[idx] += da
            
            # Repair overlaps
            new_pos, new_ang, success = _push_apart(new_pos, new_ang)
            
            if not success:
                # If repair fails, try smaller moves or skip
                continue

            new_side, new_phi, new_cx, new_cy = _compute_min_outer_side(new_pos, new_ang)
            
            delta = new_side - cur_side
            if delta < 0 or random.random() < math.exp(-delta / T):
                pos = new_pos
                ang = new_ang
                cur_side = new_side
                # Update global best
                if cur_side < best_side:
                    best_side = cur_side
                    best_positions = pos.copy()
                    best_angles = ang.copy()
                    best_phi = new_phi
                    best_cx = new_cx
                    best_cy = new_cy

            T *= alpha
            if T < T_min:
                break

    # Final polish: Coordinate Descent on the best found
    remaining_time = DEADLINE - (time.time() - t0)
    if remaining_time > 5.0 and best_positions is not None:
        # Simple local search polish
        pos = best_positions.copy()
        ang = best_angles.copy()
        cur_side = best_side
        step = 0.1
        for _ in range(100): # Max iterations
            if time.time() - t0 > DEADLINE - 5.0:
                break
            improved = False
            for idx in range(N):
                for dim in range(3):
                    for direction in [1, -1]:
                        new_pos = pos.copy()
                        new_ang = ang.copy()
                        if dim == 0:
                            new_pos[idx, 0] += direction * step
                        elif dim == 1:
                            new_pos[idx, 1] += direction * step
                        else:
                            new_ang[idx] += direction * step

                        # Repair
                        new_pos, new_ang, success = _push_apart(new_pos, new_ang)
                        if not success:
                            continue

                        new_side, new_phi, new_cx, new_cy = _compute_min_outer_side(new_pos, new_ang)
                        if new_side < cur_side - 1e-10:
                            pos = new_pos
                            ang = new_ang
                            cur_side = new_side
                            best_side = cur_side
                            best_positions = pos.copy()
                            best_angles = ang.copy()
                            best_phi = new_phi
                            best_cx = new_cx
                            best_cy = new_cy
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                step *= 0.8
                if step < 0.001:
                    break

    # Final verification
    if not _is_valid(best_positions, best_angles):
        pos0, ang0 = make_honeycomb()
        best_positions = pos0.copy()
        best_angles = ang0.copy()
        best_side, best_phi, best_cx, best_cy = _compute_min_outer_side(best_positions, best_angles)

    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), float(best_angles[i])])

    center = (float(best_cx), float(best_cy))
    side = float(best_side)
    angle = float(best_phi)

    return inner, center, side, angle
