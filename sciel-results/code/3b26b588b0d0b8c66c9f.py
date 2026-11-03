import math
import time
import numpy as np
from scipy.optimize import minimize_scalar

N = 12
DEADLINE = 145.0
_SQRT3 = math.sqrt(3)

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

    phi_vals = np.radians(np.arange(0, 60, 1))
    normal_angles = phi_vals[:, None] + np.pi / 6 + np.arange(6) * np.pi / 3
    nx = np.cos(normal_angles)
    ny = np.sin(normal_angles)
    proj = offsets[:, 0:1, None] * nx[None, :, :] + offsets[:, 1:2, None] * ny[None, :, :]
    abs_proj = np.abs(proj)
    max_projs = np.max(abs_proj, axis=0)
    side_vals = (2.0 / _SQRT3) * max_projs
    best_side = np.max(side_vals).min()
    best_idx = np.argmin(np.max(side_vals, axis=1))
    best_phi = float(best_idx)

    try:
        lo = max(0.0, best_phi - 1.0)
        hi = min(60.0, best_phi + 1.0)
        res = minimize_scalar(eval_phi, bounds=(lo, hi), method='bounded',
                              options={'xatol': 1e-10, 'maxiter': 100})
        if res.success and res.fun < best_side:
            best_side = float(res.fun)
            best_phi = float(res.x)
    except Exception:
        pass

    return float(best_side), float(best_phi), float(center[0]), float(center[1])

def _polish(pos, ang, time_limit, initial_step=0.05):
    current_side, _, _, _ = _compute_min_outer_side(pos, ang)
    best_side = current_side
    best_pos = pos.copy()
    best_ang = ang.copy()

    t_start = time.time()
    step = initial_step
    min_step = 0.001

    while time.time() - t_start < time_limit and step > min_step:
        improved = False
        for idx in range(N):
            for dim in range(3):
                for direction in [1, -1]:
                    new_pos = best_pos.copy()
                    new_ang = best_ang.copy()

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
                        if j == idx:
                            continue
                        h_other = (new_pos[j, 0], new_pos[j, 1], angles_rad_new[j])
                        if not _hexagons_disjoint_fast(h_moved, h_other):
                            valid = False
                            break

                    if not valid:
                        continue

                    new_side, _, _, _ = _compute_min_outer_side(new_pos, new_ang)

                    if new_side < best_side - 1e-12:
                        best_pos = new_pos
                        best_ang = new_ang
                        best_side = new_side
                        improved = True
                        break

                if time.time() - t_start > time_limit:
                    break
            if time.time() - t_start > time_limit:
                break
            if improved:
                break
        else:
            step *= 0.7
            continue
        break

    return best_side, best_pos, best_ang

def construct():
    t0 = time.time()
    
    # Strategy: Fixed orientation (0 degrees) and a 2x6 grid layout.
    # This drastically reduces the search space from 36D to 1D (aspect ratio) + polish.
    # The 2x6 grid is known to be efficient for hexagonal packing.
    
    rows = 2
    cols = 6
    
    # Initial aspect ratio: distance between columns / distance between rows
    # For hexagons, ideal spacing is sqrt(3) horizontally and 1.5 vertically (if offset)
    # But for a rectangular grid of aligned hexagons:
    # Horizontal spacing: sqrt(3)
    # Vertical spacing: 2.0 (if touching) or 1.5 (if staggered).
    # Let's use a rectangular grid with spacing sqrt(3) and 2.0.
    # Aspect ratio r = dx / dy.
    
    def get_grid_pos(r):
        dx = _SQRT3
        dy = 2.0 / r
        pos = np.zeros((N, 2))
        for i in range(N):
            row = i // cols
            col = i % cols
            # Center the grid
            pos[i, 0] = (col - (cols - 1) / 2.0) * dx
            pos[i, 1] = (row - (rows - 1) / 2.0) * dy
        return pos

    ang = np.zeros(N)
    
    # Optimize aspect ratio r
    # r=1 means square grid. r>1 means wider columns.
    # We want to minimize the outer hexagon side.
    
    def objective(r):
        pos = get_grid_pos(r)
        side, _, _, _ = _compute_min_outer_side(pos, ang)
        return side

    # Search for optimal r
    # r in [0.5, 2.0]
    res = minimize_scalar(objective, bounds=(0.5, 2.0), method='bounded', options={'xatol': 1e-6})
    
    best_r = res.x
    best_pos = get_grid_pos(best_r)
    best_side, best_phi, best_cx, best_cy = _compute_min_outer_side(best_pos, ang)
    
    # Polish with coordinate descent
    remaining_time = DEADLINE - (time.time() - t0)
    if remaining_time > 5.0:
        polish_time = min(remaining_time * 0.6, 30.0)
        _, polished_pos, polished_ang = _polish(best_pos, ang, polish_time)
        if _is_valid(polished_pos, polished_ang):
            best_pos = polished_pos
            ang = polished_ang
            best_side, best_phi, best_cx, best_cy = _compute_min_outer_side(best_pos, ang)
            
    # Fallback if something went wrong
    if not _is_valid(best_pos, ang):
        # Default grid
        best_pos = get_grid_pos(1.0)
        ang = np.zeros(N)
        best_side, best_phi, best_cx, best_cy = _compute_min_outer_side(best_pos, ang)

    inner = []
    for i in range(N):
        inner.append([float(best_pos[i, 0]), float(best_pos[i, 1]), float(ang[i])])

    center = (float(best_cx), float(best_cy))
    side = float(best_side)
    angle = float(best_phi)

    return inner, center, side, angle
