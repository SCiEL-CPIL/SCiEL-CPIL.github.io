import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12
DEADLINE = 148.0
SQRT3 = math.sqrt(3)

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


def _hex_disjoint_fast(h1, h2):
    dx = h1[0] - h2[0]
    dy = h1[1] - h2[1]
    dist_sq = dx * dx + dy * dy
    if dist_sq > 4.5:
        return True
    a1, a2 = h1[2], h2[2]
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
            if not _hex_disjoint_fast(h1, h2):
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
        side_vals = (2.0 / SQRT3) * max_projs
        return float(np.max(side_vals))

    phi_vals = np.radians(np.arange(0, 60, 0.5))
    normal_angles = phi_vals[:, None] + np.pi / 6 + np.arange(6) * np.pi / 3
    nx = np.cos(normal_angles)
    ny = np.sin(normal_angles)
    proj = offsets[:, 0:1, None] * nx[None, :, :] + offsets[:, 1:2, None] * ny[None, :, :]
    abs_proj = np.abs(proj)
    max_projs = np.max(abs_proj, axis=0)
    side_vals = (2.0 / SQRT3) * max_projs
    best_side = float(np.max(side_vals).min())
    best_idx = int(np.argmin(np.max(side_vals, axis=1)))
    best_phi = float(best_idx * 0.5)

    lo = max(0.0, best_phi - 0.5)
    hi = min(60.0, best_phi + 0.5)
    gr = (math.sqrt(5) - 1) / 2
    x1 = hi - gr * (hi - lo)
    x2 = lo + gr * (hi - lo)
    f1 = eval_phi(x1)
    f2 = eval_phi(x2)
    for _ in range(50):
        if f1 < f2:
            hi = x2
            x2 = x1
            f2 = f1
            x1 = hi - gr * (hi - lo)
            f1 = eval_phi(x1)
        else:
            lo = x1
            x1 = x2
            f1 = f2
            x2 = lo + gr * (hi - lo)
            f2 = eval_phi(x2)
    best_phi = (lo + hi) / 2.0
    best_side = min(best_side, eval_phi(best_phi))

    return best_side, best_phi, float(center[0]), float(center[1])


def _make_honeycomb():
    a1 = np.array([SQRT3, 0.0])
    a2 = np.array([SQRT3 / 2.0, 1.5])
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


def _make_ring():
    pos = np.zeros((N, 2))
    ang = np.zeros(N)
    for i in range(6):
        theta = i * np.pi / 3
        pos[i, 0] = 2.5 * SQRT3 * np.cos(theta)
        pos[i, 1] = 2.5 * SQRT3 * np.sin(theta)
    for i in range(6):
        theta = (i + 0.5) * np.pi / 3
        pos[6 + i, 0] = 1.2 * SQRT3 * np.cos(theta)
        pos[6 + i, 1] = 1.2 * SQRT3 * np.sin(theta)
    return pos, ang


def _make_two_rows():
    pos = np.zeros((N, 2))
    ang = np.zeros(N)
    for i in range(6):
        pos[i, 0] = i * SQRT3
        pos[i, 1] = 0.0
    for i in range(6):
        pos[6 + i, 0] = i * SQRT3 + SQRT3 / 2
        pos[6 + i, 1] = 1.5
    return pos, ang


def _make_compact_hex_cluster():
    a1 = np.array([SQRT3, 0.0])
    a2 = np.array([SQRT3 / 2.0, 1.5])
    coords = [
        (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1),
        (2, -1), (-2, 1), (2, 0), (-2, 0), (0, 2),
    ]
    pos = np.zeros((N, 2))
    ang = np.zeros(N)
    for idx, (i, j) in enumerate(coords):
        pos[idx] = i * a1 + j * a2
        ang[idx] = 0.0
    return pos, ang


def _make_random():
    pos = np.random.uniform(-3, 3, (N, 2))
    ang = np.random.uniform(-30, 30, N)
    return pos, ang


def _pos_ang_to_params(pos, ang_deg, phi_deg):
    params = np.zeros(37)
    for i in range(N):
        params[3 * i] = pos[i, 0]
        params[3 * i + 1] = pos[i, 1]
        params[3 * i + 2] = math.radians(ang_deg[i])
    params[36] = math.radians(phi_deg)
    return params


def _params_to_pos_ang(params):
    pos = np.zeros((N, 2))
    ang_deg = np.zeros(N)
    for i in range(N):
        pos[i, 0] = params[3 * i]
        pos[i, 1] = params[3 * i + 1]
        ang_deg[i] = math.degrees(params[3 * i + 2])
    phi_deg = math.degrees(params[36])
    return pos, ang_deg, phi_deg


def _try_fix_validity(pos, ang, max_iters=200):
    for _ in range(max_iters):
        if _is_valid(pos, ang):
            return True
        idx = np.random.randint(N)
        pos[idx, 0] += np.random.uniform(-0.15, 0.15)
        pos[idx, 1] += np.random.uniform(-0.15, 0.15)
        ang[idx] += np.random.uniform(-5, 5)
    return _is_valid(pos, ang)


def _objective_and_constraints(params):
    """
    params: (37,) = [x0,y0,a0, ..., x11,y11,a11, phi]
    Returns (objective, constraint_values) where constraint_values[k] >= 0 means feasible.
    Constraint: min separation between hex k1 and k2 >= 0.
    """
    positions = params[0:36:3]
    pys = params[1:36:3]
    angles = params[2:36:3]
    phi = params[36]

    # Outer hexagon side length (exact)
    c = np.cos(angles)
    s = np.sin(angles)
    rx = _UX[None, :] * c[:, None] - _UY[None, :] * s[:, None]
    ry = _UX[None, :] * s[:, None] + _UY[None, :] * c[:, None]
    verts_x = rx + positions[:, None]
    verts_y = ry + pys[:, None]
    verts_flat = np.stack((verts_x, verts_y), axis=-1).reshape(-1, 2)
    center = np.mean(verts_flat, axis=0)
    offsets = verts_flat - center

    normal_angles = phi + np.pi / 6 + np.arange(6) * np.pi / 3
    nx = np.cos(normal_angles)
    ny = np.sin(normal_angles)
    proj = offsets[:, 0:1] * nx[None, :] + offsets[:, 1:2] * ny[None, :]
    abs_proj = np.abs(proj)
    max_projs = np.max(abs_proj, axis=0)
    side_vals = (2.0 / SQRT3) * max_projs
    obj = float(np.max(side_vals))

    # Constraints: for each pair, min separation >= 0
    # We compute the minimum distance between the two hexagons using separating axes.
    # If the minimum distance is negative, they overlap; constraint value = min_dist.
    # To make it smooth for SLSQP, we use a softplus-like approximation:
    # g = min_dist + epsilon * log(1 + exp(-min_dist / epsilon))
    # But SLSQP can handle non-smooth constraints if we provide gradients.
    # Instead, we'll use a simple hinge: g = max(0, -min_dist) is not differentiable.
    # Better: use the actual min_dist as the constraint value. SLSQP can handle it.
    # Actually, scipy SLSQP expects constraints to be of form g(x) >= 0 (inequality) or h(x) = 0 (equality).
    # We want min_dist >= 0. So constraint value = min_dist.

    constraints = []
    for i in range(N):
        for j in range(i + 1, N):
            # Compute min distance between hex i and hex j
            # Using separating axis theorem: min over all axes of (max_i - min_j) where axis is from either hex's edges
            # But for two convex polygons, the minimum distance is the minimum over all separating axes of the gap.
            # If they overlap, the gap is negative.
            a1, a2 = angles[i], angles[j]
            c1, s1 = math.cos(a1), math.sin(a1)
            c2, s2 = math.cos(a2), math.sin(a2)

            v1x = positions[i] + _UX * c1 - _UY * s1
            v1y = pys[i] + _UX * s1 + _UY * c1
            v2x = positions[j] + _UX * c2 - _UY * s2
            v2y = pys[j] + _UX * s2 + _UY * c2

            en1x = _ENX * c1 - _ENY * s1
            en1y = _ENX * s1 + _ENY * c1
            en2x = _ENX * c2 - _ENY * s2
            en2y = _ENX * s2 + _ENY * c2

            min_dist = float('inf')
            # Check all 12 axes (6 from each hex)
            for k in range(6):
                # Axis from hex 1
                nx, ny = en1x[k], en1y[k]
                p1 = v1x * nx + v1y * ny
                p2 = v2x * nx + v2y * ny
                gap = min(p1.max() - p2.min(), p2.max() - p1.min())
                if gap < min_dist:
                    min_dist = gap
                # Axis from hex 2
                nx, ny = en2x[k], en2y[k]
                p1 = v1x * nx + v1y * ny
                p2 = v2x * nx + v2y * ny
                gap = min(p1.max() - p2.min(), p2.max() - p1.min())
                if gap < min_dist:
                    min_dist = gap

            # If min_dist < 0, they overlap. Constraint: min_dist >= 0
            # For SLSQP, we provide the constraint value as min_dist.
            # But SLSQP needs differentiable constraints. The min over axes is not differentiable.
            # Workaround: use a smooth approximation. We'll use a log-sum-exp of the negative gaps.
            # Actually, let's just use min_dist and hope SLSQP can handle it with finite differences.
            # Better: provide a smooth penalty in the objective and use simple bounds.
            # Alternative: use a penalty method. Add lambda * sum(max(0, -min_dist)^2) to objective.
            # This is simpler and more robust.

    # Instead of constraints, use penalty in objective
    penalty = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            a1, a2 = angles[i], angles[j]
            c1, s1 = math.cos(a1), math.sin(a1)
            c2, s2 = math.cos(a2), math.sin(a2)

            v1x = positions[i] + _UX * c1 - _UY * s1
            v1y = pys[i] + _UX * s1 + _UY * c1
            v2x = positions[j] + _UX * c2 - _UY * s2
            v2y = pys[j] + _UX * s2 + _UY * c2

            en1x = _ENX * c1 - _ENY * s1
            en1y = _ENX * s1 + _ENY * c1
            en2x = _ENX * c2 - _ENY * s2
            en2y = _ENX * s2 + _ENY * c2

            min_dist = float('inf')
            for k in range(6):
                nx, ny = en1x[k], en1y[k]
                p1 = v1x * nx + v1y * ny
                p2 = v2x * nx + v2y * ny
                gap = min(p1.max() - p2.min(), p2.max() - p1.min())
                if gap < min_dist:
                    min_dist = gap
                nx, ny = en2x[k], en2y[k]
                p1 = v1x * nx + v1y * ny
                p2 = v2x * nx + v2y * ny
                gap = min(p1.max() - p2.min(), p2.max() - p1.min())
                if gap < min_dist:
                    min_dist = gap

            if min_dist < 0:
                penalty += min_dist * min_dist

    lam = 50.0
    total = obj + lam * penalty
    return total


def _slsqp_optimize(params0, t_start, time_limit, rng):
    """
    Run SLSQP optimization on the 37D parameter vector.
    """
    best_params = params0.copy()
    best_f = _objective_and_constraints(params0)

    # Bounds for parameters
    # Positions: [-5, 5], Angles: [-pi, pi], Phi: [0, 2*pi/3]
    bounds = []
    for i in range(N):
        bounds.append((-5.0, 5.0))  # x
        bounds.append((-5.0, 5.0))  # y
        bounds.append((-math.pi, math.pi))  # angle
    bounds.append((0.0, 2 * math.pi / 3))  # phi

    # Use SLSQP with penalty method (no explicit constraints)
    try:
        result = minimize(
            _objective_and_constraints,
            params0,
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-8, 'disp': False},
            callback=lambda x: None,
        )
        if result.success and result.fun < best_f:
            best_params = result.x
            best_f = result.fun
    except Exception:
        pass

    return best_params, best_f


def construct():
    t0 = time.time()
    rng = np.random.RandomState(42)

    inits = [
        _make_honeycomb(),
        _make_ring(),
        _make_two_rows(),
        _make_compact_hex_cluster(),
        _make_random(),
        _make_random(),
    ]

    best_side = float('inf')
    best_positions = None
    best_angles = None
    best_phi = 0.0
    best_cx = 0.0
    best_cy = 0.0

    for init_idx, (pos0, ang0) in enumerate(inits):
        if time.time() - t0 > DEADLINE - 20.0:
            break

        # Ensure validity
        if not _is_valid(pos0, ang0):
            _try_fix_validity(pos0, ang0, max_iters=300)
            if not _is_valid(pos0, ang0):
                continue

        s0, phi0, cx0, cy0 = _compute_min_outer_side(pos0, ang0)
        if s0 < best_side:
            best_side = s0
            best_positions = pos0.copy()
            best_angles = ang0.copy()
            best_phi = phi0
            best_cx = cx0
            best_cy = cy0

        # SLSQP optimization
        remaining = DEADLINE - 25.0 - (time.time() - t0)
        if remaining < 5.0:
            break

        params0 = _pos_ang_to_params(pos0, ang0, phi0)
        t_slsqp_start = time.time()
        slsqp_time_limit = min(remaining * 0.7, 60.0)

        slsqp_params, slsqp_f = _slsqp_optimize(params0, t_slsqp_start, slsqp_time_limit, rng)

        slsqp_pos, slsqp_ang, slsqp_phi = _params_to_pos_ang(slsqp_params)
        if _is_valid(slsqp_pos, slsqp_ang):
            slsqp_side, slsqp_phi_opt, slsqp_cx, slsqp_cy = _compute_min_outer_side(slsqp_pos, slsqp_ang)
            if slsqp_side < best_side:
                best_side = slsqp_side
                best_positions = slsqp_pos.copy()
                best_angles = slsqp_ang.copy()
                best_phi = slsqp_phi_opt
                best_cx = slsqp_cx
                best_cy = slsqp_cy

    # Final verification
    if best_positions is None or not _is_valid(best_positions, best_angles):
        pos0, ang0 = _make_honeycomb()
        _try_fix_validity(pos0, ang0, max_iters=500)
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
