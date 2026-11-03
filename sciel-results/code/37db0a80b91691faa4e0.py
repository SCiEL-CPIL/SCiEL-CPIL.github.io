import math
import time
import numpy as np

N = 12
DEADLINE = 145.0
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


def _make_random(rng):
    pos = rng.uniform(-3, 3, (N, 2))
    ang = rng.uniform(-30, 30, N)
    return pos, ang


def _try_fix_validity(pos, ang, max_iters=200, rng=None):
    if rng is None:
        rng = np.random.RandomState()
    for _ in range(max_iters):
        if _is_valid(pos, ang):
            return True
        idx = rng.randint(N)
        pos[idx, 0] += rng.uniform(-0.15, 0.15)
        pos[idx, 1] += rng.uniform(-0.15, 0.15)
        ang[idx] += rng.uniform(-5, 5)
    return _is_valid(pos, ang)


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


def _logsumexp(values, beta=20.0):
    """Smooth max via log-sum-exp."""
    v = np.asarray(values)
    max_v = np.max(v)
    if max_v == -np.inf:
        return 0.0
    shifted = v - max_v
    return max_v + math.log(np.sum(np.exp(beta * shifted))) / beta


def _smooth_objective(params, beta_lse=20.0, beta_pen=15.0):
    """
    params: (37,) = [x0,y0,a0, ..., x11,y11,a11, phi]
    Returns a smooth scalar objective combining outer side length and collision penalties.
    """
    positions = params[0:36:3]
    pys = params[1:36:3]
    angles = params[2:36:3]
    phi = params[36]

    # Compute all vertices
    c = np.cos(angles)
    s = np.sin(angles)
    rx = _UX[None, :] * c[:, None] - _UY[None, :] * s[:, None]
    ry = _UX[None, :] * s[:, None] + _UY[None, :] * c[:, None]
    verts_x = rx + positions[:, None]
    verts_y = ry + pys[:, None]
    verts_flat = np.stack((verts_x, verts_y), axis=-1).reshape(-1, 2)
    center = np.mean(verts_flat, axis=0)
    offsets = verts_flat - center

    # Outer hexagon side length: smooth max over 6 directions
    normal_angles = phi + np.pi / 6 + np.arange(6) * np.pi / 3
    nx = np.cos(normal_angles)
    ny = np.sin(normal_angles)
    proj = offsets[:, 0:1] * nx[None, :] + offsets[:, 1:2] * ny[None, :]
    abs_proj = np.abs(proj)
    max_projs = np.max(abs_proj, axis=0)
    side_vals = (2.0 / SQRT3) * max_projs
    # Use smooth max over the 6 side values
    obj = _logsumexp(side_vals, beta_lse)

    # Collision penalty: smooth sum of squared overlaps
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

            # Collect all 12 axis gaps
            gaps = np.empty(12)
            for k in range(6):
                nx, ny = en1x[k], en1y[k]
                p1 = v1x * nx + v1y * ny
                p2 = v2x * nx + v2y * ny
                gaps[k] = min(p1.max() - p2.min(), p2.max() - p1.min())
                nx, ny = en2x[k], en2y[k]
                p1 = v1x * nx + v1y * ny
                p2 = v2x * nx + v2y * ny
                gaps[6 + k] = min(p1.max() - p2.min(), p2.max() - p1.min())

            # min_dist is the minimum gap; if negative, overlap
            # Use smooth penalty: for each axis, if gap < 0, add (gap)^2
            # To keep it smooth, use softplus on -gap for each axis, then sum
            neg_gaps = -gaps  # positive where overlap
            # Smooth penalty per axis: softplus-like
            pen_axis = np.zeros(12)
            for k in range(12):
                if neg_gaps[k] > 0:
                    # Quadratic penalty for overlap
                    pen_axis[k] = neg_gaps[k] * neg_gaps[k]
                else:
                    # Small smooth contribution from near-misses
                    pen_axis[k] = 0.0
            # Use logsumexp for smoothness across axes
            penalty += _logsumexp(pen_axis, beta_pen)

    lam = 80.0
    return obj + lam * penalty


def _cma_es_optimize(params0, t_start, time_limit, rng):
    """
    Simple CMA-ES implementation for the 37D problem.
    """
    n = len(params0)
    # Population size
    lambda_pop = max(10, int(4 + 3 * math.log(n)))
    # Initial sigma
    sigma = 0.3
    # CMA-ES parameters
    cc = (4 + n / lambda_pop) / (n + 4 + 2 * lambda_pop / n)
    cs = (lambda_pop / 2 + 1) / (n + lambda_pop / 2 + 2)
    cmu = min(1 - cc - 2 * cs, 2 / ((n + 1.3) ** 2 + lambda_pop))
    c1 = (2 + 2 * cmu / lambda_pop) / ((n + 1.6) ** 2 + lambda_pop)
    cmu_c = cmu / ((n + 1.6) ** 2 + lambda_pop)

    mean = params0.copy()
    C = np.eye(n)
    C_inv = np.eye(n)
    C_sqrt = np.eye(n)
    C_inv_sqrt = np.eye(n)
    p_sigma = np.zeros(n)
    p_c = np.zeros(n)
    chi_n = math.sqrt(n) * (1 - 1. / (4 * n) + 1. / (21 * n * n))
    best_params = params0.copy()
    best_f = _smooth_objective(params0)

    # Bounds
    x_l = np.full(n, -5.0)
    x_u = np.full(n, 5.0)
    # Angle bounds: indices 2,5,8,...,35 are angles in radians
    for i in range(N):
        x_l[3 * i + 2] = -math.pi
        x_u[3 * i + 2] = math.pi
    # Phi bound
    x_l[36] = 0.0
    x_u[36] = 2 * math.pi / 3

    def _apply_bounds(x):
        return np.clip(x, x_l, x_u)

    def _mat_sqrt(mat):
        w, v = np.linalg.eigh(mat)
        w = np.clip(w, 1e-12, None)
        return v @ np.diag(np.sqrt(w)) @ v.T

    def _mat_inv_sqrt(mat):
        w, v = np.linalg.eigh(mat)
        w = np.clip(w, 1e-12, None)
        return v @ np.diag(1.0 / np.sqrt(w)) @ v.T

    C_sqrt = _mat_sqrt(C)
    C_inv_sqrt = _mat_inv_sqrt(C)

    iteration = 0
    while time.time() - t_start < time_limit:
        iteration += 1
        # Generate population
        offspring = np.zeros((lambda_pop, n))
        for i in range(lambda_pop):
            z = rng.standard_normal(n)
            offspring[i] = mean + sigma * (C_sqrt @ z)
            offspring[i] = _apply_bounds(offspring[i])

        # Evaluate
        fitness = np.array([_smooth_objective(offspring[i]) for i in range(lambda_pop)])

        # Sort
        order = np.argsort(fitness)
        mu = max(1, lambda_pop // 2)
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mueff = 1.0 / np.sum(w ** 2)

        # Update mean
        y_w = np.zeros(n)
        for i in range(mu):
            y_w += w[i] * offspring[order[i]]
        mean_new = mean + y_w

        # Update evolution paths
        y_w_sigma = (C_inv_sqrt @ y_w) / sigma
        p_sigma = (1 - cs) * p_sigma + math.sqrt(cs * (2 - cs) * mueff) * y_w_sigma
        hsig = (np.linalg.norm(p_sigma) / math.sqrt(1 - (1 - cs) ** (iteration + 1))) / chi_n < 1.4 + 2. / (n + 1)
        if hsig:
            p_c = (1 - cc) * p_c + cc * y_w_sigma
        else:
            p_c = (1 - cc) * p_c

        # Update covariance
        delta_hsig = (1.0 if hsig else 0.0) - (1 - cc)
        C = (1 + c1 + cmu_c - c1) * C + c1 * (np.outer(p_c, p_c) - delta_hsig * C)
        for i in range(mu):
            y_i = C_inv_sqrt @ (offspring[order[i]] - mean) / sigma
            C += cmu * w[i] * np.outer(y_i, y_i)

        # Symmetrize
        C = (C + C.T) / 2.0

        # Renormalize C to keep det(C) = 1
        det_C = np.linalg.det(C)
        if det_C > 0:
            C = C * (1.0 / det_C) ** (1.0 / n)

        # Update sigma
        sigma = sigma * math.exp(cs / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
        sigma = max(1e-6, min(sigma, 5.0))

        # Update matrices
        C_sqrt = _mat_sqrt(C)
        C_inv_sqrt = _mat_inv_sqrt(C)
        mean = mean_new

        # Track best
        best_idx = order[0]
        if fitness[best_idx] < best_f:
            best_f = fitness[best_idx]
            best_params = offspring[best_idx].copy()

    return best_params, best_f


def construct():
    t0 = time.time()
    rng = np.random.RandomState(42)

    inits = [
        _make_honeycomb(),
        _make_ring(),
        _make_two_rows(),
        _make_compact_hex_cluster(),
        _make_random(rng),
        _make_random(rng),
    ]

    best_side = float('inf')
    best_positions = None
    best_angles = None
    best_phi = 0.0
    best_cx = 0.0
    best_cy = 0.0

    for init_idx, (pos0, ang0) in enumerate(inits):
        if time.time() - t0 > DEADLINE - 25.0:
            break

        # Ensure validity
        if not _is_valid(pos0, ang0):
            _try_fix_validity(pos0, ang0, max_iters=300, rng=rng)
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

        # CMA-ES optimization
        remaining = DEADLINE - 30.0 - (time.time() - t0)
        if remaining < 8.0:
            break

        params0 = _pos_ang_to_params(pos0, ang0, phi0)
        t_cma_start = time.time()
        cma_time_limit = min(remaining * 0.8, 40.0)

        cma_params, cma_f = _cma_es_optimize(params0, t_cma_start, cma_time_limit, rng)

        cma_pos, cma_ang, cma_phi = _params_to_pos_ang(cma_params)
        if _is_valid(cma_pos, cma_ang):
            cma_side, cma_phi_opt, cma_cx, cma_cy = _compute_min_outer_side(cma_pos, cma_ang)
            if cma_side < best_side:
                best_side = cma_side
                best_positions = cma_pos.copy()
                best_angles = cma_ang.copy()
                best_phi = cma_phi_opt
                best_cx = cma_cx
                best_cy = cma_cy

    # Final verification
    if best_positions is None or not _is_valid(best_positions, best_angles):
        pos0, ang0 = _make_honeycomb()
        _try_fix_validity(pos0, ang0, max_iters=500, rng=rng)
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
