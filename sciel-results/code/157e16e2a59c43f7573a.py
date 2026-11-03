import math
import time
import numpy as np

N = 12
DEADLINE = 142.0
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


def _logsumexp(x, beta=10.0):
    """Smooth approximation to max(x) using log-sum-exp."""
    x = np.asarray(x, dtype=np.float64)
    x_max = x.max()
    if x_max == -np.inf:
        return 0.0
    return x_max + np.log(np.exp(beta * (x - x_max)).sum()) / beta


def _smooth_objective(params, phi_fixed=None):
    """
    params: (36,) array = [x0,y0,a0, x1,y1,a1, ..., x11,y11,a11]
    angles in radians.
    Returns (objective_value, is_valid_flag)
    Objective = smooth_outer_side + lambda * smooth_collision_penalty
    """
    if phi_fixed is None:
        # We'll optimize phi too; for now use a fixed 0 and let CMA-ES handle it
        # Actually let's include phi as the 37th parameter
        pass

    positions = params[0:36:3]
    pys = params[1:36:3]
    angles = params[2:36:3]

    # Compute outer hexagon side (smooth)
    c = np.cos(angles)
    s = np.sin(angles)
    rx = _UX[None, :] * c[:, None] - _UY[None, :] * s[:, None]
    ry = _UX[None, :] * s[:, None] + _UY[None, :] * c[:, None]
    verts_x = rx + positions[:, None]
    verts_y = ry + pys[:, None]
    verts_flat = np.stack((verts_x, verts_y), axis=-1).reshape(-1, 2)
    center = np.mean(verts_flat, axis=0)
    offsets = verts_flat - center

    # Use a fixed set of 6 directions (optimizing phi is separate)
    # For the smooth objective, we use 6 normal directions at 0,60,...,300
    normal_angles = np.pi / 6 + np.arange(6) * np.pi / 3
    nx = np.cos(normal_angles)
    ny = np.sin(normal_angles)
    proj = offsets[:, 0:1] * nx[None, :] + offsets[:, 1:2] * ny[None, :]
    abs_proj = np.abs(proj)
    max_projs = np.max(abs_proj, axis=0)
    side_vals = (2.0 / SQRT3) * max_projs

    # Smooth max of side_vals
    outer_side_smooth = _logsumexp(side_vals, beta=20.0)

    # Collision penalty: for each pair, compute minimum distance between hexagons
    # Using a smooth approximation: if centers are far, penalty ~ 0
    # We use a soft overlap measure based on the Minkowski sum
    # For two unit hexagons, the minimum separation is 0 (touching).
    # A simple proxy: distance between centers minus the "radius" of the Minkowski sum
    # The Minkowski sum of two unit hexagons has inradius = sqrt(3) (same as one hex)
    # So if center distance < 2*sqrt(3), they might overlap.
    # More precisely, use the support function difference.

    collision_penalty = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            dx = positions[i] - positions[j]
            dy = pys[i] - pys[j]
            dist = math.sqrt(dx * dx + dy * dy)
            # The minimum distance between two unit hexagons with relative angle
            # is complex. Use a simple threshold: if dist < 2*sqrt(3) - epsilon, penalize.
            # Smooth penalty: max(0, threshold - dist)^2
            # threshold for two unit hexagons aligned: 2*sqrt(3) ≈ 3.464
            # But with rotation, the effective radius varies. Use a conservative threshold.
            threshold = 2.0 * SQRT3 - 0.1  # slightly less to allow touching
            if dist < threshold:
                overlap = threshold - dist
                collision_penalty += overlap * overlap

    lam = 50.0
    total = outer_side_smooth + lam * collision_penalty
    return total


def _smooth_objective_with_phi(params):
    """
    params: (37,) = [x0,y0,a0, ..., x11,y11,a11, phi]
    phi in radians, the outer hexagon orientation.
    """
    positions = params[0:36:3]
    pys = params[1:36:3]
    angles = params[2:36:3]
    phi = params[36]

    c = np.cos(angles)
    s = np.sin(angles)
    rx = _UX[None, :] * c[:, None] - _UY[None, :] * s[:, None]
    ry = _UX[None, :] * s[:, None] + _UY[None, :] * s[:, None]
    # Fix the rotation: ry should use c for the y-component
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

    outer_side_smooth = _logsumexp(side_vals, beta=20.0)

    collision_penalty = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            dx = positions[i] - positions[j]
            dy = pys[i] - pys[j]
            dist = math.sqrt(dx * dx + dy * dy)
            threshold = 2.0 * SQRT3 - 0.1
            if dist < threshold:
                overlap = threshold - dist
                collision_penalty += overlap * overlap

    lam = 50.0
    total = outer_side_smooth + lam * collision_penalty
    return total


def _cma_es_optimize(x0, sigma0, max_evals, time_limit, t_start, rng):
    """
    Simple CMA-ES implementation for 37D optimization.
    """
    dim = len(x0)
    pop_size = max(4 + int(3 * math.log(dim)), 12)

    # CMA-ES parameters
    mu = pop_size // 2
    weights = np.zeros(mu)
    w_sum = 0.0
    for i in range(mu):
        weights[i] = math.log(mu + 0.5) - math.log(i + 1)
        w_sum += weights[i]
    weights /= w_sum
    mu_eff = 1.0 / np.sum(weights ** 2)
    c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
    c_c = (4 - mu_eff) / (dim + 9)
    d_sigma = 1 + 2 * max(0, math.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma

    p_sigma = np.zeros(dim)
    p_c = np.zeros(dim)
    C = np.eye(dim)
    step_size = sigma0

    best_x = x0.copy()
    best_f = _smooth_objective_with_phi(x0)

    eval_count = 0
    while eval_count < max_evals:
        if time.time() - t_start > time_limit:
            break

        # Generate population
        offspring = []
        fitnesses = []
        for _ in range(pop_size):
            z = rng.standard_normal(dim)
            # Apply C matrix (Cholesky)
            L = np.linalg.cholesky(C + 1e-10 * np.eye(dim))
            x_trial = x0 + step_size * (L @ z)
            offspring.append(x_trial)
            f_trial = _smooth_objective_with_phi(x_trial)
            fitnesses.append(f_trial)
            eval_count += 1
            if f_trial < best_f:
                best_f = f_trial
                best_x = x_trial.copy()

        # Sort by fitness
        order = np.argsort(fitnesses)
        offspring_sorted = [offspring[i] for i in order]
        fitnesses_sorted = [fitnesses[i] for i in order]

        # Update mean
        new_mean = np.zeros(dim)
        for i in range(mu):
            new_mean += weights[i] * offspring_sorted[i]
        x0 = new_mean

        # Update step size (sigma)
        old_mean = best_x  # approximation
        diff = (np.linalg.norm(x0 - old_mean) / (step_size * np.sqrt(2 * (dim + 10 / c_sigma))))
        p_sigma = (1 - c_sigma) * p_sigma + c_sigma * diff
        step_size *= math.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / math.sqrt(mu_eff) - 1))
        step_size = max(1e-10, min(step_size, 10.0))

        # Update covariance
        diff_vec = (x0 - old_mean) / step_size
        p_c = (1 - c_c) * p_c + c_c * diff_vec
        rank_one = np.outer(p_c, p_c)
        rank_mu = np.zeros((dim, dim))
        for i in range(mu):
            diff_i = (offspring_sorted[i] - old_mean) / step_size
            rank_mu += weights[i] * np.outer(diff_i, diff_i)

        c1 = (2 - mu_eff / (dim + 1)) / ((dim + 1.3) ** 2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        h_sigma = 1.0 if np.linalg.norm(p_sigma) / math.sqrt(1 - (1 - c_sigma) ** 2) < (1.4 + 2 / (dim + 1)) * math.sqrt(mu_eff) else 0.0
        rank_one = h_sigma * rank_one

        C = (1 - c1 - cmu) * C + c1 * (rank_one + np.eye(dim)) + cmu * rank_mu

        # Ensure positive semi-definite
        eigvals, eigvecs = np.linalg.eigh(C)
        eigvals = np.maximum(eigvals, 1e-10)
        C = eigvecs @ np.diag(eigvals) @ eigvecs.T
        C = (C + C.T) / 2

    return best_x, best_f


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
    """Convert (positions, angles_deg, phi_deg) to 37D parameter vector."""
    params = np.zeros(37)
    for i in range(N):
        params[3 * i] = pos[i, 0]
        params[3 * i + 1] = pos[i, 1]
        params[3 * i + 2] = math.radians(ang_deg[i])
    params[36] = math.radians(phi_deg)
    return params


def _params_to_pos_ang(params):
    """Convert 37D parameter vector back to (positions, angles_deg, phi_deg)."""
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


def _coordinate_descent_polish(pos, ang_deg, phi_deg, time_limit, t_start):
    """Fine coordinate descent on the 37D parameter space."""
    params = _pos_ang_to_params(pos, ang_deg, phi_deg)
    best_params = params.copy()
    best_side, _, _, _ = _compute_min_outer_side(pos, ang_deg)
    best_side = min(best_side, _smooth_objective_with_phi(params))

    step = 0.05
    min_step = 1e-6
    no_improve = 0
    max_no_improve = 10

    while time.time() - t_start < time_limit and step > min_step:
        improved = False
        for dim in range(37):
            if time.time() - t_start > time_limit:
                break
            for direction in [1, -1]:
                trial = best_params.copy()
                trial[dim] += direction * step
                f_trial = _smooth_objective_with_phi(trial)
                # Check validity for the trial
                tp, ta, tp_phi = _params_to_pos_ang(trial)
                if _is_valid(tp, ta):
                    side_trial, _, _, _ = _compute_min_outer_side(tp, ta)
                    if side_trial < best_side - 1e-10:
                        best_params = trial
                        best_side = side_trial
                        improved = True

        if not improved:
            no_improve += 1
            if no_improve >= max_no_improve:
                step *= 0.5
                no_improve = 0
        else:
            no_improve = 0

    final_pos, final_ang, final_phi = _params_to_pos_ang(best_params)
    return final_pos, final_ang, final_phi


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

        # CMA-ES optimization
        remaining = DEADLINE - 25.0 - (time.time() - t0)
        if remaining < 5.0:
            break

        params0 = _pos_ang_to_params(pos0, ang0, phi0)
        sigma0 = 0.3
        max_evals = int(min(remaining * 30, 2000))  # ~30 evals/sec budget

        t_cma_start = time.time()
        cma_time_limit = min(remaining * 0.7, 60.0)
        cma_result, cma_f = _cma_es_optimize(params0, sigma0, max_evals, cma_time_limit, t_cma_start, rng)

        cma_pos, cma_ang, cma_phi = _params_to_pos_ang(cma_result)
        if _is_valid(cma_pos, cma_ang):
            cma_side, cma_phi_opt, cma_cx, cma_cy = _compute_min_outer_side(cma_pos, cma_ang)
            if cma_side < best_side:
                best_side = cma_side
                best_positions = cma_pos.copy()
                best_angles = cma_ang.copy()
                best_phi = cma_phi_opt
                best_cx = cma_cx
                best_cy = cma_cy

        # Coordinate descent polish
        remaining2 = DEADLINE - 15.0 - (time.time() - t0)
        if remaining2 > 3.0:
            polish_time = min(remaining2 * 0.5, 15.0)
            p_pos, p_ang, p_phi = _coordinate_descent_polish(
                best_positions, best_angles, best_phi, polish_time, time.time())
            if _is_valid(p_pos, p_ang):
                p_side, p_phi_opt, p_cx, p_cy = _compute_min_outer_side(p_pos, p_ang)
                if p_side < best_side:
                    best_side = p_side
                    best_positions = p_pos.copy()
                    best_angles = p_ang.copy()
                    best_phi = p_phi_opt
                    best_cx = p_cx
                    best_cy = p_cy

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
