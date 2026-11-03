import numpy as np
import time
from scipy.optimize import minimize

N = 26
DEADLINE = 145.0  # safety margin before 150s kill


def _softplus(z, eps=1e-4):
    """Numerically stable softplus."""
    z = np.clip(z / eps, -50, 50)
    return eps * np.log1p(np.exp(z))


def _sigmoid(z, eps=1e-4):
    """Numerically stable sigmoid."""
    z = np.clip(z / eps, -50, 50)
    return 1.0 / (1.0 + np.exp(-z))


def _enforce_validity(p):
    """Enforce all constraints: walls and pairwise non-overlap by shrinking radii."""
    p = p.copy()
    p[:, 0] = np.clip(p[:, 0], 0.0, 1.0)
    p[:, 1] = np.clip(p[:, 1], 0.0, 1.0)
    p[:, 2] = np.clip(p[:, 2], 1e-6, 0.5)

    # Clamp to walls
    wall = np.minimum(np.minimum(p[:, 0], 1.0 - p[:, 0]),
                      np.minimum(p[:, 1], 1.0 - p[:, 1]))
    p[:, 2] = np.minimum(p[:, 2], np.maximum(wall, 1e-6))

    # Iteratively resolve overlaps by shrinking the smaller circle
    for _ in range(200):
        iu = np.triu_indices(N, k=1)
        dx = p[iu[0], 0] - p[iu[1], 0]
        dy = p[iu[0], 1] - p[iu[1], 1]
        dist = np.sqrt(dx * dx + dy * dy)
        overlap = (p[iu[0], 2] + p[iu[1], 2]) - dist
        active = overlap > 1e-10
        if not np.any(active):
            break

        max_ov = 0.0
        for k in range(len(active)):
            if not active[k]:
                continue
            i = iu[0][k]
            j = iu[1][k]
            ov = overlap[k]
            max_ov = max(max_ov, ov)
            if p[i, 2] <= p[j, 2]:
                shrink = min(ov, p[i, 2] - 1e-6)
                p[i, 2] -= shrink
                rem = ov - shrink
                if rem > 0:
                    shrink2 = min(rem, p[j, 2] - 1e-6)
                    p[j, 2] -= shrink2
            else:
                shrink = min(ov, p[j, 2] - 1e-6)
                p[j, 2] -= shrink
                rem = ov - shrink
                if rem > 0:
                    shrink2 = min(rem, p[i, 2] - 1e-6)
                    p[i, 2] -= shrink2

        # Re-clamp to walls
        wall = np.minimum(np.minimum(p[:, 0], 1.0 - p[:, 0]),
                          np.minimum(p[:, 1], 1.0 - p[:, 1]))
        p[:, 2] = np.minimum(p[:, 2], np.maximum(wall, 1e-6))
        if max_ov < 1e-8:
            break

    return p


def _score_valid(p):
    """Compute sum of radii for a valid configuration."""
    return np.sum(p[:, 2])


def _bounds():
    lo = np.zeros(3 * N)
    hi = np.zeros(3 * N)
    for i in range(N):
        lo[3 * i] = 0.0
        hi[3 * i] = 1.0
        lo[3 * i + 1] = 0.0
        hi[3 * i + 1] = 1.0
        lo[3 * i + 2] = 1e-6
        hi[3 * i + 2] = 0.5
    return (lo, hi)


def _objective_penalty(params, penalty_weight, smooth_eps=1e-4):
    """
    Vectorized objective with soft penalty for constraint violations.
    params: flat array [x0, y0, r0, x1, y1, r1, ...]
    """
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]

    penalty = 0.0

    # Boundary violations (smooth)
    lx = r - x
    ux = x + r - 1.0
    ly = r - y
    uy = y + r - 1.0

    penalty += np.sum(_softplus(lx, smooth_eps) ** 2)
    penalty += np.sum(_softplus(ux, smooth_eps) ** 2)
    penalty += np.sum(_softplus(ly, smooth_eps) ** 2)
    penalty += np.sum(_softplus(uy, smooth_eps) ** 2)

    # Negative radius
    penalty += np.sum(_softplus(-r, smooth_eps) ** 2)

    # Full pairwise overlaps (vectorized)
    n = len(r)
    idx_i, idx_j = np.triu_indices(n, k=1)
    dx = x[idx_i] - x[idx_j]
    dy = y[idx_i] - y[idx_j]
    dist_sq = dx * dx + dy * dy
    dist = np.sqrt(np.maximum(dist_sq, 1e-12))
    overlap = r[idx_i] + r[idx_j] - dist
    penalty += np.sum(_softplus(overlap, smooth_eps) ** 2)

    return -np.sum(r) + penalty_weight * penalty


def _objective_grad(params, penalty_weight, smooth_eps=1e-4):
    """
    Objective and gradient for penalty method with analytic gradients.
    Returns (value, grad).
    """
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]

    val = -np.sum(r)
    grad = np.zeros_like(params)
    grad[2::3] = -1.0  # d(-sum r)/dr

    # Boundary violations
    lx = r - x
    ux = x + r - 1.0
    ly = r - y
    uy = y + r - 1.0

    # Smooth derivative of softplus(z, eps) is sigmoid(z/eps)
    # d/dx of softplus(r-x)^2 = -2*softplus(r-x)*sigmoid((r-x)/eps)
    # d/dr of softplus(r-x)^2 = +2*softplus(r-x)*sigmoid((r-x)/eps)

    # Left x boundary: softplus(r - x, eps)
    sp_lx = _softplus(lx, smooth_eps)
    sg_lx = _sigmoid(lx, smooth_eps)
    grad[0::3] += -2.0 * penalty_weight * sp_lx * sg_lx
    grad[2::3] += 2.0 * penalty_weight * sp_lx * sg_lx

    # Right x boundary: softplus(x + r - 1, eps)
    sp_ux = _softplus(ux, smooth_eps)
    sg_ux = _sigmoid(ux, smooth_eps)
    grad[0::3] += 2.0 * penalty_weight * sp_ux * sg_ux
    grad[2::3] += 2.0 * penalty_weight * sp_ux * sg_ux

    # Left y boundary: softplus(r - y, eps)
    sp_ly = _softplus(ly, smooth_eps)
    sg_ly = _sigmoid(ly, smooth_eps)
    grad[1::3] += -2.0 * penalty_weight * sp_ly * sg_ly
    grad[2::3] += 2.0 * penalty_weight * sp_ly * sg_ly

    # Right y boundary: softplus(y + r - 1, eps)
    sp_uy = _softplus(uy, smooth_eps)
    sg_uy = _sigmoid(uy, smooth_eps)
    grad[1::3] += 2.0 * penalty_weight * sp_uy * sg_uy
    grad[2::3] += 2.0 * penalty_weight * sp_uy * sg_uy

    # Negative radius
    sp_nr = _softplus(-r, smooth_eps)
    sg_nr = _sigmoid(-r, smooth_eps)
    grad[2::3] += -2.0 * penalty_weight * sp_nr * sg_nr

    # Pairwise overlaps
    n = len(r)
    idx_i, idx_j = np.triu_indices(n, k=1)
    dx = x[idx_i] - x[idx_j]
    dy = y[idx_i] - y[idx_j]
    dist_sq = dx * dx + dy * dy
    dist = np.sqrt(np.maximum(dist_sq, 1e-12))
    overlap = r[idx_i] + r[idx_j] - dist
    sp_ov = _softplus(overlap, smooth_eps)
    sg_ov = _sigmoid(overlap, smooth_eps)

    factor = 2.0 * penalty_weight * sp_ov * sg_ov
    inv_dist = 1.0 / dist

    np.add.at(grad[0::3], idx_i, -factor * dx * inv_dist)
    np.add.at(grad[0::3], idx_j, factor * dx * inv_dist)
    np.add.at(grad[1::3], idx_i, -factor * dy * inv_dist)
    np.add.at(grad[1::3], idx_j, factor * dy * inv_dist)
    np.add.at(grad[2::3], idx_i, factor)
    np.add.at(grad[2::3], idx_j, factor)

    val += penalty_weight * (
        np.sum(sp_lx ** 2) + np.sum(sp_ux ** 2) + np.sum(sp_ly ** 2) + np.sum(sp_uy ** 2) +
        np.sum(sp_nr ** 2) + np.sum(sp_ov ** 2)
    )

    return val, grad


def _hertzian_forces(p):
    """
    Compute Hertzian contact forces (non-linear, proportional to penetration^(3/2)).
    Returns force vectors (N, 2) and radial forces (N,).
    """
    x = p[:, 0]
    y = p[:, 1]
    r = p[:, 2]

    fx = np.zeros(N)
    fy = np.zeros(N)
    fr = np.zeros(N)

    # Wall forces
    # Left wall
    pen_lx = np.maximum(0, r - x)
    fx += 50.0 * pen_lx ** 1.5
    fr -= 50.0 * pen_lx ** 1.5

    # Right wall
    pen_ux = np.maximum(0, x + r - 1.0)
    fx -= 50.0 * pen_ux ** 1.5
    fr -= 50.0 * pen_ux ** 1.5

    # Bottom wall
    pen_ly = np.maximum(0, r - y)
    fy += 50.0 * pen_ly ** 1.5
    fr -= 50.0 * pen_ly ** 1.5

    # Top wall
    pen_uy = np.maximum(0, y + r - 1.0)
    fy -= 50.0 * pen_uy ** 1.5
    fr -= 50.0 * pen_uy ** 1.5

    # Pairwise Hertzian contacts
    iu = np.triu_indices(N, k=1)
    dx = x[iu[0]] - x[iu[1]]
    dy = y[iu[0]] - y[iu[1]]
    dist = np.sqrt(dx * dx + dy * dy + 1e-12)
    dist_safe = np.maximum(dist, 1e-10)
    penetration = np.maximum(0, r[iu[0]] + r[iu[1]] - dist)

    # Hertzian force magnitude: k * pen^(3/2)
    force_mag = 100.0 * penetration ** 1.5

    # Direction from j to i
    nx = dx / dist_safe
    ny = dy / dist_safe

    np.add.at(fx, iu[0], force_mag * nx)
    np.add.at(fx, iu[1], -force_mag * nx)
    np.add.at(fy, iu[0], force_mag * ny)
    np.add.at(fy, iu[1], -force_mag * ny)

    # Radial force: compression pushes radius inward
    np.add.at(fr, iu[0], -force_mag)
    np.add.at(fr, iu[1], -force_mag)

    return np.column_stack((fx, fy)), fr


def _hertzian_simulate(p, time_limit=10.0, dt=0.002, n_steps=None):
    """
    Run a Hertzian contact simulation to find a jammed configuration.
    Uses symplectic Euler integration with damping.
    """
    start_time = time.time()
    if n_steps is None:
        n_steps = int(time_limit / dt)

    p = p.copy()
    vx = np.zeros(N)
    vy = np.zeros(N)
    vr = np.zeros(N)

    damping = 0.98
    mass = 1.0

    best_p = p.copy()
    best_score = _score_valid(p)

    for step in range(n_steps):
        if time.time() - start_time > time_limit - 0.5:
            break

        # Compute forces
        (fx, fy), fr = _hertzian_forces(p)

        # Update velocities (symplectic Euler)
        vx = (vx + dt * fx / mass) * damping
        vy = (vy + dt * fy / mass) * damping
        vr = (vr + dt * fr / mass) * damping

        # Update positions
        p[:, 0] += dt * vx
        p[:, 1] += dt * vy
        p[:, 2] += dt * vr

        # Enforce hard bounds (safety)
        p[:, 0] = np.clip(p[:, 0], 0.0, 1.0)
        p[:, 1] = np.clip(p[:, 1], 0.0, 1.0)
        p[:, 2] = np.clip(p[:, 2], 1e-6, 0.5)

        # Every 100 steps, check score and enforce validity
        if step % 100 == 0:
            p_valid = _enforce_validity(p)
            s = _score_valid(p_valid)
            if s > best_score:
                best_score = s
                best_p = p_valid.copy()

    # Final enforce
    p_valid = _enforce_validity(p)
    s = _score_valid(p_valid)
    if s > best_score:
        best_score = s
        best_p = p_valid.copy()

    return best_p, best_score


def _lbfgs_optimize(p, penalty=1000.0, maxiter=1000, time_limit=5.0):
    """L-BFGS-B with smooth penalty, adaptive penalty annealing."""
    start_time = time.time()
    best_p = p.copy()
    best_score = _score_valid(p)

    lo, hi = _bounds()
    bounds_list = list(zip(lo, hi))

    # Adaptive penalty: start high, decrease as we converge
    penalties = [5000.0, 2000.0, 1000.0, 500.0, 200.0, 100.0, 50.0]
    smoothnesses = [1e-3, 5e-4, 2e-4, 1e-4, 5e-5, 2e-5, 1e-5]

    current = p.flatten().copy()

    for idx, (pen, sm) in enumerate(zip(penalties, smoothnesses)):
        if time.time() - start_time > time_limit - 0.5:
            break

        def obj(params):
            return _objective_penalty(params, pen, sm)

        def grad(params):
            _, g = _objective_grad(params, pen, sm)
            return g

        try:
            res = minimize(
                obj,
                current,
                method='L-BFGS-B',
                jac=grad,
                bounds=bounds_list,
                options={'maxiter': maxiter // len(penalties), 'ftol': 1e-14, 'gtol': 1e-10}
            )
            current = res.x
            p_new = _enforce_validity(current.reshape(N, 3))
            s_new = _score_valid(p_new)
            if s_new > best_score:
                best_score = s_new
                best_p = p_new.copy()
        except Exception:
            continue

    return best_p, best_score


def _voronoi_initial(seed):
    """Initialize using Voronoi cells to estimate max possible radii."""
    rng = np.random.RandomState(seed)
    points = rng.uniform(0.05, 0.95, size=(N, 2))

    radii = np.zeros(N)
    for i, p in enumerate(points):
        dists = np.linalg.norm(points - p, axis=1)
        dists[i] = np.inf
        min_dist_neighbor = np.min(dists)
        wall_dist = np.min([p[0], 1 - p[0], p[1], 1 - p[1]])
        radii[i] = 0.5 * np.min([min_dist_neighbor, wall_dist]) * 0.95

    params = np.empty(3 * N)
    params[0::3] = points[:, 0]
    params[1::3] = points[:, 1]
    params[2::3] = radii

    params += rng.uniform(-1e-5, 1e-5, size=params.size)
    return params


def _grid_initial(seed):
    """Grid of equal circles with small jitter."""
    rng = np.random.RandomState(seed)
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            cells.append(((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-4))
    cells = cells[:N]
    arr = np.array(cells, dtype=float).reshape(-1)
    arr += rng.uniform(-1e-5, 1e-5, size=arr.size)
    return arr


def _random_initial(seed):
    """Simple random initialization with radius estimation."""
    rng = np.random.RandomState(seed)
    points = rng.uniform(0.05, 0.95, size=(N, 2))

    radii = np.zeros(N)
    for i, p in enumerate(points):
        dists = np.linalg.norm(points - p, axis=1)
        dists[i] = np.inf
        min_dist_neighbor = np.min(dists)
        wall_dist = np.min([p[0], 1 - p[0], p[1], 1 - p[1]])
        radii[i] = 0.5 * np.min([min_dist_neighbor, wall_dist]) * 0.95

    params = np.empty(3 * N)
    params[0::3] = points[:, 0]
    params[1::3] = points[:, 1]
    params[2::3] = radii

    params += rng.uniform(-1e-5, 1e-5, size=params.size)
    return params


def _hex_initial(seed):
    """Hexagonal grid initialization."""
    rng = np.random.RandomState(seed)
    points = []
    for i in range(7):
        for j in range(7):
            x = i
            y = j * 1.5 + (0.5 if i % 2 else 0)
            points.append((x, y))

    points = np.array(points)
    points -= np.mean(points, axis=0)
    max_coord = np.max(np.abs(points))
    scale = 0.45 / max_coord
    points *= scale
    points += 0.5

    dists = np.sum((points - 0.5) ** 2, axis=1)
    indices = np.argsort(dists)[:N]

    p = points[indices]
    radii = np.full(N, 0.05)

    params = np.empty(3 * N)
    params[0::3] = p[:, 0]
    params[1::3] = p[:, 1]
    params[2::3] = radii

    params += rng.uniform(-1e-4, 1e-4, size=params.size)
    return params


def _simulated_annealing_shake(p, time_limit=15.0, seed=42):
    """
    Simulated Annealing with 'Shake and Solve'.
    1. Shake: Perturb positions/radii significantly.
    2. Solve: Run a quick local optimizer (L-BFGS) on the shaken state.
    3. Accept/Reject based on SA criterion.
    """
    rng = np.random.RandomState(seed)
    start = time.time()
    best_p = p.copy()
    best_score = _score_valid(p)
    current_p = p.copy()
    current_score = best_score

    T = 0.02
    cooling = 0.999

    while time.time() - start < time_limit:
        T *= cooling
        if T < 1e-5:
            break

        p_shake = current_p.copy()
        sigma_pos = 0.05
        sigma_rad = 0.02

        p_shake[:, :2] += rng.normal(0, sigma_pos, size=(N, 2))
        p_shake[:, 2] += rng.normal(0, sigma_rad, size=N)

        p_shake_valid = _enforce_validity(p_shake)
        s_shake_valid = _score_valid(p_shake_valid)

        inner_time = min(2.0, max(0.5, time_limit - (time.time() - start) - 1.0))
        if inner_time <= 0:
            break

        try:
            p_sol, s_sol = _lbfgs_optimize(p_shake_valid, penalty=1000.0, maxiter=200, time_limit=inner_time)
        except Exception:
            p_sol, s_sol = p_shake_valid, s_shake_valid

        delta = s_sol - current_score
        if delta > 0 or rng.random() < np.exp(delta / max(T, 1e-10)):
            current_p = p_sol
            current_score = s_sol
            if s_sol > best_score:
                best_score = s_sol
                best_p = p_sol.copy()

        if rng.random() < 0.1:
            current_p = best_p
            current_score = best_score

    return best_p, best_score


def _high_precision_polish(p, time_limit=10.0):
    """Final high-precision polish using L-BFGS with tight tolerances."""
    start = time.time()
    best_p = p.copy()
    best_score = _score_valid(p)

    for scale in [0.01, 0.005, 0.001, 0.0005]:
        if time.time() - start > time_limit - 2:
            break

        p_try = best_p.copy()
        rng = np.random.RandomState(12345)
        p_try[:, :2] += rng.normal(0, scale, size=(N, 2))
        p_try = _enforce_validity(p_try)

        p_polish, s_polish = _lbfgs_optimize(p_try, penalty=5000.0, maxiter=500, time_limit=min(5.0, time_limit - (time.time() - start) - 1))
        if s_polish > best_score:
            best_score = s_polish
            best_p = p_polish.copy()

    return best_p, best_score


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE

    best_params = None
    best_score = -np.inf

    initializers = [
        _voronoi_initial,
        _random_initial,
        _grid_initial,
        _hex_initial,
    ]

    seeds = [42, 123, 7, 999, 555, 2024, 31415, 271828, 1234, 5678,
             11, 22, 33, 44, 55, 66, 77, 88, 99, 100,
             1001, 2002, 3003, 4004, 5005, 6006, 7007, 8008, 9009, 10010]

    # Phase 1: L-BFGS with adaptive penalty
    for seed in seeds:
        if time.time() > deadline - 30:
            break
        for init_fn in initializers:
            if time.time() > deadline - 30:
                break

            x0 = init_fn(seed)
            p0 = _enforce_validity(x0)

            try:
                p, s = _lbfgs_optimize(p0, penalty=1000.0, maxiter=500, time_limit=min(3.0, max(0.5, deadline - time.time() - 35)))
                if s > best_score:
                    best_score = s
                    best_params = p.copy()
            except Exception:
                continue

    if best_params is None:
        best_params = _enforce_validity(_grid_initial(42))
        best_score = _score_valid(best_params)

    # Phase 2: Hertzian contact simulation (new structural approach)
    if time.time() < deadline - 20:
        try:
            p_hertz, s_hertz = _hertzian_simulate(best_params, time_limit=min(10.0, max(2.0, deadline - time.time() - 25)), dt=0.001)
            if s_hertz > best_score:
                best_score = s_hertz
                best_params = p_hertz
        except Exception:
            pass

    # Phase 3: Simulated Annealing with Shake-and-Solve
    if time.time() < deadline - 15:
        try:
            p_sa, s_sa = _simulated_annealing_shake(
                best_params,
                time_limit=min(15.0, max(1.0, deadline - time.time() - 15)),
                seed=42
            )
            if s_sa > best_score:
                best_score = s_sa
                best_params = p_sa
        except Exception:
            pass

    # Phase 4: High-precision polish
    if time.time() < deadline - 5:
        try:
            p_final, s_final = _high_precision_polish(
                best_params,
                time_limit=min(8.0, max(1.0, deadline - time.time() - 5))
            )
            if s_final > best_score:
                best_score = s_final
                best_params = p_final
        except Exception:
            pass

    # Phase 5: Final tiny radius increases (greedy local improvement)
    for _ in range(100):
        if time.time() > deadline - 2:
            break

        p = best_params.copy()
        p[:, 2] *= 1.0005
        p_new = _enforce_validity(p)
        s_new = _score_valid(p_new)

        if s_new > best_score:
            best_score = s_new
            best_params = p_new
        else:
            break

    return best_params.copy()
