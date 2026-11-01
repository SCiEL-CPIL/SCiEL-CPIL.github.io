"""
Optimized packing of 50 equal circles in a unit square.
Combines:
- Soft-violation objective (from inspiration) with vectorized gradients
- Simulated annealing for global search
- L-BFGS polishing with multiple beta values
- Binary search on radius with feasibility polishing
- High-precision final polish
"""
import numpy as np
import time
import scipy.optimize as opt
from scipy.spatial import cKDTree

N = 50
DEADLINE = 147.0  # Safety margin before 150s kill

# Precompute upper-triangle indices for pairwise operations
_IU = np.triu_indices(N, k=1)
_NPAIRS = len(_IU[0])


def _compute_radius(centers):
    """Compute the true maximum radius for a given configuration."""
    centers = np.asarray(centers, dtype=np.float64)
    if centers.shape != (N, 2) or not np.all(np.isfinite(centers)):
        return 0.0

    # Wall distances
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    min_wall = np.min(np.minimum(wall_dists_x, wall_dists_y))

    if min_wall < 0:
        return 0.0

    # Pairwise distances using cKDTree for speed
    tree = cKDTree(centers)
    dists, _ = tree.query(centers, k=2)
    min_pair = np.min(dists[:, 1]) / 2.0

    return min(min_pair, min_wall)


def _compute_radius_full(centers):
    """Compute radius using full pairwise (slower but exact for verification)."""
    centers = np.asarray(centers, dtype=np.float64)
    if centers.shape != (N, 2) or not np.all(np.isfinite(centers)):
        return 0.0

    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    min_wall = np.min(np.minimum(wall_dists_x, wall_dists_y))

    if min_wall < 0:
        return 0.0

    diff = centers[:, None, :] - centers[None, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    min_pair = np.sqrt(np.min(dist_sq)) / 2.0

    return min(min_pair, min_wall)


def _soft_violation_obj(x, r_target, beta):
    """
    Objective: Minimize smooth violation for target radius r_target.
    Uses softplus for smoothness.
    """
    centers = x.reshape(-1, 2)

    # Wall distances
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    walls = np.minimum(wall_dists_x, wall_dists_y)
    wall_viol = r_target - walls

    # Pairwise distances (vectorized via upper-triangle indices)
    ci = centers[_IU[0]]
    cj = centers[_IU[1]]
    diff = ci - cj
    dists = np.sqrt(np.sum(diff ** 2, axis=1))
    pair_viol = 2.0 * r_target - dists

    # Softplus: log(1 + exp(z)) / beta
    z_wall = np.clip(beta * wall_viol, -50, 50)
    z_pair = np.clip(beta * pair_viol, -50, 50)

    soft_wall = np.log1p(np.exp(z_wall)) / beta
    soft_pair = np.log1p(np.exp(z_pair)) / beta

    return float(np.sum(soft_wall) + np.sum(soft_pair))


def _soft_violation_grad(x, r_target, beta):
    """
    Gradient of the soft violation objective. Fully vectorized.
    """
    centers = x.reshape(-1, 2)
    grad = np.zeros_like(x)

    # Wall gradients
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    walls = np.minimum(wall_dists_x, wall_dists_y)
    wall_viol = r_target - walls

    z_wall = np.clip(beta * wall_viol, -50, 50)
    sig_wall = 1.0 / (1.0 + np.exp(-z_wall))

    # Vectorized wall gradient contribution
    wx = wall_dists_x < wall_dists_y
    wy = wall_dists_y < wall_dists_x
    wt = ~(wx | wy)

    idx_wx = np.nonzero(wx)[0]
    idx_wy = np.nonzero(wy)[0]
    idx_wt = np.nonzero(wt)[0]

    if len(idx_wx) > 0:
        grad[2 * idx_wx] += sig_wall[wx] * np.sign(centers[idx_wx, 0])
    if len(idx_wy) > 0:
        grad[2 * idx_wy + 1] += sig_wall[wy] * np.sign(centers[idx_wy, 1])
    if len(idx_wt) > 0:
        grad[2 * idx_wt] += 0.5 * sig_wall[wt] * np.sign(centers[idx_wt, 0])
        grad[2 * idx_wt + 1] += 0.5 * sig_wall[wt] * np.sign(centers[idx_wt, 1])

    # Pair gradients (vectorized)
    ci = centers[_IU[0]]
    cj = centers[_IU[1]]
    diff_vec = ci - cj  # shape (_NPAIRS, 2)
    dists = np.sqrt(np.sum(diff_vec ** 2, axis=1))
    dists_safe = np.maximum(dists, 1e-12)
    pair_viol = 2.0 * r_target - dists

    z_pair = np.clip(beta * pair_viol, -50, 50)
    sig_pair = 1.0 / (1.0 + np.exp(-z_pair))

    # factor_k = sig_pair[k] / dists[k]
    factor = sig_pair / dists_safe
    # contribution per pair: factor * diff_vec
    contrib = factor[:, None] * diff_vec

    # Scatter: for pair (i,j), grad_i -= contrib, grad_j += contrib
    np.add.at(grad, 2 * _IU[0], -contrib[:, 0])
    np.add.at(grad, 2 * _IU[0] + 1, -contrib[:, 1])
    np.add.at(grad, 2 * _IU[1], contrib[:, 0])
    np.add.at(grad, 2 * _IU[1] + 1, contrib[:, 1])

    return grad


def _soft_min_objective(x, tau=0.008):
    """
    Objective: minimize -soft_min(minimum distance to constraint boundary).
    x: flattened array of 2*N coordinates
    """
    centers = x.reshape(-1, 2)

    # Wall distances
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    walls = np.minimum(wall_dists_x, wall_dists_y)

    # Pairwise distances using upper triangle
    ci = centers[_IU[0]]
    cj = centers[_IU[1]]
    diff = ci - cj
    dists = np.sqrt(np.sum(diff ** 2, axis=1))
    pair_dists = dists / 2.0

    # Combine all constraint distances
    all_dists = np.concatenate([walls, pair_dists])

    # Soft-min using log-sum-exp
    min_d = np.min(all_dists)
    shifted = all_dists - min_d
    log_sum = np.log(np.sum(np.exp(-shifted / tau)))
    soft_min = min_d - tau * log_sum

    return -soft_min


def _grad_soft_min(x, tau=0.008):
    """
    Compute gradient of -soft_min objective analytically.
    """
    centers = x.reshape(-1, 2)

    # Wall distances
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    walls = np.minimum(wall_dists_x, wall_dists_y)

    # Pairwise distances
    ci = centers[_IU[0]]
    cj = centers[_IU[1]]
    diff = ci - cj
    dists = np.sqrt(np.sum(diff ** 2, axis=1))
    pair_dists = dists / 2.0

    all_dists = np.concatenate([walls, pair_dists])

    # Soft-min weights: w_k = exp(-d_k/tau) / sum(exp(-d_j/tau))
    min_d = np.min(all_dists)
    shifted = all_dists - min_d
    exp_vals = np.exp(-shifted / tau)
    weights = exp_vals / np.sum(exp_vals)

    grad = np.zeros_like(x)

    # Wall gradient contributions
    for i in range(N):
        w_wall = weights[i]
        if walls[i] == wall_dists_x[i]:
            grad[2 * i] += w_wall * np.sign(centers[i, 0])
        if walls[i] == wall_dists_y[i]:
            grad[2 * i + 1] += w_wall * np.sign(centers[i, 1])

    # Pairwise gradient contributions
    for k in range(_NPAIRS):
        w_pair = weights[N + k]
        i, j = _IU[0][k], _IU[1][k]
        d = dists[k]
        if d > 1e-12:
            factor = 0.5 * w_pair / d
            diff_ij = diff[k]
            grad[2 * i:2 * i + 2] -= factor * diff_ij
            grad[2 * j:2 * j + 2] += factor * diff_ij

    return grad


def _initial_grid():
    """Create initial square grid layout with slight jitter."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    grid = grid[:N].copy()

    rng = np.random.default_rng(12345)
    grid = grid + rng.normal(0, 0.005, grid.shape)
    grid = np.clip(grid, -0.49, 0.49)

    return grid


def _hex_init():
    """Create a hexagonal packing initial layout with multiple attempts."""
    best_grid = None
    best_r = 0

    rng = np.random.default_rng(42)

    for attempt in range(8):
        rows = rng.integers(5, 10)
        cols = int(np.ceil(N / rows))

        s_width = 1.0 / (cols + 0.5)
        s_height = 1.0 / (rows * 0.866 + 0.5)
        s = min(s_width, s_height)

        centers = []
        for j in range(rows):
            for i in range(cols):
                x = (i - (cols - 1) / 2) * s
                y = (j - (rows - 1) / 2) * s * 0.866
                if j % 2 == 1:
                    x += s / 2
                centers.append([x, y])

        centers = np.array(centers[:N])
        if len(centers) < N:
            while len(centers) < N:
                centers = np.vstack([centers, rng.uniform(-0.45, 0.45, 2)])

        centers = np.clip(centers, -0.49, 0.49)
        r = _compute_radius(centers)
        if r > best_r:
            best_r = r
            best_grid = centers.copy()

    return best_grid if best_grid is not None else _initial_grid()


def _random_init():
    """Create a random initial layout."""
    rng = np.random.default_rng(999)
    centers = rng.uniform(-0.45, 0.45, (N, 2))
    return centers


def _simulated_annealing(centers, rng, time_budget):
    """
    Simulated annealing to optimize the minimum radius directly.
    Uses vectorized radius computation.
    """
    start_time = time.time()
    current = centers.copy()
    best = current.copy()
    best_r = _compute_radius(current)

    T_start = 0.02
    T_end = 0.0001
    n_steps = 50000
    cooling = (T_end / T_start) ** (1.0 / max(n_steps, 1))
    T = T_start

    for step in range(n_steps):
        if time.time() - start_time > time_budget:
            break

        # Quick vectorized radius check
        wall_x = 0.5 - np.abs(current[:, 0])
        wall_y = 0.5 - np.abs(current[:, 1])
        min_wall = np.min(np.minimum(wall_x, wall_y))

        # Vectorized pairwise min distance using full matrix
        d = current[:, None, :] - current[None, :, :]
        dist_sq = np.einsum('ijk,ijk->ij', d, d)
        np.fill_diagonal(dist_sq, np.inf)
        min_pair = np.sqrt(np.min(dist_sq)) / 2.0
        curr_r = min(min_pair, min_wall)

        # Move generation: pick a random point and perturb it
        i = rng.integers(0, N)
        move_size = 0.003 + 0.015 * T / T_start
        direction = rng.normal(size=2)
        norm = np.linalg.norm(direction)
        if norm > 1e-15:
            direction /= norm

        trial = current.copy()
        trial[i] += direction * move_size
        trial = np.clip(trial, -0.5, 0.5)

        trial_r = _compute_radius(trial)
        delta_r = trial_r - curr_r

        if delta_r >= 0:
            current = trial
            if trial_r > best_r:
                best_r = trial_r
                best = trial.copy()
        else:
            # Accept with probability exp(delta_r / T)
            if delta_r > -50 * T:  # Avoid underflow
                if rng.random() < np.exp(delta_r / max(T, 1e-10)):
                    current = trial

        T *= cooling

    return best, best_r


def _lbfgs_violation_polish(x0, r_target, beta, time_budget):
    """
    L-BFGS optimization to minimize violation for a target radius.
    """
    start_time = time.time()

    def obj_and_grad(x):
        if time.time() - start_time > time_budget:
            return 0.0, np.zeros_like(x)
        f = _soft_violation_obj(x, r_target, beta)
        g = _soft_violation_grad(x, r_target, beta)
        return f, g

    try:
        result = opt.minimize(
            obj_and_grad, x0, jac=True, method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * (2 * N),
            options={'maxiter': 300, 'ftol': 1e-16, 'gtol': 1e-14,
                     'maxfun': 1000, 'disp': False}
        )
        polished = result.x.reshape(-1, 2)
        polished = np.clip(polished, -0.5, 0.5)
        return polished
    except Exception:
        return x0.reshape(-1, 2)


def _lbfgs_softmin_polish(x0, tau, time_budget):
    """
    L-BFGS optimization on the soft-min objective.
    """
    start_time = time.time()

    def obj_and_grad(x):
        if time.time() - start_time > time_budget:
            return 0.0, np.zeros_like(x)
        f = _soft_min_objective(x, tau)
        g = _grad_soft_min(x, tau)
        return f, g

    try:
        result = opt.minimize(
            obj_and_grad, x0, jac=True, method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * (2 * N),
            options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-14,
                     'maxfun': 2000, 'disp': False}
        )
        polished = result.x.reshape(-1, 2)
        polished = np.clip(polished, -0.5, 0.5)
        return polished
    except Exception:
        return x0.reshape(-1, 2)


def _binary_search_radius(centers, lo, hi, time_budget):
    """
    Binary search for maximum feasible radius.
    Uses L-BFGS with violation objective to find feasible configurations.
    """
    start_time = time.time()
    best = centers.copy()
    best_r = lo

    for _ in range(20):
        if time.time() - start_time > time_budget:
            break

        mid = (lo + hi) / 2
        trial = best.copy()

        # Try to find a feasible configuration for radius mid
        # Use violation minimization with increasing beta
        for beta in [20, 50, 100, 200]:
            if time.time() - start_time > time_budget:
                break

            x = trial.flatten().copy()
            polished = _lbfgs_violation_polish(x, mid, beta, time_budget=min(8, deadline_remaining(start_time, time_budget)))
            trial = polished
            trial_r = _compute_radius(trial)

            if trial_r >= mid - 1e-10:
                break

        trial_r = _compute_radius(trial)

        if trial_r >= mid - 1e-10:
            lo = mid
            best = trial
            best_r = trial_r
        else:
            hi = mid

    return best, best_r


def deadline_remaining(start_time, budget):
    """Return remaining time budget."""
    elapsed = time.time() - start_time
    return max(0, budget - elapsed)


def _high_precision_polish(centers, time_budget):
    """Final high-precision polish using very tight tolerances with soft-min."""
    start_time = time.time()
    current = centers.copy()
    best = current.copy()
    best_r = _compute_radius(current)

    # Use very small tau for high precision
    for tau in [0.0005, 0.0002, 0.0001]:
        if time.time() - start_time > time_budget:
            break

        def obj_and_grad(x):
            if time.time() - start_time > time_budget:
                return 0.0, np.zeros_like(x)
            f = _soft_min_objective(x, tau)
            g = _grad_soft_min(x, tau)
            return f, g

        try:
            result = opt.minimize(
                obj_and_grad, current.flatten(), jac=True, method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (2 * N),
                options={'maxiter': 1000, 'ftol': 1e-18, 'gtol': 1e-16,
                         'maxfun': 5000, 'disp': False}
            )
            polished = result.x.reshape(-1, 2)
            polished = np.clip(polished, -0.5, 0.5)

            r_polished = _compute_radius(polished)
            if r_polished > best_r:
                best_r = r_polished
                best = polished
                current = polished
        except Exception:
            pass

    return best, best_r


def _contact_polish(centers, time_budget):
    """
    SLSQP-based polish that identifies contact pairs and optimizes
    with explicit distance constraints.
    """
    start_time = time.time()
    current = centers.copy()
    best = current.copy()
    best_r = _compute_radius(current)

    # Identify likely contact pairs (within 1.1x of current min distance)
    tree = cKDTree(current)
    min_d = _compute_radius(current) * 2.0
    threshold = min_d * 1.15

    # Get all pairs within threshold
    pairs = tree.query_pairs(threshold)

    # Also identify wall contacts
    wall_x = 0.5 - np.abs(current[:, 0])
    wall_y = 0.5 - np.abs(current[:, 1])
    wall_min = np.minimum(wall_x, wall_y)

    # Build constraint list
    constraints = []
    for i, j in pairs:
        constraints.append({
            'type': 'eq',
            'fun': lambda x, i=i, j=j: np.sqrt(np.sum((x[2 * i:2 * i + 2] - x[2 * j:2 * j + 2]) ** 2)) - min_d,
            'jac': lambda x, i=i, j=j: _contact_jac(x, i, j)
        })

    # Wall constraints for circles touching walls
    for i in range(N):
        if wall_min[i] < min_d * 0.95:
            if wall_x[i] <= wall_y[i]:
                # Touching left or right wall
                side = 1 if current[i, 0] > 0 else -1
                constraints.append({
                    'type': 'eq',
                    'fun': lambda x, i=i, side=side: side * x[2 * i] + 0.5 - min_d,
                    'jac': lambda x, i=i, side=side: _wall_jac_x(x, i, side)
                })
            else:
                # Touching top or bottom wall
                side = 1 if current[i, 1] > 0 else -1
                constraints.append({
                    'type': 'eq',
                    'fun': lambda x, i=i, side=side: side * x[2 * i + 1] + 0.5 - min_d,
                    'jac': lambda x, i=i, side=side: _wall_jac_y(x, i, side)
                })

    if not constraints:
        return best, best_r

    # Objective: maximize minimum distance (soft-min surrogate)
    tau = 0.002

    def obj(x):
        return _soft_min_objective(x, tau)

    def obj_grad(x):
        return _grad_soft_min(x, tau)

    try:
        result = opt.minimize(
            obj, current.flatten(), jac=obj_grad, method='SLSQP',
            bounds=[(-0.5, 0.5)] * (2 * N),
            constraints=constraints,
            options={'maxiter': 300, 'ftol': 1e-14, 'disp': False}
        )
        polished = np.clip(result.x.reshape(-1, 2), -0.5, 0.5)
        r_polished = _compute_radius(polished)
        if r_polished > best_r:
            best_r = r_polished
            best = polished
    except Exception:
        pass

    return best, best_r


def _contact_jac(x, i, j):
    """Jacobian of distance constraint for pair (i,j)."""
    grad = np.zeros(2 * N)
    diff = x[2 * i:2 * i + 2] - x[2 * j:2 * j + 2]
    d = np.sqrt(np.sum(diff ** 2))
    if d > 1e-12:
        grad[2 * i:2 * i + 2] = diff / d
        grad[2 * j:2 * j + 2] = -diff / d
    return grad


def _wall_jac_x(x, i, side):
    """Jacobian of wall constraint (x-direction)."""
    grad = np.zeros(2 * N)
    grad[2 * i] = side
    return grad


def _wall_jac_y(x, i, side):
    """Jacobian of wall constraint (y-direction)."""
    grad = np.zeros(2 * N)
    grad[2 * i + 1] = side
    return grad


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE

    # Generate diverse initializations
    rng = np.random.default_rng(777)

    best_centers = None
    best_radius = 0.0

    # Try multiple initializations
    inits = [
        _hex_init(),
        _initial_grid(),
        _random_init(),
    ]

    for init in inits:
        r = _compute_radius(init)
        if r > best_radius:
            best_radius = r
            best_centers = init

    # Phase 1: Simulated Annealing for global search
    if time.time() < deadline - 60:
        sa_budget = min(45.0, deadline - time.time() - 60)
        if sa_budget > 5:
            best_centers, best_radius = _simulated_annealing(best_centers, rng, sa_budget)

    # Phase 2: L-BFGS with soft-min objective at multiple tau values
    if time.time() < deadline - 45:
        x0 = best_centers.flatten()
        for tau in [0.02, 0.015, 0.01, 0.008, 0.005, 0.003]:
            if time.time() > deadline - 30:
                break
            budget = min(8, deadline - time.time() - 25)
            if budget < 2:
                break
            trial = _lbfgs_softmin_polish(x0, tau, budget)
            trial_r = _compute_radius(trial)
            if trial_r > best_radius:
                best_radius = trial_r
                best_centers = trial
            x0 = best_centers.flatten()

    # Phase 3: Binary search on radius with violation-based feasibility
    if time.time() < deadline - 20:
        lo = best_radius
        hi = min(0.075, best_radius * 1.015)
        budget = min(15, deadline - time.time() - 15)
        if budget > 3:
            best_centers, best_radius = _binary_search_radius(
                best_centers, lo, hi, budget
            )

    # Phase 4: Contact-based SLSQP polish
    if time.time() < deadline - 10:
        budget = min(8, deadline - time.time() - 8)
        if budget > 2:
            best_centers, best_radius = _contact_polish(best_centers, budget)

    # Phase 5: High-precision polish
    if time.time() < deadline - 3:
        budget = min(6, deadline - time.time() - 2)
        if budget > 1:
            best_centers, best_radius = _high_precision_polish(best_centers, budget)

    # Final validation
    if not np.all(np.isfinite(best_centers)) or best_centers.shape != (N, 2):
        return _initial_grid()

    # Ensure all points are within bounds
    best_centers = np.clip(best_centers, -0.5, 0.5)

    return best_centers
