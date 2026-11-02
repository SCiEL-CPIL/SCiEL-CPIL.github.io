"""
Optimized packing of 50 equal circles in a unit square.
Structural change: replaces the softplus violation + binary-search refinement with a
differential-evolution global search (no gradient needed, robust to non-smoothness),
followed by a smooth soft-minimum epigraph polish with progressively increasing
sharpness, and a final contact-graph-aware push.
"""
import numpy as np
import time
import scipy.optimize as opt

N = 50
DEADLINE = 147.0

_IU = np.triu_indices(N, k=1)
_NPAIRS = len(_IU[0])


def _compute_radius(centers):
    centers = np.asarray(centers, dtype=np.float64)
    if centers.shape != (N, 2) or not np.all(np.isfinite(centers)):
        return 0.0
    wx = 0.5 - np.abs(centers[:, 0])
    wy = 0.5 - np.abs(centers[:, 1])
    min_wall = np.min(np.minimum(wx, wy))
    if min_wall < 0:
        return 0.0
    diff = centers[:, None, :] - centers[None, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    min_pair = np.sqrt(np.min(dist_sq)) / 2.0
    return float(min(min_pair, min_wall))


def _soft_min_obj_and_grad(x, beta):
    """
    Objective: minimize -soft_min of all constraint values.
    Constraints: wall_i >= 0, half_dist_ij >= 0.
    soft_min = -logsumexp(-beta * vals) / beta
    We minimize -soft_min, so gradient = -sum_k w_k * d(val_k)/dx
    """
    centers = x.reshape(-1, 2)

    wx = 0.5 - np.abs(centers[:, 0])
    wy = 0.5 - np.abs(centers[:, 1])
    walls = np.minimum(wx, wy)

    ci = centers[_IU[0]]
    cj = centers[_IU[1]]
    diff = ci - cj
    dists = np.sqrt(np.sum(diff ** 2, axis=1))
    half_dists = dists / 2.0

    all_vals = np.concatenate([walls, half_dists])
    n_walls = N

    z = -beta * all_vals
    z_max = np.max(z)
    z_shifted = z - z_max
    exp_z = np.exp(z_shifted)
    sum_exp = np.sum(exp_z)
    weights = exp_z / sum_exp

    obj_val = float(z_max / beta + np.log(sum_exp) / beta)

    grad = np.zeros_like(x)

    # Wall gradients
    w_weights = weights[:n_walls]
    is_x_limit = wx < wy
    is_y_limit = wy < wx
    is_tie = ~(is_x_limit | is_y_limit)

    idx_x = np.nonzero(is_x_limit)[0]
    idx_y = np.nonzero(is_y_limit)[0]
    idx_t = np.nonzero(is_tie)[0]

    if len(idx_x) > 0:
        grad[2 * idx_x] -= w_weights[idx_x] * np.sign(centers[idx_x, 0])
    if len(idx_y) > 0:
        grad[2 * idx_y + 1] -= w_weights[idx_y] * np.sign(centers[idx_y, 1])
    if len(idx_t) > 0:
        grad[2 * idx_t] -= 0.5 * w_weights[idx_t] * np.sign(centers[idx_t, 0])
        grad[2 * idx_t + 1] -= 0.5 * w_weights[idx_t] * np.sign(centers[idx_t, 1])

    # Pair gradients
    p_weights = weights[n_walls:]
    dists_safe = np.maximum(dists, 1e-12)
    factor = p_weights / dists_safe
    contrib = factor[:, None] * diff

    np.add.at(grad, 2 * _IU[0], contrib[:, 0])
    np.add.at(grad, 2 * _IU[0] + 1, contrib[:, 1])
    np.add.at(grad, 2 * _IU[1], -contrib[:, 0])
    np.add.at(grad, 2 * _IU[1] + 1, -contrib[:, 1])

    return obj_val, grad


def _violation_obj_and_grad(x, r_target, beta):
    """Minimize sum of softplus violations for target radius."""
    centers = x.reshape(-1, 2)

    wx = 0.5 - np.abs(centers[:, 0])
    wy = 0.5 - np.abs(centers[:, 1])
    walls = np.minimum(wx, wy)
    wall_viol = r_target - walls

    ci = centers[_IU[0]]
    cj = centers[_IU[1]]
    diff = ci - cj
    dists = np.sqrt(np.sum(diff ** 2, axis=1))
    pair_viol = 2.0 * r_target - dists

    zw = np.clip(beta * wall_viol, -50, 50)
    zp = np.clip(beta * pair_viol, -50, 50)
    soft_w = np.log1p(np.exp(zw)) / beta
    soft_p = np.log1p(np.exp(zp)) / beta

    obj_val = float(np.sum(soft_w) + np.sum(soft_p))

    grad = np.zeros_like(x)

    sig_w = 1.0 / (1.0 + np.exp(-zw))
    is_x_limit = wx < wy
    is_y_limit = wy < wx
    is_tie = ~(is_x_limit | is_y_limit)
    idx_x = np.nonzero(is_x_limit)[0]
    idx_y = np.nonzero(is_y_limit)[0]
    idx_t = np.nonzero(is_tie)[0]

    if len(idx_x) > 0:
        grad[2 * idx_x] += sig_w[idx_x] * np.sign(centers[idx_x, 0])
    if len(idx_y) > 0:
        grad[2 * idx_y + 1] += sig_w[idx_y] * np.sign(centers[idx_y, 1])
    if len(idx_t) > 0:
        grad[2 * idx_t] += 0.5 * sig_w[idx_t] * np.sign(centers[idx_t, 0])
        grad[2 * idx_t + 1] += 0.5 * sig_w[idx_t] * np.sign(centers[idx_t, 1])

    sig_p = 1.0 / (1.0 + np.exp(-zp))
    dists_safe = np.maximum(dists, 1e-12)
    factor = sig_p / dists_safe
    contrib = factor[:, None] * diff

    np.add.at(grad, 2 * _IU[0], -contrib[:, 0])
    np.add.at(grad, 2 * _IU[0] + 1, -contrib[:, 1])
    np.add.at(grad, 2 * _IU[1], contrib[:, 0])
    np.add.at(grad, 2 * _IU[1] + 1, contrib[:, 1])

    return obj_val, grad


def _initial_grid():
    m = 8
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    grid = grid[:N].copy()
    rng = np.random.default_rng(12345)
    grid = grid + rng.normal(0, 0.004, grid.shape)
    return np.clip(grid, -0.49, 0.49)


def _hex_init():
    best = None
    best_r = 0
    rng = np.random.default_rng(42)
    for attempt in range(12):
        rows = rng.integers(5, 9)
        cols = int(np.ceil(N / rows))
        s_w = 1.0 / (cols + 0.5)
        s_h = 1.0 / (rows * 0.866 + 0.5)
        s = min(s_w, s_h)
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
            pad = rng.uniform(-0.45, 0.45, (N - len(centers), 2))
            centers = np.vstack([centers, pad])
        centers = np.clip(centers, -0.49, 0.49)
        r = _compute_radius(centers)
        if r > best_r:
            best_r = r
            best = centers.copy()
    return best if best is not None else _initial_grid()


def _differential_evolution(centers, rng, time_budget):
    """
    Global search via differential evolution on the soft-min objective.
    This avoids gradient issues and explores the landscape broadly.
    """
    start_time = time.time()
    pop_size = 30
    x = centers.flatten()

    # Initialize population around the starting point
    pop = np.zeros((pop_size, 2 * N))
    pop[0] = x
    for i in range(1, pop_size):
        pop[i] = x + rng.normal(0, 0.03, 2 * N)
        pop[i] = np.clip(pop[i], -0.5, 0.5)

    # Evaluate initial population
    beta = 300.0
    fitness = np.zeros(pop_size)
    for i in range(pop_size):
        f, _ = _soft_min_obj_and_grad(pop[i], beta)
        fitness[i] = f

    best_idx = np.argmin(fitness)
    best_x = pop[best_idx].copy()
    best_f = fitness[best_idx]

    # DE parameters
    F = 0.8
    CR = 0.9
    gen = 0

    while True:
        if time.time() - start_time > time_budget:
            break
        gen += 1

        # Adaptive parameters
        F_adapt = F * (0.95 ** (gen // 50))
        CR_adapt = CR

        for i in range(pop_size):
            if time.time() - start_time > time_budget:
                break

            # Select 3 distinct random individuals
            r1, r2, r3 = rng.choice(pop_size, 3, replace=False)
            while r1 == r2 or r1 == r3 or r2 == r3:
                r1, r2, r3 = rng.choice(pop_size, 3, replace=False)

            # Mutation
            v = pop[r1] + F_adapt * (pop[r2] - pop[r3])
            v = np.clip(v, -0.5, 0.5)

            # Crossover
            trial = pop[i].copy()
            j_rand = rng.integers(0, 2 * N)
            for j in range(2 * N):
                if j == j_rand or rng.random() < CR_adapt:
                    trial[j] = v[j]

            # Evaluate trial
            f_trial, _ = _soft_min_obj_and_grad(trial, beta)

            # Selection
            if f_trial < fitness[i]:
                pop[i] = trial
                fitness[i] = f_trial
                if f_trial < best_f:
                    best_f = f_trial
                    best_x = trial.copy()

        # Increase beta periodically for sharper minima
        if gen % 30 == 0:
            beta *= 1.5
            for i in range(pop_size):
                f, _ = _soft_min_obj_and_grad(pop[i], beta)
                fitness[i] = f
            best_idx = np.argmin(fitness)
            best_f = fitness[best_idx]
            best_x = pop[best_idx].copy()

    return best_x.reshape(-1, 2)


def _lbfgs_soft_min_polish(x0, beta, time_budget):
    start_time = time.time()

    def obj_and_grad(x):
        if time.time() - start_time > time_budget:
            return 0.0, np.zeros_like(x)
        return _soft_min_obj_and_grad(x, beta)

    try:
        result = opt.minimize(
            obj_and_grad, x0, jac=True, method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * (2 * N),
            options={'maxiter': 3000, 'ftol': 1e-18, 'gtol': 1e-14,
                     'maxfun': 20000, 'maxls': 50, 'disp': False}
        )
        polished = np.clip(result.x.reshape(-1, 2), -0.5, 0.5)
        return polished
    except Exception:
        return x0.reshape(-1, 2)


def _lbfgs_violation_polish(x0, r_target, beta, time_budget):
    start_time = time.time()

    def obj_and_grad(x):
        if time.time() - start_time > time_budget:
            return 0.0, np.zeros_like(x)
        return _violation_obj_and_grad(x, r_target, beta)

    try:
        result = opt.minimize(
            obj_and_grad, x0, jac=True, method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * (2 * N),
            options={'maxiter': 3000, 'ftol': 1e-18, 'gtol': 1e-14,
                     'maxfun': 20000, 'maxls': 50, 'disp': False}
        )
        polished = np.clip(result.x.reshape(-1, 2), -0.5, 0.5)
        return polished
    except Exception:
        return x0.reshape(-1, 2)


def _binary_search_refine(centers, time_budget):
    """Binary search on radius with violation polish."""
    start_time = time.time()
    best = centers.copy()
    best_r = _compute_radius(best)

    lo = best_r
    hi = min(0.075, best_r * 1.08)

    for i in range(20):
        if time.time() - start_time > time_budget:
            break
        mid = (lo + hi) / 2
        progress = i / 19.0
        beta = 300.0 + 700.0 * progress

        remaining = time_budget - (time.time() - start_time)
        if remaining < 5:
            break
        lbfgs_budget = min(10.0, remaining * 0.85)

        trial = _lbfgs_violation_polish(best.flatten(), mid, beta, lbfgs_budget)
        trial_r = _compute_radius(trial)

        if trial_r >= mid - 1e-9:
            lo = mid
            best = trial
            best_r = trial_r
        else:
            hi = mid

    return best, best_r


def _progressive_polish(centers, time_budget):
    """
    Progressive soft-min polish: increase beta geometrically.
    This is the core polishing phase.
    """
    start_time = time.time()
    best = centers.copy()
    best_r = _compute_radius(best)

    betas = [100, 200, 400, 800, 1500, 3000, 5000, 8000, 12000, 18000, 25000]

    for beta in betas:
        if time.time() - start_time > time_budget:
            break
        remaining = time_budget - (time.time() - start_time)
        if remaining < 4:
            break
        lbfgs_budget = min(8.0, remaining * 0.85)

        trial = _lbfgs_soft_min_polish(best.flatten(), beta, lbfgs_budget)
        trial_r = _compute_radius(trial)

        if trial_r > best_r:
            best_r = trial_r
            best = trial

    return best, best_r


def _contact_push(centers, time_budget):
    """
    Final push: identify the contact graph (pairs at minimum distance and
    walls at minimum distance), then do targeted L-BFGS with very high beta
    violation objective targeting slightly above current radius.
    """
    start_time = time.time()
    best = centers.copy()
    best_r = _compute_radius(best)

    # Try a few pushes above current best
    for frac in [1.0005, 1.001, 1.002, 1.003, 1.005, 1.007, 1.01, 1.013, 1.016]:
        if time.time() - start_time > time_budget - 2:
            break
        r_push = best_r * frac
        if r_push > 0.075:
            break
        remaining = time_budget - (time.time() - start_time)
        if remaining < 3:
            break
        lbfgs_budget = min(6.0, remaining * 0.85)
        trial = _lbfgs_violation_polish(best.flatten(), r_push, 8000.0, lbfgs_budget)
        trial_r = _compute_radius(trial)
        if trial_r > best_r:
            best_r = trial_r
            best = trial

    return best, best_r


def _simulated_annealing(centers, rng, time_budget):
    """
    SA with incremental evaluation for local refinement.
    """
    start_time = time.time()
    current = centers.copy()
    best = current.copy()
    best_r = _compute_radius(current)

    T_start = 0.012
    T_end = 0.00003
    est_steps = int(time_budget / 0.0003)
    n_steps = max(15000, min(80000, est_steps))
    cooling = (T_end / T_start) ** (1.0 / max(n_steps, 1))
    T = T_start

    point_pair_idx = [[] for _ in range(N)]
    for k in range(_NPAIRS):
        a, b = _IU[0][k], _IU[1][k]
        point_pair_idx[a].append(k)
        point_pair_idx[b].append(k)

    ci_all = current[_IU[0]]
    cj_all = current[_IU[1]]
    diff_all = ci_all - cj_all
    dist_sq_all = np.einsum('ij,ij->i', diff_all, diff_all)

    wall_x = 0.5 - np.abs(current[:, 0])
    wall_y = 0.5 - np.abs(current[:, 1])
    min_wall = np.min(np.minimum(wall_x, wall_y))
    min_pair = np.sqrt(np.min(dist_sq_all)) / 2.0
    curr_r = min(min_pair, min_wall)

    for step in range(n_steps):
        if time.time() - start_time > time_budget:
            break

        i = rng.integers(0, N)
        move_size = 0.002 + 0.007 * (T / T_start)
        if rng.random() < 0.04:
            move_size *= 3.0

        direction = rng.normal(size=2)
        norm = np.linalg.norm(direction)
        if norm > 1e-12:
            direction /= norm
        else:
            direction = np.array([1.0, 0.0])

        old_pos = current[i].copy()
        new_pos = old_pos + direction * move_size
        new_pos = np.clip(new_pos, -0.5, 0.5)

        twx = 0.5 - abs(new_pos[0])
        twy = 0.5 - abs(new_pos[1])
        trial_wall_i = min(twx, twy)

        affected = point_pair_idx[i]
        if len(affected) > 0:
            other_pts = []
            for k in affected:
                if _IU[0][k] == i:
                    other_pts.append(_IU[1][k])
                else:
                    other_pts.append(_IU[0][k])
            other_pos = current[other_pts]
            diffs = new_pos[None, :] - other_pos
            new_dists = np.sqrt(np.sum(diffs ** 2, axis=1))
            min_affected_pair = np.min(new_dists) / 2.0
        else:
            min_affected_pair = np.inf

        if len(affected) < _NPAIRS:
            mask = np.ones(_NPAIRS, dtype=bool)
            mask[affected] = False
            if np.any(mask):
                min_unaffected = np.sqrt(np.min(dist_sq_all[mask])) / 2.0
            else:
                min_unaffected = np.inf
            trial_min_pair = min(min_unaffected, min_affected_pair)
        else:
            trial_min_pair = min_affected_pair

        old_wall_i = min(wall_x[i], wall_y[i])
        if trial_wall_i >= old_wall_i:
            trial_min_wall = min_wall
        else:
            trial_walls = np.minimum(wall_x, wall_y).copy()
            trial_walls[i] = trial_wall_i
            trial_min_wall = np.min(trial_walls)

        trial_r = min(trial_min_pair, trial_min_wall)
        delta_r = trial_r - curr_r

        if delta_r >= 0 or (rng.random() < np.exp(delta_r / max(T, 1e-10))):
            current[i] = new_pos
            wall_x[i] = 0.5 - abs(new_pos[0])
            wall_y[i] = 0.5 - abs(new_pos[1])

            if len(affected) > 0:
                other_pts = []
                for k in affected:
                    if _IU[0][k] == i:
                        other_pts.append(_IU[1][k])
                    else:
                        other_pts.append(_IU[0][k])
                other_pos = current[other_pts]
                diffs = new_pos[None, :] - other_pos
                dist_sq_all[affected] = np.sum(diffs ** 2, axis=1)

            min_wall = np.min(np.minimum(wall_x, wall_y))
            min_pair = np.sqrt(np.min(dist_sq_all)) / 2.0
            curr_r = min(min_pair, min_wall)

            if curr_r > best_r:
                best_r = curr_r
                best = current.copy()

        T *= cooling

    return best, best_r


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE

    rng = np.random.default_rng(2024)

    # 1. Initialization - pick best of several starts
    inits = [
        _initial_grid(),
        _hex_init(),
    ]
    rng_init = np.random.default_rng(777)
    inits.append(rng_init.uniform(-0.45, 0.45, (N, 2)))

    best_centers = None
    best_radius = 0
    for init in inits:
        r = _compute_radius(init)
        if r > best_radius:
            best_radius = r
            best_centers = init.copy()

    if best_centers is None:
        best_centers = _initial_grid()
        best_radius = _compute_radius(best_centers)

    # 2. Differential Evolution global search (structural change)
    de_budget = min(40.0, (deadline - time.time()) * 0.35)
    if de_budget > 10:
        de_result = _differential_evolution(best_centers, rng, de_budget)
        de_r = _compute_radius(de_result)
        if de_r > best_radius:
            best_radius = de_r
            best_centers = de_result

    # 3. SA local refinement from best
    if time.time() < deadline - 35:
        sa_budget = min(25.0, (deadline - time.time()) - 40)
        if sa_budget > 5:
            c, r = _simulated_annealing(best_centers, rng, sa_budget)
            if r > best_radius:
                best_radius = r
                best_centers = c

        # Second SA restart
        if time.time() < deadline - 25:
            rng2 = np.random.default_rng(54321)
            perturbed = best_centers + rng2.normal(0, 0.008, (N, 2))
            perturbed = np.clip(perturbed, -0.49, 0.49)
            sa2_budget = min(15.0, (deadline - time.time()) - 30)
            if sa2_budget > 5:
                c2, r2 = _simulated_annealing(perturbed, rng2, sa2_budget)
                if r2 > best_radius:
                    best_radius = r2
                    best_centers = c2

    # 4. Progressive soft-min polish
    if time.time() < deadline - 20:
        polish_budget = min(35.0, (deadline - time.time()) - 15)
        if polish_budget > 5:
            best_centers, best_radius = _progressive_polish(best_centers, polish_budget)

    # 5. Binary search radius refinement
    if time.time() < deadline - 15:
        bs_budget = min(25.0, (deadline - time.time()) - 10)
        if bs_budget > 5:
            best_centers, best_radius = _binary_search_refine(best_centers, bs_budget)

    # 6. Final contact push
    if time.time() < deadline - 5:
        push_budget = min(10.0, (deadline - time.time()) - 3)
        if push_budget > 2:
            best_centers, best_radius = _contact_push(best_centers, push_budget)

    # Final validation
    if not np.all(np.isfinite(best_centers)) or best_centers.shape != (N, 2):
        return _initial_grid()

    return best_centers
