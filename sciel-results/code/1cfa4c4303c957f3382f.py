"""
Optimized packing of 50 equal circles in a unit square.
Combines Simulated Annealing for global search with soft-min (log-sum-exp)
L-BFGS polishing that directly maximizes the minimum distance metric.
"""
import numpy as np
import time
import scipy.optimize as opt
from scipy.spatial import cKDTree

N = 50
DEADLINE = 145.0  # Safety margin before 150s kill

# Precompute upper-triangle indices for pairwise operations
_IU = np.triu_indices(N, k=1)
_NPAIRS = len(_IU[0])


def _compute_radius(centers):
    """Compute the true maximum radius for a given configuration."""
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


def _soft_min_obj_and_grad(x, tau):
    """
    Objective: minimize -soft_min(all constraint distances).
    Uses log-sum-exp for smooth approximation of min.
    Returns (f, g) where f = -soft_min, g = gradient.
    """
    centers = x.reshape(-1, 2)

    # Wall distances per circle
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    walls = np.minimum(wall_dists_x, wall_dists_y)  # (N,)

    # Pairwise distances (upper triangle)
    ci = centers[_IU[0]]
    cj = centers[_IU[1]]
    diff = ci - cj
    dists = np.sqrt(np.sum(diff ** 2, axis=1))  # (_NPAIRS,)
    pair_dists = dists / 2.0  # radius contribution

    # All constraint distances: N walls + NPAIRS pairs
    all_dists = np.concatenate([walls, pair_dists])  # (N + _NPAIRS,)

    # Soft-min via log-sum-exp
    min_d = np.min(all_dists)
    shifted = all_dists - min_d
    # For numerical stability
    exp_vals = np.exp(-shifted / tau)
    sum_exp = np.sum(exp_vals)
    weights = exp_vals / sum_exp
    soft_min = min_d - tau * np.log(sum_exp)

    f = -soft_min

    # Gradient
    grad = np.zeros_like(x)

    # Wall gradient contributions
    for i in range(N):
        w = weights[i]
        if w < 1e-12:
            continue
        if wall_dists_x[i] < wall_dists_y[i]:
            grad[2 * i] += w * np.sign(centers[i, 0])
        elif wall_dists_y[i] < wall_dists_x[i]:
            grad[2 * i + 1] += w * np.sign(centers[i, 1])
        else:
            grad[2 * i] += 0.5 * w * np.sign(centers[i, 0])
            grad[2 * i + 1] += 0.5 * w * np.sign(centers[i, 1])

    # Pair gradient contributions (vectorized)
    pair_weights = weights[N:]  # (_NPAIRS,)
    # factor_k = weight_k / dist_k  (since d(pair_dist)/d(coord) involves 1/dist)
    dists_safe = np.maximum(dists, 1e-12)
    factor = pair_weights / dists_safe  # (_NPAIRS,)
    contrib = factor[:, None] * diff  # (_NPAIRS, 2)

    np.add.at(grad, 2 * _IU[0], -contrib[:, 0])
    np.add.at(grad, 2 * _IU[0] + 1, -contrib[:, 1])
    np.add.at(grad, 2 * _IU[1], contrib[:, 0])
    np.add.at(grad, 2 * _IU[1] + 1, contrib[:, 1])

    return f, grad


def _initial_grid():
    """Create initial grid layout with slight jitter."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    grid = grid[:N].copy()

    rng = np.random.default_rng(12345)
    grid = grid + rng.normal(0, 0.005, grid.shape)
    grid = np.clip(grid, -0.49, 0.49)

    return grid


def _hex_init():
    """Create a hexagonal packing initial layout."""
    best_grid = None
    best_r = 0

    rng = np.random.default_rng(42)

    for attempt in range(8):
        rows = rng.integers(5, 9)
        cols = int(np.ceil(N / rows))

        s_width = 1.0 / (cols + 0.3)
        s_height = 1.0 / (rows * 0.866 + 0.3)
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


def _simulated_annealing(centers, rng, time_budget):
    """
    Simulated annealing with incremental distance updates for speed.
    Maintains pairwise distance array and updates only affected entries.
    """
    start_time = time.time()
    current = centers.copy()
    best = current.copy()
    best_r = _compute_radius(current)

    # Precompute initial pairwise distances (upper triangle)
    ci = current[_IU[0]]
    cj = current[_IU[1]]
    diff = ci - cj
    pair_dists = np.sqrt(np.sum(diff ** 2, axis=1))  # (_NPAIRS,)

    # Wall distances
    wall_x = 0.5 - np.abs(current[:, 0])
    wall_y = 0.5 - np.abs(current[:, 1])
    walls = np.minimum(wall_x, wall_y)  # (N,)

    # Track minimum pair distance and minimum wall distance
    min_pair = np.min(pair_dists) / 2.0
    min_wall = np.min(walls)
    curr_r = min(min_pair, min_wall)

    T_start = 0.02
    T_end = 0.00003
    # Estimate steps based on budget; each step ~0.05-0.1ms with incremental updates
    est_steps = int(time_budget / 0.00008)
    n_steps = max(10000, min(200000, est_steps))
    cooling = (T_end / T_start) ** (1.0 / max(n_steps, 1))
    T = T_start

    # Build mapping: for each point i, which pair indices involve it
    # pair_idx for (i,j) where i<j: we need to find all pairs involving point i
    # Precompute for speed: for each point, list of pair indices
    point_to_pairs = [[] for _ in range(N)]
    for k in range(_NPAIRS):
        i = _IU[0][k]
        j = _IU[1][k]
        point_to_pairs[i].append(k)
        point_to_pairs[j].append(k)

    for step in range(n_steps):
        if time.time() - start_time > time_budget:
            break

        # Move generation: pick a point and perturb it
        i = rng.integers(0, N)
        move_size = 0.003 + 0.012 * (T / T_start)
        direction = rng.normal(size=2)
        norm = np.linalg.norm(direction)
        if norm > 1e-12:
            direction /= norm
        else:
            direction = np.array([1.0, 0.0])

        old_pos = current[i].copy()
        new_pos = old_pos + direction * move_size
        new_pos = np.clip(new_pos, -0.5, 0.5)

        # Update pairwise distances involving point i
        for k in point_to_pairs[i]:
            j = _IU[1][k] if _IU[0][k] == i else _IU[0][k]
            old_diff = current[i] - current[j]
            new_diff = new_pos - current[j]
            pair_dists[k] = np.sqrt(np.sum(new_diff ** 2))

        # Update wall distance for point i
        new_wall_x = 0.5 - abs(new_pos[0])
        new_wall_y = 0.5 - abs(new_pos[1])
        new_wall_i = min(new_wall_x, new_wall_y)
        old_wall_i = walls[i]
        walls[i] = new_wall_i

        # Compute new min pair and min wall (approximate: only need to check if min changed)
        # For speed, track running min; if the moved point's pair was the min, recompute
        # Simple approach: just take min over all (fast with numpy for 1225 pairs)
        new_min_pair = np.min(pair_dists) / 2.0
        new_min_wall = np.min(walls)
        new_r = min(new_min_pair, new_min_wall)

        delta_r = new_r - curr_r

        if delta_r >= 0:
            current[i] = new_pos
            curr_r = new_r
            min_pair = new_min_pair
            min_wall = new_min_wall
            if new_r > best_r:
                best_r = new_r
                best = current.copy()
        else:
            if rng.random() < np.exp(delta_r / max(T, 1e-10)):
                current[i] = new_pos
                curr_r = new_r
                min_pair = new_min_pair
                min_wall = new_min_wall
            else:
                # Revert: restore old distances
                for k in point_to_pairs[i]:
                    j = _IU[1][k] if _IU[0][k] == i else _IU[0][k]
                    old_diff = old_pos - current[j]
                    pair_dists[k] = np.sqrt(np.sum(old_diff ** 2))
                walls[i] = old_wall_i

        T *= cooling

        # Periodic shake: every ~5000 steps, do a small random perturbation of all points
        if step % 5000 == 4999 and step < n_steps - 5000:
            shake = rng.normal(0, 0.005, (N, 2))
            trial = np.clip(current + shake, -0.5, 0.5)
            trial_r = _compute_radius(trial)
            if trial_r > curr_r * 0.95:  # accept if not too bad
                current = trial
                # Recompute all distances
                ci = current[_IU[0]]
                cj = current[_IU[1]]
                diff = ci - cj
                pair_dists = np.sqrt(np.sum(diff ** 2, axis=1))
                wall_x = 0.5 - np.abs(current[:, 0])
                wall_y = 0.5 - np.abs(current[:, 1])
                walls = np.minimum(wall_x, wall_y)
                min_pair = np.min(pair_dists) / 2.0
                min_wall = np.min(walls)
                curr_r = min(min_pair, min_wall)
                if curr_r > best_r:
                    best_r = curr_r
                    best = current.copy()

    # Final exact computation
    final_r = _compute_radius(best)
    if final_r > best_r:
        best_r = final_r

    return best, best_r


def _soft_min_polish(x0, tau, time_budget):
    """
    Run L-BFGS-B to minimize -soft_min objective (i.e., maximize soft-min radius).
    """
    start_time = time.time()

    def obj_and_grad(x):
        if time.time() - start_time > time_budget:
            return 0.0, np.zeros_like(x)
        return _soft_min_obj_and_grad(x, tau)

    try:
        result = opt.minimize(
            obj_and_grad, x0, jac=True, method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * (2 * N),
            options={'maxiter': 5000, 'ftol': 1e-18, 'gtol': 1e-16,
                     'maxfun': 25000, 'disp': False}
        )
        polished = result.x.reshape(-1, 2)
        polished = np.clip(polished, -0.5, 0.5)
        return polished
    except Exception:
        return x0.reshape(-1, 2)


def _refine_with_soft_min(centers, time_budget):
    """
    Progressive soft-min polishing with decreasing tau for higher precision.
    """
    start_time = time.time()
    best = centers.copy()
    best_r = _compute_radius(best)

    # Start with moderate tau, decrease for sharper min approximation
    # More aggressive tau schedule for better convergence
    taus = [0.004, 0.002, 0.001, 0.0005, 0.0002, 0.0001, 0.00005]

    for tau in taus:
        if time.time() - start_time > time_budget:
            break

        remaining = time_budget - (time.time() - start_time)
        if remaining < 2:
            break
        lbfgs_budget = min(15.0, remaining * 0.85)

        trial = _soft_min_polish(best.flatten(), tau, lbfgs_budget)
        trial_r = _compute_radius(trial)

        if trial_r > best_r:
            best_r = trial_r
            best = trial

    return best, best_r


def _violation_polish(x0, r_target, beta, time_budget):
    """
    Fallback: L-BFGS-B to minimize softplus violation for target radius.
    Used for final tightening when soft-min stalls.
    """
    start_time = time.time()

    def obj_and_grad(x):
        if time.time() - start_time > time_budget:
            return 0.0, np.zeros_like(x)
        centers = x.reshape(-1, 2)

        # Wall violations
        wall_dists_x = 0.5 - np.abs(centers[:, 0])
        wall_dists_y = 0.5 - np.abs(centers[:, 1])
        walls = np.minimum(wall_dists_x, wall_dists_y)
        wall_viol = r_target - walls

        # Pair violations
        ci = centers[_IU[0]]
        cj = centers[_IU[1]]
        diff = ci - cj
        dists = np.sqrt(np.sum(diff ** 2, axis=1))
        pair_viol = 2.0 * r_target - dists

        z_wall = np.clip(beta * wall_viol, -50, 50)
        z_pair = np.clip(beta * pair_viol, -50, 50)

        soft_wall = np.log1p(np.exp(z_wall)) / beta
        soft_pair = np.log1p(np.exp(z_pair)) / beta

        f = float(np.sum(soft_wall) + np.sum(soft_pair))

        # Gradient
        grad = np.zeros_like(x)

        sig_wall = 1.0 / (1.0 + np.exp(-z_wall))
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

        sig_pair = 1.0 / (1.0 + np.exp(-z_pair))
        dists_safe = np.maximum(dists, 1e-12)
        factor = sig_pair / dists_safe
        contrib = factor[:, None] * diff

        np.add.at(grad, 2 * _IU[0], -contrib[:, 0])
        np.add.at(grad, 2 * _IU[0] + 1, -contrib[:, 1])
        np.add.at(grad, 2 * _IU[1], contrib[:, 0])
        np.add.at(grad, 2 * _IU[1] + 1, contrib[:, 1])

        return f, grad

    try:
        result = opt.minimize(
            obj_and_grad, x0, jac=True, method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * (2 * N),
            options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-12,
                     'maxfun': 10000, 'disp': False}
        )
        polished = result.x.reshape(-1, 2)
        polished = np.clip(polished, -0.5, 0.5)
        return polished
    except Exception:
        return x0.reshape(-1, 2)


def _final_tighten(centers, time_budget):
    """
    Final high-precision tightening using violation-based polish
    targeting slightly above current best radius.
    """
    start_time = time.time()
    best = centers.copy()
    best_r = _compute_radius(best)

    # Try to push slightly above current best with finer increments
    for push_frac in [1.0005, 1.001, 1.002, 1.003, 1.005]:
        if time.time() - start_time > time_budget:
            break

        r_push = best_r * push_frac
        if r_push > 0.08:
            break

        remaining = time_budget - (time.time() - start_time)
        if remaining < 2:
            break
        lbfgs_budget = min(12.0, remaining * 0.85)

        trial = _violation_polish(best.flatten(), r_push, 3000.0, lbfgs_budget)
        trial_r = _compute_radius(trial)

        if trial_r > best_r:
            best_r = trial_r
            best = trial

    # Also try soft-min with very small tau for final precision
    if time.time() - start_time < time_budget - 4:
        remaining = time_budget - (time.time() - start_time)
        if remaining > 6:
            trial = _soft_min_polish(best.flatten(), 0.00003, min(10.0, remaining * 0.9))
            trial_r = _compute_radius(trial)
            if trial_r > best_r:
                best_r = trial_r
                best = trial

    return best, best_r


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE

    rng = np.random.default_rng(777)

    # 1. Initialization - try multiple seeds for diversity
    inits = [
        _initial_grid(),
        _hex_init(),
    ]

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

    # 2. Simulated Annealing (Global Search) - give it the majority of time
    # With incremental updates, SA is much faster, so we can run longer
    sa_budget = min(65.0, (deadline - time.time()) * 0.50)
    if sa_budget > 5:
        best_centers, best_radius = _simulated_annealing(best_centers, rng, sa_budget)

    # 3. Soft-Min Progressive Polish (Primary refinement)
    remaining = deadline - time.time()
    if remaining > 25:
        polish_budget = min(40.0, remaining * 0.65)
        best_centers, best_radius = _refine_with_soft_min(
            best_centers, time_budget=polish_budget
        )

    # 4. Final Tightening (Violation-based push + ultra-fine soft-min)
    remaining = deadline - time.time()
    if remaining > 10:
        tighten_budget = min(18.0, remaining * 0.80)
        best_centers, best_radius = _final_tighten(
            best_centers, time_budget=tighten_budget
        )

    # Final Validation
    if not np.all(np.isfinite(best_centers)) or best_centers.shape != (N, 2):
        return _initial_grid()

    return best_centers
