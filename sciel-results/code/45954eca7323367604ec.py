"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import scipy.optimize as opt

N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize as opt

N = 50


def _compute_radius(centers):
    """
    Compute the true maximum radius for a given configuration.
    """
    # Wall distances
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    min_wall = np.min(np.minimum(wall_dists_x, wall_dists_y))

    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0

    return min(min_pair, min_wall)


def _violation(centers, r):
    """
    Compute sum of squared violations for target radius r.
    Wall violation: max(0, r - wall_dist)
    Pair violation: max(0, 2r - pair_dist)
    """
    wall_x = 0.5 - np.abs(centers[:, 0])
    wall_y = 0.5 - np.abs(centers[:, 1])
    walls = np.minimum(wall_x, wall_y)
    wall_viol = np.maximum(0.0, r - walls)

    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    pair_viol = np.maximum(0.0, 2.0 * r - dists)

    return np.sum(wall_viol**2) + np.sum(pair_viol**2)


def _soft_min_obj(x, tau):
    """
    Objective function: Minimize -soft_min(distances)/2.
    x: flattened array of centers (2*N)
    tau: temperature for soft-min
    """
    centers = x.reshape(-1, 2)

    # Wall distances
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    walls = np.minimum(wall_dists_x, wall_dists_y)

    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    d_flat = dists[~np.eye(N, dtype=bool)]

    all_dists = np.concatenate([walls, d_flat])

    # Soft min via log-sum-exp
    vals = -all_dists / tau
    vals_max = np.max(vals)
    sum_exp = np.sum(np.exp(vals - vals_max))
    soft_min_val = -np.min(all_dists) - tau * (np.log(sum_exp) + vals_max)

    return -soft_min_val / 2.0


def _soft_min_grad(x, tau):
    """
    Gradient of -soft_min(distances)/2 with respect to x.
    """
    centers = x.reshape(-1, 2)

    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    walls = np.minimum(wall_dists_x, wall_dists_y)

    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    d_flat = dists[~np.eye(N, dtype=bool)]

    all_dists = np.concatenate([walls, d_flat])

    vals = -all_dists / tau
    vals_max = np.max(vals)
    exp_vals = np.exp(vals - vals_max)
    sum_exp = np.sum(exp_vals)
    weights = exp_vals / sum_exp

    # d(soft_min)/d(dist_i) = -weights[i]
    # soft_min = min approximation, we want to maximize it
    # objective = -soft_min / 2
    # d(objective)/d(dist_i) = weights[i] / 2

    # Now compute d(objective)/d(x) via chain rule
    grad = np.zeros_like(x)

    # Wall contributions
    # wall_dist_i = min(0.5 - |c_ix|, 0.5 - |c_iy|)
    # d(wall_i)/d(c_ix) = -sign(c_ix) if x-constraint is active
    # d(wall_i)/d(c_iy) = -sign(c_iy) if y-constraint is active
    wall_weights = weights[:N]  # weights for wall distances

    for i in range(N):
        cx, cy = centers[i, 0], centers[i, 1]
        wx = 0.5 - np.abs(cx)
        wy = 0.5 - np.abs(cy)
        w = wall_weights[i]
        if wx <= wy:
            # x-constraint active
            if cx > 0:
                grad[2*i] += w / 2.0
            else:
                grad[2*i] -= w / 2.0
        else:
            # y-constraint active
            if cy > 0:
                grad[2*i+1] += w / 2.0
            else:
                grad[2*i+1] -= w / 2.0

    # Pair contributions
    pair_weights = weights[N:]  # weights for pair distances
    # Reconstruct pair index mapping
    pair_idx = []
    for i in range(N):
        for j in range(i+1, N):
            pair_idx.append((i, j))

    for k, (i, j) in enumerate(pair_idx):
        w = pair_weights[k]
        d_ij = centers[i] - centers[j]
        dist_ij = dists[i, j]
        if dist_ij < 1e-15:
            continue
        # d(dist_ij)/d(c_i) = d_ij / dist_ij
        # d(dist_ij)/d(c_j) = -d_ij / dist_ij
        # d(objective)/d(c_i) += w/2 * d_ij/dist_ij
        # d(objective)/d(c_j) += w/2 * (-d_ij/dist_ij)
        factor = w / (2.0 * dist_ij)
        grad[2*i] += factor * d_ij[0]
        grad[2*i+1] += factor * d_ij[1]
        grad[2*j] -= factor * d_ij[0]
        grad[2*j+1] -= factor * d_ij[1]

    return grad


def _sa_search(centers, rng, time_budget):
    """
    Simulated annealing search to improve the configuration.
    """
    start_time = time.time()
    current_centers = centers.copy()
    current_radius = _compute_radius(current_centers)
    best_centers = current_centers.copy()
    best_radius = current_radius

    T_start = 0.03
    T_end = 1e-7
    max_steps = 800000

    alpha = (T_end / T_start) ** (1.0 / max_steps)

    for step in range(max_steps):
        if time.time() > start_time + time_budget:
            break

        T = T_start * (alpha ** step)

        # Move: perturb 1 to 3 random circles
        num_movers = rng.integers(1, 4)
        indices = rng.choice(N, size=num_movers, replace=False)

        step_size = 0.04 * (0.3 + 0.7 * T / T_start)
        delta = rng.normal(0, step_size, (num_movers, 2))

        new_centers = current_centers.copy()
        new_centers[indices] += delta
        new_centers = np.clip(new_centers, -0.5 + 1e-9, 0.5 - 1e-9)

        new_radius = _compute_radius(new_centers)

        if new_radius > current_radius:
            current_centers = new_centers
            current_radius = new_radius
            if current_radius > best_radius:
                best_radius = current_radius
                best_centers = new_centers.copy()
        else:
            delta_r = new_radius - current_radius
            if delta_r < -15 * T:
                prob = 0.0
            else:
                prob = np.exp(delta_r / T)

            if rng.random() < prob:
                current_centers = new_centers
                current_radius = new_radius

    return best_centers, best_radius


def _gradient_polish(centers, rng, time_budget):
    """
    Gradient-based polish using L-BFGS-B with soft-min objective.
    """
    start_time = time.time()
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    x0 = best_centers.flatten()

    # Try multiple tau values from coarse to fine
    tau_values = [0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001, 0.00005]

    for tau in tau_values:
        if time.time() > start_time + time_budget:
            break

        def obj(x):
            return _soft_min_obj(x, tau)

        def grad(x):
            return _soft_min_grad(x, tau)

        bounds = [(-0.5, 0.5)] * (2 * N)

        try:
            res = opt.minimize(
                obj,
                x0,
                jac=grad,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            centers_try = res.x.reshape(-1, 2)
            r_try = _compute_radius(centers_try)
            if r_try > best_radius:
                best_radius = r_try
                best_centers = centers_try.copy()
                x0 = res.x
        except Exception:
            pass

    return best_centers, best_radius


def _binary_search_polish(centers, best_radius, rng, time_budget):
    """
    Binary search on radius with penalty-based L-BFGS-B.
    """
    start_time = time.time()
    x0 = centers.flatten()
    r_low = best_radius
    r_high = min(0.08, best_radius * 1.02)

    def _penalty_obj(x, r_target, penalty_weight):
        c = x.reshape(-1, 2)
        wall_x = 0.5 - np.abs(c[:, 0])
        wall_y = 0.5 - np.abs(c[:, 1])
        walls = np.minimum(wall_x, wall_y)
        wall_viol = np.maximum(0.0, r_target - walls)

        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        pair_viol = np.maximum(0.0, 2.0 * r_target - dists)

        return penalty_weight * (np.sum(wall_viol**2) + np.sum(pair_viol**2))

    def _is_feasible(c, r_target, tol=1e-9):
        wall_x = 0.5 - np.abs(c[:, 0])
        wall_y = 0.5 - np.abs(c[:, 1])
        walls = np.minimum(wall_x, wall_y)
        if np.min(walls) < r_target - tol:
            return False
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        if np.min(dists) < 2.0 * r_target - tol:
            return False
        return True

    penalty_weight = 1000.0
    max_bisect_iters = 20

    for bisect_iter in range(max_bisect_iters):
        if time.time() > start_time + time_budget:
            break

        r_mid = (r_low + r_high) / 2.0
        success = False

        for attempt in range(3):
            if time.time() > start_time + time_budget:
                break

            x_start = x0.copy()
            if attempt > 0:
                x_start += rng.normal(0, 0.003, x_start.shape)
                x_start = np.clip(x_start, -0.5 + 1e-9, 0.5 - 1e-9)

            def obj(x):
                return _penalty_obj(x, r_mid, penalty_weight)

            bounds = [(-0.5, 0.5)] * (2 * N)

            try:
                res = opt.minimize(
                    obj,
                    x_start,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 400, 'ftol': 1e-13, 'gtol': 1e-11}
                )

                centers_try = res.x.reshape(-1, 2)
                r_actual = _compute_radius(centers_try)

                if _is_feasible(centers_try, r_mid, tol=1e-8):
                    if r_actual > best_radius:
                        best_radius = r_actual
                        centers = centers_try.copy()
                    x0 = res.x
                    success = True
                    break
                elif r_actual > best_radius:
                    best_radius = r_actual
                    centers = centers_try.copy()
                    x0 = res.x
            except Exception:
                pass

        if success:
            r_low = r_mid
        else:
            r_high = r_mid

        if r_high - r_low < 1e-8:
            break

    return centers, best_radius


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s limit

    rng = np.random.default_rng(42)

    # 1. Initialization: try multiple starting configurations
    best_centers = None
    best_radius = 0.0

    # Initialization 1: Jittered grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    init1 = grid[:N].copy()
    init1 += rng.normal(0, 0.015, init1.shape)
    init1 = np.clip(init1, -0.49, 0.49)

    # Initialization 2: Hexagonal lattice
    points = []
    spacing = 0.19
    for i in range(10):
        for j in range(10):
            x = -0.5 + j * spacing + (i % 2) * spacing / 2
            y = -0.5 + i * spacing * 0.866
            if -0.5 + 1e-9 <= x <= 0.5 - 1e-9 and -0.5 + 1e-9 <= y <= 0.5 - 1e-9:
                points.append([x, y])
    rng.shuffle(points)
    init2 = np.array(points[:N])
    init2 += rng.normal(0, 0.015, init2.shape)
    init2 = np.clip(init2, -0.49, 0.49)

    # Initialization 3: Random uniform
    init3 = rng.uniform(-0.45, 0.45, (N, 2))

    # Evaluate all inits
    inits = [init1, init2, init3]
    for init in inits:
        r = _compute_radius(init)
        if r > best_radius:
            best_radius = r
            best_centers = init.copy()

    # 2. Simulated Annealing
    sa_budget = min(50.0, deadline - start_time - 60.0)
    if sa_budget > 10.0:
        best_centers, best_radius = _sa_search(best_centers, rng, sa_budget)

    # 3. Gradient polish
    if time.time() < deadline - 15.0:
        grad_budget = min(40.0, deadline - time.time() - 15.0)
        if grad_budget > 5.0:
            best_centers, best_radius = _gradient_polish(best_centers, rng, grad_budget)

    # 4. Binary search polish
    if time.time() < deadline - 5.0:
        bisect_budget = min(30.0, deadline - time.time() - 5.0)
        if bisect_budget > 5.0:
            best_centers, best_radius = _binary_search_polish(best_centers, best_radius, rng, bisect_budget)

    # 5. Final high-precision polish
    if time.time() < deadline - 3.0:
        final_budget = max(1.0, deadline - time.time() - 3.0)
        best_centers, best_radius = _gradient_polish(best_centers, rng, final_budget)

    # Ensure all values are finite and within bounds
    best_centers = np.clip(best_centers, -0.5 + 1e-12, 0.5 - 1e-12)

    return best_centers
# EVOLVE-BLOCK-END
