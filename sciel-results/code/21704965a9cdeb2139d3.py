"""50 equal circles in a unit square: binary-search feasibility with projected gradient descent."""
import numpy as np
import time
import scipy.optimize as opt
from scipy.spatial import cKDTree

N = 50
DEADLINE = 142.0


def _compute_radius(centers):
    """Compute the true maximum common radius for a configuration."""
    centers = np.asarray(centers, dtype=np.float64)
    if centers.shape != (N, 2) or not np.all(np.isfinite(centers)):
        return 0.0
    wall_x = 0.5 - np.abs(centers[:, 0])
    wall_y = 0.5 - np.abs(centers[:, 1])
    min_wall = np.min(np.minimum(wall_x, wall_y))
    if min_wall < 0:
        return 0.0
    tree = cKDTree(centers)
    dists, _ = tree.query(centers, k=2)
    min_pair = np.min(dists[:, 1]) / 2.0
    return min(min_pair, min_wall)


def _feasibility_check(centers, r):
    """Check if circles of radius r fit: no overlap and inside square."""
    if np.any(centers < -0.5 + r - 1e-12) or np.any(centers > 0.5 - r + 1e-12):
        return False
    tree = cKDTree(centers)
    dists, _ = tree.query(centers, k=2)
    return np.all(dists[:, 1] >= 2.0 * r - 1e-12)


def _hex_init():
    """Generate a hexagonal lattice seed."""
    r_est = 0.07
    dx = 2.0 * r_est
    dy = dx * np.sqrt(3.0) / 2.0
    pts = []
    row = 0
    while len(pts) < N:
        offset = (row % 2) * dx / 2.0
        col = 0
        while len(pts) < N:
            x = -0.5 + r_est + dx / 2.0 + col * dx + offset
            y = -0.5 + r_est + dy / 2.0 + row * dy
            if x < 0.5 - r_est and y < 0.5 - r_est:
                pts.append([x, y])
            col += 1
        row += 1
    return np.array(pts[:N])


def _grid_init():
    """Generate a grid seed."""
    m = 8
    ticks = (np.arange(m) + 0.5) / m - 0.5
    pts = []
    for y in ticks:
        for x in ticks:
            pts.append([x, y])
            if len(pts) >= N:
                break
        if len(pts) >= N:
            break
    return np.array(pts[:N])


def _project(centers, r):
    """Project centers to be inside the feasible region for radius r."""
    centers = np.clip(centers, -0.5 + r, 0.5 - r)
    return centers


def _pgd_feasibility(centers0, r, time_budget, rng):
    """
    Try to find a feasible configuration for circles of radius r
    using projected gradient descent on the violation objective.
    Returns (best_centers, best_violation).
    """
    start = time.time()
    cur = centers0.copy()
    best = cur.copy()
    best_viol = _violation(cur, r)

    step = 0.005
    damping = 0.7
    momentum = np.zeros_like(cur)

    for it in range(50000):
        if time.time() - start > time_budget:
            break

        # Compute pairwise distances and wall violations
        diff = cur[:, np.newaxis, :] - cur[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)

        # Pairwise violation: dist < 2r
        pair_viol = np.maximum(0.0, 2.0 * r - dists)

        # Wall violation: center outside [-0.5+r, 0.5-r]
        wall_viol = np.zeros_like(cur)
        for dim in range(2):
            wall_viol[:, dim] = np.maximum(0.0, (-0.5 + r) - cur[:, dim]) + \
                                np.maximum(0.0, cur[:, dim] - (0.5 - r))

        total_viol = np.sum(pair_viol) + np.sum(wall_viol)

        if total_viol < best_viol:
            best_viol = total_viol
            best = cur.copy()

        if total_viol < 1e-10:
            break

        # Gradient of violation w.r.t. centers
        grad = np.zeros_like(cur)

        # Pairwise gradient
        iu, ju = np.triu_indices(N, k=1)
        for k in range(len(iu)):
            i, j = iu[k], ju[k]
            d = dists[i, j]
            if d < 2.0 * r and d > 1e-12:
                viol = 2.0 * r - d
                # gradient of max(0, 2r - d) w.r.t. positions
                unit = (cur[i] - cur[j]) / d
                grad[i] += unit
                grad[j] -= unit

        # Wall gradient
        for dim in range(2):
            lo_viol = np.maximum(0.0, (-0.5 + r) - cur[:, dim])
            hi_viol = np.maximum(0.0, cur[:, dim] - (0.5 - r))
            grad[:, dim] += (-lo_viol + hi_viol)

        # Update with momentum
        momentum = damping * momentum + grad
        cur += step * momentum
        cur = _project(cur, r)

        # Adaptive step
        new_viol = _violation(cur, r)
        if new_viol > total_viol * 1.5:
            step *= 0.5
        elif new_viol < total_viol * 0.5:
            step = min(step * 1.2, 0.02)

    return best, best_viol


def _violation(centers, r):
    """Compute total violation for circles of radius r."""
    # Wall violations
    wall_viol = 0.0
    for dim in range(2):
        wall_viol += np.sum(np.maximum(0.0, (-0.5 + r) - centers[:, dim]))
        wall_viol += np.sum(np.maximum(0.0, centers[:, dim] - (0.5 - r)))

    # Pairwise violations
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    pair_viol = np.sum(np.maximum(0.0, 2.0 * r - dists))

    return wall_viol + pair_viol


def _binary_search_radius(centers_init, time_budget, rng):
    """
    Binary search on radius r, using PGD to check feasibility at each step.
    Returns the best feasible radius and corresponding centers.
    """
    start = time.time()
    r_lo = 0.05
    r_hi = 0.08

    # First, verify that r_lo is feasible
    trial_centers, trial_viol = _pgd_feasibility(centers_init, r_lo, 2.0, rng)
    if trial_viol > 1e-8:
        # Even low radius is hard; try to improve
        r_lo = 0.04

    best_r = r_lo
    best_centers = centers_init.copy()

    # Binary search
    for _ in range(20):
        if time.time() - start > time_budget:
            break
        r_mid = (r_lo + r_hi) / 2.0
        remaining = time_budget - (time.time() - start)
        if remaining < 2.0:
            break

        # Try to find feasible config for r_mid
        trial_centers, trial_viol = _pgd_feasibility(best_centers, r_mid,
                                                      min(remaining * 0.4, 5.0), rng)
        if trial_viol < 1e-8:
            r_lo = r_mid
            best_r = r_mid
            best_centers = trial_centers
        else:
            r_hi = r_mid

    return best_r, best_centers


def _sa_search(centers, r_target, time_budget, rng):
    """
    Simulated annealing to find feasible configuration for target radius.
    """
    start = time.time()
    cur = centers.copy()
    best = cur.copy()
    best_viol = _violation(cur, r_target)

    temp = 0.002
    cool = 0.997

    for _ in range(20000):
        if time.time() - start > time_budget:
            break

        i = rng.integers(N)
        trial = cur.copy()
        trial[i] += rng.normal(0, temp, 2)
        trial[i] = _project(trial[i], r_target)

        r_t_viol = _violation(trial, r_target)
        r_c_viol = _violation(cur, r_target)

        # Accept if better or with probability
        if r_t_viol <= r_c_viol or rng.random() < np.exp((r_c_viol - r_t_viol) / max(temp, 1e-6)):
            cur = trial

        if r_t_viol < best_viol:
            best_viol = r_t_viol
            best = cur.copy()

        temp *= cool
        if temp < 1e-6:
            temp = 1e-6

    return best, best_viol


def _lbfgs_polish(centers, r_target, budget):
    """
    Use L-BFGS to minimize violation for a given target radius.
    """
    start = time.time()

    def obj_and_grad(x):
        if time.time() - start > budget:
            return 0.0, np.zeros_like(x)
        c = x.reshape(-1, 2)
        # Violation
        wall_viol = np.zeros_like(c)
        for dim in range(2):
            wall_viol[:, dim] = np.maximum(0.0, (-0.5 + r_target) - c[:, dim]) + \
                                np.maximum(0.0, c[:, dim] - (0.5 - r_target))
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        pair_viol = np.maximum(0.0, 2.0 * r_target - dists)
        total = np.sum(wall_viol) + np.sum(pair_viol)

        # Gradient
        grad = np.zeros_like(c)
        for dim in range(2):
            lo = np.maximum(0.0, (-0.5 + r_target) - c[:, dim])
            hi = np.maximum(0.0, c[:, dim] - (0.5 - r_target))
            grad[:, dim] += (-lo + hi)

        iu, ju = np.triu_indices(N, k=1)
        for k in range(len(iu)):
            i, j = iu[k], ju[k]
            d = dists[i, j]
            if d < 2.0 * r_target and d > 1e-12:
                unit = (c[i] - c[j]) / d
                grad[i] += unit
                grad[j] -= unit

        return total, grad.flatten()

    x0 = centers.flatten()
    try:
        res = opt.minimize(obj_and_grad, x0, jac=True, method='L-BFGS-B',
                           bounds=[(-0.5 + r_target, 0.5 - r_target)] * (2 * N),
                           options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12, 'disp': False})
        polished = np.clip(res.x.reshape(-1, 2), -0.5 + r_target, 0.5 - r_target)
        return polished
    except:
        return centers


def construct():
    t0 = time.time()
    deadline = t0 + DEADLINE
    rng = np.random.default_rng(12345)

    # Generate initial seeds
    hex_seed = _hex_init()
    grid_seed = _grid_init()

    # Use hex seed as primary
    centers = hex_seed.copy()
    best_r = _compute_radius(centers)
    best_centers = centers.copy()

    # Phase 1: Binary search on radius with PGD feasibility
    remaining = deadline - time.time()
    if remaining > 30:
        r_found, centers = _binary_search_radius(centers, min(remaining * 0.5, 50.0), rng)
        if r_found > best_r:
            best_r = r_found
            best_centers = centers.copy()

    # Phase 2: SA to further improve
    remaining = deadline - time.time()
    if remaining > 15:
        # Try SA at current best radius
        r_try = best_r + 0.0005
        sa_centers, sa_viol = _sa_search(best_centers, r_try, min(remaining * 0.4, 12.0), rng)
        if sa_viol < 1e-8:
            # Feasible at higher radius
            new_r = _compute_radius(sa_centers)
            if new_r > best_r:
                best_r = new_r
                best_centers = sa_centers.copy()

    # Phase 3: L-BFGS polish at current best radius
    remaining = deadline - time.time()
    if remaining > 10:
        polished = _lbfgs_polish(best_centers, best_r, min(remaining * 0.5, 8.0))
        new_r = _compute_radius(polished)
        if new_r > best_r:
            best_r = new_r
            best_centers = polished.copy()

    # Phase 4: Another round of binary search from improved starting point
    remaining = deadline - time.time()
    if remaining > 20:
        r_found2, centers2 = _binary_search_radius(best_centers, min(remaining * 0.5, 25.0), rng)
        if r_found2 > best_r:
            best_r = r_found2
            best_centers = centers2.copy()

    # Final L-BFGS polish
    remaining = deadline - time.time()
    if remaining > 5:
        polished = _lbfgs_polish(best_centers, best_r, min(remaining * 0.6, 5.0))
        new_r = _compute_radius(polished)
        if new_r > best_r:
            best_r = new_r
            best_centers = polished.copy()

    # Ensure validity
    if not np.all(np.isfinite(best_centers)) or best_centers.shape != (N, 2):
        return hex_seed
    # Final safety clip
    best_centers = np.clip(best_centers, -0.4999, 0.4999)
    return best_centers
