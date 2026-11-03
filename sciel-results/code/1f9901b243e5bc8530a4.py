"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

DEADLINE = 145.0  # Safety margin before 150s limit


def _true_score(centers):
    """Compute the true packing radius for a given configuration."""
    centers = np.asarray(centers, dtype=np.float64)
    if centers.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(np.abs(centers) > 0.5):
        return 0.0
    boundary = (0.5 - np.abs(centers)).min()
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = (diffs ** 2).sum(axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    min_dist = np.sqrt(dists_sq.min())
    return min(boundary, min_dist / 2.0)


def _soft_min(values, beta=50.0):
    """Soft minimum using log-sum-exp for differentiability."""
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return 0.0
    v_min = v.min()
    shifted = v - v_min
    shifted = np.clip(shifted, 0, 100.0 / beta)
    log_sum = np.log(np.sum(np.exp(-beta * shifted))) / beta
    return v_min - log_sum


def _soft_radius(centers, beta=50.0):
    """Differentiable soft approximation of the packing radius."""
    centers = np.asarray(centers, dtype=np.float64)
    boundary_dists = 0.5 - np.abs(centers)
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = (diffs ** 2).sum(axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    iu = np.triu_indices(N, k=1)
    pair_dists = np.sqrt(dists_sq[iu]) / 2.0
    all_dists = np.concatenate([boundary_dists.ravel(), pair_dists])
    return _soft_min(all_dists, beta)


def _negative_soft_radius(x, beta=50.0):
    """Negative soft radius for minimization."""
    centers = x.reshape(-1, 2)
    return -_soft_radius(centers, beta)


def _init_grid():
    """Grid initialization trying multiple configurations."""
    best_pts = None
    best_r = -1

    for rows, cols in [(7, 8), (8, 7), (5, 10), (10, 5), (6, 9), (9, 6), (7, 7)]:
        total = rows * cols
        if total < N:
            continue

        xs = (np.arange(cols) + 0.5) / cols - 0.5
        ys = (np.arange(rows) + 0.5) / rows - 0.5
        grid = np.array([(x, y) for y in ys for x in xs])

        if total == N:
            pts = grid
        else:
            dists_to_center = np.sqrt((grid ** 2).sum(axis=1))
            indices = np.argsort(dists_to_center)[:N]
            pts = grid[indices]

        r = _true_score(pts)
        if r > best_r:
            best_r = r
            best_pts = pts.copy()

    if best_pts is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_pts = grid[:N].copy()

    return best_pts


def _init_hexagonal():
    """Hexagonal lattice initialization."""
    s = 1.0 / 7.0
    pts = []
    for i in range(9):
        for j in range(9):
            x = j * s + (i % 2) * s / 2 - 0.5 + s / 4
            y = i * s * np.sqrt(3) / 2 - 0.5 + s * np.sqrt(3) / 4
            if abs(x) <= 0.5 and abs(y) <= 0.5:
                pts.append([x, y])

    pts = np.array(pts)
    if len(pts) >= N:
        selected = []
        remaining = pts.copy()
        for _ in range(N):
            if len(remaining) == 0:
                break
            if len(selected) == 0:
                idx = 0
            else:
                sel = np.array(selected)
                dists = np.min(np.linalg.norm(remaining[:, np.newaxis, :] - sel[np.newaxis, :, :], axis=2), axis=1)
                idx = np.argmax(dists)
            selected.append(remaining[idx])
            remaining = np.delete(remaining, idx, axis=0)

        if len(selected) == N:
            return np.array(selected)

    return _init_grid()


def _init_random(seed=0):
    """Create a random initialization."""
    rng = np.random.RandomState(seed)
    pts = rng.uniform(-0.45, 0.45, (N, 2))
    return pts


def _force_relax(centers, n_steps=300, lr_init=0.01, lr_decay=0.999):
    """Force-directed relaxation to resolve overlaps and keep in bounds."""
    centers = centers.copy()
    n = len(centers)
    lr = lr_init

    for step in range(n_steps):
        forces = np.zeros_like(centers)

        for k in range(2):
            mask_r = centers[:, k] > 0.45
            forces[mask_r, k] -= (centers[mask_r, k] - 0.45) * 20
            mask_l = centers[:, k] < -0.45
            forces[mask_l, k] += ((-0.45) - centers[mask_l, k]) * 20

        diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt((diffs ** 2).sum(axis=2))
        np.fill_diagonal(dists, np.inf)

        threshold = 0.12
        mask = dists < threshold
        np.fill_diagonal(mask, False)

        force_mag = np.zeros_like(dists)
        force_mag[mask] = (threshold - dists[mask]) / (dists[mask] + 1e-10) ** 2
        force_mag = np.minimum(force_mag, 50)

        forces += (force_mag[:, :, np.newaxis] * diffs).sum(axis=1)

        centers += forces * lr
        centers = np.clip(centers, -0.5, 0.5)

        lr *= lr_decay

    return centers


def _optimize_lbfgsb(centers, time_budget=15.0):
    """Optimize using L-BFGS-B with soft-min surrogate and progressive beta."""
    start_time = time.time()

    x0 = centers.ravel()
    best_x = x0.copy()
    best_score = _true_score(centers)

    # Progressive beta schedule: start smooth, end sharp
    betas = [20, 40, 80, 150, 300, 500]

    for beta in betas:
        elapsed = time.time() - start_time
        remaining = time_budget - elapsed
        if remaining < 2:
            break

        def objective(x):
            return _negative_soft_radius(x, beta)

        try:
            maxiter = 500 if beta < 100 else 800
            ftol = 1e-12
            gtol = 1e-10

            result = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (2 * N),
                options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol}
            )

            new_centers = result.x.reshape(-1, 2)
            new_score = _true_score(new_centers)
            if new_score > best_score:
                best_score = new_score
                best_x = result.x.copy()
            x0 = result.x.copy()
        except Exception:
            pass

    # Final high-precision polish with very sharp beta
    elapsed = time.time() - start_time
    if elapsed < time_budget - 3:
        try:
            def objective_sharp(x):
                return _negative_soft_radius(x, 1000.0)

            result = minimize(
                objective_sharp,
                x0,
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (2 * N),
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12}
            )

            new_centers = result.x.reshape(-1, 2)
            new_score = _true_score(new_centers)
            if new_score > best_score:
                best_score = new_score
                best_x = result.x.copy()
        except Exception:
            pass

    return best_x.reshape(-1, 2), best_score


def _optimize_sa(centers, time_budget=10.0):
    """Simulated annealing refinement with adaptive step size."""
    start_time = time.time()
    centers = centers.copy()
    best_centers = centers.copy()
    best_score = _true_score(centers)

    T = 0.005
    T_min = 0.00005
    cooling = 0.995
    step_scale = 0.3

    while time.time() - start_time < time_budget and T > T_min:
        idx = np.random.randint(N)
        delta = np.random.randn(2) * T * step_scale
        new_centers = centers.copy()
        new_centers[idx] = np.clip(centers[idx] + delta, -0.5, 0.5)

        new_score = _true_score(new_centers)
        delta_score = new_score - best_score

        if delta_score > 0 or np.random.rand() < np.exp(delta_score / T):
            centers = new_centers
            if new_score > best_score:
                best_score = new_score
                best_centers = new_centers.copy()

        T *= cooling

    return best_centers, best_score


def _local_refine(centers, time_budget=20.0):
    """Local refinement with small perturbations (hill climbing)."""
    start_time = time.time()
    best_centers = centers.copy()
    best_score = _true_score(best_centers)

    step_sizes = [0.001, 0.0005, 0.0002, 0.0001, 0.00005, 0.00002]

    for step in step_sizes:
        if time.time() - start_time > time_budget:
            break

        improved = True
        while improved and time.time() - start_time < time_budget:
            improved = False
            for idx in range(N):
                if time.time() - start_time > time_budget:
                    break
                for axis in range(2):
                    for sign in [1, -1]:
                        new_centers = best_centers.copy()
                        new_centers[idx, axis] += sign * step
                        if -0.5 <= new_centers[idx, axis] <= 0.5:
                            new_score = _true_score(new_centers)
                            if new_score > best_score + 1e-14:
                                best_centers = new_centers
                                best_score = new_score
                                improved = True

    return best_centers, best_score


def _radius_expansion(centers, max_iter=1000, deadline=None):
    """Iteratively expand the radius by pushing points apart."""
    centers = centers.copy()
    current_r = _true_score(centers)

    for _ in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break

        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dists, np.inf)

        wall_dists = np.minimum(
            np.minimum(0.5 - centers[:, 0], 0.5 + centers[:, 0]),
            np.minimum(0.5 - centers[:, 1], 0.5 + centers[:, 1])
        )

        min_pair_idx = np.unravel_index(np.argmin(dists), dists.shape)
        min_wall_idx = np.argmin(wall_dists)

        min_pair_dist = dists[min_pair_idx]
        min_wall_dist = 2.0 * wall_dists[min_wall_idx]

        if min_pair_dist < min_wall_dist:
            i, j = min_pair_idx
            direction = (centers[i] - centers[j]) / (min_pair_dist + 1e-15)
            shift = 0.001
            centers[i] += direction * shift
            centers[j] -= direction * shift
        else:
            i = min_wall_idx
            for axis in range(2):
                if centers[i, axis] > 0:
                    centers[i, axis] -= 0.001
                else:
                    centers[i, axis] += 0.001

        centers = np.clip(centers, -0.5, 0.5)

        new_r = _true_score(centers)
        if new_r > current_r:
            current_r = new_r
        else:
            break

    return centers, current_r


def construct():
    start_time = time.time()

    # Multiple diverse initializations
    inits = [
        _init_grid(),
        _init_hexagonal(),
        _init_random(0),
        _init_random(1),
        _init_random(2),
        _init_random(3),
    ]

    # Phase 1: Quick passes on all initializations to find a good starting point
    best_pts = None
    best_score = 0.0

    for init in inits:
        if time.time() > DEADLINE - 40:
            break

        relaxed = _force_relax(init, n_steps=150)
        lbfgsb_pts, lbfgsb_score = _optimize_lbfgsb(relaxed, time_budget=8.0)
        if lbfgsb_score > best_score:
            best_pts = lbfgsb_pts.copy()
            best_score = lbfgsb_score

    if best_pts is None:
        best_pts = _init_grid()
        best_score = _true_score(best_pts)

    # Phase 2: Deep optimization on the best result
    if time.time() < DEADLINE - 20:
        sa_pts, sa_score = _optimize_sa(best_pts, time_budget=10.0)
        if sa_score > best_score:
            best_pts = sa_pts.copy()
            best_score = sa_score

    if time.time() < DEADLINE - 15:
        ref_pts, ref_score = _local_refine(best_pts, time_budget=15.0)
        if ref_score > best_score:
            best_pts = ref_pts.copy()
            best_score = ref_score

    if time.time() < DEADLINE - 5:
        exp_pts, exp_score = _radius_expansion(best_pts, max_iter=500, deadline=DEADLINE - 2)
        if exp_score > best_score:
            best_pts = exp_pts.copy()
            best_score = exp_score

    # Phase 3: Final high-precision polish
    if time.time() < DEADLINE - 2:
        final_pts, final_score = _local_refine(best_pts, time_budget=5.0)
        if final_score > best_score:
            best_pts = final_pts
            best_score = final_score

    return best_pts
# EVOLVE-BLOCK-END
