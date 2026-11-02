"""Circle packing in unit square: maximize sum of radii for 26 disjoint circles."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26
DEADLINE = 145.0


def _smooth_min_dist(pos):
    """For each circle, smooth min distance to others (log-sum-exp surrogate)."""
    n = pos.shape[0]
    diff = pos[:, None, :] - pos[None, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    # Soft min via log-sum-exp: min_j d_ij ≈ -beta * log(sum_j exp(-beta * d_ij))
    beta = 50.0
    dist = np.sqrt(dist_sq)
    np.fill_diagonal(dist, np.inf)
    log_sum = np.log(np.sum(np.exp(-beta * dist), axis=1))
    return -log_sum / beta


def _pack_objective(x, beta=50.0):
    """Objective: -sum(radii) + penalty for overlaps and wall violations."""
    n = x.shape[0]
    pos = x[:2 * n].reshape(-1, 2)
    radii = x[2 * n:]

    # Wall penalty
    wall_viol = np.maximum(0.0, radii - pos[:, 0]) + np.maximum(0.0, radii - (1.0 - pos[:, 0]))
    wall_viol += np.maximum(0.0, radii - pos[:, 1]) + np.maximum(0.0, radii - (1.0 - pos[:, 1]))
    wall_pen = np.sum(wall_viol ** 2)

    # Overlap penalty (smooth)
    diff = pos[:, None, :] - pos[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2) + 1e-12)
    np.fill_diagonal(dist, np.inf)
    overlap = radii[:, None] + radii[None, :] - dist
    overlap = np.maximum(overlap, 0.0)
    overlap_pen = np.sum(overlap ** 2)

    # Maximize sum of radii → minimize -sum(radii)
    return -np.sum(radii) + 100.0 * wall_pen + 100.0 * overlap_pen


def _pack_grad(x, beta=50.0):
    n = x.shape[0]
    pos = x[:2 * n].reshape(-1, 2)
    radii = x[2 * n:]

    grad_pos = np.zeros_like(pos)
    grad_r = np.zeros_like(radii)

    # Wall gradient
    for i in range(n):
        for axis in range(2):
            # r - pos[i,axis]
            if radii[i] > pos[i, axis]:
                grad_r[i] += 200.0 * (radii[i] - pos[i, axis])
                grad_pos[i, axis] -= 200.0 * (radii[i] - pos[i, axis])
            # r - (1-pos[i,axis])
            if radii[i] > 1.0 - pos[i, axis]:
                grad_r[i] += 200.0 * (radii[i] - (1.0 - pos[i, axis]))
                grad_pos[i, axis] += 200.0 * (radii[i] - (1.0 - pos[i, axis]))

    # Overlap gradient
    diff = pos[:, None, :] - pos[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2) + 1e-12)
    np.fill_diagonal(dist, np.inf)
    overlap = radii[:, None] + radii[None, :] - dist
    mask = overlap > 0
    coeff = 200.0 * overlap * mask
    np.fill_diagonal(coeff, 0.0)
    grad_pos += np.sum(coeff[:, :, None] * diff / dist[:, :, None], axis=1)
    grad_r += np.sum(coeff, axis=1)

    # Objective gradient: -sum(r) → -1 for each r
    grad_r -= 1.0

    return np.concatenate([grad_pos.reshape(-1), grad_r])


def construct():
    start_time = time.time()

    # Multi-start
    best_x = None
    best_val = np.inf

    # Start 1: grid
    side = 6
    cells = np.array([(0.5 + i, 0.5 + j) for i in range(side) for j in range(side)][:N]) / side
    r0 = np.full(N, 0.5 / side - 1e-3)
    x0 = np.concatenate([cells.reshape(-1), r0])

    starts = [x0]

    # Start 2: jittered grid
    rng = np.random.default_rng(42)
    x0b = x0.copy()
    x0b[:2 * N] += rng.normal(0, 0.02, 2 * N)
    x0b[:2 * N] = np.clip(x0b[:2 * N], 0.05, 0.95)
    starts.append(x0b)

    # Start 3: random
    x0c = np.concatenate([rng.uniform(0.1, 0.9, 2 * N), rng.uniform(0.02, 0.15, N)])
    starts.append(x0c)

    bounds = [(0.0, 1.0)] * (2 * N) + [(1e-6, 0.5)] * N

    for x_init in starts:
        if time.time() - start_time > DEADLINE - 10:
            break
        try:
            res = minimize(_pack_objective, x_init, jac=_pack_grad, method='L-BFGS-B',
                           bounds=bounds, options={'maxiter': 500, 'gtol': 1e-8, 'ftol': 1e-12})
            val = _pack_objective(res.x)
            if val < best_val:
                best_val = val
                best_x = res.x.copy()
        except Exception:
            continue

    if best_x is None:
        # Fallback: grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    pos = best_x[:2 * N].reshape(-1, 2)
    radii = best_x[2 * N:]

    # Polish: set radii to max feasible, de-overlap
    for _ in range(5):
        if time.time() - start_time > DEADLINE - 3:
            break
        # Max feasible radii
        wall = np.minimum(np.minimum(pos[:, 0], 1.0 - pos[:, 0]),
                          np.minimum(pos[:, 1], 1.0 - pos[:, 1]))
        diff = pos[:, None, :] - pos[None, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2) + 1e-12)
        np.fill_diagonal(dist, np.inf)
        pair_limit = 0.5 * np.min(dist, axis=1)
        radii = np.minimum(wall, pair_limit)
        # De-overlap
        def fun(x):
            p = x.reshape(-1, 2)
            d = np.sqrt(np.sum((p[:, None, :] - p[None, :, :]) ** 2, axis=2) + 1e-12)
            np.fill_diagonal(d, np.inf)
            ov = np.maximum(radii[:, None] + radii[None, :] - d, 0)
            return np.sum(ov ** 2)

        def jac(x):
            p = x.reshape(-1, 2)
            diff = p[:, None, :] - p[None, :, :]
            d = np.sqrt(np.sum(diff ** 2, axis=2) + 1e-12)
            np.fill_diagonal(d, np.inf)
            ov = radii[:, None] + radii[None, :] - d
            mask = ov > 0
            coeff = 2.0 * ov * mask / d
            np.fill_diagonal(coeff, 0.0)
            return np.sum(coeff[:, :, None] * diff, axis=1).reshape(-1)

        res = minimize(fun, pos.reshape(-1), jac=jac, method='L-BFGS-B',
                       bounds=[(0.0, 1.0)] * (2 * N), options={'maxiter': 200, 'gtol': 1e-9})
        pos = res.x.reshape(-1, 2)

    # Clamp
    radii = np.maximum(radii, 1e-6)
    pos[:, 0] = np.clip(pos[:, 0], radii, 1.0 - radii)
    pos[:, 1] = np.clip(pos[:, 1], radii, 1.0 - radii)

    return np.column_stack([pos, radii])
