"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _radius(centers):
    """Compute the maximum common radius for given centers (vectorized)."""
    n = len(centers)
    # Pairwise distances (upper triangle)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    d2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(d2, np.inf)
    min_pair_dist = np.sqrt(np.min(d2))
    # Distance to walls
    wall = 0.5 - np.max(np.abs(centers), axis=1)
    min_wall = np.min(wall)
    return min(min_pair_dist / 2.0, min_wall)


def _relax(centers, time_budget):
    """Force-directed relaxation to push spheres apart and away from walls."""
    start = time.time()
    centers = centers.copy()
    n = len(centers)
    best_r = _radius(centers)
    best = centers.copy()

    lr = 0.02
    decay = 0.998
    while time.time() - start < time_budget:
        # Pairwise repulsion
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        d2 = np.sum(diff * diff, axis=2)
        np.fill_diagonal(d2, 1e-12)
        d = np.sqrt(d2)
        # Force magnitude ~ 1/d^2 (strong when close)
        mag = 1.0 / (d2 + 1e-6)
        forces = np.sum(mag[:, :, np.newaxis] * diff / d[:, :, np.newaxis], axis=1)

        # Wall repulsion
        margin = 0.05
        for k in range(3):
            over = centers[:, k] - (0.5 - margin)
            forces[:, k] -= np.where(over > 0, over * 5.0, 0.0)
            under = (-0.5 + margin) - centers[:, k]
            forces[:, k] += np.where(under > 0, under * 5.0, 0.0)

        centers += lr * forces
        centers = np.clip(centers, -0.499, 0.499)
        r = _radius(centers)
        if r > best_r:
            best_r = r
            best = centers.copy()
        lr *= decay

    return best, best_r


def _local_opt(x0, time_budget):
    """Nelder-Mead local optimization of the negative radius."""
    start = time.time()

    def objective(x):
        c = x.reshape(N, 3)
        if np.any(np.abs(c) > 0.5):
            return 1e10
        return -_radius(c)

    try:
        result = minimize(
            objective, x0, method='Nelder-Mead',
            options={'maxiter': 3000, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True}
        )
        c = result.x.reshape(N, 3)
        if np.all(np.abs(c) <= 0.5) and np.all(np.isfinite(c)):
            return c, _radius(c)
    except Exception:
        pass
    return x0.reshape(N, 3), _radius(x0.reshape(N, 3))


def construct():
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s kill

    # Initial configuration: 4x4x4 grid (64 points), take 50, with small perturbation
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    centers = grid[:N].copy()
    np.random.seed(42)
    centers += np.random.normal(0, 0.005, centers.shape)
    centers = np.clip(centers, -0.49, 0.49)

    best_centers = centers.copy()
    best_r = _radius(best_centers)

    # Phase 1: force-directed relaxation (cheap, fast)
    if time.time() - start_time < deadline - 30:
        relaxed, r = _relax(centers, time_budget=min(20.0, deadline - 30 - (time.time() - start_time)))
        if r > best_r:
            best_r = r
            best_centers = relaxed.copy()

    # Phase 2: local optimization from the relaxed configuration
    if time.time() - start_time < deadline - 5:
        budget = min(30.0, deadline - 5 - (time.time() - start_time))
        c, r = _local_opt(best_centers.flatten(), budget)
        if r > best_r:
            best_r = r
            best_centers = c.copy()

    # Phase 3: a couple of Nelder-Mead restarts from different random starts if time remains
    for attempt in range(3):
        if time.time() - start_time > deadline - 5:
            break
        np.random.seed(100 + attempt)
        x0 = np.random.uniform(-0.4, 0.4, (N, 3)).flatten()
        budget = min(15.0, deadline - 5 - (time.time() - start_time))
        c, r = _local_opt(x0, budget)
        if r > best_r:
            best_r = r
            best_centers = c.copy()

    # Final validity check
    if not np.all(np.isfinite(best_centers)) or np.any(np.abs(best_centers) > 0.5):
        best_centers = grid[:N].copy()

    return best_centers
# EVOLVE-BLOCK-END
