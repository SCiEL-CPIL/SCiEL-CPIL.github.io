"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import time
import numpy as np


def _score(pos):
    """Largest common radius for a placement (validity assumed)."""
    if pos.shape != (N, 2) or not np.all(np.isfinite(pos)):
        return 0.0
    d = pos[:, None, :] - pos[None, :, :]
    dist = np.sqrt(np.sum(d * d, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = dist.min()
    min_wall = (0.5 - np.abs(pos)).min()
    return min(0.5 * min_pair, min_wall)


def _force_step(pos, r, alpha):
    """One force-directed relaxation step for radius r."""
    n = pos.shape[0]
    d = pos[:, None, :] - pos[None, :, :]
    dist = np.sqrt(np.sum(d * d, axis=2))
    np.fill_diagonal(dist, 1.0)
    # Repulsive force: proportional to 1/dist^3, applied only when overlapping.
    overlap = (dist < 2 * r)
    force = np.zeros_like(pos)
    for i in range(n):
        for j in range(i + 1, n):
            if overlap[i, j]:
                gap = dist[i, j]
                if gap < 1e-12:
                    continue
                mag = (2 * r - gap) / (gap ** 3)
                diff = d[i, j]
                force[i] -= mag * diff
                force[j] += mag * diff
    # Attractive force toward center (weak).
    force -= 0.05 * pos
    # Adaptive step size: limit movement to avoid oscillation.
    max_move = 0.5 * r * alpha
    step = force * max_move
    step_norm = np.sqrt(np.sum(step * step, axis=1))
    for i in range(n):
        if step_norm[i] > max_move:
            step[i] *= max_move / step_norm[i]
    pos = pos + step
    # Clamp inside the square (leaving radius margin).
    lim = 0.5 - r
    pos = np.clip(pos, -lim, lim)
    return pos


def _force_layout(r, iters, alpha0, alpha1):
    """Build a non-overlapping layout for radius r via force-directed relaxation."""
    np.random.seed(12345)
    pos = np.random.uniform(-0.5 + r, 0.5 - r, size=(N, 2))
    alpha = alpha0
    for it in range(iters):
        alpha = alpha0 * (alpha1 / alpha0) ** (it / max(1, iters - 1))
        pos = _force_step(pos, r, alpha)
    return pos


def _anneal(pos, r, time_budget):
    """Simulated annealing to maximize the minimum radius around r."""
    best_pos = pos.copy()
    best_r = _score(pos)
    cur_pos = pos.copy()
    cur_r = best_r
    np.random.seed(98765)
    t_start = time.time()
    temp = 0.002
    cool = 0.9995
    step = 0.01
    while time.time() - t_start < time_budget:
        # Propose a move for a random circle.
        i = np.random.randint(N)
        old = cur_pos[i].copy()
        cur_pos[i] = old + np.random.normal(0, step, size=2)
        lim = 0.5 - 0.07
        cur_pos[i] = np.clip(cur_pos[i], -lim, lim)
        new_r = _score(cur_pos)
        if new_r > cur_r or np.random.rand() < np.exp((new_r - cur_r) / max(temp, 1e-9)):
            cur_r = new_r
            if new_r > best_r:
                best_r = new_r
                best_pos = cur_pos.copy()
        else:
            cur_pos[i] = old
        temp *= cool
    return best_pos, best_r


def construct():
    t_start = time.time()
    deadline = 120.0
    # Baseline grid.
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    best_pos = grid[:N].copy()
    best_r = _score(best_pos)

    # Force-directed initialization at the best radius found so far.
    if time.time() - t_start < deadline - 20:
        pos = _force_layout(best_r, 4000, 1.0, 0.05)
        r = _score(pos)
        if r > best_r:
            best_r, best_pos = r, pos

    # Simulated annealing to push the radius higher, with deadline guard.
    if time.time() - t_start < deadline - 10:
        budget = deadline - 10 - (time.time() - t_start)
        pos, r = _anneal(best_pos, best_r, budget)
        if r > best_r:
            best_r, best_pos = r, pos

    # Final safety: ensure validity (finite, inside square).
    best_pos = np.clip(best_pos, -0.5 + 0.07, 0.5 - 0.07)
    if not np.all(np.isfinite(best_pos)):
        return grid[:N]
    return best_pos
# EVOLVE-BLOCK-END
