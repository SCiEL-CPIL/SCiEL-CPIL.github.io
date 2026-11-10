"""Force-directed relaxation for 100-sphere packing in unit cube."""
N = 100
import time

# EVOLVE-BLOCK-START
import numpy as np


def _radius_score(centres):
    """Largest common radius for given centres (validity assumed by caller)."""
    c = np.asarray(centres, dtype=float)
    if c.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(c)):
        return 0.0
    # wall clearance
    wall = 0.5 - np.abs(c).max(axis=1)
    r_wall = wall.min()
    # pairwise
    d = c[:, None, :] - c[None, :, :]
    dist = np.linalg.norm(d, axis=2)
    np.fill_diagonal(dist, np.inf)
    r_pair = dist.min() / 2.0
    return min(r_wall, r_pair)


def construct():
    t0 = time.time()
    deadline = t0 + 135.0  # safety margin before 150s kill

    # Initial dense grid: 5 x 5 x 4 = 100
    nx, ny, nz = 5, 5, 4
    xs = (np.arange(nx) + 0.5) / nx - 0.5
    ys = (np.arange(ny) + 0.5) / ny - 0.5
    zs = (np.arange(nz) + 0.5) / nz - 0.5
    c = np.array([(x, y, z) for z in zs for y in ys for x in xs], dtype=float)

    best_c = c.copy()
    best_r = _radius_score(best_c)

    # Force-directed relaxation
    # Parameters
    r_target = 0.115  # slightly above known best to encourage expansion
    rep_k = 50.0      # repulsion strength
    wall_k = 100.0    # wall repulsion strength
    wall_margin = 0.01
    step = 0.01
    min_step = 1e-6
    max_iter = 20000

    for it in range(max_iter):
        if time.time() > deadline:
            break

        # Compute forces
        forces = np.zeros_like(c)
        for i in range(N):
            ci = c[i]
            # Wall forces
            for k in range(3):
                if ci[k] > 0.5 - r_target - wall_margin:
                    forces[i, k] -= wall_k * (ci[k] - (0.5 - r_target - wall_margin))
                elif ci[k] < -0.5 + r_target + wall_margin:
                    forces[i, k] += wall_k * ((-0.5 + r_target + wall_margin) - ci[k])

            # Pairwise repulsion
            for j in range(i + 1, N):
                diff = ci - c[j]
                d = np.linalg.norm(diff)
                if d < 1e-12:
                    # Jitter
                    forces[i] += np.random.randn(3) * 0.001
                    forces[j] -= np.random.randn(3) * 0.001
                    continue
                if d < 2 * r_target + 0.05:  # interaction cutoff
                    # Repulsive force: strong when overlapping
                    overlap = (2 * r_target - d)
                    if overlap > 0:
                        f = rep_k * overlap / d
                    else:
                        f = 0.0
                    forces[i] += f * diff
                    forces[j] -= f * diff

        # Adaptive step: try to move, accept if radius improves
        disp = forces * step
        # Clip displacement to keep inside cube
        new_c = c + disp
        new_c = np.clip(new_c, -0.5 + 1e-9, 0.5 - 1e-9)

        new_r = _radius_score(new_c)

        if new_r > best_r:
            best_c = new_c.copy()
            best_r = new_r
            c = new_c
            # Increase step slightly on improvement
            step = min(step * 1.1, 0.05)
        else:
            # Reduce step
            step *= 0.9
            if step < min_step:
                # Try a small random perturbation to escape local min
                if it % 100 == 0:
                    c = c + np.random.randn(N, 3) * 0.005
                    c = np.clip(c, -0.5 + 1e-9, 0.5 - 1e-9)
                else:
                    c = new_c  # accept anyway to explore
                    new_r2 = _radius_score(c)
                    if new_r2 > best_r:
                        best_c = c.copy()
                        best_r = new_r2
                    step = min_step * 2

        # Periodic re-evaluation of best
        if it % 500 == 0:
            cur_r = _radius_score(c)
            if cur_r > best_r:
                best_r = cur_r
                best_c = c.copy()

    # Final: ensure validity and return best
    final_r = _radius_score(best_c)
    if final_r <= 0:
        # Fallback: return original grid
        return c

    return best_c
# EVOLVE-BLOCK-END
