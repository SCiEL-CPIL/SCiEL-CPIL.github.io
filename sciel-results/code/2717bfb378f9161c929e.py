"""Improved packer: simple-cubic seed + repulsion-based local optimization."""
import time
import numpy as np

N = 100


def _score(points):
    """Largest common radius: min of half pairwise distances and wall clearance."""
    # pairwise
    d = points[:, None, :] - points[None, :, :]
    dists = np.sqrt(np.sum(d * d, axis=2))
    iu = np.triu_indices(N, k=1)
    min_pair = dists[iu].min()
    # walls
    clearance = (0.5 - np.abs(points)).min()
    return min(min_pair * 0.5, clearance)


def construct():
    t0 = time.time()
    DEADLINE = 115.0  # safety margin under 120s

    # ---- seed: simple cubic 5x5x5 grid ----
    m = 5
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    pts = grid[:N].copy()

    best_pts = pts.copy()
    best_r = _score(best_pts)

    # ---- repulsion-based optimization ----
    # forces: pairwise 1/r^2 (repulsive) + wall repulsion
    def forces(p):
        f = np.zeros_like(p)
        # pairwise repulsion
        d = p[:, None, :] - p[None, :, :]
        dists = np.sqrt(np.sum(d * d, axis=2))
        np.fill_diagonal(dists, 1.0)
        # repulsion magnitude ~ 1/d^2, direction along d
        mag = 1.0 / (dists * dists)
        mag[np.diag_indices_from(mag)] = 0.0
        f += np.sum(mag[:, :, None] * d, axis=1)
        # wall repulsion: push inward when near wall
        # soft wall: for each coordinate, if p > 0.5 - margin push left, etc.
        margin = 0.2
        for k in range(3):
            x = p[:, k]
            # near +wall
            near_p = np.clip((x - (0.5 - margin)) / margin, 0.0, None)
            f[:, k] -= 50.0 * near_p ** 2
            # near -wall
            near_n = np.clip(((0.5 - margin) - x) / margin, 0.0, None)
            f[:, k] += 50.0 * near_n ** 2
        return f

    # adaptive step
    step = 0.005
    for it in range(200000):
        if time.time() - t0 > DEADLINE:
            break
        f = forces(pts)
        # normalize force magnitude to keep steps stable
        fmag = np.linalg.norm(f)
        if fmag > 1e-12:
            f = f / fmag
        new_pts = pts + step * f
        # clamp strictly inside cube (with tiny epsilon) to keep valid
        new_pts = np.clip(new_pts, -0.5 + 1e-9, 0.5 - 1e-9)
        r = _score(new_pts)
        if r > best_r:
            best_r = r
            best_pts = new_pts.copy()
            step = min(step * 1.05, 0.02)  # increase step if improving
        else:
            step = max(step * 0.85, 1e-5)  # decrease step if not improving
        pts = new_pts

    # final safety: ensure best_pts is strictly inside and finite
    best_pts = np.clip(best_pts, -0.5 + 1e-9, 0.5 - 1e-9)
    if not np.all(np.isfinite(best_pts)):
        # fallback to grid
        best_pts = grid[:N].copy()
    return best_pts
