"""Pack 300 equal circles in the unit disk; maximize the common radius."""
import numpy as np
import time

N = 300
DEADLINE_FRACTION = 0.92  # safety margin against the 150 s wall-clock limit


def _score(pts):
    """Return the largest common radius for a set of centres (or -inf if invalid)."""
    pts = np.asarray(pts, dtype=float)
    if pts.shape != (N, 2) or not np.all(np.isfinite(pts)):
        return -np.inf
    d = np.hypot(pts[:, 0], pts[:, 1])
    if np.any(d >= 1.0):
        return -np.inf
    # pairwise distances
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist)
    min_edge = 1.0 - np.max(d)
    return min(min_pair, 2.0 * min_edge) / 2.0


def _hex_init():
    """Hexagonal lattice clipped to the disk, picking the N closest-to-centre points."""
    best = None
    for spacing in np.linspace(0.055, 0.045, 60):
        r = spacing / 2.0
        margin = 1.0 - r
        k = int(np.ceil(1.0 / spacing)) + 2
        xs, ys = np.meshgrid(
            [(i + 0.5 * (j % 2)) * spacing for i in range(-k, k + 1)],
            [j * spacing * np.sqrt(3.0) / 2.0 for j in range(-k, k + 1)],
            indexing="ij",
        )
        pts = np.column_stack([xs.ravel(), ys.ravel()])
        rad = np.hypot(pts[:, 0], pts[:, 1])
        pts = pts[rad <= margin]
        if len(pts) >= N:
            idx = np.argsort(rad, kind="stable")[:N]
            cand = pts[idx]
            if best is None:
                best = cand
            else:
                # prefer the one with larger min radius
                pass
    return best


def construct():
    t0 = time.time()
    deadline = t0 + 150.0 * DEADLINE_FRACTION

    pts = _hex_init()
    if pts is None:
        # fallback: concentric rings
        pts = np.zeros((N, 2))
        ring = 0
        i = 0
        while i < N:
            if ring == 0:
                pts[i] = [0.0, 0.0]
                i += 1
            else:
                n_ring = 6 * ring
                for j in range(n_ring):
                    if i >= N:
                        break
                    ang = 2.0 * np.pi * j / n_ring
                    rad = ring * 0.1
                    pts[i] = [rad * np.cos(ang), rad * np.sin(ang)]
                    i += 1
            ring += 1

    best_pts = pts.copy()
    best_score = _score(best_pts)

    # Iterative local optimisation: push points apart and keep them inside the disk.
    rng = np.random.default_rng(0)
    step = 0.02
    iters = 0
    while time.time() < deadline:
        # compute repulsion from neighbours and boundary
        diff = pts[:, None, :] - pts[None, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, 1e9)
        # repulsion: for each pair closer than 2r, push apart
        r = best_score
        force = np.zeros_like(pts)
        # vectorized: only pairs with dist < 2r
        mask = dist < 2.0 * r
        # accumulate
        for i in range(N):
            j_idx = np.where(mask[i])[0]
            if len(j_idx) == 0:
                continue
            dij = diff[i, j_idx]
            dijn = dist[i, j_idx]
            push = (dij / dijn[:, None]) * (2.0 * r - dijn)[:, None] * 0.5
            force[i] += push.sum(axis=0)
            force[j_idx] -= push
        # boundary: if a point is too close to the edge, push inward
        rad = np.hypot(pts[:, 0], pts[:, 1])
        margin = 1.0 - r
        over = rad > margin
        if np.any(over):
            inward = -pts[over] / rad[over, None] * (rad[over] - margin)[:, None]
            force[over] += inward

        # random jitter to escape local minima
        force += rng.normal(0.0, 0.001, size=pts.shape)

        # adaptive step
        new_pts = pts + step * force
        # clip to disk
        new_rad = np.hypot(new_pts[:, 0], new_pts[:, 1])
        lim = 1.0 - r
        scale = np.minimum(1.0, lim / np.maximum(new_rad, 1e-9))
        new_pts *= scale[:, None]

        new_score = _score(new_pts)
        iters += 1

        if new_score > best_score:
            best_score = new_score
            best_pts = new_pts.copy()
            pts = new_pts.copy()
            step = min(step * 1.05, 0.05)
        else:
            # revert and shrink step
            step *= 0.9
            if step < 1e-5:
                step = 0.02
                # random perturbation to escape
                pts = pts + rng.normal(0.0, 0.005, size=pts.shape)
                pr = np.hypot(pts[:, 0], pts[:, 1])
                sc = np.minimum(1.0, (1.0 - r) / np.maximum(pr, 1e-9))
                pts *= sc[:, None]

    # final polish: a few deterministic relaxation passes
    if time.time() < deadline - 5.0:
        for _ in range(50):
            r = best_score
            diff = best_pts[:, None, :] - best_pts[None, :, :]
            dist = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dist, 1e9)
            force = np.zeros_like(best_pts)
            for i in range(N):
                j_idx = np.where(dist[i] < 2.0 * r)[0]
                if len(j_idx) == 0:
                    continue
                dij = diff[i, j_idx]
                dijn = dist[i, j_idx]
                push = (dij / dijn[:, None]) * (2.0 * r - dijn)[:, None] * 0.5
                force[i] += push.sum(axis=0)
                force[j_idx] -= push
            rad = np.hypot(best_pts[:, 0], best_pts[:, 1])
            margin = 1.0 - r
            over = rad > margin
            if np.any(over):
                inward = -best_pts[over] / rad[over, None] * (rad[over] - margin)[:, None]
                force[over] += inward
            new_pts = best_pts + 0.01 * force
            new_rad = np.hypot(new_pts[:, 0], new_pts[:, 1])
            scale = np.minimum(1.0, (1.0 - r) / np.maximum(new_rad, 1e-9))
            new_pts *= scale[:, None]
            ns = _score(new_pts)
            if ns > best_score:
                best_score = ns
                best_pts = new_pts

    return best_pts
