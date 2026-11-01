"""Combined solver: FCC initialization + Voronoi-Lloyd equalization + smooth max-min polish + local search.
Entry point: construct() -> (N,3) array of sphere centres in [-0.5,0.5]^3."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay

DEADLINE_MARGIN = 15.0  # safety margin before 150s kill


def _compute_radius(c):
    """Exact common radius for centres c (N,3) in cube [-0.5,0.5]^3."""
    diff = c[:, None, :] - c[None, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    wall = 0.5 - np.abs(c)
    min_wall = np.min(wall)
    return min(min_pair / 2.0, min_wall)


def _pair_dist_matrix(c):
    """(N,N) pairwise distance matrix, diagonal = inf."""
    diff = c[:, None, :] - c[None, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(d, np.inf)
    return d


def _generate_initial_positions():
    """Generate a good initial configuration using FCC lattice points closest to origin."""
    points = []
    max_idx = 6
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            for k in range(-max_idx, max_idx + 1):
                # FCC basis
                points.append([i, j, k])
                points.append([i + 0.5, j + 0.5, k])
                points.append([i + 0.5, j, k + 0.5])
                points.append([i, j + 0.5, k + 0.5])

    points = np.array(points)

    # Center at origin
    points -= np.mean(points, axis=0)

    # Sort by distance from origin
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)

    # Take N closest
    pts = points[idx[:N]].copy()

    # Scale to fit inside cube with margin
    max_coord = np.max(np.abs(pts))
    if max_coord > 0:
        scale = 0.42 / max_coord
        pts *= scale

    return pts


def _lloyd_optimize(c, deadline):
    """Voronoi-Lloyd equalization: move each center toward its Voronoi cell centroid."""
    c = c.copy()
    c = np.clip(c, -0.49, 0.49)
    best = c.copy()
    best_r = _compute_radius(c)

    for damp in (0.7, 0.5, 0.35, 0.25, 0.15, 0.1, 0.06, 0.03):
        if time.time() > deadline:
            break
        steps = max(30, int(200 * damp))
        for _ in range(steps):
            if time.time() > deadline:
                break
            try:
                tri = Delaunay(c)
            except Exception:
                # Fallback: simple repulsion
                d = _pair_dist_matrix(c)
                inv = 1.0 / np.maximum(d, 1e-6) ** 2
                diff = c[:, None, :] - c[None, :, :]
                F = np.sum(diff * inv[:, :, None], axis=1)
                c = np.clip(c + 0.005 * F, -0.49, 0.49)
                continue

            simplices = tri.simplices
            cell_centroids = np.zeros((N, 3))
            cell_counts = np.zeros(N)

            for si in simplices:
                v0, v1, v2, v3 = c[si[0]], c[si[1]], c[si[2]], c[si[3]]
                # Circumcenter
                A = np.array([v1 - v0, v2 - v0, v3 - v0])
                b = np.array([
                    np.dot(v1, v1) - np.dot(v0, v0),
                    np.dot(v2, v2) - np.dot(v0, v0),
                    np.dot(v3, v3) - np.dot(v0, v0)
                ])
                try:
                    cc = np.linalg.solve(2.0 * A, b)
                except np.linalg.LinAlgError:
                    cc = np.mean([v0, v1, v2, v3])

                for vi in si:
                    cell_centroids[vi] += cc
                    cell_counts[vi] += 1

            valid = cell_counts > 0
            if np.any(valid):
                targets = cell_centroids[valid] / cell_counts[valid, None]
                c[valid] = c[valid] + damp * (targets - c[valid])

            c = np.clip(c, -0.49, 0.49)

        r = _compute_radius(c)
        if r > best_r:
            best_r = r
            best = c.copy()

    return best, best_r


def _softmin_radius(c, t):
    """Smooth approximation of the true radius via log-sum-exp."""
    d = _pair_dist_matrix(c)
    pair = d / 2.0
    wall = 0.5 - np.abs(c)
    vals = np.concatenate([pair.ravel(), wall.ravel()])
    vals = vals[np.isfinite(vals)]
    m = np.min(vals)
    s = np.sum(np.exp(-(vals - m) / t))
    return m - t * np.log(s)


def _smooth_obj(x, t):
    """Objective to MINIMIZE: negative smooth radius plus boundary penalty."""
    c = x.reshape(N, 3)
    r = _softmin_radius(c, t)
    pen = np.sum(np.maximum(0.0, np.abs(c) - 0.5 + 1e-6) ** 2)
    return -r + 1e3 * pen


def _smooth_polish(c, deadline):
    """Polish with smooth softmin max-min via L-BFGS-B, annealing temperature."""
    c = c.copy()
    c = np.clip(c, -0.49, 0.49)
    best = c.copy()
    best_r = _compute_radius(c)
    x0 = c.flatten()
    bounds = [(-0.49, 0.49)] * (N * 3)

    for t in (0.05, 0.02, 0.008, 0.003, 0.001, 0.0005, 0.0001):
        if time.time() > deadline:
            break
        try:
            res = minimize(
                _smooth_obj,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                args=(t,),
                options={'maxiter': 1500, 'ftol': 1e-14, 'gtol': 1e-11,
                         'maxls': 50},
            )
            c = np.clip(res.x.reshape(N, 3), -0.49, 0.49)
            r = _compute_radius(c)
            if r > best_r:
                best_r = r
                best = c.copy()
            x0 = c.flatten()
        except Exception:
            pass
    return best, best_r


def _repulsion_anneal(c, deadline, steps=2000):
    """Vectorized repulsion + wall-attraction gradient descent."""
    c = c.copy()
    c = np.clip(c, -0.49, 0.49)

    for it in range(steps):
        if time.time() > deadline:
            break
        d = _pair_dist_matrix(c)
        inv = 1.0 / np.maximum(d, 1e-6) ** 2
        diff = c[:, None, :] - c[None, :, :]
        F = np.sum(diff * inv[:, :, None], axis=1)

        wall_force = np.zeros_like(c)
        for k in range(3):
            sgn = np.sign(c[:, k])
            sgn[sgn == 0] = 1.0
            wall_force[:, k] = sgn * (0.5 - np.abs(c[:, k]))

        frac = it / max(steps - 1, 1)
        rep_scale = 1e-3 * (1.0 - 0.7 * frac)
        wall_scale = 1e-3 * (0.3 + 1.0 * frac)
        lr = 1e-2 * (1.0 - 0.5 * frac)
        c = c + lr * (rep_scale * F + wall_scale * wall_force)
        c = np.clip(c, -0.49, 0.49)

    return c


def _local_search(c, deadline, iters=3000):
    """Local search: perturb binding constraints, accept if radius improves."""
    c = c.copy()
    c = np.clip(c, -0.499, 0.499)
    best_r = _compute_radius(c)
    best_c = c.copy()
    rng = np.random.default_rng(42)

    for it in range(iters):
        if time.time() > deadline:
            break

        d = _pair_dist_matrix(c)
        min_pair_idx = np.unravel_index(np.argmin(d), d.shape)
        min_pair_dist = d[min_pair_idx]

        wall = 0.5 - np.abs(c)
        min_wall_idx = np.unravel_index(np.argmin(wall), wall.shape)
        min_wall_val = wall[min_wall_idx]

        if min_pair_dist / 2.0 < min_wall_val:
            i, j = min_pair_idx
            direction = c[i] - c[j]
            norm = np.linalg.norm(direction)
            if norm > 1e-12:
                direction /= norm
                # Push apart
                step = 0.002 * rng.uniform(0.5, 1.5)
                c[i] += step * direction
                c[j] -= step * direction
        else:
            idx = min_wall_idx
            axis = idx[1]
            # Push away from wall
            step = 0.002 * rng.uniform(0.5, 1.5)
            if c[idx[0], axis] > 0:
                c[idx[0], axis] -= step
            else:
                c[idx[0], axis] += step

        c = np.clip(c, -0.499, 0.499)
        r = _compute_radius(c)
        if r > best_r:
            best_r = r
            best_c = c.copy()

    return best_c, best_r


def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin

    # 1. Initialize with FCC-like cluster
    best_pts = _generate_initial_positions()
    best_r = _compute_radius(best_pts)

    # 2. Voronoi-Lloyd equalization
    if time.time() < deadline:
        pts, r = _lloyd_optimize(best_pts, deadline)
        if r > best_r:
            best_r = r
            best_pts = pts

    # 3. Repulsion anneal
    if time.time() < deadline:
        pts = _repulsion_anneal(best_pts, deadline, steps=1500)
        r = _compute_radius(pts)
        if r > best_r:
            best_r = r
            best_pts = pts

    # 4. Smooth max-min polish
    if time.time() < deadline:
        pts, r = _smooth_polish(best_pts, deadline)
        if r > best_r:
            best_r = r
            best_pts = pts

    # 5. Local search on binding constraints
    if time.time() < deadline:
        pts, r = _local_search(best_pts, deadline, iters=2000)
        if r > best_r:
            best_r = r
            best_pts = pts

    # 6. Final smooth polish if time remains
    if time.time() < deadline:
        pts, r = _smooth_polish(best_pts, deadline)
        if r > best_r:
            best_r = r
            best_pts = pts

    # Ensure final result is valid
    final_r = _compute_radius(best_pts)
    if final_r > best_r:
        best_r = final_r

    return best_pts
# EVOLVE-BLOCK-END
