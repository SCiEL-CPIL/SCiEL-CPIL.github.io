"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

N = 50

def _hexagonal_grid(n, spacing=1.0):
    """Generate a hexagonal grid of n points, centered at origin."""
    points = []
    dx = spacing
    dy = spacing * np.sqrt(3) / 2
    row = 0
    while len(points) < n:
        for col in range(-row - 1, row + 2):
            x = (col + 0.5 * (row % 2)) * dx
            y = row * dy
            points.append([x, y])
        row += 1
        if len(points) >= n:
            break
    # Also add negative rows
    points2 = []
    for row in range(0, -n, -1):
        for col in range(-abs(row) - 1, abs(row) + 2):
            x = (col + 0.5 * (abs(row) % 2)) * dx
            y = row * dy
            points2.append([x, y])
    all_pts = np.array(points[:n] + points2[:max(0, n - len(points))])
    # Center
    center = all_pts.mean(axis=0)
    all_pts = all_pts - center
    return all_pts[:n]


def _soft_min(objs, k):
    """Log-sum-exp approximation to min. Returns -logsumexp(-k*x)/k which approximates min(x) as k->inf."""
    objs = np.asarray(objs)
    # clip for numerical stability
    objs_clipped = np.clip(objs, 1e-15, 10)
    max_val = np.max(objs_clipped)
    # log-sum-exp trick: -1/k * log(sum(exp(-k*x)))
    exp_vals = np.exp(-k * (objs_clipped - max_val))
    lse = max_val - (1.0 / k) * np.log(np.sum(exp_vals))
    # This approximates min as k increases
    return lse


def _score(pts):
    """Compute the packing score: min pairwise distance / 2, bounded by wall distance."""
    pts = np.asarray(pts)
    if len(pts) != N:
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    # Check bounds
    if np.any(pts < -0.5 - 1e-9) or np.any(pts > 0.5 + 1e-9):
        return 0.0
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    # Wall distances
    wall_dists = np.min([
        np.min(pts[:, 0] + 0.5),
        np.min(0.5 - pts[:, 0]),
        np.min(pts[:, 1] + 0.5),
        np.min(0.5 - pts[:, 1])
    ])
    return min(min_pair_dist / 2.0, wall_dists)


def _objective(x, k):
    """Objective to minimize: negative of soft-min of all constraints."""
    pts = x.reshape(-1, 2)
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    # Wall distances
    walls = np.concatenate([
        pts[:, 0] + 0.5,
        0.5 - pts[:, 0],
        pts[:, 1] + 0.5,
        0.5 - pts[:, 1]
    ])
    all_constr = np.concatenate([dists.flatten(), walls])
    # Soft min approximation
    sm = _soft_min(all_constr, k)
    return -sm


def _refine(x0, k_schedule, max_iters_per_stage=50, deadline=None):
    """Progressive refinement with increasing k."""
    x = x0.copy()
    best_score = _score(x)
    for k in k_schedule:
        if deadline is not None and time.time() > deadline:
            break
        res = minimize(
            _objective, x, args=(k,),
            method='L-BFGS-B',
            bounds=[(-0.49, 0.49)] * (2 * N),
            options={'maxiter': max_iters_per_stage, 'gtol': 1e-6}
        )
        if res.success:
            x = res.x
        score = _score(x)
        if score > best_score:
            best_score = score
    return x, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 135  # 15s margin before 150s limit

    # Initial guess: hexagonal arrangement scaled to fit
    hex_pts = _hexagonal_grid(N, spacing=0.14)
    # Scale to fit in [-0.5, 0.5]
    max_extent = np.max(np.abs(hex_pts))
    if max_extent > 0:
        hex_pts = hex_pts * (0.45 / max_extent)

    best_x = hex_pts.copy().flatten()
    best_score = _score(best_x)

    # K schedule: progressive tightening
    k_schedule = [5, 10, 20, 50, 100, 200, 500]

    if time.time() < deadline:
        x1, s1 = _refine(best_x, k_schedule, max_iters_per_stage=30, deadline=deadline)
        if s1 > best_score:
            best_x, best_score = x1, s1

    # Additional restart with perturbed hexagonal
    if time.time() < deadline - 20:
        rng = np.random.default_rng(42)
        for restart in range(3):
            if time.time() >= deadline:
                break
            # Perturb initial hex
            perturbed = hex_pts.copy()
            perturbed += rng.normal(0, 0.01, size=perturbed.shape)
            # Re-center
            perturbed -= perturbed.mean(axis=0)
            # Clamp
            perturbed = np.clip(perturbed, -0.48, 0.48)
            k_sched_short = [10, 30, 100, 300]
            x2, s2 = _refine(perturbed.flatten(), k_sched_short, max_iters_per_stage=20, deadline=deadline)
            if s2 > best_score:
                best_x, best_score = x2, s2

    # Final polish with very high k
    if time.time() < deadline - 5:
        x3, s3 = _refine(best_x, [500, 1000, 2000], max_iters_per_stage=30, deadline=deadline)
        if s3 > best_score:
            best_x, best_score = x3, s3

    result = best_x.reshape(-1, 2)
    # Final validation
    if not np.all(np.isfinite(result)):
        return hex_pts
    if np.any(result < -0.5) or np.any(result > 0.5):
        result = np.clip(result, -0.5, 0.5)
    return result
# EVOLVE-BLOCK-END
