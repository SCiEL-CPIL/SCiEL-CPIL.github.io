"""Pack 200 equal circles in a unit square to maximize common radius.

Strategy: Direct 400D coordinate optimization with multi-start from
various grid initializations. Use smooth soft-min surrogate for the
radius objective with L-BFGS-B, plus targeted push-apart refinement
for the minimum pairwise distance.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import time

N = 200
_TIME_LIMIT = 135  # safety margin before 150s kill


def _compute_radius(pts):
    """Compute the achievable radius for a set of points."""
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape != (N, 2):
        return 0.0
    if np.any(~np.isfinite(pts)):
        return 0.0
    if np.any(pts < -0.5) or np.any(pts > 0.5):
        return 0.0
    wall = np.min([
        np.min(pts[:, 0]) + 0.5,
        0.5 - np.max(pts[:, 0]),
        np.min(pts[:, 1]) + 0.5,
        0.5 - np.max(pts[:, 1]),
    ])
    D = cdist(pts, pts)
    np.fill_diagonal(D, np.inf)
    min_pair = np.min(D)
    r = min(wall, min_pair / 2.0)
    return max(r, 0.0)


def _neg_soft_radius(params, k=80.0):
    """
    Negative soft-min radius using log-sum-exp.
    params: flat array of shape (2N,)
    Returns negative of soft radius (for minimization).
    """
    pts = params.reshape(N, 2)
    if np.any(~np.isfinite(pts)):
        return 1e10
    if np.any(pts < -0.5) or np.any(pts > 0.5):
        # Penalty for being outside
        out = np.maximum(0.0, np.abs(pts) - 0.5)
        return 1e6 * np.sum(out) + 1.0

    # Wall distances (4 values)
    wall_dists = np.array([
        np.min(pts[:, 0]) + 0.5,
        0.5 - np.max(pts[:, 0]),
        np.min(pts[:, 1]) + 0.5,
        0.5 - np.max(pts[:, 1]),
    ])
    # Pairwise distances / 2
    D = cdist(pts, pts)
    np.fill_diagonal(D, np.inf)
    pair_dists = D / 2.0

    all_dists = np.concatenate([wall_dists, pair_dists.ravel()])
    all_dists = np.clip(all_dists, -1.0, 10.0)

    neg_k_d = -k * all_dists
    m = np.max(neg_k_d)
    m = min(m, 500.0)
    log_sum = m + np.log(np.sum(np.exp(neg_k_d - m)))
    r_soft = -log_sum / k
    return -r_soft


def _neg_radius_direct(params):
    """Negative of the actual (non-smooth) radius. Used for verification."""
    pts = params.reshape(N, 2)
    if np.any(~np.isfinite(pts)):
        return 1e10
    if np.any(pts < -0.5) or np.any(pts > 0.5):
        out = np.maximum(0.0, np.abs(pts) - 0.5)
        return 1e6 * np.sum(out) + 1.0
    r = _compute_radius(pts)
    return -r


def _push_apart(pts, n_iters=150, strength=0.004):
    """Iteratively push apart the closest pairs of points."""
    pts = pts.copy().astype(np.float64)
    for it in range(n_iters):
        D = cdist(pts, pts)
        np.fill_diagonal(D, np.inf)
        min_idx = np.unravel_index(np.argmin(D), D.shape)
        i, j = min_idx
        min_dist = D[i, j]
        if min_dist < 1e-12:
            pts[j] += np.random.randn(2) * 0.01
            continue
        direction = pts[j] - pts[i]
        norm = np.linalg.norm(direction)
        if norm < 1e-14:
            continue
        direction /= norm
        move = strength * direction
        new_i = pts[i] - move
        new_j = pts[j] + move
        if np.all(np.abs(new_i) <= 0.5) and np.all(np.abs(new_j) <= 0.5):
            pts[i] = new_i
            pts[j] = new_j
        else:
            for frac in [0.5, 0.25, 0.1]:
                move2 = strength * frac * direction
                ni = pts[i] - move2
                nj = pts[j] + move2
                if np.all(np.abs(ni) <= 0.5) and np.all(np.abs(nj) <= 0.5):
                    pts[i] = ni
                    pts[j] = nj
                    break
    return pts


def _optimize_from_start(pts0, k=80.0, maxiter=300, time_budget=None):
    """Run L-BFGS-B from a starting configuration. Returns (pts, r)."""
    t0 = time.time()
    if time_budget is not None and time.time() - t0 > time_budget:
        return pts0, _compute_radius(pts0)

    x0 = pts0.flatten().astype(np.float64)
    # Clip to valid range
    x0 = np.clip(x0, -0.5, 0.5)

    try:
        res = minimize(_neg_soft_radius, x0, args=(k,), method='L-BFGS-B',
                       options={'maxiter': maxiter, 'gtol': 1e-9, 'ftol': 1e-12,
                                'maxls': 20})
        new_pts = res.x.reshape(N, 2)
        new_pts = np.clip(new_pts, -0.5, 0.5)
        r = _compute_radius(new_pts)
        if r > 0:
            return new_pts, r
    except Exception:
        pass
    return pts0, _compute_radius(pts0)


def _make_rect_grid(rows, cols, jitter=0.0, seed=0):
    """Create a rectangular grid of N points."""
    rng = np.random.RandomState(seed)
    xs = np.linspace(-0.5 + 1.0/(2*cols), 0.5 - 1.0/(2*cols), cols)
    ys = np.linspace(-0.5 + 1.0/(2*rows), 0.5 - 1.0/(2*rows), rows)
    pts = np.array([(x, y) for y in ys for x in xs])[:N]
    if jitter > 0:
        pts = pts + rng.randn(N, 2) * jitter
        pts = np.clip(pts, -0.49, 0.49)
    return pts


def _make_hex_grid(rows, cols, jitter=0.0, seed=0):
    """Create a hexagonal (staggered) grid of N points."""
    rng = np.random.RandomState(seed)
    dx = 1.0 / cols
    dy = dx * np.sqrt(3) / 2.0
    # Adjust dy to fit
    if (rows - 1) * dy > 1.0:
        dy = 1.0 / (rows - 1) if rows > 1 else 1.0
    pts = []
    for i in range(rows):
        y = -0.5 + dy/2 + i * dy
        shift = dx / 2.0 if i % 2 == 1 else 0.0
        xs = -0.5 + dx/2 + shift + np.arange(cols) * dx
        for x in xs:
            if -0.5 <= x <= 0.5:
                pts.append((x, y))
    pts = np.array(pts[:N])
    if len(pts) < N:
        # Pad with remaining grid points
        extra = _make_rect_grid(rows, cols, seed=seed+100)
        pts = np.vstack([pts, extra[N - len(pts):]]) if len(pts) < N else pts
        pts = pts[:N]
    if jitter > 0:
        pts = pts + rng.randn(N, 2) * jitter
        pts = np.clip(pts, -0.49, 0.49)
    return pts


def _make_random_grid(n_per_side, jitter_frac=0.3, seed=0):
    """Random perturbation of a square grid."""
    rng = np.random.RandomState(seed)
    n_per_side = int(np.ceil(np.sqrt(N)))
    grid = []
    for i in range(n_per_side):
        for j in range(n_per_side):
            x = -0.5 + (j + 0.5) / n_per_side
            y = -0.5 + (i + 0.5) / n_per_side
            grid.append((x, y))
    grid = np.array(grid[:N])
    if len(grid) < N:
        extra = rng.uniform(-0.45, 0.45, (N - len(grid), 2))
        grid = np.vstack([grid, extra])
    grid = grid[:N]
    if jitter_frac > 0:
        spacing = 1.0 / n_per_side
        grid = grid + rng.randn(N, 2) * jitter_frac * spacing
        grid = np.clip(grid, -0.49, 0.49)
    return grid


def _refine(pts, remaining_time):
    """Multi-stage refinement of a configuration."""
    t0 = time.time()
    best_r = _compute_radius(pts)
    best_pts = pts.copy()

    # Stage 1: Push apart to fix obvious overlaps
    if time.time() - t0 < remaining_time - 10:
        pts2 = _push_apart(best_pts, n_iters=100, strength=0.003)
        r2 = _compute_radius(pts2)
        if r2 > best_r:
            best_r = r2
            best_pts = pts2

    # Stage 2: L-BFGS-B with moderate k
    if time.time() - t0 < remaining_time - 8:
        pts3, r3 = _optimize_from_start(best_pts, k=60.0, maxiter=200,
                                        time_budget=remaining_time - 8)
        if r3 > best_r:
            best_r = r3
            best_pts = pts3

    # Stage 3: Push apart again
    if time.time() - t0 < remaining_time - 5:
        pts4 = _push_apart(best_pts, n_iters=80, strength=0.002)
        r4 = _compute_radius(pts4)
        if r4 > best_r:
            best_r = r4
            best_pts = pts4

    # Stage 4: L-BFGS-B with higher k (sharper)
    if time.time() - t0 < remaining_time - 3:
        pts5, r5 = _optimize_from_start(best_pts, k=120.0, maxiter=150,
                                        time_budget=remaining_time - 3)
        if r5 > best_r:
            best_r = r5
            best_pts = pts5

    # Stage 5: Final push
    if time.time() - t0 < remaining_time - 1:
        pts6 = _push_apart(best_pts, n_iters=50, strength=0.001)
        r6 = _compute_radius(pts6)
        if r6 > best_r:
            best_r = r6
            best_pts = pts6

    return best_pts, best_r


def construct():
    t0 = time.time()
    best_r = 0.0
    best_pts = None

    # Generate diverse starting configurations
    starts = []

    # Rectangular grids with various aspect ratios
    for rows, cols in [(20, 10), (10, 20), (25, 8), (8, 25), (16, 13), (13, 16)]:
        if rows * cols >= N:
            starts.append(_make_rect_grid(rows, cols, jitter=0.0, seed=0))
            starts.append(_make_rect_grid(rows, cols, jitter=0.01, seed=42))

    # Hexagonal grids
    for rows, cols in [(20, 10), (10, 20), (14, 15), (15, 14)]:
        if rows * cols >= N:
            starts.append(_make_hex_grid(rows, cols, jitter=0.0, seed=0))
            starts.append(_make_hex_grid(rows, cols, jitter=0.01, seed=7))

    # Random perturbation of square grid
    for n_side in [14, 15]:
        starts.append(_make_random_grid(n_side, jitter_frac=0.3, seed=123))
        starts.append(_make_random_grid(n_side, jitter_frac=0.5, seed=456))

    # Pure random
    rng = np.random.RandomState(99)
    starts.append(rng.uniform(-0.45, 0.45, (N, 2)))

    # Deduplicate starts (keep first occurrence of similar configs)
    unique_starts = []
    for s in starts:
        s = np.clip(s, -0.5, 0.5)
        is_dup = False
        for u in unique_starts:
            if np.max(np.abs(s - u)) < 0.05:
                is_dup = True
                break
        if not is_dup:
            unique_starts.append(s)

    # Evaluate each start with refinement
    n_starts = len(unique_starts)
    for idx, start in enumerate(unique_starts):
        elapsed = time.time() - t0
        remaining = _TIME_LIMIT - elapsed - 5  # leave 5s safety
        if remaining < 3:
            break

        # Quick evaluation first
        r_quick = _compute_radius(start)
        if r_quick > best_r:
            best_r = r_quick
            best_pts = start.copy()

        # Refine
        pts_ref, r_ref = _refine(start, remaining)
        if r_ref > best_r:
            best_r = r_ref
            best_pts = pts_ref

    # If we still have time, do additional multi-start from the best result
    elapsed = time.time() - t0
    remaining = _TIME_LIMIT - elapsed - 3
    if remaining > 10 and best_pts is not None:
        # Try perturbing the best result and re-optimizing
        rng = np.random.RandomState(2024)
        for perturb_scale in [0.005, 0.01, 0.02]:
            if time.time() - t0 > _TIME_LIMIT - 5:
                break
            perturbed = best_pts + rng.randn(N, 2) * perturb_scale
            perturbed = np.clip(perturbed, -0.49, 0.49)
            pts_p, r_p = _refine(perturbed, _TIME_LIMIT - (time.time() - t0) - 3)
            if r_p > best_r:
                best_r = r_p
                best_pts = pts_p

    # Final fallback
    if best_pts is None or best_r <= 0:
        best_pts = _make_rect_grid(20, 10, jitter=0.0, seed=0)
        best_r = _compute_radius(best_pts)

    # Ensure valid output
    best_pts = np.clip(best_pts, -0.5, 0.5)
    if best_pts.shape != (N, 2) or np.any(~np.isfinite(best_pts)):
        best_pts = _make_rect_grid(20, 10, jitter=0.0, seed=0)

    return best_pts
