"""Fifty equal spheres in a unit cube: BCC superlattice init + repulsive-force optimization."""
import time
import numpy as np

N = 50
HALF = 0.5  # cube is [-0.5, 0.5]^3


def _valid_radius(centers):
    """Compute the true (hard) radius for a set of centers; None if invalid."""
    c = np.asarray(centers, dtype=float)
    if c.shape != (N, 3):
        return None
    if not np.all(np.isfinite(c)):
        return None
    # wall distances
    wall = HALF - np.abs(c)  # shape (N,3)
    wall_min = wall.min()
    # pair distances
    diff = c[:, None, :] - c[None, :, :]
    d = np.linalg.norm(diff, axis=2)
    # set diagonal to inf
    np.fill_diagonal(d, np.inf)
    pair_min = d.min() / 2.0
    r = min(wall_min, pair_min)
    if not np.isfinite(r):
        return None
    return float(r)


def _bcc_init():
    """Build a BCC superlattice of 50 points, scaled to fit the cube, then jitter."""
    rng = np.random.default_rng(12345)
    # BCC: for a cubic cell of side a, points at corners and center.
    # We build a superlattice of M^3 cells, each contributing 2 points (corner + center)
    # but corners are shared. Simpler: generate a grid of cell centers plus body centers.
    # For M cells per axis: total points = M^3 * 2 approximately (corners shared).
    # We'll just generate a dense BCC set and take the first N, then optimize.
    M = 4  # 4x4x4 cells -> 64 corners shared + 64 centers. Use a compact scheme.
    a = 1.0 / M  # cell side so total extent = 1
    pts = []
    for i in range(M):
        for j in range(M):
            for k in range(M):
                # corner of this cell (offset so overall centered)
                cx = (i + 0.0) * a - 0.5
                cy = (j + 0.0) * a - 0.5
                cz = (k + 0.0) * a - 0.5
                pts.append([cx, cy, cz])
                # body center
                pts.append([cx + 0.5 * a, cy + 0.5 * a, cz + 0.5 * a])
    pts = np.array(pts)
    # We have 2*M^3 = 128 points; take a well-spread subset of N.
    # To get a good spread, we can pick every k-th or use a simple selection.
    # Better: scale the whole BCC to fill the cube nicely and take N.
    # Let's instead just take the first N after sorting by a space-filling order.
    # For simplicity, take a strided selection to ensure spread.
    order = np.arange(len(pts))
    # Use a deterministic space-filling-ish selection: sort by (i+j+k) then take evenly.
    # Simpler: just take evenly spaced indices.
    sel = np.linspace(0, len(pts) - 1, N).astype(int)
    pts = pts[sel]
    # Add small random jitter to break symmetry and avoid degenerate minima
    pts = pts + rng.normal(0, 0.01, size=pts.shape)
    # Clamp into cube with margin
    pts = np.clip(pts, -HALF + 1e-3, HALF - 1e-3)
    return pts


def _soft_min_radius(centers, beta):
    """
    Smooth approximation of the radius using a soft-min over all constraints.
    Constraints: wall distances (3N) and half-pair-distances (N(N-1)/2).
    Returns (value, grad) where grad is d(value)/d(centers).
    """
    c = centers  # (N,3)
    # Wall terms: for each point and each axis, distance to nearest wall.
    # dist_wall_ik = HALF - |c_ik|
    abs_c = np.abs(c)
    wall = HALF - abs_c  # (N,3)
    # Pair terms: 0.5 * |c_i - c_j| for i<j
    n = c.shape[0]
    iu = np.triu_indices(n, k=1)
    diff = c[iu[0]] - c[iu[1]]  # (P,3)
    pdist = np.linalg.norm(diff, axis=1) * 0.5  # (P,)
    # Combine all constraints into one vector
    all_c = np.concatenate([wall.reshape(-1), pdist])  # (3N + P,)
    # Soft-min: f = (1/beta) * log(sum(exp(-beta * x)))  -> approximates min x
    # Use log-sum-exp for stability
    m = all_c.min()
    s = np.sum(np.exp(-beta * (all_c - m)))
    value = m - np.log(s) / beta
    # Gradient of soft-min w.r.t. each constraint: p_k = exp(-beta*(x_k - m)) / s
    p = np.exp(-beta * (all_c - m)) / s  # weights, sum to 1
    # Now backprop through each constraint to centers.
    grad = np.zeros_like(c)
    # Wall part: for point i, axis k: constraint = HALF - |c_ik|
    # d(constraint)/d(c_ik) = -sign(c_ik)
    w = p[:3 * n].reshape(n, 3)
    grad += -w * np.sign(c)
    # Pair part: constraint = 0.5 * |c_i - c_j|
    # d/d(c_i) = 0.5 * (c_i - c_j)/|c_i - c_j|
    # d/d(c_j) = 0.5 * (c_j - c_i)/|c_i - c_j|
    pw = p[3 * n:]  # (P,)
    di = iu[0]
    dj = iu[1]
    dist_full = np.linalg.norm(diff, axis=1)  # (P,)
    # avoid div by zero
    dist_safe = np.where(dist_full < 1e-12, 1e-12, dist_full)
    unit = diff / dist_safe[:, None]  # (P,3)
    contrib = 0.5 * pw[:, None] * unit  # (P,3)
    # accumulate: c_i gets +contrib, c_j gets -contrib
    np.add.at(grad, di, contrib)
    np.add.at(grad, dj, -contrib)
    return value, grad


def _optimize(centers, deadline, rng):
    """
    Gradient ascent on the soft-min radius, with periodic hard checks.
    Returns best valid center set found.
    """
    best = centers.copy()
    best_r = _valid_radius(best)
    if best_r is None:
        best_r = -1.0
    beta = 80.0  # softness; larger = sharper
    lr = 0.02
    t0 = time.time()
    it = 0
    while time.time() < deadline - 5.0:
        it += 1
        val, grad = _soft_min_radius(centers, beta)
        # Gradient ascent step
        centers = centers + lr * grad
        # Keep inside cube (project)
        centers = np.clip(centers, -HALF + 1e-6, HALF - 1e-6)
        # Periodically evaluate hard radius
        if it % 20 == 0:
            r = _valid_radius(centers)
            if r is not None and r > best_r:
                best_r = r
                best = centers.copy()
        # Decay learning rate slowly
        lr = max(0.001, lr * 0.999)
    # Final check
    r = _valid_radius(centers)
    if r is not None and r > best_r:
        best_r = r
        best = centers.copy()
    return best


def construct():
    start = time.time()
    deadline = start + 145.0  # safety margin before 150s
    rng = np.random.default_rng(2024)

    # Start from BCC init
    centers = _bcc_init()
    best = centers.copy()
    best_r = _valid_radius(best)
    if best_r is None:
        best_r = -1.0

    # Run a few restarts with different seeds if time permits
    for seed in [2024, 7, 42, 99, 314]:
        if time.time() > deadline - 20.0:
            break
        rng = np.random.default_rng(seed)
        # Re-init with different jitter
        c0 = _bcc_init()
        c0 = c0 + rng.normal(0, 0.02, size=c0.shape)
        c0 = np.clip(c0, -HALF + 1e-3, HALF - 1e-3)
        local_best = _optimize(c0, deadline, rng)
        r = _valid_radius(local_best)
        if r is not None and r > best_r:
            best_r = r
            best = local_best

    # Safety: ensure best is valid; if not, fall back to a known-valid grid
    r = _valid_radius(best)
    if r is None:
        # Fallback: simple cubic grid
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best = grid[:N]
    return best
