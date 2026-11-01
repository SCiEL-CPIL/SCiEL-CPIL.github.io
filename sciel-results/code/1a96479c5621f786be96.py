"""Packing of 50 equal circles in a unit square via force-based repulsion + accelerated SA."""
import numpy as np
import time

N = 50
HALF = 0.5


def _compute_r(pts):
    """Compute the feasible radius for a configuration (double precision)."""
    if pts.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    d = np.minimum(np.minimum(HALF - pts[:, 0], HALF - pts[:, 1]),
                   np.minimum(HALF + pts[:, 0], HALF + pts[:, 1]))
    r = float(np.min(d))
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    r = min(r, float(np.min(dist)) / 2.0)
    return r


def _compute_forces(pts, r_target=None):
    """
    Compute repulsion forces on each circle.
    If r_target is given, use a hard constraint model: circles closer than
    r_target*2 get strong repulsion. Otherwise use inverse-distance repulsion.
    Returns array of shape (N, 2) with force on each circle.
    """
    forces = np.zeros((N, 2))
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, 1e15)

    if r_target is not None:
        # Hard constraint: repel if dist < 2*r_target
        min_dist = 2.0 * r_target
        overlap = np.maximum(0.0, min_dist - dist)
        # Force proportional to overlap, direction = normalized diff
        inv_dist = np.where(dist > 1e-10, 1.0 / dist, 0.0)
        force_mag = overlap * inv_dist
        force_mag[~np.isfinite(force_mag)] = 0.0
        forces = np.sum(force_mag[:, :, None] * diff, axis=1)
    else:
        # Inverse-distance repulsion (soft)
        inv_dist = np.where(dist > 1e-10, 1.0 / (dist ** 2 + 1e-8), 0.0)
        force_mag = inv_dist
        force_mag[~np.isfinite(force_mag)] = 0.0
        forces = np.sum(force_mag[:, :, None] * diff, axis=1)

    # Wall repulsion
    for i in range(N):
        for dim in range(2):
            if pts[i, dim] < -HALF + r_target if r_target else -HALF + 0.02:
                forces[i, dim] += 50.0 * ((-HALF + (r_target if r_target else 0.02)) - pts[i, dim])
            if pts[i, dim] > HALF - r_target if r_target else HALF - 0.02:
                forces[i, dim] -= 50.0 * (pts[i, dim] - (HALF - (r_target if r_target else 0.02)))

    return forces


def _force_optimize(pts0, r_target=None, n_steps=5000, lr=0.001, deadline=None):
    """
    Force-based optimization: iteratively move circles along repulsion forces.
    Much faster than L-BFGS with finite-difference gradients.
    """
    pts = pts0.copy()
    best_pts = pts.copy()
    best_r = _compute_r(pts)

    for step in range(n_steps):
        if deadline is not None and time.time() > deadline:
            break

        forces = _compute_forces(pts, r_target)
        # Normalize forces to avoid exploding steps
        fmag = np.linalg.norm(forces, axis=1, keepdims=True)
        fmag = np.maximum(fmag, 1e-10)
        forces = forces / fmag

        pts = pts + lr * forces
        pts = np.clip(pts, -HALF + 1e-9, HALF - 1e-9)

        if step % 50 == 0:
            r = _compute_r(pts)
            if r > best_r:
                best_r = r
                best_pts = pts.copy()

    r = _compute_r(pts)
    if r > best_r:
        best_r = r
        best_pts = pts.copy()

    return best_pts, best_r


def _accelerated_sa(pts0, rng, deadline, t_start=0.1, t_end=0.0001, n_iter=None):
    """
    Accelerated simulated annealing with vectorized distance computation.
    Uses incremental updates where possible.
    """
    pts = pts0.copy()
    current_r = _compute_r(pts)
    best_pts = pts.copy()
    best_r = current_r

    if n_iter is None:
        n_iter = 100000

    for it in range(n_iter):
        if time.time() > deadline:
            break

        progress = it / n_iter
        t = t_start * (t_end / t_start) ** progress

        # Perturb 1-3 random circles for more exploration
        n_perturb = rng.integers(1, 4)
        idxs = rng.choice(N, size=n_perturb, replace=False)

        step_size = 0.015 * (1.0 - progress * 0.7)
        new_pts = pts.copy()
        for idx in idxs:
            direction = rng.normal(0, 1, size=2)
            direction /= (np.linalg.norm(direction) + 1e-15)
            new_pts[idx] += direction * step_size

        new_pts = np.clip(new_pts, -HALF + 1e-9, HALF - 1e-9)
        new_r = _compute_r(new_pts)

        delta = new_r - current_r
        if delta > 0:
            pts = new_pts
            current_r = new_r
            if current_r > best_r:
                best_r = current_r
                best_pts = pts.copy()
        else:
            if t > 1e-15:
                prob = np.exp(delta / t)
                if rng.random() < prob:
                    pts = new_pts
                    current_r = new_r

    return best_pts, best_r


def _hex_init(rng, jitter=0.0):
    """Hexagonal close-packed lattice, jittered."""
    pts = []
    a = 0.12
    dx = a
    dy = a * np.sqrt(3) / 2
    row = 0
    while len(pts) < N:
        offset = (dx / 2) if (row % 2) else 0.0
        x = -HALF + a + offset
        while len(pts) < N and x < HALF - a + 1e-9:
            pts.append([x, -HALF + a + row * dy])
            x += dx
        row += 1
    pts = np.array(pts[:N])
    if jitter > 0:
        pts = pts + rng.normal(0, jitter, size=pts.shape)
    return np.clip(pts, -HALF + 1e-9, HALF - 1e-9)


def _grid_init(rng, jitter=0.0):
    """Grid initialization."""
    m = 7
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    extra = np.array([[0.0, 0.0]])
    grid = np.vstack([grid, extra])[:N]
    if jitter > 0:
        grid = grid + rng.normal(0, jitter, size=grid.shape)
    return np.clip(grid, -HALF + 1e-9, HALF - 1e-9)


def _random_init(rng):
    """Random initialization."""
    return rng.uniform(-HALF + 0.05, HALF - 0.05, size=(N, 2))


def _binary_search_radius(pts0, rng, deadline, n_iter=10):
    """
    Binary search on radius: for each candidate radius, run force optimization
    with that as a hard constraint, then check if the achieved radius meets it.
    """
    best_pts = pts0.copy()
    best_r = _compute_r(pts0)

    r_low = best_r
    r_high = min(best_r + 0.015, 0.12)

    for iteration in range(n_iter):
        if time.time() > deadline:
            break
        r_mid = (r_low + r_high) / 2.0

        # Force-optimize with r_mid as target
        pts, r_achieved = _force_optimize(
            best_pts, r_target=r_mid,
            n_steps=3000, lr=0.0005, deadline=deadline
        )

        # Also try SA polish
        if time.time() < deadline - 5:
            pts, r_achieved = _accelerated_sa(
                pts, rng, deadline,
                t_start=0.05, t_end=0.0001, n_iter=20000
            )

        r_achieved = _compute_r(pts)
        if r_achieved > best_r:
            best_r = r_achieved
            best_pts = pts.copy()

        if r_achieved >= r_mid - 1e-9:
            r_low = r_mid
        else:
            r_high = r_mid

    return best_pts, best_r


def construct():
    start = time.time()
    deadline = start + 140.0
    rng = np.random.default_rng(12345)

    best_pts = None
    best_r = 0.0

    def try_update(pts):
        nonlocal best_pts, best_r
        r = _compute_r(pts)
        if r > best_r:
            best_r = r
            best_pts = pts.copy()

    # Phase 1: Multi-start with force optimization (fast, no gradient overhead)
    inits = [
        _hex_init(rng, jitter=0.005),
        _hex_init(rng, jitter=0.02),
        _hex_init(rng, jitter=0.05),
        _grid_init(rng, jitter=0.01),
        _grid_init(rng, jitter=0.04),
        _random_init(rng),
        _random_init(rng),
        _random_init(rng),
        _random_init(rng),
    ]

    for i, init in enumerate(inits):
        if time.time() > deadline - 30.0:
            break
        pts, r = _force_optimize(init, n_steps=5000, lr=0.002, deadline=deadline)
        try_update(pts)

    if best_pts is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]

    # Phase 2: Accelerated SA from best point
    if time.time() < deadline - 40.0:
        sa_pts, sa_r = _accelerated_sa(
            best_pts, rng, deadline,
            t_start=0.08, t_end=0.0001, n_iter=80000
        )
        try_update(sa_pts)

    # Phase 3: Force polish with decreasing learning rate
    if time.time() < deadline - 20.0:
        for lr in [0.001, 0.0005, 0.0002]:
            if time.time() > deadline - 10.0:
                break
            pts, r = _force_optimize(best_pts, n_steps=4000, lr=lr, deadline=deadline)
            try_update(pts)

    # Phase 4: Binary search on radius with force + SA
    if time.time() < deadline - 20.0:
        bs_pts, bs_r = _binary_search_radius(best_pts, rng, deadline, n_iter=8)
        try_update(bs_pts)

    # Phase 5: Final SA polish
    if time.time() < deadline - 5.0:
        sa_pts, sa_r = _accelerated_sa(
            best_pts, rng, deadline,
            t_start=0.03, t_end=0.00005, n_iter=50000
        )
        try_update(sa_pts)

    # Phase 6: Final force polish
    if time.time() < deadline - 3.0:
        pts, r = _force_optimize(best_pts, n_steps=3000, lr=0.0001, deadline=deadline)
        try_update(pts)

    return best_pts
