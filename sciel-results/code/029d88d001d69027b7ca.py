"""Packing 300 equal circles in a unit circle.

Strategy:
1. Generate a hexagonal lattice with fine spacing; pick the N closest-to-origin
   points that maximise the achievable radius (min of pairwise/2 and boundary).
2. Refine with a vectorised local-search / repulsion loop:
   - Each iteration picks a random point, proposes a small perturbation,
     and accepts if the computed radius improves.
   - Also run a repulsion-based relaxation pass to untangle close pairs.
3. Deadline guard ensures we return the best configuration found before
   the 150 s wall-clock kill.
"""
import numpy as np
import time

N = 300


def _compute_radius(pts):
    """Return the max common radius for a configuration of N points."""
    if not np.all(np.isfinite(pts)):
        return 0.0
    dists_origin = np.hypot(pts[:, 0], pts[:, 1])
    boundary_min = 1.0 - np.max(dists_origin)
    if boundary_min <= 0.0:
        return 0.0
    # Pairwise minimum distance via cdist (fast for N=300)
    try:
        from scipy.spatial.distance import cdist
        D = cdist(pts, pts)
        np.fill_diagonal(D, np.inf)
        min_pair = float(np.min(D))
    except Exception:
        # Fallback: loop-based
        min_pair = 1.0
        for i in range(N):
            dx = pts[i, 0] - pts[:i, 0]
            dy = pts[i, 1] - pts[:i, 1]
            d = np.hypot(dx, dy)
            if len(d):
                m = float(np.min(d))
                if m < min_pair:
                    min_pair = m
    return min(min_pair / 2.0, boundary_min)


def _hex_initial(spacing, rotation=0.0):
    """Generate a hexagonal lattice, keep points inside unit circle, return N closest."""
    sqrt3 = np.sqrt(3.0)
    k = int(1.0 / spacing) + 4
    pts = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * sqrt3 / 2.0
            if rotation != 0.0:
                cr, sr = np.cos(rotation), np.sin(rotation)
                x, y = x * cr - y * sr, x * sr + y * cr
            if np.hypot(x, y) < 1.0:
                pts.append((x, y))
    pts = np.array(pts)
    if len(pts) < N:
        return None
    dists = np.hypot(pts[:, 0], pts[:, 1])
    idx = np.argsort(dists, kind='stable')[:N]
    return pts[idx]


def _local_search(pts, deadline, max_iters=8000):
    """Vectorised local search: random perturbation, accept if radius improves."""
    current = pts.copy()
    current_r = _compute_radius(current)
    best = current.copy()
    best_r = current_r
    rng = np.random.default_rng(12345)

    for _ in range(max_iters):
        if time.time() > deadline:
            break
        idx = int(rng.integers(N))
        old = current[idx].copy()
        # Adaptive step: start small, can grow if stuck
        step = 0.008
        new = old + rng.uniform(-step, step, 2)
        mag = np.hypot(new[0], new[1])
        if mag > 0.995:
            new *= 0.995 / mag
        current[idx] = new
        r = _compute_radius(current)
        if r > current_r:
            current_r = r
            if r > best_r:
                best_r = r
                best = current.copy()
        else:
            current[idx] = old
    return best, best_r


def _repulsion_relax(pts, deadline, max_iters=2000):
    """Repulsion-based relaxation: push apart points that are too close."""
    current = pts.copy()
    current_r = _compute_radius(current)
    best = current.copy()
    best_r = current_r
    rng = np.random.default_rng(99999)

    for it in range(max_iters):
        if time.time() > deadline:
            break
        # Compute all pairwise distances
        try:
            from scipy.spatial.distance import cdist
            D = cdist(current, current)
        except Exception:
            D = np.sqrt(np.sum((current[:, None, :] - current[None, :, :])**2, axis=-1))
        np.fill_diagonal(D, np.inf)
        # Find closest pair
        min_d = float(np.min(D))
        if min_d > 2.0 * current_r * 1.01:
            break  # no conflicts
        # Identify points involved in close pairs
        close_mask = D < 2.0 * current_r * 1.05
        np.fill_diagonal(close_mask, False)
        # Compute repulsion forces
        forces = np.zeros_like(current)
        for i in range(N):
            for j in range(N):
                if close_mask[i, j]:
                    d = D[i, j]
                    if d > 1e-12:
                        diff = current[i] - current[j]
                        forces[i] += diff / d * (2.0 * current_r - d) * 0.5
        # Boundary force
        mags = np.hypot(current[:, 0], current[:, 1])
        for i in range(N):
            if mags[i] > 1.0 - current_r:
                forces[i] -= current[i] / mags[i] * (mags[i] - (1.0 - current_r)) * 0.5
        # Apply forces with damping
        lr = 0.05
        current += forces * lr
        # Project back inside circle
        mags = np.hypot(current[:, 0], current[:, 1])
        too_far = mags > 0.999
        if np.any(too_far):
            current[too_far] *= 0.999 / mags[too_far]
        r = _compute_radius(current)
        if r > best_r:
            best_r = r
            best = current.copy()
        current_r = r
    return best, best_r


def construct():
    deadline = time.time() + 140.0  # safety margin before 150 s kill

    # --- Stage 1: Find best hexagonal lattice initialisation ---
    best_pts = None
    best_r = 0.0

    # Try a range of spacings and rotations
    for trial in range(60):
        if time.time() > deadline:
            break
        spacing = 0.085 + trial * 0.0012
        rotation = (trial % 10) * np.pi / 30.0  # 0 to 90 degrees in 10 steps
        pts = _hex_initial(spacing, rotation)
        if pts is None:
            continue
        r = _compute_radius(pts)
        if r > best_r:
            best_r = r
            best_pts = pts.copy()

    if best_pts is None:
        # Fallback: random
        angles = np.random.uniform(0, 2 * np.pi, N)
        radii = np.sqrt(np.random.uniform(0, 1, N))
        best_pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        best_r = _compute_radius(best_pts)

    # --- Stage 2: Repulsion relaxation to untangle close pairs ---
    best_pts, best_r = _repulsion_relax(best_pts, deadline, max_iters=1500)

    # --- Stage 3: Local search refinement ---
    best_pts, best_r = _local_search(best_pts, deadline, max_iters=6000)

    # --- Stage 4: Another short repulsion pass to polish ---
    best_pts, best_r = _repulsion_relax(best_pts, deadline, max_iters=800)

    # --- Final validation ---
    # Ensure strictly inside
    dists_origin = np.hypot(best_pts[:, 0], best_pts[:, 1])
    max_allowed = 1.0 - 1e-9
    mask = dists_origin >= max_allowed
    if np.any(mask):
        best_pts[mask] *= (max_allowed / dists_origin[mask]) * 0.9999

    # Ensure no coincident points
    for i in range(N):
        for j in range(i + 1, N):
            d = np.hypot(best_pts[i, 0] - best_pts[j, 0],
                         best_pts[i, 1] - best_pts[j, 1])
            if d < 1e-8:
                angle = np.random.uniform(0, 2 * np.pi)
                best_pts[i, 0] += 0.01 * np.cos(angle)
                best_pts[i, 1] += 0.01 * np.sin(angle)
                best_pts[j, 0] -= 0.01 * np.cos(angle)
                best_pts[j, 1] -= 0.01 * np.sin(angle)

    if not np.all(np.isfinite(best_pts)) or best_pts.shape != (N, 2):
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        radii = np.sqrt(np.linspace(0, 1, N))
        best_pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        best_pts *= 0.99

    return best_pts
