"""Rewritten program for the 300-circles-in-a-circle packing task."""
import numpy as np

N = 300


def _radius(pts):
    """Largest common radius for given centres."""
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape != (N, 2) or not np.all(np.isfinite(pts)):
        return 0.0
    # Pairwise distances via broadcasting (300x300 is fine)
    diff = pts[:, None, :] - pts[None, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(d, np.inf)
    min_pair = d.min() / 2.0
    # Distance to container boundary
    r = np.sqrt(np.sum(pts * pts, axis=1))
    min_bound = (1.0 - r).min()
    return float(min(min_pair, min_bound))


def _ring_init(n_rings):
    """Deterministic ring-based initial placement."""
    pts = []
    # Centre point
    pts.append([0.0, 0.0])
    remaining = N - 1
    # Distribute remaining across rings with increasing radius
    ring_count = n_rings
    # Approximate ring radii: pack rings so total ~ N
    # Use roughly uniform angular spacing per ring
    radii = np.linspace(0.15, 0.92, ring_count)
    # Number of points per ring proportional to circumference
    weights = radii
    weights = weights / weights.sum()
    counts = np.round(weights * remaining).astype(int)
    # Fix rounding to match total
    diff = remaining - counts.sum()
    # Add/remove from largest rings
    order = np.argsort(-radii)
    idx = 0
    while diff != 0:
        o = order[idx % len(order)]
        if diff > 0:
            counts[o] += 1
            diff -= 1
        else:
            if counts[o] > 0:
                counts[o] -= 1
                diff += 1
        idx += 1
        if idx > 10000:
            break
    for k in range(ring_count):
        c = counts[k]
        if c <= 0:
            continue
        rad = radii[k]
        angles = 2.0 * np.pi * np.arange(c) / c + (0.5 * np.pi / max(c,1)) * (k % 2)
        pts.append(np.column_stack([rad * np.cos(angles), rad * np.sin(angles)]))
    pts = np.vstack(pts)
    # Trim or pad if needed
    if len(pts) > N:
        pts = pts[:N]
    else:
        # Pad with small random points near origin
        while len(pts) < N:
            ang = np.random.uniform(0, 2 * np.pi)
            rr = np.random.uniform(0.0, 0.05)
            pts = np.vstack([pts, [[rr * np.cos(ang), rr * np.sin(ang)]]])
    return pts[:N].astype(np.float64)


def _hex_init():
    """Hexagonal lattice initial placement (fallback)."""
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
        if len(points) >= N:
            return points[np.argsort(radius, kind="stable")[:N]]
        spacing *= 0.98


def _repulsion_step(pts, r_target, dt):
    """Move points to increase minimum radius. Returns new pts."""
    pts = pts.copy()
    diff = pts[:, None, :] - pts[None, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(d, np.inf)
    # Force: repel when closer than 2*r_target, attract slightly when far
    # Use a smooth force
    force = np.zeros_like(pts)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            dij = d[i, j]
            if dij < 1e-9:
                continue
            # Repulsive force proportional to (2*r_target - dij)
            overlap = (2.0 * r_target - dij)
            if overlap > 0:
                f = overlap / dij
                force[i] += f * diff[i, j]
    # Wall force: push inward if near boundary
    rr = np.sqrt(np.sum(pts * pts, axis=1))
    for i in range(N):
        if rr[i] > 1.0 - r_target:
            if rr[i] > 1e-9:
                force[i] -= (rr[i] - (1.0 - r_target)) * pts[i] / rr[i]
    pts = pts + dt * force
    return pts


def _local_search(pts, start_time, deadline):
    """Vectorized-ish local refinement with adaptive step size."""
    best = pts.copy()
    best_r = _radius(best)
    dt = 0.002
    # Precompute for speed
    while (time.time() - start_time) < deadline:
        # Compute current radius
        cur_r = _radius(best)
        if cur_r > best_r:
            best_r = cur_r
            # Keep dt or increase slightly
            dt = min(dt * 1.05, 0.01)
        else:
            dt *= 0.7
        # Do a batch of small steps
        steps = 3
        for _ in range(steps):
            new_pts = _repulsion_step(best, best_r, dt)
            new_r = _radius(new_pts)
            if new_r >= best_r - 1e-12:
                best = new_pts
                best_r = max(best_r, new_r)
            else:
                break
    return best, best_r


import time


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s

    # Generate multiple initial candidates
    candidates = []
    # Ring-based with different ring counts
    for nr in [8, 10, 12, 14]:
        candidates.append(_ring_init(nr))
    # Hex lattice
    try:
        candidates.append(_hex_init())
    except Exception:
        pass

    # Keep the best initial
    best_pts = None
    best_r = -1.0
    for cand in candidates:
        r = _radius(cand)
        if r > best_r:
            best_r = r
            best_pts = cand.copy()

    # Local search from the best start
    if (time.time() - start_time) < deadline:
        best_pts, best_r = _local_search(best_pts, start_time, deadline)

    # If time remains, try a second local search from a different start
    if (time.time() - start_time) < deadline - 20.0:
        # Pick the second-best initial
        scored = sorted([(_radius(c), c) for c in candidates], reverse=True)
        if len(scored) > 1:
            second_pts = scored[1][1].copy()
            # Quick local search with shorter budget
            sub_deadline = min(time.time() + 30.0, deadline)
            sp, sr = _local_search(second_pts, time.time(), sub_deadline)
            if sr > best_r:
                best_pts, best_r = sp, sr

    return best_pts


if __name__ == "__main__":
    pts = construct()
    print(_radius(pts))
