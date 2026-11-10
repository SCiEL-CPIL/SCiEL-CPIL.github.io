"""Ring-based packing for 300 equal circles in the unit disk, with local refinement."""
import numpy as np
import time

N = 300
_TIME_BUDGET = 110.0  # seconds, safety margin before the 120 s kill


def _score(pts):
    """Return the largest feasible common radius (min pairwise dist/2, boundary margin)."""
    if pts is None or len(pts) != N:
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    r = 1.0 - np.linalg.norm(pts, axis=1)
    s = np.sum(pts * pts, axis=1)
    d2 = s[:, None] + s[None, :] - 2.0 * (pts @ pts.T)
    np.fill_diagonal(d2, 1.0)
    d2 = np.maximum(d2, 0.0)
    d = np.sqrt(d2)
    iu = np.triu_indices(N, k=1)
    return min(float(np.min(r)), float(np.min(d[iu])) / 2.0)


def _ring_build(rings, phase):
    """Build point array from ring radii `rings` (list of (n, radius)) and angular phase."""
    pts = np.empty((N, 2))
    idx = 0
    for n, rad in rings:
        if n <= 0:
            continue
        if n == 1:
            pts[idx, 0] = rad * np.cos(phase)
            pts[idx, 1] = rad * np.sin(phase)
            idx += 1
        else:
            th = phase + 2.0 * np.pi * np.arange(n) / n
            pts[idx:idx + n, 0] = rad * np.cos(th)
            pts[idx:idx + n, 1] = rad * np.sin(th)
            idx += n
    return pts


def _local_refine(pts, t0):
    """Subgradient ascent on the (smoothed) minimum distance with boundary handling."""
    pts = pts.astype(float).copy()
    t = 0.004
    it = 0
    while time.time() - t0 < _TIME_BUDGET - 5.0:
        s = np.sum(pts * pts, axis=1)
        d2 = s[:, None] + s[None, :] - 2.0 * (pts @ pts.T)
        np.fill_diagonal(d2, 1.0)
        d2 = np.maximum(d2, 0.0)
        d = np.sqrt(d2)
        d = d + t
        inv = 1.0 / d
        W = inv * inv
        np.fill_diagonal(W, 0.0)
        # pairwise attractive force (pull apart)
        grad = (W @ pts) - (W * pts[:, None, 0]) @ np.ones((N, 1))  # placeholder, do properly below
        # proper: grad_i = sum_j W_ij (p_i - p_j)
        grad = np.zeros((N, 2))
        for j in range(N):
            grad += W[:, j] * (pts - pts[j])
        # boundary: push inward if close to edge
        rad = np.linalg.norm(pts, axis=1)
        mask = rad > 0.96
        if np.any(mask):
            grad[mask] -= (pts[mask] / rad[mask][:, None]) * 0.6
        # normalize steps
        gn = np.linalg.norm(grad, axis=1, keepdims=True)
        gn = np.maximum(gn, 1e-12)
        step = 0.0015 * (grad / gn)
        pts += step
        # clamp inside
        rad = np.linalg.norm(pts, axis=1)
        m = rad > 0.999
        if np.any(m):
            pts[m] *= (0.999 / rad[m])[:, None]
        it += 1
        if it % 40 == 0:
            t *= 0.85
    return pts


def _anneal_build(rings, phase, t0):
    """Simulated-annealing style local search on ring radii and phase."""
    pts = _ring_build(rings, phase)
    best = _score(pts)
    best_pts = pts.copy()
    cur = best
    T = 0.0004
    ring_radii = [r for _, r in rings]
    n_rings = len(ring_radii)
    it = 0
    while time.time() - t0 < _TIME_BUDGET - 6.0:
        # perturb: change one ring radius or the phase
        if np.random.rand() < 0.3:
            phase = (phase + np.random.normal(0, 0.05)) % (2 * np.pi)
            ring_radii = [r for _, r in rings]
        else:
            k = np.random.randint(0, n_rings)
            ring_radii = list(ring_radii)
            ring_radii[k] = max(0.0, ring_radii[k] + np.random.normal(0, 0.008))
            rings = [rings[i] if i != k else (rings[i][0], ring_radii[k]) for i in range(n_rings)]
        pts = _ring_build(rings, phase)
        sc = _score(pts)
        if sc >= cur or np.random.rand() < np.exp((sc - cur) / max(T, 1e-12)):
            cur = sc
            if sc > best:
                best = sc
                best_pts = pts.copy()
        T *= 0.995
        it += 1
    return best_pts, best


def construct():
    t0 = time.time()
    np.random.seed(0)

    # Build a sensible initial ring layout: rings from center outward.
    # Outermost ring near boundary, inner rings denser.
    # We'll choose ring counts to sum to N, with outer rings having more points.
    # Target radii: outer ring near 1 - r, spacing shrinks inward.
    # Start with a guess and let anneal refine.

    # Determine number of rings and per-ring counts.
    # Use ~11 rings; outer rings hold more points.
    # counts roughly proportional to ring radius.
    n_rings = 11
    # weights ~ radius of each ring
    base_radii = np.linspace(0.10, 0.985, n_rings)
    weights = base_radii
    counts = np.maximum(1, np.round(weights / weights.sum() * (N - n_rings)))
    counts[0] = 1  # single center point
    # adjust to sum to N
    diff = N - counts.sum()
    # add/remove from the largest rings
    order = np.argsort(-counts)
    i = 0
    while diff != 0:
        k = order[i % n_rings]
        if diff > 0:
            counts[k] += 1
            diff -= 1
        else:
            if counts[k] > 1:
                counts[k] -= 1
                diff += 1
        i += 1
    if counts[0] != 1:
        # force center single
        counts[0] = 1
        diff = N - counts.sum()
        order = np.argsort(-counts)
        i = 0
        while diff != 0 and i < 1000:
            k = order[i % n_rings]
            if diff > 0:
                counts[k] += 1
                diff -= 1
            elif counts[k] > 1:
                counts[k] -= 1
                diff += 1
            i += 1

    rings = [(int(counts[i]), float(base_radii[i])) for i in range(n_rings)]
    phase = 0.0

    # Anneal ring layout
    best_pts, best = _anneal_build(rings, phase, t0)

    # Local refine the best
    best_pts = _local_refine(best_pts, t0)
    final_score = _score(best_pts)

    if time.time() - t0 > _TIME_BUDGET - 2.0:
        return best_pts

    # A few more refinement passes if time remains
    for _ in range(3):
        if time.time() - t0 > _TIME_BUDGET - 2.0:
            break
        cand = _local_refine(best_pts, t0)
        if _score(cand) >= final_score:
            best_pts = cand
            final_score = _score(cand)

    return best_pts
